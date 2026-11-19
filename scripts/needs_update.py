"""Decide whether the hourly trigger should run the full update.

Runs in seconds with the standard library only. Prints a reason and writes
`run=true|false` to $GITHUB_OUTPUT (when set).

Run when any of:
  - a GFS cycle newer than the published manifest's latestCycle has its f024
    posted on AWS (the window's whole forecast is available);
  - the published manifest is missing or unreadable;
  - the published manifest is older than STALE_HOURS (a safety net, so the
    rolling window keeps moving even if NOAA posting is unusual).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
SITE = "https://josht3113.github.io/contextclimate-trajectory-data/"
FORECAST_HOURS = 24          # must match build_tiles.FORECAST_HOURS
STALE_HOURS = 9
UA = {"User-Agent": "contextclimate-trajectory-data (github.com/josht3113)"}


def idx_posted(cycle: datetime, fh: int) -> bool:
    d, h = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    url = f"{BUCKET}/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p50.f{fh:03d}.idx"
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return False
        except (urllib.error.URLError, TimeoutError):
            pass
    return False


def newest_complete_cycle(now: datetime) -> datetime | None:
    c = now.replace(minute=0, second=0, microsecond=0, hour=now.hour - now.hour % 6)
    for _ in range(5):
        if idx_posted(c, FORECAST_HOURS):
            return c
        c -= timedelta(hours=6)
    return None


def published_manifest() -> dict | None:
    req = urllib.request.Request(SITE + "manifest.json", headers={**UA, "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        print(f"manifest unreadable: {e}")
        return None


def decide(now: datetime) -> tuple[bool, str]:
    m = published_manifest()
    if not m or not m.get("latestCycle") or not m.get("generated"):
        return True, "no usable published manifest"
    pub = datetime.strptime(m["latestCycle"], "%Y%m%d%H")
    gen = datetime.strptime(m["generated"], "%Y-%m-%dT%H:%MZ")
    age = (now - gen).total_seconds() / 3600
    c = newest_complete_cycle(now)
    if c and c > pub:
        return True, f"new cycle {c:%Y%m%d%H} (published {m['latestCycle']})"
    if age > STALE_HOURS:
        return True, f"published data is {age:.1f} h old"
    return False, f"up to date: published {m['latestCycle']}, newest complete {c:%Y%m%d%H}" if c else \
        f"no newer complete cycle on AWS; published {m['latestCycle']}, {age:.1f} h old"


def main() -> int:
    force = os.environ.get("FORCE", "").lower() == "true"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    run, why = (True, "manual run") if force else decide(now)
    print(("RUN: " if run else "skip: ") + why)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if run else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
