"""Build the trajectory city list from ContextClimate's shared station roster.

Stations within MERGE_KM of a higher-priority station are folded into it (kept as
search aliases), so each metro gets one arrival point. Priority: KEEP and PREFER
lists first, then stations named "Intl", then roster order. KEEP stations are never
folded into another station.

Writes cities.json and CITIES.md (a review table of every merge).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import requests

ROSTER_URL = "https://raw.githubusercontent.com/josht3113/context-climate/main/public/station-roster.js"
HERE = os.path.dirname(__file__)
FALLBACK = os.path.join(HERE, "station-roster.snapshot.js")

MERGE_KM = 70

# Always kept as their own arrival point, even near a bigger city.
KEEP = ["KISP"]

# Representative station for a metro when several roster stations fall together.
PREFER = [
    "KNYC", "KLAX", "KORD", "KSFO", "KBOS", "KMIA", "KIAH", "KDFW", "KPHL", "KATL",
    "KSEA", "KDEN", "KDCA", "KDTW", "KMSP", "KPHX", "KSAN", "KSTL", "KPIT", "KCLE",
    "KMCI", "KMSY", "KLAS", "KSLC", "KPDX", "KCVG", "KIND", "KCMH", "KBNA", "KMEM",
    "KCLT", "KRDU", "KTPA", "KMCO", "KJAX", "KSAT", "KAUS", "KOKC", "KMKE", "KBUF",
    "PANC", "PHNL", "TJSJ",
]

ROW = re.compile(r'\["([A-Z0-9]{3,4})","([^"]+)","([A-Z]{2})",(-?[\d.]+),(-?[\d.]+),"([A-Z]{2})"\]')


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def load_roster():
    try:
        r = requests.get(ROSTER_URL, timeout=30)
        r.raise_for_status()
        text, src = r.text, ROSTER_URL
        with open(FALLBACK, "w") as f:
            f.write(text)
    except Exception as e:  # network trouble: use the last good copy
        print(f"roster fetch failed ({e}); using snapshot", file=sys.stderr)
        with open(FALLBACK) as f:
            text, src = f.read(), "snapshot"
    rows = [{"id": m[0], "name": m[1], "st": m[2], "lat": float(m[3]), "lon": float(m[4]), "tz": m[5]}
            for m in ROW.findall(text)]
    if len(rows) < 100:
        raise RuntimeError(f"roster parse found only {len(rows)} stations")
    return rows, src


def build(rows):
    order = {s["id"]: i for i, s in enumerate(rows)}

    def rank(s):
        if s["id"] in KEEP:
            return (0, KEEP.index(s["id"]))
        if s["id"] in PREFER:
            return (1, PREFER.index(s["id"]))
        return (2 if "Intl" in s["name"] else 3, order[s["id"]])

    reps = []
    for s in sorted(rows, key=rank):
        # KEEP stations are added unconditionally and never block another station
        if s["id"] in KEEP or all(km(s, r) > MERGE_KM for r in reps if r["id"] not in KEEP):
            reps.append(dict(s, aliases=[]))
    for s in rows:
        if any(r["id"] == s["id"] for r in reps):
            continue
        nearest = min(reps, key=lambda r: km(s, r))
        nearest["aliases"].append({"id": s["id"], "name": s["name"], "st": s["st"],
                                   "km": round(km(s, nearest))})
    reps.sort(key=lambda r: (r["st"], r["name"]))
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cities.json")
    ap.add_argument("--review", default=os.path.join(HERE, "..", "CITIES.md"))
    args = ap.parse_args()
    rows, src = load_roster()
    reps = build(rows)
    out = [{"id": r["id"], "name": r["name"], "st": r["st"], "lat": r["lat"], "lon": r["lon"],
            "tz": r["tz"], "aliases": [a["id"] for a in r["aliases"]],
            "aliasNames": [a["name"] for a in r["aliases"]]} for r in reps]
    with open(args.out, "w") as f:
        json.dump({"source": src, "mergeKm": MERGE_KM, "cities": out}, f, separators=(",", ":"))
    lines = [f"# Trajectory cities", "",
             f"{len(rows)} roster stations → {len(reps)} arrival points (stations within {MERGE_KM} km folded into "
             "the representative; folded stations stay searchable).", "",
             f"Edit `KEEP` / `PREFER` in `scripts/cities.py` to change representatives.", "",
             "| Arrival point | State | Folded in (distance) |", "|---|---|---|"]
    for r in reps:
        folded = ", ".join(f"{a['name']} ({a['km']} km)" for a in r["aliases"])
        lines.append(f"| {r['name']} ({r['id']}) | {r['st']} | {folded} |")
    with open(args.review, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(rows)} stations -> {len(reps)} cities ({src})")


if __name__ == "__main__":
    main()
