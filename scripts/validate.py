"""Backfill the validation cases from the GFS archive, run the reference trajectory
model, and compare against HYSPLIT (READY) endpoints.

Writes a markdown report to stdout (and to $GITHUB_STEP_SUMMARY when set) and a JSON
report to --report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(__file__))
import gfs_common as G  # noqa: E402
from trajectory import TileStore, back_trajectory  # noqa: E402

HERE = os.path.dirname(__file__)


def gc_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="validation_data")
    ap.add_argument("--report", default="validation.json")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args()

    with open(os.path.join(HERE, "ready_reference.json")) as f:
        ref = json.load(f)["cases"]
    cases = args.cases or list(ref)

    report = {"cases": {}}
    lines = ["## Validation vs HYSPLIT (READY, GFS 0.25°)", "",
             "Separation between our trajectory and HYSPLIT's at each reference hour (km), "
             "and as a share of HYSPLIT's straight-line distance from the arrival point.", ""]
    for name in cases:
        c = ref[name]
        t0 = G.parse_tkey(c["arrival"])
        need = [t0 - timedelta(hours=h) for h in range(0, args.hours + 4, 3)]
        work = os.path.join(args.work, name)
        subprocess.run([sys.executable, os.path.join(HERE, "build_tiles.py"), "--out", work,
                        "--grids", "na", "--times", *[G.tkey(t) for t in need]], check=True)
        store = TileStore(work)
        lines += [f"### {c['label']} — arrival {c['arrival']}Z", "",
                  "| Height | " + " | ".join(f"{p[0]} h" for p in c["points"]["500"]) + " |",
                  "|---|" + "---|" * len(c["points"]["500"])]
        report["cases"][name] = {}
        for h, refpts in c["points"].items():
            tr = back_trajectory(store, c["lat"], c["lon"], t0, float(h), hours=args.hours)
            ours = {p["h"]: p for p in tr["points"]}
            cells, rows = [], []
            for age, la, lo, agl in refpts:
                if age not in ours:
                    cells.append("—")
                    continue
                o = ours[age]
                sep = gc_km((o["lat"], o["lon"]), (la, lo))
                dist = gc_km((c["lat"], c["lon"]), (la, lo))
                pct = 100 * sep / dist if dist > 0 else 0
                cells.append(f"{sep:.0f} ({pct:.0f}%)")
                rows.append({"age": age, "sep_km": round(sep, 1), "pct": round(pct, 1),
                             "ours": [round(o["lat"], 3), round(o["lon"], 3), round(o["agl"])],
                             "hysplit": [la, lo, agl]})
            # pad if this height has fewer reference points
            cells += ["—"] * (len(c["points"]["500"]) - len(cells))
            lines.append(f"| {h} m | " + " | ".join(cells) + " |")
            report["cases"][name][h] = {"end": tr["end"], "rows": rows}
            if os.environ.get("GITHUB_ACTIONS"):
                detail = "; ".join(f"{r['age']}h {r['sep_km']:.0f}km ({r['pct']:.0f}%) ours {r['ours']} hy {r['hysplit']}"
                                   for r in rows)
                print(f"::notice title=validate {name} {h}m::{detail} [{tr['end']}]")
        lines.append("")

    md = "\n".join(lines)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
