"""Maintain the rolling window of GFS wind tiles.

Run after each GFS cycle. Builds any 3-hourly valid time that is missing or whose
source has been superseded, prunes times that left the window, and rewrites the
manifest. Safe to re-run: unchanged times are skipped.

Window: arrivals back to ARRIVAL_DAYS ago with trajectories up to TRAJ_HOURS long,
plus the latest cycle's forecast out to FORECAST_HOURS.

Backfill mode (--times ...) builds specific valid times into --out without pruning;
used for validation cases and event presets.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gfs_common as G  # noqa: E402

ARRIVAL_DAYS = 7
TRAJ_HOURS = 120
FORECAST_HOURS = 24
MARGIN_HOURS = 6


def floor6(t):
    return t.replace(minute=0, second=0, microsecond=0, hour=t.hour - t.hour % 6)


def latest_cycle(now):
    """Newest cycle whose f000 and f003 are both posted."""
    c = floor6(now)
    for _ in range(6):
        if G.fetch_idx(c, 3) is not None and G.fetch_idx(c, 0) is not None:
            return c
        c -= timedelta(hours=6)
    raise RuntimeError("no complete GFS cycle found in the last 36 h")


def source_for(valid):
    """Analysis-window source for a valid time: f000 on cycle hours, else f003 of prior cycle."""
    if valid.hour % 6 == 0:
        return valid, 0
    return valid - timedelta(hours=3), 3


def desired_sources(now):
    L = latest_cycle(now)
    start = L - timedelta(days=ARRIVAL_DAYS, hours=TRAJ_HOURS + MARGIN_HOURS)
    want = {}
    t = start - timedelta(hours=start.hour % 3)
    while t <= L + timedelta(hours=3):
        want[t] = (source_for(t), "analysis")
        t += timedelta(hours=3)
    for fh in range(6, FORECAST_HOURS + 1, 3):
        if G.fetch_idx(L, fh) is None:
            break
        want[L + timedelta(hours=fh)] = ((L, fh), "forecast")
    return L, want


def write_time(out, valid, cube, grids):
    scales = G.cube_scales()
    k = G.tkey(valid)
    for gname in grids:
        d = os.path.join(out, gname, k)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for la, lo in G.grid_tiles(gname):
            blob = G.encode(G.cut_tile(cube, gname, la, lo), scales)
            with open(os.path.join(d, G.tile_name(la, lo) + ".bin.gz"), "wb") as f:
                f.write(blob)


def write_terrain(out, terrain, grids):
    for gname in grids:
        d = os.path.join(out, gname, "terrain")
        os.makedirs(d, exist_ok=True)
        for la, lo in G.grid_tiles(gname):
            blob = G.encode(G.cut_tile(terrain, gname, la, lo), 1.0)
            with open(os.path.join(d, G.tile_name(la, lo) + ".bin.gz"), "wb") as f:
                f.write(blob)


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


def manifest(out, state, latest, grids):
    times = []
    for k in sorted(state["valid"]):
        src = state["valid"][k]
        times.append({"t": k, "cycle": src["cycle"], "fh": src["fh"], "kind": src["kind"]})
    gdesc = {}
    for gname in grids:
        g = dict(G.GRIDS[gname])
        g["n"] = int(round(g["tile"] / g["res"])) + 1
        g["tiles"] = [G.tile_name(la, lo) for la, lo in G.grid_tiles(gname)]
        gdesc[gname] = g
    return {
        "version": 1,
        "generated": G.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "source": "NOAA GFS 0.5° pressure-level analyses (f000) and 3-h forecasts (f003); "
                  "latest cycle forecast to +%d h. NOAA Open Data Dissemination (AWS)." % FORECAST_HOURS,
        "latestCycle": G.tkey(latest) if latest else None,
        "levels": G.LEVELS,
        "vars": G.VARS,
        "units": G.UNITS,
        "scale": G.SCALE,
        "layout": "int16 LE [var][level][lat][lon], lat/lon ascending from tile SW corner, "
                  "edges included; byte-shuffled (low bytes then high bytes); gzip",
        "grids": gdesc,
        "terrain": {"path": "{grid}/terrain/{tile}.bin.gz", "units": "m", "scale": 1},
        "tilePath": "{grid}/{t}/{tile}.bin.gz",
        "times": times,
        "bytes": dir_size(out),
    }


def load_state(out):
    import json
    p = os.path.join(out, "state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"valid": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--times", nargs="*", help="backfill these valid times (YYYYMMDDHH) only")
    ap.add_argument("--grids", nargs="*", default=list(G.GRIDS))
    ap.add_argument("--budget-min", type=float, default=300, help="stop building after this many minutes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    state = load_state(args.out)
    t_start = time.time()

    if args.times:
        latest = None
        want = {G.parse_tkey(k): (source_for(G.parse_tkey(k)), "archive") for k in args.times}
    else:
        latest, want = desired_sources(G.utcnow())
        print(f"latest cycle {G.tkey(latest)}; window {G.tkey(min(want))} .. {G.tkey(max(want))} ({len(want)} times)")

    built = skipped = missing = 0
    terrain_done = all(os.path.isdir(os.path.join(args.out, g, "terrain")) for g in args.grids)
    # newest first, so a cold start publishes the most useful data first if it runs out of budget
    for valid in sorted(want, reverse=True):
        (cycle, fh), kind = want[valid]
        k = G.tkey(valid)
        src = {"cycle": G.tkey(cycle), "fh": fh, "kind": kind}
        if state["valid"].get(k) == src and all(os.path.isdir(os.path.join(args.out, g, k)) for g in args.grids):
            skipped += 1
            continue
        if (time.time() - t_start) / 60 > args.budget_min:
            print("time budget reached; remaining times left for the next run")
            break
        t0 = time.time()
        res = G.fetch_fields(cycle, fh)
        if res is None:
            print(f"  {k}: source gfs {G.tkey(cycle)} f{fh:03d} not available, skipped")
            missing += 1
            continue
        cube, terrain = res
        if not np.isfinite(cube).all():
            raise RuntimeError(f"non-finite values in {k}")
        write_time(args.out, valid, cube, args.grids)
        if not terrain_done:
            write_terrain(args.out, terrain, args.grids)
            terrain_done = True
        state["valid"][k] = src
        G.write_json(os.path.join(args.out, "state.json"), state)
        built += 1
        print(f"  {k} <- gfs {G.tkey(cycle)} f{fh:03d} [{kind}] {time.time() - t0:.1f}s")

    pruned = 0
    if not args.times:
        keep = {G.tkey(v) for v in want}
        for k in list(state["valid"]):
            if k not in keep:
                for g in args.grids:
                    shutil.rmtree(os.path.join(args.out, g, k), ignore_errors=True)
                del state["valid"][k]
                pruned += 1
        # drop stray directories that are not in the state (e.g. from an interrupted run)
        for g in args.grids:
            gd = os.path.join(args.out, g)
            if os.path.isdir(gd):
                for k in os.listdir(gd):
                    if k != "terrain" and k not in state["valid"]:
                        shutil.rmtree(os.path.join(gd, k), ignore_errors=True)
        G.write_json(os.path.join(args.out, "state.json"), state)

    if not state["valid"]:
        raise RuntimeError("no valid times built; refusing to publish an empty manifest")
    m = manifest(args.out, state, latest, args.grids)
    G.write_json(os.path.join(args.out, "manifest.json"), m)
    print(f"built {built}, skipped {skipped}, missing {missing}, pruned {pruned}; "
          f"{len(m['times'])} times, {m['bytes'] / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
