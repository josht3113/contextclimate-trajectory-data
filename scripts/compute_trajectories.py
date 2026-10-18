"""Compute back trajectories for every city and arrival time, and write the site files.

Arrival times: every 3-hourly data time that has 120 h of data behind it (≈7 days of
analyses plus the latest cycle's forecast). An arrival is recomputed when the data
sources behind it change (a forecast replaced by an analysis), otherwise reused.

Site layout (all binary files: int16 little-endian, gzip):
  manifest.json            arrivals, heights, ensemble, encoding
  cities.json              arrival points (from cities.py)
  traj/{city}.bin.gz       centre paths  [arrival][height][hour 0..120][lat, lon, agl]
  ens/{city}/{t}.bin.gz    members       [height][member 1..M-1][every 3 h][lat, lon, agl]
Series are delta-encoded along time (first value absolute); lat/lon in 0.01°, longitude
unwrapped (may run past ±180), agl in metres; -32768 marks "no data" (parcel left the
domain) and holds for the rest of the series.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gfs_common as G  # noqa: E402
import batch_traj as BT  # noqa: E402

HOURS = 120
HEIGHTS = [500, 1500, 3000]
MEMBER_EVERY = 3
NODATA = -32768


# ---------------------------------------------------------------- encoding
def encode_series(arr):
    """arr (..., T, 3) float with NaN -> int16 delta-encoded along T."""
    a = np.array(arr, np.float64)
    a[..., 1] = np.degrees(np.unwrap(np.radians(np.nan_to_num(a[..., 1], nan=0.0)), axis=-1)) \
        * np.where(np.isnan(a[..., 1]), np.nan, 1.0)
    q = np.empty(a.shape, np.int64)
    q[..., 0] = np.rint(a[..., 0] * 100)
    q[..., 1] = np.rint(a[..., 1] * 100)
    q[..., 2] = np.rint(np.clip(a[..., 2], -500, 30000))
    nan = np.isnan(a).any(axis=-1)
    d = np.diff(q, axis=-2, prepend=0)
    d = np.clip(d, -32767, 32767)
    # a series stays "no data" once it starts
    nan = np.maximum.accumulate(nan, axis=-1)
    d[nan] = NODATA
    return d.astype("<i2")


def decode_series(d):
    d = d.astype(np.int64)
    nan = (d == NODATA).all(axis=-1)
    d = np.where(d == NODATA, 0, d)
    q = np.cumsum(d, axis=-2).astype(np.float64)
    q[..., 0] /= 100
    q[..., 1] /= 100
    q[nan] = np.nan
    return q


def write_bin(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(gzip.compress(np.ascontiguousarray(arr).tobytes(), compresslevel=9, mtime=0))


# ---------------------------------------------------------------- worker
_model = None


def _init(data_dir):
    global _model
    with open(os.path.join(data_dir, "manifest.json")) as f:
        m = json.load(f)
    _model = BT.Model(BT.Grids(data_dir, m))


def _compute(args):
    key, lats, lons = args
    t = G.parse_tkey(key)
    S = len(lats)
    LA = np.repeat(lats, len(HEIGHTS))
    LO = np.repeat(lons, len(HEIGHTS))
    H = np.tile(np.array(HEIGHTS, float), S)
    t0 = time.time()
    r = BT.run(_model, LA, LO, H, t, hours=HOURS, member_record_every=MEMBER_EVERY)
    centre = r["centre"].reshape(S, len(HEIGHTS), HOURS + 1, 3).astype(np.float32)
    members = r["members"].reshape(S, len(HEIGHTS), len(BT.ENSEMBLE) - 1, HOURS // MEMBER_EVERY + 1, 3)
    return key, centre, members.astype(np.float32), time.time() - t0


# ---------------------------------------------------------------- driver
def signature(data_manifest, t):
    srcs = {x["t"]: f'{x["cycle"]}f{x["fh"]}' for x in data_manifest["times"]}
    lo = t - timedelta(hours=HOURS + 3)
    keys = [k for k in sorted(srcs) if lo <= G.parse_tkey(k) <= t]
    s = "|".join(f"{k}:{srcs[k]}" for k in keys) + f"|cities:{data_manifest.get('_cities_sig', '')}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cities", default="data/site/cities.json")
    ap.add_argument("--site", default="data/site")
    ap.add_argument("--store", default="data/trajstore")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--budget-min", type=float, default=200)
    args = ap.parse_args()

    with open(os.path.join(args.data, "manifest.json")) as f:
        dm = json.load(f)
    with open(args.cities) as f:
        cities = json.load(f)["cities"]
    ids = [c["id"] for c in cities]
    lats = np.array([c["lat"] for c in cities])
    lons = np.array([c["lon"] for c in cities])
    dm["_cities_sig"] = hashlib.sha1(json.dumps([ids, lats.tolist(), lons.tolist()]).encode()).hexdigest()[:12]

    times = [G.parse_tkey(x["t"]) for x in dm["times"]]
    kinds = {x["t"]: x["kind"] for x in dm["times"]}
    first = times[0]
    arrivals = [t for t in times if t - timedelta(hours=HOURS) >= first]
    if not arrivals:
        raise RuntimeError("no arrival time has 120 h of data behind it yet")

    os.makedirs(args.store, exist_ok=True)
    state_p = os.path.join(args.store, "state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {}
    todo = []
    for t in sorted(arrivals, reverse=True):
        k = G.tkey(t)
        sig = signature(dm, t)
        if state.get(k) != sig or not os.path.exists(os.path.join(args.store, k + ".npz")):
            todo.append((k, sig))
    print(f"{len(arrivals)} arrival times ({G.tkey(arrivals[0])}..{G.tkey(arrivals[-1])}), "
          f"{len(todo)} to compute, {len(ids)} cities x {len(HEIGHTS)} heights x {len(BT.ENSEMBLE)} members")

    t_start = time.time()
    sigs = dict(todo)
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.data,)) as ex:
            futs = []
            for k, _ in todo:
                futs.append(ex.submit(_compute, (k, lats, lons)))
            for fu in as_completed(futs):
                k, centre, members, sec = fu.result()
                np.savez(os.path.join(args.store, k + ".npz"), centre=centre,
                         members=members.astype(np.float16), ids=np.array(ids))
                state[k] = sigs[k]
                with open(state_p, "w") as f:
                    json.dump(state, f)
                print(f"  {k} computed in {sec:.0f}s")
                if (time.time() - t_start) / 60 > args.budget_min:
                    print("time budget reached; cancelling the rest")
                    for x in futs:
                        x.cancel()
                    break

    # drop expired arrivals from the store
    keep = {G.tkey(t) for t in arrivals}
    for fn in os.listdir(args.store):
        if fn.endswith(".npz") and fn[:-4] not in keep:
            os.remove(os.path.join(args.store, fn))
            state.pop(fn[:-4], None)
    with open(state_p, "w") as f:
        json.dump(state, f)

    # ---------------- write the site from the store
    done = [G.tkey(t) for t in sorted(arrivals) if os.path.exists(os.path.join(args.store, G.tkey(t) + ".npz"))
            and state.get(G.tkey(t))]
    if not done:
        raise RuntimeError("no computed arrivals; refusing to publish")
    shutil.rmtree(os.path.join(args.site, "traj"), ignore_errors=True)
    shutil.rmtree(os.path.join(args.site, "ens"), ignore_errors=True)
    n_cities = len(ids)
    centre_all = np.full((n_cities, len(done), len(HEIGHTS), HOURS + 1, 3), np.nan, np.float32)
    for ai, k in enumerate(done):
        z = np.load(os.path.join(args.store, k + ".npz"))
        sid = list(z["ids"])
        idx = [sid.index(c) if c in sid else -1 for c in ids]
        for ci, j in enumerate(idx):
            if j < 0:
                continue
            centre_all[ci, ai] = z["centre"][j]
            write_bin(os.path.join(args.site, "ens", ids[ci], k + ".bin.gz"),
                      encode_series(z["members"][j].astype(np.float32)))
    for ci, cid in enumerate(ids):
        write_bin(os.path.join(args.site, "traj", cid + ".bin.gz"), encode_series(centre_all[ci]))

    manifest = {
        "version": 1,
        "generated": G.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "latestCycle": dm.get("latestCycle"),
        "arrivals": [{"t": k, "kind": kinds[k]} for k in done],
        "heights": HEIGHTS,
        "hours": HOURS,
        "memberEvery": MEMBER_EVERY,
        "ensemble": {"members": BT.ENSEMBLE, "note": "met offsets: (lat cells, lon cells, hPa); member 0 unperturbed"},
        "model": "kinematic, pressure coordinate with GFS omega; Heun 15-min steps; GFS 0.5° (North America) / 1°",
        "source": dm["source"],
        "encoding": {"traj": "[arrival][height][hour][lat,lon,agl]", "ens": "[height][member][step][lat,lon,agl]",
                     "dtype": "int16 LE, gzip", "delta": "along time; first value absolute",
                     "scale": {"lat": 0.01, "lon": 0.01, "agl": 1}, "nodata": NODATA,
                     "lonNote": "unwrapped; may run past ±180"},
    }
    G.write_json(os.path.join(args.site, "manifest.json"), manifest)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.site) for f in fs)
    print(f"site: {len(done)} arrivals, {n_cities} cities, {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
