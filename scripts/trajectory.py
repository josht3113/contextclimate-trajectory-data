"""Reference kinematic trajectory model over the published tiles.

Used to validate the data against HYSPLIT and as the specification the browser
implementation must match. Method:
  - horizontal: bilinear in lat/lon on the tile grid (0.5° inside the North America
    box, 1° elsewhere); vertical: linear in ln(p) between pressure levels; time: linear
    between 3-hourly steps
  - vertical coordinate: pressure, dp/dt = omega (model vertical velocity)
  - integration: Heun (RK2) with a fixed step, positions advanced on the sphere
  - parcels that would go below terrain + 10 m are put back at terrain + 10 m;
    parcels above the top level (300 hPa) are held at it
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gfs_common as G  # noqa: E402

R_EARTH = 6371000.0
VI = {v: i for i, v in enumerate(G.VARS)}
LNP = np.log(np.array(G.LEVELS, np.float64))       # descending pressure -> ascending -lnp


class TileStore:
    def __init__(self, root: str, manifest: dict | None = None):
        self.root = root
        if manifest is None:
            with open(os.path.join(root, "manifest.json")) as f:
                manifest = json.load(f)
        self.m = manifest
        self.times = [G.parse_tkey(x["t"]) for x in manifest["times"]]
        self._cache = {}
        self._terrain = {}

    # ---- grid selection
    def grid_for(self, lat, lon):
        na = self.m["grids"].get("na")
        if na and na["lat"][0] <= lat <= na["lat"][1] and na["lon"][0] <= lon <= na["lon"][1]:
            return "na"
        return "nh"

    def _locate(self, gname, lat, lon):
        g = self.m["grids"][gname]
        T, res = g["tile"], g["res"]
        la0 = g["lat"][0] + math.floor((lat - g["lat"][0]) / T) * T
        lo0 = g["lon"][0] + math.floor((((lon - g["lon"][0]) % 360) if gname == "nh" else (lon - g["lon"][0])) / T) * T
        la0 = min(la0, g["lat"][1] - T)
        if gname == "na":
            lo0 = min(lo0, g["lon"][1] - T)
        fy = (lat - la0) / res
        dlon = (lon - lo0) % 360 if gname == "nh" else lon - lo0
        fx = dlon / res
        n = g["n"]
        iy = min(int(math.floor(fy)), n - 2)
        ix = min(int(math.floor(fx)), n - 2)
        return G.tile_name(la0, lo0), iy, ix, fy - iy, fx - ix

    def _tile(self, gname, t, tile):
        key = (gname, t, tile)
        if key not in self._cache:
            path = os.path.join(self.root, gname, G.tkey(t), tile + ".bin.gz")
            with open(path, "rb") as f:
                self._cache[key] = G.decode(f.read(), G.tile_shape(gname), G.cube_scales())
        return self._cache[key]

    def _terrain_tile(self, gname, tile):
        key = (gname, tile)
        if key not in self._terrain:
            path = os.path.join(self.root, gname, "terrain", tile + ".bin.gz")
            n = self.m["grids"][gname]["n"]
            with open(path, "rb") as f:
                self._terrain[key] = G.decode(f.read(), (n, n), 1.0)
        return self._terrain[key]

    # ---- sampling
    def column(self, lat, lon, t):
        """Bilinear/linear-in-time columns: returns array [var, level]."""
        gname = self.grid_for(lat, lon)
        tile, iy, ix, wy, wx = self._locate(gname, lat, lon)
        k = bisect.bisect_right(self.times, t) - 1
        if k < 0:
            raise LookupError("time outside data window")
        if self.times[k] == t:
            pairs = ((t, 1.0),)
        elif k >= len(self.times) - 1:
            raise LookupError("time outside data window")
        else:
            t0, t1 = self.times[k], self.times[k + 1]
            ft = (t - t0) / (t1 - t0)
            pairs = ((t0, 1 - ft), (t1, ft))
        out = 0
        for tt, wt in pairs:
            if wt == 0:
                continue
            a = self._tile(gname, tt, tile)
            c = (a[:, :, iy, ix] * (1 - wy) * (1 - wx) + a[:, :, iy, ix + 1] * (1 - wy) * wx
                 + a[:, :, iy + 1, ix] * wy * (1 - wx) + a[:, :, iy + 1, ix + 1] * wy * wx)
            out = out + wt * c
        return out

    def terrain(self, lat, lon):
        gname = self.grid_for(lat, lon)
        tile, iy, ix, wy, wx = self._locate(gname, lat, lon)
        a = self._terrain_tile(gname, tile)
        return float(a[iy, ix] * (1 - wy) * (1 - wx) + a[iy, ix + 1] * (1 - wy) * wx
                     + a[iy + 1, ix] * wy * (1 - wx) + a[iy + 1, ix + 1] * wy * wx)


# ---------------------------------------------------------------- vertical helpers
P_BOTTOM = 1080.0   # parcels may sit below the 1000 hPa surface (sea-level highs)


def interp_p(col_var, p):
    """Linear in ln p; holds the end value outside the level range (winds, omega)."""
    x = math.log(p)
    # LNP descends (1000 -> 300 hPa); use reversed arrays for np.interp
    return float(np.interp(-x, -LNP, col_var))


def height_at_p(z_col, p):
    """Geopotential height at p, extrapolated linearly in ln p below the lowest level."""
    x = math.log(p)
    if x > LNP[0]:
        slope = (z_col[1] - z_col[0]) / (LNP[1] - LNP[0])
        return float(z_col[0] + slope * (x - LNP[0]))
    return float(np.interp(-x, -LNP, z_col))


def p_at_height(z_col, z):
    """Pressure (hPa) where geopotential height equals z (inverse of height_at_p)."""
    if z < z_col[0]:
        slope = (z_col[1] - z_col[0]) / (LNP[1] - LNP[0])
        return float(min(math.exp(LNP[0] + (z - z_col[0]) / slope), P_BOTTOM))
    # z_col ascends with level index (1000 hPa lowest)
    return float(math.exp(np.interp(z, z_col, LNP)))


def move(lat, lon, u, v, dt):
    """Advance a point on the sphere by velocity (u east, v north) over dt seconds."""
    phi, lam = math.radians(lat), math.radians(lon)
    r = np.array([math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)])
    e = np.array([-math.sin(lam), math.cos(lam), 0.0])
    n = np.array([-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)])
    d = (u * e + v * n) * dt / R_EARTH
    ang = float(np.linalg.norm(d))
    if ang > 0:
        r = r * math.cos(ang) + d / ang * math.sin(ang)
    r /= np.linalg.norm(r)
    return math.degrees(math.asin(max(-1.0, min(1.0, r[2])))), math.degrees(math.atan2(r[1], r[0]))


# ---------------------------------------------------------------- integration
def back_trajectory(store: TileStore, lat0, lon0, t_arrive, agl, hours=120, dt_min=15,
                    p_top=None, clearance=10.0):
    p_top = p_top or min(G.LEVELS)
    dt = -dt_min * 60.0

    def state_vel(lat, lon, p, t):
        c = store.column(lat, lon, t)
        return (interp_p(c[VI["UGRD"]], p), interp_p(c[VI["VGRD"]], p),
                interp_p(c[VI["VVEL"]], p) / 100.0, c)          # omega -> hPa/s

    def fix_vertical(lat, lon, p, t):
        c = store.column(lat, lon, t)
        ground = store.terrain(lat, lon)
        z = height_at_p(c[VI["HGT"]], p)
        if z < ground + clearance:
            p = p_at_height(c[VI["HGT"]], ground + clearance)
        return min(max(p, p_top), P_BOTTOM), c, ground

    c = store.column(lat0, lon0, t_arrive)
    ground = store.terrain(lat0, lon0)
    p = p_at_height(c[VI["HGT"]], ground + agl)
    lat, lon, t = lat0, lon0, t_arrive
    pts = [{"h": 0, "lat": lat, "lon": lon, "agl": agl, "p": p}]
    steps_per_hour = int(60 / dt_min)
    end_reason = "complete"
    for k in range(1, hours * steps_per_hour + 1):
        try:
            u1, v1, w1, _ = state_vel(lat, lon, p, t)
            la_p, lo_p = move(lat, lon, u1, v1, dt)
            p_p, _, _ = fix_vertical(la_p, lo_p, p + w1 * dt, t + timedelta(seconds=dt))
            u2, v2, w2, _ = state_vel(la_p, lo_p, p_p, t + timedelta(seconds=dt))
            lat, lon = move(lat, lon, (u1 + u2) / 2, (v1 + v2) / 2, dt)
            t = t + timedelta(seconds=dt)
            p, c, ground = fix_vertical(lat, lon, p + (w1 + w2) / 2 * dt, t)
        except (LookupError, FileNotFoundError, KeyError):
            end_reason = "left data window or domain"
            break
        if k % steps_per_hour == 0:
            z = height_at_p(c[VI["HGT"]], p)
            pts.append({"h": -k // steps_per_hour, "lat": lat, "lon": lon,
                        "agl": z - ground, "p": p})
    return {"points": pts, "end": end_reason}
