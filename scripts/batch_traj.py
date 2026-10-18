"""Vectorised kinematic back-trajectory model with a HYSPLIT-style ensemble.

Same numerics as trajectory.py (the readable reference), applied to many parcels at
once: pressure vertical coordinate with dp/dt = omega; bilinear horizontal, linear in
ln p vertical, linear in time; Heun steps on the sphere; parcels held >= terrain + 10 m
and between 300 and 1080 hPa.

Ensemble: each member samples the meteorology from a grid offset by one grid cell
horizontally and/or a pressure offset vertically, while the parcel itself starts at the
arrival point (the scheme HYSPLIT's trajectory ensemble uses). Member 0 is unperturbed.
"""
from __future__ import annotations

import bisect
import gzip
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gfs_common as G  # noqa: E402

R_EARTH = 6371000.0
NV, NL = len(G.VARS), len(G.LEVELS)
IU, IV, IW, IZ = (G.VARS.index(v) for v in ("UGRD", "VGRD", "VVEL", "HGT"))
LNP = np.log(np.array(G.LEVELS, np.float64))      # descending
XL = -LNP                                          # ascending
P_TOP, P_BOTTOM, CLEARANCE = float(min(G.LEVELS)), 1080.0, 10.0
SCALES = np.repeat(np.array([G.SCALE[v] for v in G.VARS], np.float32), NL)   # (68,) var-major

# (dy cells, dx cells, dp hPa): unperturbed + 4 horizontal + 2 vertical + 8 combined
ENSEMBLE = [(0, 0, 0)] + [(dy, dx, 0) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))] \
    + [(0, 0, dp) for dp in (10, -10)] \
    + [(dy, dx, dp) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) for dp in (10, -10)]


class Grids:
    """Full-domain arrays assembled from tiles: na (0.5°) and nh (1°), each (ny, nx, 68) int16."""

    def __init__(self, root: str, manifest: dict):
        self.root = root
        self.m = manifest
        self.times = [G.parse_tkey(x["t"]) for x in manifest["times"]]
        self.na = manifest["grids"]["na"]
        self.nh = manifest["grids"]["nh"]
        self.na_ny = int(round((self.na["lat"][1] - self.na["lat"][0]) / self.na["res"])) + 1
        self.na_nx = int(round((self.na["lon"][1] - self.na["lon"][0]) / self.na["res"])) + 1
        self.nh_ny = int(round((self.nh["lat"][1] - self.nh["lat"][0]) / self.nh["res"])) + 1
        self.nh_nx = 360
        self.terrain_na, self.terrain_nh = self._assemble("terrain", terrain=True)
        self._cache = {}

    def _read(self, grid, folder, tile, terrain):
        n = self.m["grids"][grid]["n"]
        with open(os.path.join(self.root, grid, folder, tile + ".bin.gz"), "rb") as f:
            raw = gzip.decompress(f.read())
        half = len(raw) // 2
        b = np.empty((half, 2), np.uint8)
        b[:, 0] = np.frombuffer(raw[:half], np.uint8)
        b[:, 1] = np.frombuffer(raw[half:], np.uint8)
        a = b.reshape(-1).view("<i2")
        if terrain:
            return a.reshape(n, n)
        return a.reshape(NV * NL, n, n).transpose(1, 2, 0)      # (n, n, 68)

    def _assemble(self, folder, terrain=False):
        na, nh = self.na, self.nh
        shape_tail = () if terrain else (NV * NL,)
        A = np.zeros((self.na_ny, self.na_nx) + shape_tail, np.int16)
        T = na["tile"]
        step = int(round(T / na["res"]))
        for la, lo in G.grid_tiles("na"):
            a = self._read("na", folder, G.tile_name(la, lo), terrain)
            iy = int(round((la - na["lat"][0]) / na["res"]))
            ix = int(round((lo - na["lon"][0]) / na["res"]))
            A[iy:iy + step + 1, ix:ix + step + 1] = a
        B = np.zeros((self.nh_ny, self.nh_nx) + shape_tail, np.int16)
        stepn = int(round(nh["tile"] / nh["res"]))
        for la, lo in G.grid_tiles("nh"):
            a = self._read("nh", folder, G.tile_name(la, lo), terrain)
            iy = int(round((la - nh["lat"][0]) / nh["res"]))
            ix = int(round((lo - nh["lon"][0]) / nh["res"]))
            cols = (ix + np.arange(stepn + 1)) % 360
            B[iy:iy + stepn + 1, cols] = a
        # fill the nh cells inside the na box (tiles not written) from na at 1°
        k = int(round(nh["res"] / na["res"]))
        sub = A[::k, ::k]
        iy0 = int(round((na["lat"][0] - nh["lat"][0]) / nh["res"]))
        ix0 = int(round((na["lon"][0] - nh["lon"][0]) / nh["res"]))
        cols = (ix0 + np.arange(sub.shape[1])) % 360
        B[iy0:iy0 + sub.shape[0], cols] = sub
        return A, B

    def fields(self, t):
        key = G.tkey(t)
        if key not in self._cache:
            if len(self._cache) > 48:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = self._assemble(key)
        return self._cache[key]

    def bracket(self, t):
        k = bisect.bisect_right(self.times, t) - 1
        if k < 0:
            return None
        if self.times[k] == t:
            return ((self.times[k], 1.0),)
        if k >= len(self.times) - 1:
            return None
        t0, t1 = self.times[k], self.times[k + 1]
        f = (t - t0) / (t1 - t0)
        return ((t0, 1.0 - f), (t1, f))


def _locate(g, lat, lon, oy, ox):
    """Fractional indices on grid g ('na' dict or 'nh' dict) with member offsets in cells."""
    res = g["res"]
    fy = (lat - g["lat"][0]) / res + oy
    if g["lon"][1] - g["lon"][0] >= 360:
        fx = np.mod(lon - g["lon"][0], 360.0) / res + ox
    else:
        fx = (lon - g["lon"][0]) / res + ox
    return fy, fx


def _bilinear(A, fy, fx, wrap):
    ny, nx = A.shape[0], A.shape[1]
    fy = np.clip(fy, 0, ny - 1 - 1e-9)
    iy = np.minimum(np.floor(fy).astype(np.int64), ny - 2)
    wy = (fy - iy)
    if wrap:
        fx = np.mod(fx, nx)
        ix = np.floor(fx).astype(np.int64) % nx
        ix1 = (ix + 1) % nx
    else:
        fx = np.clip(fx, 0, nx - 1 - 1e-9)
        ix = np.minimum(np.floor(fx).astype(np.int64), nx - 2)
        ix1 = ix + 1
    wx = (fx - np.floor(fx)) if wrap else (fx - ix)
    if A.ndim == 3:
        wy, wx = wy[:, None], wx[:, None]
    return ((A[iy, ix] * (1 - wy) + A[iy + 1, ix] * wy) * (1 - wx)
            + (A[iy, ix1] * (1 - wy) + A[iy + 1, ix1] * wy) * wx)


class Model:
    def __init__(self, grids: Grids):
        self.g = grids

    def in_na(self, lat, lon):
        na = self.g.na
        return (lat >= na["lat"][0]) & (lat <= na["lat"][1]) & (lon >= na["lon"][0]) & (lon <= na["lon"][1])

    def column(self, lat, lon, t, oy, ox):
        """(N, 68) float columns [var-major], or None if t is outside the data window."""
        br = self.g.bracket(t)
        if br is None:
            return None
        m = self.in_na(lat, lon)
        out = np.zeros((lat.size, NV * NL), np.float32)
        for tt, wt in br:
            if wt == 0:
                continue
            A, B = self.g.fields(tt)
            if m.any():
                fy, fx = _locate(self.g.na, lat[m], lon[m], oy[m], ox[m])
                out[m] += wt * _bilinear(A, fy, fx, wrap=False)
            if (~m).any():
                fy, fx = _locate(self.g.nh, lat[~m], lon[~m], oy[~m], ox[~m])
                out[~m] += wt * _bilinear(B, fy, fx, wrap=True)
        return out * SCALES

    def terrain(self, lat, lon):
        m = self.in_na(lat, lon)
        z = np.empty(lat.size, np.float32)
        zero = np.zeros(lat.size)
        if m.any():
            fy, fx = _locate(self.g.na, lat[m], lon[m], zero[m], zero[m])
            z[m] = _bilinear(self.g.terrain_na, fy, fx, wrap=False)
        if (~m).any():
            fy, fx = _locate(self.g.nh, lat[~m], lon[~m], zero[~m], zero[~m])
            z[~m] = _bilinear(self.g.terrain_nh, fy, fx, wrap=True)
        return z


# ---------------------------------------------------------------- vertical helpers (vectorised)
def _lev_weights(p):
    x = -np.log(p)
    k = np.clip(np.searchsorted(XL, x) - 1, 0, NL - 2)
    w = np.clip((x - XL[k]) / (XL[k + 1] - XL[k]), 0.0, 1.0)
    return k, w


def interp_var(col, vi, p):
    k, w = _lev_weights(p)
    c = col[:, vi * NL:(vi + 1) * NL]
    r = np.arange(p.size)
    return c[r, k] * (1 - w) + c[r, k + 1] * w


def height_at_p(col, p):
    z = col[:, IZ * NL:(IZ + 1) * NL]
    lnp = np.log(p)
    k, w = _lev_weights(p)
    r = np.arange(p.size)
    inside = z[r, k] * (1 - w) + z[r, k + 1] * w
    slope = (z[:, 1] - z[:, 0]) / (LNP[1] - LNP[0])
    below = z[:, 0] + slope * (lnp - LNP[0])
    return np.where(lnp > LNP[0], below, inside)


def p_at_height(col, zt):
    z = col[:, IZ * NL:(IZ + 1) * NL]
    r = np.arange(zt.size)
    k = np.clip((z < zt[:, None]).sum(axis=1) - 1, 0, NL - 2)
    z0, z1 = z[r, k], z[r, k + 1]
    w = np.clip((zt - z0) / np.where(z1 == z0, 1, z1 - z0), 0.0, 1.0)
    lnp = LNP[k] * (1 - w) + LNP[k + 1] * w
    slope = (z[:, 1] - z[:, 0]) / (LNP[1] - LNP[0])
    lnp_below = LNP[0] + (zt - z[:, 0]) / slope
    lnp = np.where(zt < z[:, 0], lnp_below, lnp)
    return np.minimum(np.exp(lnp), P_BOTTOM)


def move(lat, lon, u, v, dt):
    phi, lam = np.radians(lat), np.radians(lon)
    cp, sp, cl, sl = np.cos(phi), np.sin(phi), np.cos(lam), np.sin(lam)
    r = np.stack([cp * cl, cp * sl, sp], 1)
    e = np.stack([-sl, cl, np.zeros_like(sl)], 1)
    n = np.stack([-sp * cl, -sp * sl, cp], 1)
    d = (u[:, None] * e + v[:, None] * n) * (dt / R_EARTH)
    ang = np.linalg.norm(d, axis=1)
    safe = np.where(ang > 0, ang, 1.0)
    r = r * np.cos(ang)[:, None] + d / safe[:, None] * np.sin(ang)[:, None]
    r /= np.linalg.norm(r, axis=1)[:, None]
    return np.degrees(np.arcsin(np.clip(r[:, 2], -1, 1))), np.degrees(np.arctan2(r[:, 1], r[:, 0]))


# ---------------------------------------------------------------- integration
def run(model: Model, lat0, lon0, agl, t_arrive, hours=120, dt_min=15, members=ENSEMBLE,
        record_every=1, member_record_every=3):
    """Back trajectories for every (start, member) pair.

    lat0, lon0, agl: arrays of start points (S,). Returns dict with
      'centre': (S, hours/record_every + 1, 3) [lat, lon, agl] for member 0
      'members': (S, M-1, hours/member_record_every + 1, 3) for the perturbed members
      'alive_hours': (S, M) hours each parcel stayed inside the data domain
    """
    S, M = len(lat0), len(members)
    lat = np.repeat(np.asarray(lat0, np.float64), M)
    lon = np.repeat(np.asarray(lon0, np.float64), M)
    agl_start = np.repeat(np.asarray(agl, np.float64), M)
    off = np.array(members, np.float64)
    oy = np.tile(off[:, 0], S)
    ox = np.tile(off[:, 1], S)
    op = np.tile(off[:, 2], S)
    N = S * M
    zero = np.zeros(N)
    dt = -dt_min * 60.0
    steps_h = int(round(60 / dt_min))
    nrec = hours // record_every + 1
    nrec_m = hours // member_record_every + 1
    centre = np.full((S, nrec, 3), np.nan)
    memb = np.full((S, M - 1, nrec_m, 3), np.nan)
    alive_h = np.zeros(N, np.int32)

    # start: pressure at terrain + agl from the unperturbed column (all members share the start)
    col = model.column(lat, lon, t_arrive, zero, zero)
    if col is None:
        raise LookupError("arrival time outside data window")
    ground = model.terrain(lat, lon)
    p = p_at_height(col, ground + agl_start)
    alive = np.ones(N, bool)

    def rec(h, col_unpert, ground):
        z = height_at_p(col_unpert, p) - ground
        vals = np.stack([lat, lon, z], 1).reshape(S, M, 3)
        if h % record_every == 0:
            centre[:, h // record_every] = np.where(alive.reshape(S, M)[:, 0, None], vals[:, 0], np.nan)
        if h % member_record_every == 0:
            ok = alive.reshape(S, M)[:, 1:, None]
            memb[:, :, h // member_record_every] = np.where(ok, vals[:, 1:], np.nan)

    rec(0, col, ground)
    t = t_arrive
    from datetime import timedelta
    step = timedelta(seconds=dt)

    def sample(la, lo, pp, tt):
        c = model.column(la, lo, tt, oy, ox)
        if c is None:
            return None, None
        ps = np.clip(pp + op, P_TOP, P_BOTTOM)
        return c, (interp_var(c, IU, ps), interp_var(c, IV, ps), interp_var(c, IW, ps) / 100.0)

    def fix_vertical(la, lo, pp, tt):
        c0 = model.column(la, lo, tt, zero, zero)     # unperturbed column: heights & terrain test
        g = model.terrain(la, lo)
        z = height_at_p(c0, pp)
        low = z < g + CLEARANCE
        if low.any():
            pp = np.where(low, p_at_height(c0, g + CLEARANCE), pp)
        return np.clip(pp, P_TOP, P_BOTTOM), c0, g

    c_pert, vel1 = sample(lat, lon, p, t)
    for k in range(1, hours * steps_h + 1):
        if vel1 is None:
            break
        u1, v1, w1 = vel1
        la_p, lo_p = move(lat, lon, u1, v1, dt)
        p_p, _, _ = fix_vertical(la_p, lo_p, p + w1 * dt, t + step)
        _, vel2 = sample(la_p, lo_p, p_p, t + step)
        if vel2 is None:
            break
        u2, v2, w2 = vel2
        nla, nlo = move(lat, lon, (u1 + u2) / 2, (v1 + v2) / 2, dt)
        t = t + step
        npp, c0, g = fix_vertical(nla, nlo, p + (w1 + w2) / 2 * dt, t)
        # parcels leaving the data domain (south of the nh grid) stop
        out = nla < model.g.nh["lat"][0]
        alive &= ~out
        lat = np.where(alive, nla, lat)
        lon = np.where(alive, nlo, lon)
        p = np.where(alive, npp, p)
        if k % steps_h == 0:
            h = k // steps_h
            alive_h[alive] = h
            rec(h, c0, g)
        c_pert, vel1 = sample(lat, lon, p, t)
    return {"centre": centre, "members": memb, "alive_hours": alive_h.reshape(S, M)}
