"""Shared configuration and I/O for the ContextClimate back-trajectory wind data.

Source: NOAA GFS 0.5-degree pressure-level GRIB2 on the NOAA Open Data AWS bucket.
Output: gzip-compressed int16 tiles, one file per grid tile per 3-hourly valid time.

Tile file layout (all grids):
  int16 little-endian, array order [var][level][lat][lon]
  lat ascending from the tile's south edge, lon ascending from its west edge,
  both edges included (tile + 1 points per side).
  Bytes are shuffled before gzip (all low bytes, then all high bytes).
  value = stored * SCALE[var]
"""
from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
          600, 550, 500, 450, 400, 350, 300]
VARS = ["UGRD", "VGRD", "VVEL", "HGT"]          # m/s, m/s, Pa/s, gpm
SCALE = {"UGRD": 0.01, "VGRD": 0.01, "VVEL": 0.001, "HGT": 1.0}
UNITS = {"UGRD": "m/s", "VGRD": "m/s", "VVEL": "Pa/s", "HGT": "m"}

# Grids. "na" is cut directly from the 0.5-degree field; "nh" subsamples it to 1 degree.
# nh tiles that lie entirely inside the na box are not written.
GRIDS = {
    "na": {"res": 0.5, "lat": [10, 80], "lon": [-170, -50], "tile": 10},
    "nh": {"res": 1.0, "lat": [-10, 90], "lon": [-180, 180], "tile": 10, "skip": "na"},
}

SRC_RES = 0.5  # source grid spacing (degrees)


# ---------------------------------------------------------------- time helpers
def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def tkey(t: datetime) -> str:
    return t.strftime("%Y%m%d%H")


def parse_tkey(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H")


def file_url(cycle: datetime, fh: int) -> str:
    d, h = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    return f"{BUCKET}/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p50.f{fh:03d}"


# ---------------------------------------------------------------- HTTP
_session = requests.Session()
_session.headers["User-Agent"] = "contextclimate-trajectory-data (github.com/josht3113)"


def http_get(url: str, headers: dict | None = None, tries: int = 4, timeout: int = 60) -> requests.Response | None:
    """GET with retries. Returns None on 404/403 (file not posted yet)."""
    for k in range(tries):
        try:
            r = _session.get(url, headers=headers or {}, timeout=timeout)
            if r.status_code in (403, 404):
                return None
            if r.status_code in (200, 206):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {url}")


def fetch_idx(cycle: datetime, fh: int) -> list[tuple[int, int, str, str]] | None:
    """Return [(start, end_or_-1, var, level)] for the file, or None if not posted."""
    r = http_get(file_url(cycle, fh) + ".idx")
    if r is None:
        return None
    rows = [ln.split(":") for ln in r.text.strip().splitlines()]
    out = []
    for i, p in enumerate(rows):
        start = int(p[1])
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else -1
        out.append((start, end, p[3], p[4]))
    return out


def wanted_messages(idx):
    """Pick (key, start, end) for the variables/levels we store, plus surface height."""
    want = {}
    lev_names = {f"{L} mb": L for L in LEVELS}
    for start, end, var, lev in idx:
        if var in VARS and lev in lev_names:
            want[(var, lev_names[lev])] = (start, end)
        elif var == "HGT" and lev == "surface":
            want[("HGT", "surface")] = (start, end)
    missing = [(v, L) for v in VARS for L in LEVELS if (v, L) not in want]
    if missing or ("HGT", "surface") not in want:
        raise RuntimeError(f"idx missing fields: {missing[:6]}")
    return want


# ---------------------------------------------------------------- GRIB decode
def decode_message(buf: bytes) -> np.ndarray:
    """Decode one GRIB2 message to a (lat, lon) float32 array on the global
    0.5-degree grid with lat ascending -90..90 and lon 0..359.5."""
    import eccodes
    gid = eccodes.codes_new_from_message(buf)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        lat1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lon1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        dj = eccodes.codes_get(gid, "jDirectionIncrementInDegrees")
        di = eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
        j_pos = eccodes.codes_get(gid, "jScansPositively")
        i_neg = eccodes.codes_get(gid, "iScansNegatively")
        vals = eccodes.codes_get_values(gid).astype(np.float32)
    finally:
        eccodes.codes_release(gid)
    if abs(di - SRC_RES) > 1e-6 or abs(dj - SRC_RES) > 1e-6:
        raise RuntimeError(f"unexpected grid spacing {di}x{dj}")
    a = vals.reshape(nj, ni)
    if not j_pos:          # rows run north -> south
        a = a[::-1]
        lat1 = lat1 - (nj - 1) * dj
    if i_neg:
        a = a[:, ::-1]
        lon1 = lon1 - (ni - 1) * di
    if abs(lat1 + 90) > 1e-6 or nj != int(180 / SRC_RES) + 1:
        raise RuntimeError(f"unexpected latitude coverage: first {lat1}, nj {nj}")
    shift = int(round((lon1 % 360) / SRC_RES))
    if shift:
        a = np.roll(a, shift, axis=1)
    if ni != int(360 / SRC_RES):
        raise RuntimeError(f"unexpected Ni {ni}")
    return a


def fetch_fields(cycle: datetime, fh: int, workers: int = 8):
    """Download and decode one forecast file's fields.
    Returns (cube[var, level, lat, lon] float32, terrain[lat, lon]) or None if not posted."""
    idx = fetch_idx(cycle, fh)
    if idx is None:
        return None
    want = wanted_messages(idx)
    url = file_url(cycle, fh)

    def get(item):
        key, (s, e) = item
        rng = f"bytes={s}-{e}" if e >= 0 else f"bytes={s}-"
        r = http_get(url, headers={"Range": rng}, timeout=120)
        if r is None:
            raise RuntimeError(f"range fetch vanished: {url} {key}")
        return key, decode_message(r.content)

    with ThreadPoolExecutor(workers) as ex:
        fields = dict(ex.map(get, want.items()))
    shape = fields[("HGT", "surface")].shape
    cube = np.empty((len(VARS), len(LEVELS)) + shape, np.float32)
    for vi, v in enumerate(VARS):
        for li, L in enumerate(LEVELS):
            cube[vi, li] = fields[(v, L)]
    return cube, fields[("HGT", "surface")]


# ---------------------------------------------------------------- tiling
def grid_tiles(name: str):
    """Yield (lat0, lon0) south-west corners of the tiles written for a grid."""
    g = GRIDS[name]
    T = g["tile"]
    skip = GRIDS[g["skip"]] if g.get("skip") else None
    for la in range(g["lat"][0], g["lat"][1], T):
        for lo in range(g["lon"][0], g["lon"][1], T):
            if skip and skip["lat"][0] <= la and la + T <= skip["lat"][1] \
                    and skip["lon"][0] <= lo and lo + T <= skip["lon"][1]:
                continue
            yield la, lo


def tile_name(la: int, lo: int) -> str:
    return f"{la}_{lo}"


def cut_tile(a: np.ndarray, name: str, la: int, lo: int) -> np.ndarray:
    """Cut a tile from arrays whose last two axes are the global 0.5-degree (lat, lon) grid."""
    g = GRIDS[name]
    step = int(round(g["res"] / SRC_RES))
    n = int(round(g["tile"] / g["res"])) + 1
    i0 = int(round((la + 90) / SRC_RES))
    rows = i0 + step * np.arange(n)
    j0 = int(round((lo % 360) / SRC_RES))
    cols = (j0 + step * np.arange(n)) % a.shape[-1]
    rows = np.clip(rows, 0, a.shape[-2] - 1)  # the 90N edge row duplicates the pole
    return a[..., rows[:, None], cols[None, :]]


def encode(arr: np.ndarray, scales: np.ndarray | float) -> bytes:
    """Quantise to int16, byte-shuffle, gzip."""
    q = np.ascontiguousarray(np.clip(np.rint(arr / scales), -32768, 32767).astype("<i2"))
    b = q.reshape(-1).view(np.uint8).reshape(-1, 2)
    return gzip.compress(b[:, 0].tobytes() + b[:, 1].tobytes(), compresslevel=9, mtime=0)


def decode(blob: bytes, shape: tuple, scales: np.ndarray | float) -> np.ndarray:
    raw = gzip.decompress(blob)
    half = len(raw) // 2
    b = np.empty((half, 2), np.uint8)
    b[:, 0] = np.frombuffer(raw[:half], np.uint8)
    b[:, 1] = np.frombuffer(raw[half:], np.uint8)
    return b.reshape(-1).view("<i2").reshape(shape).astype(np.float32) * scales


def cube_scales() -> np.ndarray:
    return np.array([SCALE[v] for v in VARS], np.float32)[:, None, None, None]


def tile_shape(name: str) -> tuple:
    g = GRIDS[name]
    n = int(round(g["tile"] / g["res"])) + 1
    return (len(VARS), len(LEVELS), n, n)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
