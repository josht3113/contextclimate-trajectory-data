# contextclimate-trajectory-data

Rolling window of GFS wind data for the ContextClimate back-trajectory map.
Published to GitHub Pages; the map computes trajectories in the browser from these files.

## What's published

- `manifest.json` — available valid times, grids, levels, encoding.
- `{grid}/{YYYYMMDDHH}/{lat}_{lon}.bin.gz` — one 10°×10° tile per valid time.
- `{grid}/terrain/{lat}_{lon}.bin.gz` — model surface height.

| Grid | Spacing | Area |
|---|---|---|
| `na` | 0.5° | 10–80°N, 170–50°W |
| `nh` | 1° | 10°S–90°N, all longitudes (tiles inside `na` omitted) |

Fields: u, v (m/s), omega (Pa/s), geopotential height (m) on 17 pressure levels,
1000–300 hPa. Tiles are int16 little-endian `[var][level][lat][lon]`, edges included,
byte-shuffled, gzip; value = stored × `scale[var]`.

Times are 3-hourly: GFS analyses (f000) and 3-h forecasts (f003) for arrivals up to
7 days back plus 120 h of trajectory, and the latest cycle's forecast to +24 h.

Source: NOAA Global Forecast System, NOAA Open Data Dissemination program (AWS).

## Workflows

- **Update trajectory data** — every 6 h after each GFS cycle. Restores the previous
  window from the Actions cache, builds what's new, prunes what expired, deploys Pages.
  The first run builds the whole window (~105 times) and can take 30–60 minutes.
- **Validate against HYSPLIT** — manual. Backfills three reference cases from the GFS
  archive and compares the reference trajectory model (`scripts/trajectory.py`)
  against READY/HYSPLIT endpoints (`scripts/ready_reference.json`).

## One-time setup

Settings → Pages → Build and deployment → Source: **GitHub Actions**.
