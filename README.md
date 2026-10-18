# contextclimate-trajectory-data

Precomputed back trajectories for the ContextClimate trajectory map, refreshed after
every GFS cycle and published to GitHub Pages.

## What's published

- `manifest.json` — arrival times (analysis / forecast), heights, ensemble, encoding.
- `cities.json` — 292 arrival points built from ContextClimate's shared station roster
  (stations within 70 km folded into one representative; folded names kept as aliases).
  See [CITIES.md](CITIES.md).
- `traj/{city}.bin.gz` — centre trajectories for every arrival time:
  `[arrival][height][hour 0..120][lat, lon, agl]`.
- `ens/{city}/{YYYYMMDDHH}.bin.gz` — ensemble members for one arrival:
  `[height][member][every 3 h][lat, lon, agl]`.

Binary files are int16 little-endian, gzip, delta-encoded along time (first value
absolute); lat/lon in 0.01° (longitude unwrapped), agl in metres; −32768 = no data.

Heights 500 / 1500 / 3000 m above ground; 120 h; arrivals every 3 h for the past 7 days
plus the latest cycle's 24 h forecast.

## Model

Kinematic trajectories in pressure coordinates using GFS vertical velocity (omega);
bilinear in space (0.5° over North America, 1° elsewhere), linear in ln p and time;
Heun integration, 15-minute steps; parcels kept at least 10 m above model terrain.
Ensemble: 15 members sampling the meteorology offset by one grid cell and/or ±10 hPa
(the scheme HYSPLIT's trajectory ensemble uses). `scripts/trajectory.py` is the
readable reference; `scripts/batch_traj.py` is the vectorised production model.

Source: NOAA Global Forecast System, NOAA Open Data Dissemination program (AWS).

## Workflows

- **Update trajectory data** — every 6 h after each GFS cycle: restores the window from
  the Actions cache, downloads new GFS fields into wind tiles (kept in the cache, not
  published), recomputes arrivals whose data changed, deploys Pages.
- **Validate against HYSPLIT** — manual: backfills three reference cases from the GFS
  archive and compares with READY/HYSPLIT endpoints (`scripts/ready_reference.json`).

## One-time setup

Settings → Pages → Build and deployment → Source: **GitHub Actions**.
