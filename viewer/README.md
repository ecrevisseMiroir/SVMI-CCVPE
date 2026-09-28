# SVMI+ viewer

A static page with no build step. It plays the SVMI+ camera frames next to the IGN orthophoto. The map shows the GT trajectory and pose. It also overlays method predictions when `results/manifest.json` exists.

## 1. Generate the viewer assets (once)

```bash
.venv/bin/python scripts/make_viewer_frames.py
```

The script writes:

- `data/viewer_frames/000000.jpg … 002798.jpg`: 640x360 JPEGs at quality 80, about 96 MB in total. Only missing or stale frames are converted, so `--force` is needed to redo them all.
- `data/map/ign_z19_viewer.jpg` and `.json`: the orthophoto at half resolution. `scale` is the number of viewer pixels per `ign_z19.jpg` pixel. The page loads this image first. It loads the full-resolution mosaic only when you zoom in past it.

Options: `--size 640x360`, `--quality 80`, `--map-scale 0.5`, `--skip-frames`, `--skip-map`.

## 2. Serve and open

```bash
python3 -m http.server 8000      # from the repo root
```

Then open <http://localhost:8000/viewer/>. You can link to a frame with `#f=1234`.

## Controls

| Key or action | Effect |
|---|---|
| Space | play / pause |
| ← / → | step back / forward 1 frame |
| Shift+← / Shift+→ | step back / forward 10 frames |
| Home / End | jump to the first / last frame |
| `[` / `]` | lower / raise the speed (0.25x–8x; 1x = 10.14 fps, real time) |
| F | follow the drone |
| R | fit the trajectory in view |
| + / − or mouse wheel | zoom the map |
| Drag the map | pan |
| Click the GT path | jump to the nearest frame |
| Click or drag the chart | scrub |

## Method predictions

`results/manifest.json` is optional. Without it, the page shows GT only.

```json
{"methods": [{"name": "CCVPE (VIGOR samearea)", "file": "ccvpe_vigor_samearea/predictions.json", "color": "#e4572e"}]}
```

`file` is a path relative to `results/`. Each predictions file follows this format:

```json
{"method": "...", "params": {}, "summary": {},
 "frames": [{"i": 0, "lat": 0, "lon": 0, "north": 0, "east": 0,
             "heading_deg": null, "err_m": 0, "heading_err_deg": null}]}
```

`frames` can be any subset of indices. If `lat`/`lon` are missing, the viewer computes them from `north`/`east` with the constants in `svmi/geo.py`. If `err_m` is missing, the viewer computes it against GT. The median and mean errors in the table are computed over the frames that have predictions. Up to three scalar `summary` fields are listed under each method name. Hover over a method name to see its `params` and `summary`.
