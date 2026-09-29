# Zero-shot satellite localization baselines on SVMI+ (Matrice600Pro)

2799 dual-fisheye frames (Ricoh Theta S under a DJI Matrice 600 Pro, 0–90 m
above ground, fields near Vaux-en-Amiénois). Reference imagery: IGN
orthophotos (Géoplateforme WMTS, zoom 19, 0.19 m/px). Ground truth is used
only for scoring (and, for CCVPE, to pick the aerial tile, see below).

## Ground-truth conventions (important)

The dataset Readme does not match the data. Established empirically
(`svmi/poses.py`):

- `camera_poses.txt` columns 1–3 are **(down, south, east)**, not north. With
  the Readme's reading the drone would fly away from the forest that the
  images show it approaching, and the sun's world azimuth drifts over the
  flight (circular resultant 0.43). With column 2 = south and the heading
  mirrored accordingly (`heading = 180° − logged`), the sun stays fixed at
  az ≈ 108°, el ≈ 20° for the whole flight (resultant 0.99).
- Column 1 is an absolute GPS altitude; the drone is on the ground in frames
  0–~15 and at the end, so height above ground = value − value(frame 0).
- North is quantized in 11.1 m steps (1e-4° latitude) and GNSS is low-rate,
  so position errors below ~6 m cannot be resolved.
- Origin (takeoff) = 49.967068, 2.266771, supplied by the dataset owner.

## Protocols

**CCVPE** [1], VIGOR weights, unknown orientation (`ori_noise=180`).
Ground input: 320×640 equirectangular panorama from the dual fisheye
(level-flight assumption, no GT attitude). Aerial input: 640 px north-up
tile, centred on GT + a fixed random offset up to ±¼ tile (VIGOR "positive"
protocol [2]). This is a *local* task: the answer is known to lie inside the
tile. Baseline: always answer the tile centre.

**Game4Loc** [3], GTA-UAV cross-area weights, retrieval only.
Query: virtual nadir pinhole view (512×384, 60° HFOV, as in GTA-UAV).
Gallery: 3165 north-up IGN tiles (60/120/240 m, 75 % overlap) covering the
whole 889×732 m area. This is a *global* task. Baseline: a random tile.

## Results (median position error in m; baseline in brackets)

| Method | on ground (<5 m, 216 fr) | 5–30 m (563 fr) | >30 m (2020 fr) | all | median heading err |
|---|---|---|---|---|---|
| CCVPE VIGOR-cross, 0.111 m/px | **5.9** (14.3) | 16.2 (14.2) | 23.3 (14.1) | 19.7 (14.1) | 55° |
| CCVPE VIGOR-cross, 0.25 m/px | **5.2** (32.1) | **18.7** (32.0) | 45.8 (31.7) | 38.6 (31.7) | 63° |
| CCVPE VIGOR-cross, 0.5 m/px | **11.1** (64.3) | **32.3** (64.0) | 80.4 (63.3) | 66.4 (63.5) | 61° |
| CCVPE VIGOR-same, 0.111 m/px | 12.4 (14.3) | 20.6 (14.2) | 26.8 (14.1) | 24.2 (14.1) | 75° |
| CCVPE VIGOR-same, 0.111, IMU-levelled | 9.2 (14.3) | 19.7 (14.2) | 25.4 (14.1) | 22.5 (14.1) | 79° |
| CCVPE VIGOR-same, 0.25 m/px | **7.8** (32.1) | 28.2 (32.0) | 57.4 (31.7) | 46.5 (31.7) | 77° |
| CCVPE VIGOR-same, 0.5 m/px | **18.3** (64.3) | **40.2** (64.0) | 100.9 (63.3) | 79.4 (63.5) | 68° |
| Game4Loc GTA-cross, 60° nadir | 444.8 | 336.4 | 271.2 | 310.7 (304.4 random) | — |

Heading baselines: uniform random = 90°; the best *constant* heading (an
oracle chosen from the GT distribution, 18°) = 30° in flight.

## Takeaways

- CCVPE genuinely localizes only while the drone sits on the ground, i.e.
  at street level like its training data (5–6 m vs 14–32 m baseline).
  In flight it is worse than guessing the tile centre at every scale.
- CCVPE heading carries no usable information in flight (worse than the
  constant-heading oracle).
- Game4Loc retrieval is at chance: its training altitudes (80–650 m) exceed
  this flight, the scene is repetitive farmland, and the crops in the
  orthophoto differ from the flight date.

## Reproduce

```bash
.venv/bin/python scripts/prepare_gt.py
.venv/bin/python scripts/fetch_ign_mosaic.py --zoom 19 --margin 500 --name ign_z19_wide
.venv/bin/python ccvpe/run_ccvpe.py --area crossarea --m-per-px 0.111
.venv/bin/python game4loc/run_game4loc.py --area cross_area
python3 -m http.server 8000   # then open http://localhost:8000/viewer/
```

## References

[1] Z. Xia, O. Booij, J. F. P. Kooij, "Convolutional Cross-View Pose Estimation," IEEE T-PAMI 46(5):3813–3831, 2024.
[2] S. Zhu, T. Yang, C. Chen, "VIGOR: Cross-View Image Geo-localization beyond One-to-one Retrieval," CVPR 2021.
[3] Y. Ji, B. He, Z. Tan, L. Wu, "Game4Loc: A UAV Geo-Localization Benchmark from Game Data," AAAI 2025.
