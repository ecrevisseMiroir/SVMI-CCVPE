# UniK3D on SVMIS+ (Matrice600Pro, Ricoh Theta S dual-fisheye)

Calibration source: `SVMI/SVMISplus/SVMISplus/Matrice600Pro/camera_calibration.txt`
(unified/Mei model: K + xi, no distortion). Mapped to UniK3D `MEI` params
`[fx, fy, cx, cy, k1, k2, p1, p2, xi]` with k/p = 0.

| crop (640x640) | calib    | json             | note                         |
|----------------|----------|------------------|------------------------------|
| left  (x 0:640)    | camera 1 | `cam_left.json`  | as-is                        |
| right (x 640:1280) | camera 0 | `cam_right.json` | cx shifted by -640 (966.95 -> 326.95) |

Both crops keep rows 0:640, so cy is unchanged.

```bash
python unik3d/split_dual_fisheye.py SVMI/SVMISplus/SVMISplus/Matrice600Pro/SixDOFs/Images split
# in the UniK3D repo:
python scripts/infer.py --input split/left/000000.png  --output out_left  --config-file configs/eval/vitl.json --camera-path <repo>/unik3d/cam_left.json  --save --save-ply
python scripts/infer.py --input split/right/000000.png --output out_right --config-file configs/eval/vitl.json --camera-path <repo>/unik3d/cam_right.json --save --save-ply
```

Unverified: that UniK3D's `xi` convention matches the dataset's calibration. If clouds look warped, check that first.
`c2Mc1` in the calibration file gives the left<->right extrinsic for fusing both halves.
