# Badminton AI Analytics

End-to-end computer-vision pipeline that turns a broadcast badminton video into match analytics: shuttle tracking, player tracking with pose, a live court mini-map, automatic hit / serve detection, rally segmentation, landing points, scoring and a live stats dashboard video.

![pipeline](https://img.shields.io/badge/pipeline-TrackNetV3%20%2B%20YOLO%20Pose%20%2B%20BoT--SORT-blue)

## Features

- **Shuttle tracking** with TrackNetV3 (+ InpaintNet for occlusions)
- **Player detection, pose and tracking** with YOLOv8-pose + BoT-SORT
- **Court homography**: 4 clicked corners map pixels to real court metres (6.1 × 13.4 m)
- **Player filtering**: umpire, line judges, crowd and on-screen graphics removed; players labelled by court half
- **Hit detection** from sudden shuttle velocity changes, player proximity and racket reach
- **Serve detection** from the "both players still" set-up and the moment the held shuttle is launched
- **Rally logic**: P1/P2 alternation, recovery of soft net touches, overlapping-box disambiguation, between-point taps ignored
- **Landing point + result** (in / out / net), rally winner, running score (cross-checked with "winner serves next")
- **Dashboard video**: score with server indicator, live rally timer, shots and distance per player, movement heatmap, hit positions, landing points and rally results

## Pipeline

```
match video
   │
   ├─ TrackNetV3 ──────────────► outputs/clip_ball.csv       (shuttle x, y per frame)
   ├─ align.py ────────────────► clip_new.mp4                (video aligned to the CSV)
   ├─ players.py ──────────────► outputs/players.csv         (YOLO pose + BoT-SORT)
   ├─ court_pick.py ───────────► court.json                  (4 court corners)
   ├─ players_clean.py ────────► outputs/players_clean.csv   (2 players, court metres)
   ├─ hits.py ─────────────────► outputs/hits.csv            (serves + hits per rally)
   ├─ rally.py ────────────────► outputs/rallies.csv         (landing, winner, score)
   └─ dashboard.py ────────────► outputs/dashboard.mp4       (final video)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Shuttle tracking uses [TrackNetV3](https://github.com/qaz812345/TrackNetV3). Clone it into this folder and put its checkpoints in `TrackNetV3/ckpts/` (`TrackNet_best.pt`, `InpaintNet_best.pt`). On Windows set `num_workers=0` in `predict.py`.

## Usage

```bash
# 1. shuttle tracking
cd TrackNetV3
python predict.py --video_file ..\clip.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir ..\outputs
cd ..

# 2. align video with the shuttle CSV (only needed if the frame count differs)
python align.py
python check.py                  # optional: draws the shuttle to verify alignment

# 3. players
python players.py

# 4. court + clean players
python court_pick.py             # click far-left, far-right, near-right, near-left (outer lines)
python players_clean.py

# 5. hits, serves, rallies
python hits.py

# 6. landing points, winners, score
python rally.py

# 7. dashboard video
python dashboard.py
```

Each step also writes a `*_check.mp4` video in `outputs/` to verify it visually.
`serve_debug.py <start> <end>` prints the shuttle data for a frame range, useful for tuning.

## Tuning

The thresholds are at the top of each script:

| Script | Main settings |
|---|---|
| `hits.py` | `DV_MIN`, `NEAR_PX`, `GAP`, `MIN_TRAVEL`, `RALLY_GAP`, serve `HOLD_*` / `SERVE_*` |
| `rally.py` | `NAMES`, `START_SCORE`, `STILL_*`, `TOL` |
| `dashboard.py` | `NAMES`, `START_SCORE`, `COL`, `TITLE` |

## Output files

| File | Content |
|---|---|
| `hits.csv` | rally, frame, time, player, serve flag, shuttle position, player court position |
| `rallies.csv` | rally, server, start/end frame, duration, shots per player, last hitter, landing point, result, winner, score |
| `dashboard.mp4` | match video with the live analytics panel |

## Limitations

- Single broadcast camera angle; the court corners are set once per video.
- The shuttle in the air cannot be placed on the mini-map (homography is only valid on the floor), so hit positions use the hitter's feet.
- If the video ends before the shuttle lands, the last point is marked as estimated.

## Tech stack

Python · OpenCV · PyTorch · Ultralytics YOLO (pose) · BoT-SORT · TrackNetV3 · pandas · NumPy

## Author

**Muhammad Rahman Shahid** – Computer Vision & AI · [GitHub](https://github.com/muhammadrahman-hehe)
