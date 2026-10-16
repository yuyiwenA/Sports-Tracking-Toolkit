# Sports-Tracking-Toolkit

A sports video analysis toolkit. It detects and tracks players and the ball,
rebuilds a continuous ball path from fragmented detections, assigns players to
teams by jersey colour, detects match events (passes, possession changes,
sprints, changes of direction), and scores the quality of its own output without
any labelled data.

```
video ─▶ Detector ─▶ Tracker ─▶ Trajectory ─▶ ┌ annotated video
                              │               ├ trajectory.csv
                              ▼               └ session_report.json
                      ball stitching
                              │
                              ▼
                    ┌─────────────────────┐
                    │ events · teams ·    │  ─▶ events.csv / events.json
                    │ self-check          │  ─▶ selfcheck.json
                    └─────────────────────┘  ─▶ events.png
                              │              ─▶ dashboard.html
                              ▼
```

## Results on two clips

The same pipeline on two very different pieces of footage. The comparison shows
which conditions each layer needs before it becomes usable.

|  | Aerial football (3840x2160, panning drone) | Broadcast match (1080x1920, cropped) |
|---|---|---|
| Processed as | 1600x900, `--imgsz 1600` | 1080x864 crop, `--imgsz 1280` |
| Players per frame | 5.74, peak 9 | 17.43, peak 22 |
| Unique track ids | 41 | 114 |
| Ball path coverage, raw to stitched | 59% to 80% | 69% to 100% |
| Jersey team separation | strong, silhouette 0.42 | weak, silhouette 0.18 |
| Possession segments | 0 | 3, with 2 changes |
| Events | 15 sprints, 5 turns, 6 ball-speed peaks | 10 sprints, 15 turns, 8 peaks |
| Track ids reused across classes | 15 | 0 |
| Self-check | 40.6 / 100 (D) | 49.7 / 100 (D) |

Both clips score poorly, for different reasons.

The aerial clip misses roughly a third of the players (recall proxy 0.64). Its
figures are also depressed by a tracker defect: 15 of its track ids appear as
both a person and a ball. An id is supposed to identify one object, so any
analysis keyed on the raw id blends the two sets. That defect produced a ball
coverage reading of 2300%, which is impossible; it is now reported in
`selfcheck.json` as `track_ids_spanning_classes`.

The broadcast clip finds people reliably (recall proxy 0.79, ball path fully
continuous) but its identities fragment: 114 track ids for about 22 players, or
3.5 ids issued per player. Its team separation is also weak enough to be labelled
`weak`, which makes the labels usable for colouring a diagram but not for
inferring a turnover.

Two clips failing in different ways is why one clip cannot judge a tracker.

## Capabilities

### Match events

`src/events.py` derives passes, possession, turnovers, sprints, changes of
direction and ball-speed peaks from relative geometry: ball-to-player distance,
ball-to-previous-possessor distance, and each player's own speed history.

Relative rules were chosen because the footage moves. In the aerial clip the
penalty area drifts from x≈160-800 to x≈80-650 across the sequence, so a rule
such as "the ball entered this region" measures the camera.
Rules built on relationships survive camera motion.

Every event carries the evidence that produced it:

```
frame,time_s,kind,team,track_id,partner_id,partner_team,magnitude,confidence,detail
1841,76.708,pass,A,89,100,A,268.4,0.882,"ball travelled 268 px over 11 frames (same team (A)); observed in 96% of the interval"
```

`confidence` is the fraction of the interval in which the ball was observed
without interpolation, multiplied by the detection confidence. It reports how
much of the evidence is real.

### Ball path stitching

On the aerial clip the ball detector produced 68 unbroken runs across 26 track
ids and covered 59% of the clip. Passes and possession both need to know where
the ball was continuously, so analysing those fragments would mean analysing
noise.

`src/ball_path.py` rejoins fragments with a gate that depends on how long the
ball went unobserved. Four frames of occlusion allow only a small jump; two
seconds of occlusion allow a large one, because the ball may have been kicked.
Coverage goes from 59% to 80% on the aerial clip and from 69% to 100% on the
broadcast clip.

Three-frame runs and shorter are discarded before stitching. A ball detection
that survives for one frame is usually grass texture or a boot, and keeping such
detections lets the stitcher chain unrelated blips into a plausible-looking path.

### Team assignment

A pass is only a pass if the receiver is a teammate. If the receiver is an
opponent the same observation is a turnover, which means the opposite thing, so
the event layer needs team labels.

`src/teams.py` samples the torso of each track, histograms hue and saturation,
masks out pitch-green pixels and clusters the result. No labels, no GPU and no
network call are involved.

Separation is reported as a strength rather than a yes/no verdict, because the
dangerous case is not failure. The dangerous case is clusters that separate just
enough to pass a threshold and not enough to act on. On the broadcast clip the
red kit (`#ad3a3f`) and the white kit do separate, at silhouette 0.18, so the
result is reported as `weak`. On the aerial clip separation is `strong` (0.42)
with dark-green and olive kits, sampled at 33 px per player. When clustering
fails the assigner labels every track `unknown` and team-dependent reasoning is
disabled, since a wrong team label inverts the meaning of the events derived
from it.

Two bugs in this module produced confident wrong answers:

- The pipeline can crop, and CSV coordinates are crop-relative.
  `FrameTransform` restores them. The first version applied the scale without the
  crop offset, sampled the crowd and the grass, and still returned a clustering
  result, because a colour histogram cannot tell that it was given the wrong
  pixels.
- Letting the cluster count float settled on three clusters on the broadcast
  clip, after which the labelling filed the third group of 24 players as match
  officials. A match has two teams, so the default is now two, and a non-small
  extra cluster is never labelled as officials.

### Self-assessment without labels

Knowing whether a tracker improved normally requires annotated data, and
annotating sports footage is expensive. `src/selfcheck.py` computes five proxies
from the output alone.

| Metric | Aerial | Broadcast | Meaning |
|---|---|---|---|
| `detection_recall` | 0.637 | 0.792 | players found per frame, over the typical peak count |
| `identity_fragmentation` | 3.00 | 3.50 | track ids issued per player, where 1.0 is ideal |
| `trajectory_jitter` | 0.418 | 0.355 | path wobble relative to step size |
| `ball_continuity` | 0.836 | 1.000 | share of the ball path observed directly |
| `track_longevity` | 0.697 | 0.651 | share of rows in long-lived track ids |

The value lies in comparing two versions. Fragmentation rising from 3.0 to 3.5 is
a regression even without knowing the absolute quality. An earlier tuning pass
raised player recall while pushing ball false positives from 1.9 to 5.7 per
frame, and without a number attached that regression would have shipped.

Read the caveats in `selfcheck.json` before quoting these figures. They are
proxies. IDF1 and HOTA remain the reference. They cannot see a consistent bias: a tracker
that always lags three frames scores well here. They are also all ratios, so they
can be insensitive to changes that do not move a ratio. Removing 653 crowd false
positives visibly cleaned the video and barely moved the overall score.

## Install

```bash
pip install -r requirements.txt
```

`filterpy` was removed; a dependency-free Kalman filter lives in
`src/kalman.py`. `mediapipe` is optional and never imported eagerly.

## Run

```bash
# aerial football
python demo.py --video 13386302_3840_2160_24fps.mp4 \
               --max-width 1600 --imgsz 1600 \
               --conf 0.18 --conf-override "sports ball=0.55" \
               --classes person,"sports ball" --trail 60

# broadcast match: crop the crowd away, ignore the advertising boards
python demo.py --video 14604731_1080_1920_30fps.mp4 \
               --crop 0,0.55,1,1 --ignore-top 0.09 --imgsz 1280 \
               --conf 0.28 --conf-override "sports ball=0.35" \
               --classes person,"sports ball" --sport soccer --trail 45 \
               --out outputs_broadcast

# analysis, without re-running inference
python tools/analyze.py   --trajectory outputs/trajectory.csv --video <clip> --out outputs
python tools/dashboard.py --trajectory outputs/trajectory.csv --video <clip> --out outputs
python tools/screenshot.py outputs/result.mp4 -n 6 -o outputs/screenshots
```

`python demo.py --help` lists every option.

### Why `--crop` and `--ignore-top` exist

The broadcast clip is 1080x1920 portrait and about 55% of it is crowd. Those
pixels consume the detector's input budget and carry nothing detectable, so
`--crop 0,0.55,1,1` discards them and raises the effective resolution on the
players at no extra inference cost. The remaining strip of advertising boards
still produced 653 spurious person detections. `--ignore-top 0.09` removes them
without cropping the picture; cropping further would have cut the goal off.

### Resolution

A wide shot puts every player at a handful of pixels, and YOLO resizes its input
to `--imgsz` before inference. On the aerial clip, averaged over 20 sampled
frames:

| processing width | `--imgsz` | persons detected | median box width | time per frame |
|---|---|---|---|---|
| 1280 | 640 | 0 | | 0.14 s |
| 1280 | 1280 | 7 | 26 px | 0.51 s |
| 1600 | 1600 | 7.4 | 33 px | 0.77 s |
| 1920 | 1920 | 7.6 | 40 px | 1.06 s |

At the default `--imgsz 640` nothing is found at all. What matters is the size of
the subject in the network input, so raising the resolution is the effective
change and adjusting `--conf` is not. Gains flatten past 1600 px.

### Per-class confidence

`--conf` sets the default and `--conf-override` gives any class its own gate. The
aerial clip contains one ball, yet at 0.18 the detector produced 5.7 ball
candidates per frame:

| threshold | candidates kept | per frame |
|---|---|---|
| 0.18 | 114 | 5.70 |
| 0.45 | 39 | 1.95 |
| 0.55 | 22 | 1.10 |
| 0.60 | 16 | 0.80 |

0.55 is the operating point. At 0.60 the real ball starts being discarded. The
false candidates are large boxes of 20 to 35 px while the true ball measures 9 to
13 px, which indicates a detector firing on grass texture and white markings.

## Court calibration

Metric distance and speed are computed by projecting consecutive positions into
court space and measuring the step there. That step is exact under any
perspective. Scaling a pixel step by a single metres-per-pixel constant would
bias every speed, because a homography scales differently along x and y and
differently again at each location.

Two behaviours are intentional:

- A failed auto-detection leaves the pipeline uncalibrated. Metre values derived
  from an invented court would look authoritative while being wrong, so none are
  written. The state is recorded in `session_report.json`.
- Calibration is computed once, which requires a static camera. All five clips
  available here pan, so every result above reports pixel metrics.

`--calibrate auto` segments the playing surface as its primary cue and falls back
to Hough line detection. Quads touching the frame border are rejected, since a
cropped surface has unknown extent. On a synthetic 105x68 m pitch with known
ground truth it recovers the quad to within 4.2 px, measuring 105.51 m against a
true 105.00 m.

## Output

```
outputs/
├── result.mp4            annotated video: boxes, ids, trails, court overlay, mini-map
├── trajectory.csv        frame, time_s, track_id, class, conf, px, py, w, h,
│                         court_x, court_y, distance_px, speed_px_s, speed_m_s, calib_mode
├── session_report.json   throughput, calibration state, per-track kinematics
├── events.csv/.json      match events with the evidence behind each one
├── selfcheck.json        label-free quality scorecard and its caveats
├── heatmap.png           spatial occupancy
├── summary.png           session analysis board
├── events.png            event board: ball speed, possession, per-player, timeline
├── dashboard.html        interactive replay: scrub the timeline, click an event
└── screenshots/          sampled frames + contact_sheet.png
```

`session_report.json` reports throughput twice. `elapsed_s` includes model
loading and first-inference warm-up, while `processing_fps` covers only the span
of processed frames. On a 12-second clip the two differ by a factor of five or
more, and reporting only the first would understate the pipeline.

## Interactive dashboard

`outputs/dashboard.html` is one self-contained file with no CDN, no web fonts and
no charting library. It scrubs through the tracked positions, marks events,
colours players by team and names the player in possession. Open it directly; it
behaves the same offline and on GitHub Pages. A dashboard that fails when a CDN
is blocked is worse than a plainer one that always renders.

The view is a schematic map of tracked positions rather than an overlay on the
footage. Embedding 20 MB of video as base64 would make the page unusable, and the
geometry is the subject.

## Module map

| Module | Responsibility |
|---|---|
| `src/detector.py` | YOLOv8 wrapper, per-class confidence gates |
| `src/tracker.py` | `Track` objects; ByteTrack with an IoU+Kalman fallback |
| `src/kalman.py` | dependency-free constant-velocity filter |
| `src/trajectory_data.py` | the CSV parsing layer every tool reads |
| `src/ball_path.py` | ball fragment stitching and interpolation |
| `src/teams.py` | jersey-colour team assignment with a strength verdict |
| `src/events.py` | the semantic layer, plus possession diagnostics |
| `src/selfcheck.py` | label-free quality proxies |
| `src/court_homography.py` | pixel to metres, four calibration strategies |
| `src/trajectory.py` | smoothing, distance, speed |
| `src/analytics.py` | session statistics, heat-map, analysis board |
| `src/visualizer.py` | boxes, labels, trails, HUD, court mini-map |
| `src/pose_estimator.py` | optional MediaPipe keypoints, never imported eagerly |
| `src/pipeline.py` | the run loop and CLI |

Tools: `analyze.py` (events, self-check, charts), `dashboard.py`,
`render_report.py` (re-render charts from a CSV without inference),
`screenshot.py`.

Both `demo.py` and `src/demo.py` are thin wrappers over `src/pipeline.py`, so the
two entry points cannot drift apart.
