"""Command-line pipeline: detect -> track -> smooth -> measure -> render.

This module holds the single implementation of the run loop. Both ``demo.py``
at the repository root and ``src/demo.py`` are thin wrappers around
:func:`main`, so the two entry points cannot drift apart. That drift is how the
original repository ended up with a duplicate ``demo.py`` and a
missing ``detector.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

from .analytics import SessionAnalytics
from .court_homography import SPORT_PRESETS, CourtHomography
from .detector import DEFAULT_CLASSES, Detector
from .trajectory import TrailBuffer, TrajectoryAnalyzer
from .tracker import Tracker
from .visualizer import Visualizer

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python demo.py",
        description="Sports-Tracking-Toolkit: multi-object tracking and kinematics for sports video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=None,
                        help="input video; auto-detects the first video file next to demo.py")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    parser.add_argument("--conf", type=float, default=0.30, help="detection confidence threshold")
    parser.add_argument("--conf-override", default=None,
                        help='per-class confidence, e.g. "sports ball=0.45,person=0.15"')
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference resolution")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu / cuda:0")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                        help="comma separated class names or ids to keep ('all' disables filtering)")
    parser.add_argument("--tracker", choices=("bytetrack", "iou"), default="bytetrack",
                        help="tracking back-end; falls back to 'iou' automatically")
    parser.add_argument("--track-buffer", type=int, default=30, help="frames a lost track survives")
    parser.add_argument("--trail", type=int, default=90, help="trail length in frames")

    parser.add_argument("--sport", default="basketball", choices=sorted(SPORT_PRESETS),
                        help="court preset used for metric conversion")
    parser.add_argument("--calibrate", choices=("none", "auto", "default", "manual"), default="auto",
                        help="court calibration strategy")
    parser.add_argument("--court-points", default=None,
                        help='manual corners "x1,y1;x2,y2;x3,y3;x4,y4" ordered TL,TR,BR,BL')
    parser.add_argument("--court-json", default=None, help="JSON file with court points")

    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--crop", default=None,
                        help='crop before processing, as normalised "x1,y1,x2,y2" in 0..1 '
                             '(use it to discard crowd or sky, which cost resolution but carry '
                             'no information)')
    parser.add_argument("--ignore-top", type=float, default=0.0, metavar="FRACTION",
                        help="discard detections lying entirely above this fraction of the frame. "
                             "Broadcast crops often keep a strip of spectators or advertising "
                             "boards, which the detector reports as people and which inflate the "
                             "track count; this removes them without cropping the picture")
    parser.add_argument("--max-width", type=int, default=0,
                        help="downscale frames wider than this before processing (0 = keep native size)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--no-video", dest="save_video", action="store_false", help="skip writing the result video")
    parser.add_argument("--no-csv", dest="save_csv", action="store_false", help="skip writing the CSV")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false", help="skip report charts")
    parser.add_argument("--pose", action="store_true", help="draw MediaPipe pose keypoints (optional)")
    parser.add_argument("--show", action="store_true", help="open a preview window (needs a display)")
    parser.set_defaults(save_video=True, save_csv=True, save_plots=True)
    return parser


def _autodetect_video(anchor_dir: str) -> Optional[str]:
    for name in sorted(os.listdir(anchor_dir)):
        if name.lower().endswith(VIDEO_SUFFIXES):
            return os.path.join(anchor_dir, name)
    return None


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    anchor = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(anchor)

    video_path = args.video
    if video_path is None:
        video_path = _autodetect_video(repo_root) or _autodetect_video(os.getcwd())
        if video_path is None:
            print("[error] no --video given and no video file found next to demo.py", file=sys.stderr)
            return 2
        print(f"[info] auto-detected video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] cannot open video: {video_path}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1e-3:  # 0 / NaN guard
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Optional crop. On a phone-shot or stand-heavy clip most of the frame is
    # crowd or sky, and those pixels consume the detector's input budget without
    # carrying anything detectable. Cropping raises the effective resolution on
    # the players at no extra inference cost.
    crop_box = None
    if args.crop:
        try:
            values = [float(v) for v in args.crop.split(",")]
        except ValueError:
            values = []
        if len(values) != 4:
            print("[error] --crop expects 'x1,y1,x2,y2' with values in 0..1", file=sys.stderr)
            return 2
        nx1, ny1, nx2, ny2 = values
        crop_box = (
            max(0, min(int(nx1 * width), width - 1)),
            max(0, min(int(ny1 * height), height - 1)),
            max(1, min(int(nx2 * width), width)),
            max(1, min(int(ny2 * height), height)),
        )
        if crop_box[2] - crop_box[0] < 32 or crop_box[3] - crop_box[1] < 32:
            print("[error] --crop produced a region smaller than 32x32", file=sys.stderr)
            return 2

    base_w = (crop_box[2] - crop_box[0]) if crop_box else width
    base_h = (crop_box[3] - crop_box[1]) if crop_box else height

    # Optional downscale: 4K footage costs a lot of decode/draw/write time while
    # contributing nothing to accuracy, since the detector resizes to --imgsz
    # internally anyway. Dimensions are rounded to even numbers for the codec.
    proc_w, proc_h = base_w, base_h
    if args.max_width and base_w > args.max_width:
        scale = args.max_width / float(base_w)
        proc_w = int(base_w * scale) // 2 * 2
        proc_h = int(base_h * scale) // 2 * 2
    resized = (proc_w, proc_h) != (base_w, base_h)

    # With --stride N only every Nth frame is observed, so the time between
    # consecutive observations is N/fps, not 1/fps. Using the raw fps would
    # under-report every speed by a factor of N.
    stride = max(int(args.stride), 1)
    effective_fps = fps / stride
    ignore_top_px = int(round(max(0.0, min(0.95, args.ignore_top)) * proc_h))
    suppressed = [0]  # mutable counter, reported at the end

    print(f"[info] {width}x{height} @ {fps:.2f} fps, {total or '?'} frames")
    if crop_box:
        print(f"[info] cropping to x{crop_box[0]}..{crop_box[2]}, y{crop_box[1]}..{crop_box[3]} "
              f"({base_w}x{base_h})")
    if resized:
        print(f"[info] downscaling to {proc_w}x{proc_h} for processing")
    if stride > 1:
        print(f"[info] stride {stride}: effective sampling rate {effective_fps:.2f} fps")

    os.makedirs(args.out, exist_ok=True)

    # --- components -------------------------------------------------------
    detector = Detector(
        model_path=args.model,
        conf_thres=args.conf,
        iou_thres=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )
    try:
        class_ids = detector.resolve_classes(args.classes.split(","))
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    detector.classes = class_ids
    class_names = detector.names

    if args.conf_override:
        overrides = {}
        for token in args.conf_override.split(","):
            token = token.strip()
            if not token:
                continue
            name, sep, value = token.partition("=")
            if not sep:
                print(f"[error] --conf-override expects 'class=threshold', got {token!r}",
                      file=sys.stderr)
                return 2
            try:
                resolved = detector.resolve_classes([name.strip()])
                overrides[resolved[0]] = float(value)
            except (ValueError, TypeError, IndexError) as exc:
                print(f"[error] bad --conf-override entry {token!r}: {exc}", file=sys.stderr)
                return 2
        detector.class_conf = overrides

    print(f"[info] keeping classes: {[class_names.get(c, c) for c in class_ids] if class_ids else 'all'}")
    if detector.class_conf:
        print("[info] confidence thresholds: default "
              + str(detector.conf_thres) + ", "
              + ", ".join(
                  f"{class_names.get(c, c)}={t}" for c, t in sorted(detector.class_conf.items())
              ))

    tracker = Tracker(
        frame_rate=fps,
        track_buffer=args.track_buffer,
        backend=args.tracker,
    )

    court = CourtHomography(sport=args.sport)
    calibration_notes = "skipped"
    if args.court_json:
        court.calibrate_from_json(args.court_json)
        calibration_notes = "manual (json)"
    elif args.court_points:
        court.calibrate_from_string(args.court_points)
        calibration_notes = "manual (cli)"

    analyzer = TrajectoryAnalyzer(fps=effective_fps)
    trails = TrailBuffer(maxlen=args.trail)
    analytics = SessionAnalytics(fps=effective_fps, frame_size=(proc_h, proc_w), class_names=class_names)

    pose = None
    if args.pose:
        from .pose_estimator import PoseEstimator

        pose = PoseEstimator()
        if not pose.available:
            print(f"[warn] {PoseEstimator.install_hint()}")
            pose = None

    # --- writers ----------------------------------------------------------
    writer = None
    video_out = os.path.join(args.out, "result.mp4")
    if args.save_video:
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, (proc_w, proc_h))
        if not writer.isOpened():
            print("[warn] video writer failed to open; continuing without video output")
            writer = None

    csv_handle = csv_writer = None
    csv_out = os.path.join(args.out, "trajectory.csv")
    if args.save_csv:
        csv_handle = open(csv_out, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(
            ["frame", "time_s", "track_id", "class", "conf", "px", "py", "w", "h",
             "court_x", "court_y", "distance_px", "speed_px_s", "speed_m_s", "calib_mode"]
        )

    # --- run loop ---------------------------------------------------------
    frame_id = 0
    processed = 0
    attempted_calibration = court.calibrated  # manual points count as done
    t_first = t_last = None
    window = "sports-tracking"
    show = bool(args.show)
    if show:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("[warn] no display available; running headless")
            show = False

    started = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if crop_box:
                frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            if resized:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            if args.stride > 1 and frame_id % args.stride:
                frame_id += 1
                continue

            # One-shot calibration, attempted on the first usable frame.
            #
            # When detection fails the pipeline stays *uncalibrated* instead of
            # falling back to a fabricated geometry. Writing metre values derived
            # from an invented court into the CSV would be worse than writing
            # none, because it looks authoritative. The heuristic remains
            # available on request.
            if not court.calibrated and not attempted_calibration and args.calibrate != "none":
                attempted_calibration = True
                if args.calibrate == "auto":
                    if court.auto_calibrate(frame):
                        calibration_notes = "auto (court quad detected)"
                    else:
                        calibration_notes = (
                            "auto detection failed; left uncalibrated "
                            "(use --calibrate default to force the heuristic)"
                        )
                elif args.calibrate == "default":
                    court.calibrate_default(frame.shape)
                    calibration_notes = "default heuristic (approximate by construction)"
                else:  # manual, but no points were supplied
                    calibration_notes = (
                        "manual requested but --court-points/--court-json missing; "
                        "left uncalibrated"
                    )
                print(f"[info] court calibration: {calibration_notes}")

            # Attach the homography as soon as one exists. This also covers the
            # case where court points were supplied on the command line, so no
            # calibration attempt is needed inside the loop at all.
            if analyzer.pixel2court is None and court.calibrated:
                analyzer.pixel2court = court.pixel2court

            detections = detector.detect(frame)
            if ignore_top_px > 0:
                before_n = len(detections)
                detections = [d for d in detections if d[3] > ignore_top_px]
                suppressed[0] += before_n - len(detections)
            tracks = tracker.update(detections, (proc_h, proc_w))

            frame_centers = {}
            frame_court = {}
            frame_speeds = {}
            frame_speed_m = {}

            for track in tracks:
                cx, cy = track.center
                sx, sy = analyzer.add_point(
                    track.track_id,
                    cx,
                    cy,
                    cls=track.cls,
                    frame_id=frame_id,
                )
                state = analyzer.get(track.track_id)
                trails.push(track.track_id, (sx, sy))

                frame_centers[track.track_id] = (sx, sy)
                frame_speeds[track.track_id] = state.speed_px_s
                frame_speed_m[track.track_id] = state.speed_m_s

                court_xy = court.pixel2court(sx, sy)
                if court_xy is not None:
                    frame_court[track.track_id] = court_xy
                if csv_writer is not None:
                    # The box and its confidence are written out alongside the
                    # centre. Downstream analysis needs the box to crop the
                    # player (jersey colour, pose) and the confidence to weight
                    # how much an event derived from this row should be trusted;
                    # a centre point alone cannot recover either.
                    csv_writer.writerow(
                        [
                            frame_id,
                            round(frame_id / fps, 4),
                            track.track_id,
                            class_names.get(track.cls, track.cls),
                            round(track.conf, 4),
                            round(sx, 2),
                            round(sy, 2),
                            track.width,
                            track.height,
                            round(court_xy[0], 3) if court_xy else "",
                            round(court_xy[1], 3) if court_xy else "",
                            round(state.distance_px, 2),
                            round(state.speed_px_s, 2),
                            round(state.speed_m_s, 3) if court.calibrated else "",
                            court.mode,
                        ]
                    )

            # Speeds are reported in metres per second whenever the court plane
            # is known; pixel rates are the only honest option otherwise.
            if court.calibrated:
                label_speeds, label_unit = frame_speed_m, "m/s"
            else:
                label_speeds, label_unit = frame_speeds, "px/s"

            analytics.update(tracks, speeds=label_speeds, centers=frame_centers)

            # --- render ---------------------------------------------------
            canvas = frame.copy()
            Visualizer.draw_court_overlay(canvas, court)
            for track in tracks:
                Visualizer.draw_trail(canvas, trails.get(track.track_id), cls=track.cls)
            Visualizer.draw_tracks(
                canvas,
                tracks,
                class_names=class_names,
                speeds=label_speeds,
                speed_unit=label_unit,
            )
            if pose is not None:
                landmarks = pose.estimate(canvas)
                pose.draw(canvas, landmarks)

            n_tracks = len(tracks)
            Visualizer.draw_hud(
                canvas,
                [
                    f"frame {frame_id}  |  {frame_id / fps:6.2f} s",
                    f"tracks {n_tracks}   backend {tracker.backend}",
                    f"calib {court.mode}   {len(analyzer.tracks)} ids seen",
                    f"source {os.path.basename(video_path)}",
                ],
            )
            Visualizer.draw_minimap(
                canvas,
                court,
                frame_court,
                class_of=analytics.track_class,
            )

            if writer is not None:
                writer.write(canvas)
            if show:
                cv2.imshow(window, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[info] interrupted by user")
                    break

            processed += 1
            now = time.time()
            if t_first is None:
                t_first = now
            t_last = now
            if processed % 30 == 0:
                span = max(now - t_first, 1e-6)
                print(f"[info] {processed} frames processed ({(processed - 1) / span:.1f} fps)")

            frame_id += 1
            if args.max_frames and processed >= args.max_frames:
                print(f"[info] reached --max-frames={args.max_frames}")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_handle is not None:
            csv_handle.close()
        if pose is not None:
            pose.close()
        if show:
            cv2.destroyAllWindows()

    # --- reports ----------------------------------------------------------
    elapsed = time.time() - started
    analytics.frames_processed = processed
    analytics.elapsed_s = elapsed
    analytics.processing_span_s = (t_last - t_first) if (t_first and t_last) else 0.0

    traj_summary = analyzer.summary(class_names=class_names)
    extras = {
        "source": os.path.basename(video_path),
        "preprocessing": {
            "source_size": [width, height],
            "crop_px": list(crop_box) if crop_box else None,
            "processing_size": [proc_w, proc_h],
            "ignore_top_fraction": args.ignore_top,
            "detections_suppressed_by_ignore_top": suppressed[0],
            "stride": stride,
        },
        "calibration": {**court.describe(), "notes": calibration_notes},
        "tracker": {
            "requested": args.tracker,
            "used": tracker.backend,
            "track_buffer": args.track_buffer,
        },
        "per_track": traj_summary,
    }
    report_path = os.path.join(args.out, "session_report.json")
    report = analytics.write_report(report_path, extras)

    artifacts = [csv_out if csv_writer else None, video_out if writer else None, report_path]

    if args.save_plots:
        heat = os.path.join(args.out, "heatmap.png")
        if analytics.write_heatmap(heat):
            artifacts.append(heat)
        board = os.path.join(args.out, "summary.png")
        unit = "m/s" if court.calibrated else "px/s"
        if analytics.write_summary_board(board, traj_summary, speed_unit=unit):
            artifacts.append(board)

    # --- console summary --------------------------------------------------
    print("\n" + "=" * 62)
    print("  SESSION SUMMARY")
    print("=" * 62)
    span = analytics.processing_span_s
    steady = (processed - 1) / span if (processed > 1 and span > 0) else 0.0
    print(f"  frames processed : {processed} ({processed / fps:.2f} s of footage)")
    print(f"  wall time        : {elapsed:.2f} s (model load + warm-up included)")
    print(f"  steady throughput: {steady:.2f} fps  ({steady / fps:.2f}x real time)")
    print(f"  unique tracks    : {len(analyzer.tracks)}")
    print(f"  tracker used     : {tracker.backend}")
    print(f"  calibration      : {court.mode} ({calibration_notes})")
    if suppressed[0]:
        print(f"  suppressed       : {suppressed[0]} detections removed by --ignore-top")
    if analytics.per_class_counts:
        print("  detections       : " + ", ".join(
            f"{class_names.get(c, c)}={n}" for c, n in sorted(analytics.per_class_counts.items())
        ))
    if traj_summary:
        print("\n  most active objects")
        print("  " + "-" * 58)
        print(f"  {'id':>4}  {'class':<12} {'frames':>7} {'dist(px)':>10} {'max speed':>11}")
        for row in traj_summary[:5]:
            print(
                f"  {row['track_id']:>4}  {str(row['class']):<12} {row['frames_seen']:>7} "
                f"{row['distance_px']:>10.1f} {row['max_speed_px_s']:>11.1f}"
            )
    print("\n  artifacts")
    for path in artifacts:
        if path:
            print(f"    - {path}")
    print("=" * 62)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[info] aborted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
