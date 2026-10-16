"""Rebuild the session report charts from a saved trajectory CSV.

Inference is the expensive part of this pipeline and presentation is not.
Separating them means the summary board can be re-rendered, re-styled or
re-tuned in a second instead of re-processing the whole clip.

Usage::

    python tools/render_report.py outputs/trajectory.csv --out outputs
    python tools/render_report.py outputs/trajectory.csv --out . --width 1600 --height 900 --fps 24

Frame size and fps are read from ``session_report.json`` sitting next to the
CSV when it is present; otherwise pass ``--width`` / ``--height`` / ``--fps``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics import SessionAnalytics  # noqa: E402


class _Row:
    """Minimal stand-in for a tracker row, so analytics can be reused as-is."""

    __slots__ = ("track_id", "cls")

    def __init__(self, track_id: int, cls: int) -> None:
        self.track_id = track_id
        self.cls = cls


def load_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-render session charts from trajectory.csv")
    parser.add_argument("csv_path", help="trajectory.csv produced by the pipeline")
    parser.add_argument("--report", default=None, help="session_report.json (defaults to a sibling file)")
    parser.add_argument("--out", default=None, help="output directory (defaults to the CSV's directory)")
    parser.add_argument("--width", type=int, default=None, help="frame width, if no report JSON is available")
    parser.add_argument("--height", type=int, default=None, help="frame height, if no report JSON is available")
    parser.add_argument("--fps", type=float, default=None, help="source fps, if no report JSON is available")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv_path))
    os.makedirs(out_dir, exist_ok=True)

    report_path = args.report or os.path.join(
        os.path.dirname(os.path.abspath(args.csv_path)), "session_report.json"
    )
    meta = {}
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        print(f"[info] read metadata from {report_path}")

    video = meta.get("video", {})
    fps = args.fps or video.get("fps")
    width = args.width or video.get("frame_width")
    height = args.height or video.get("frame_height")
    if not (fps and width and height):
        print("[error] frame size and fps unknown; pass --fps/--width/--height", file=sys.stderr)
        return 2

    rows = load_rows(args.csv_path)
    if not rows:
        print("[error] CSV contains no rows", file=sys.stderr)
        return 2

    # Stable class-id assignment across CSV and report.
    detected = meta.get("detections", {}) or {}
    names = sorted({r["class"] for r in rows} | set(detected))
    class_ids = {name: i for i, name in enumerate(names)}
    class_names = {i: name for name, i in class_ids.items()}

    analytics = SessionAnalytics(fps=float(fps), frame_size=(int(height), int(width)),
                                class_names=class_names)

    per_frame = defaultdict(list)
    for row in rows:
        if not row.get("px") or not row.get("py"):
            continue
        tid = int(row["track_id"])
        cls = class_ids.get(row["class"], 0)
        speed = float(row["speed_px_s"] or 0.0)
        per_frame[int(row["frame"])].append((tid, cls, float(row["px"]), float(row["py"]), speed))

    for frame in sorted(per_frame):
        entries = per_frame[frame]
        if not entries:
            continue
        analytics.update(
            [_Row(t, c) for t, c, _, _, _ in entries],
            speeds={t: s for t, _, _, _, s in entries},
            centers={t: (x, y) for t, _, x, y, _ in entries},
        )

    # Prefer the true per-class detection counts from the report: the CSV holds
    # *tracked* objects, which is a different (smaller) number.
    if detected:
        counts = Counter()
        for name, number in detected.items():
            counts[class_ids[name]] += int(number)
        analytics.per_class_counts = counts

    traj_summary = meta.get("per_track") or []
    if not traj_summary:
        # Rebuild a minimal summary straight from the CSV if the report is absent.
        seen = {}
        for row in rows:
            tid = int(row["track_id"])
            state = seen.setdefault(tid, {"track_id": tid, "class": row["class"],
                                          "frames_seen": 0, "distance_px": 0.0,
                                          "max_speed_px_s": 0.0})
            state["frames_seen"] += 1
            state["distance_px"] = float(row["distance_px"] or 0.0)
            state["max_speed_px_s"] = max(state["max_speed_px_s"], float(row["speed_px_s"] or 0.0))
        traj_summary = sorted(seen.values(), key=lambda r: r["distance_px"], reverse=True)

    unit = "m/s" if any(r.get("speed_m_s") for r in rows[:1000]) else "px/s"
    written = []
    heat = os.path.join(out_dir, "heatmap.png")
    if analytics.write_heatmap(heat):
        written.append(heat)
    board = os.path.join(out_dir, "summary.png")
    if analytics.write_summary_board(board, traj_summary, speed_unit=unit):
        written.append(board)

    print(f"[ok] rendered {len(written)} chart(s) from {len(rows)} CSV rows")
    for path in written:
        print(f"     - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
