"""Shared reading layer for trajectory data.

Event detection, the self-check, the dashboard and the chart re-renderer all read
the same CSV, so parsing it lives in one place.

The original pipeline wrote only a centre point per row. A centre point cannot be
cropped, cannot be weighted by confidence and cannot distinguish a wide box from
a tall one, so :class:`TrackRow` also carries the box, its size and the detection
confidence. Rows written by older versions leave the newer fields at their
defaults, so old CSVs keep working.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

PERSON = "person"
SPORTS_BALL = "sports ball"


@dataclass
class TrackRow:
    """One tracked object in one frame."""

    frame: int
    time_s: float
    track_id: int
    cls: str
    conf: float
    x: float
    y: float
    w: int
    h: int
    distance_px: float
    speed_px_s: float
    speed_m_s: Optional[float] = None

    @property
    def is_ball(self) -> bool:
        return self.cls == SPORTS_BALL

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Approximate box, recovered from the centre and the size."""
        half_w, half_h = self.w // 2, self.h // 2
        return (int(self.x) - half_w, int(self.y) - half_h, int(self.x) + half_w, int(self.y) + half_h)


class TrajectoryData:
    """All rows of a trajectory CSV, grouped in the ways the analyses need."""

    def __init__(self, rows: List[TrackRow], fps: float) -> None:
        self.rows = rows
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.dt = 1.0 / self.fps

        self.by_frame: Dict[int, List[TrackRow]] = {}
        self.by_track: Dict[int, List[TrackRow]] = {}
        self.class_of: Dict[int, str] = {}
        # Grouped by class *and* track id, not by track id alone. Trackers reuse
        # ids across classes: on the reference clip the same id carried 1081
        # person rows alongside its ball rows, so grouping by id alone silently
        # mixes people into the ball set. That bug produced a "2300% coverage"
        # reading before it was caught, which is the sort of nonsense that only
        # shows up as an absurd number instead of as an exception.
        self.by_class_track: Dict[str, Dict[int, List[TrackRow]]] = {}
        for row in rows:
            self.by_frame.setdefault(row.frame, []).append(row)
            self.by_track.setdefault(row.track_id, []).append(row)
            self.by_class_track.setdefault(row.cls, {}).setdefault(row.track_id, []).append(row)
            self.class_of[row.track_id] = row.cls
        for bucket in self.by_track.values():
            bucket.sort(key=lambda r: r.frame)
        for grouped in self.by_class_track.values():
            for bucket in grouped.values():
                bucket.sort(key=lambda r: r.frame)

        self.frames = sorted(self.by_frame)
        self.first_frame = self.frames[0] if self.frames else 0
        self.last_frame = self.frames[-1] if self.frames else 0

    # ------------------------------------------------------------------ views
    @property
    def ball_rows(self) -> List[TrackRow]:
        return [r for r in self.rows if r.is_ball]

    @property
    def person_rows(self) -> List[TrackRow]:
        return [r for r in self.rows if r.is_person]

    def ball_tracks(self) -> Dict[int, List[TrackRow]]:
        """Ball rows only, keyed by track id."""
        return dict(self.by_class_track.get(SPORTS_BALL, {}))

    def person_tracks(self) -> Dict[int, List[TrackRow]]:
        """Person rows only, keyed by track id."""
        return dict(self.by_class_track.get(PERSON, {}))

    def ids_spanning_classes(self) -> List[int]:
        """Track ids that carry more than one class.

        An id should mean one object. An id that is sometimes a person and
        sometimes a ball means the tracker recycled it, and any analysis keyed
        on the raw id will silently blend the two.
        """
        counts: Dict[int, set] = {}
        for row in self.rows:
            counts.setdefault(row.track_id, set()).add(row.cls)
        return sorted(tid for tid, classes in counts.items() if len(classes) > 1)

    def frame_span(self) -> int:
        return self.last_frame - self.first_frame + 1

    def mean_people_per_frame(self) -> float:
        """Average number of tracked people per frame that has any person."""
        counts = [sum(1 for r in self.by_frame[f] if r.is_person) for f in self.frames]
        counts = [c for c in counts if c > 0]
        return float(sum(counts) / len(counts)) if counts else 0.0

    def median_people_per_frame(self) -> float:
        counts = sorted(sum(1 for r in self.by_frame[f] if r.is_person) for f in self.frames)
        if not counts:
            return 0.0
        mid = len(counts) // 2
        return float(counts[mid])

    def peak_people_per_frame(self, q: float = 90.0) -> float:
        """90th-percentile simultaneous player count.

        Used as the denominator for identity and recall proxies. The median was
        tried first and rejected: on frames where detection fails badly only a
        few players are found, which shrinks the denominator and makes identity
        churn *look better* as recall gets worse. A percentile of the peak is
        robust to those frames, so improving recall moves the metric in the
        direction a reader expects.
        """
        counts = [sum(1 for r in self.by_frame[f] if r.is_person) for f in self.frames]
        if not counts:
            return 0.0
        return float(max(1.0, float(np.percentile(counts, q))))

    def frame_size(self) -> Tuple[int, int]:
        """Best-effort ``(width, height)`` from the observed point spread."""
        if not self.rows:
            return (1, 1)
        xs = [r.x for r in self.rows]
        ys = [r.y for r in self.rows]
        return (max(int(max(xs)), 1), max(int(max(ys)), 1))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    got = _to_float(value)
    return default if got is None else int(got)


def load_trajectory(csv_path: str, fps: Optional[float] = None) -> TrajectoryData:
    """Read a trajectory CSV.

    ``fps`` is inferred from the ``time_s`` column when not supplied, so a CSV
    carries enough information to be analysed without its companion report.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"{csv_path} contains no rows")

    rows: List[TrackRow] = []
    for entry in raw:
        x = _to_float(entry.get("px"))
        y = _to_float(entry.get("py"))
        if x is None or y is None:
            continue
        rows.append(
            TrackRow(
                frame=_to_int(entry.get("frame")),
                time_s=_to_float(entry.get("time_s"), 0.0) or 0.0,
                track_id=_to_int(entry.get("track_id")),
                cls=(entry.get("class") or "").strip(),
                # older CSVs have no confidence column; treat them as certain
                conf=_to_float(entry.get("conf"), 1.0) or 1.0,
                x=x,
                y=y,
                # older CSVs have no box: fall back to a nominal square so that
                # code depending on size degrades instead of crashing
                w=_to_int(entry.get("w"), 0),
                h=_to_int(entry.get("h"), 0),
                distance_px=_to_float(entry.get("distance_px"), 0.0) or 0.0,
                speed_px_s=_to_float(entry.get("speed_px_s"), 0.0) or 0.0,
                speed_m_s=_to_float(entry.get("speed_m_s"), None),
            )
        )

    if fps is None:
        fps = infer_fps(raw)
    return TrajectoryData(rows, fps)


def infer_fps(rows: Sequence[dict]) -> float:
    """Derive fps from the ``time_s`` column."""
    times = sorted(
        {( _to_int(r.get("frame")), _to_float(r.get("time_s"))) for r in rows if _to_float(r.get("time_s"))}
    )
    deltas = []
    for (f0, t0), (f1, t1) in zip(times, times[1:]):
        if f1 > f0 and t1 > t0:
            deltas.append((t1 - t0) / (f1 - f0))
    if not deltas:
        return 25.0
    deltas.sort()
    return 1.0 / deltas[len(deltas) // 2]


def load_report(report_path: str) -> dict:
    """Read ``session_report.json``; returns ``{}`` when it is absent."""
    import json

    if not os.path.exists(report_path):
        return {}
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sibling_report(csv_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), "session_report.json")


@dataclass
class FrameTransform:
    """Maps coordinates in the processed frame back to the source video.

    Any tool that needs to look at the pixels, for jersey colours, pose or a
    re-read crop, must undo whatever preprocessing the pipeline applied. The pipeline
    can crop and it can downscale, and the two compose in a fixed order: crop in
    the source frame, then resize. Applying only the scale produces boxes that
    land in the wrong place whenever a crop was used, which is how the team
    assigner came to sample the crowd and the grass and still report a confident
    clustering result. A colour histogram cannot tell that it was given the wrong
    pixels.
    """

    src_width: int
    src_height: int
    proc_width: int
    proc_height: int
    crop: Optional[Tuple[int, int, int, int]] = None

    @property
    def base_width(self) -> int:
        return (self.crop[2] - self.crop[0]) if self.crop else self.src_width

    @property
    def base_height(self) -> int:
        return (self.crop[3] - self.crop[1]) if self.crop else self.src_height

    def to_source(self, x: float, y: float) -> Tuple[float, float]:
        x = x * self.base_width / max(self.proc_width, 1)
        y = y * self.base_height / max(self.proc_height, 1)
        if self.crop:
            x += self.crop[0]
            y += self.crop[1]
        return x, y

    def box_to_source(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1 = self.to_source(box[0], box[1])
        x2, y2 = self.to_source(box[2], box[3])
        return int(x1), int(y1), int(x2), int(y2)

    def is_identity(self) -> bool:
        return (
            self.crop is None
            and self.proc_width == self.src_width
            and self.proc_height == self.src_height
        )

    def describe(self) -> str:
        if self.is_identity():
            return "identity"
        parts = []
        if self.crop:
            parts.append(f"crop {self.crop}")
        if (self.proc_width, self.proc_height) != (self.base_width, self.base_height):
            parts.append(f"resize {self.base_width}x{self.base_height} -> "
                         f"{self.proc_width}x{self.proc_height}")
        return ", ".join(parts) or "identity"


def transform_from_report(report: dict, src_width: int, src_height: int) -> FrameTransform:
    """Rebuild the preprocessing transform recorded in ``session_report.json``."""
    video = report.get("video", {})
    pre = report.get("preprocessing", {})
    proc_w = int(video.get("frame_width") or src_width)
    proc_h = int(video.get("frame_height") or src_height)
    crop = pre.get("crop_px")
    crop_tuple = tuple(int(v) for v in crop) if crop and len(crop) == 4 else None
    return FrameTransform(src_width, src_height, proc_w, proc_h, crop_tuple)
