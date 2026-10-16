"""Quality assessment that needs no ground truth.

Knowing whether a tracker improved normally requires annotated data, and
annotating sports footage is expensive. This module computes proxies from the
output alone, so a change can be judged before any labelling budget is spent:

``detection_recall``         players found per frame, over the typical peak count
``identity_fragmentation``   track ids issued per player
``track_longevity``          share of rows in long-lived track ids
``trajectory_jitter``        path wobble relative to step size
``ball_continuity``          share of the ball path observed directly

Caveats are recorded in the report by :func:`caveats`. The two that matter most:

* these are proxies. IDF1 or HOTA on annotated data remain the reference, and the
  index cannot see a consistent bias, since a tracker that always lags three
  frames by a fixed amount scores well here;
* the sub-score mappings are simple and monotonic. They are intended for
  comparing two versions of this pipeline, not for comparison against a
  published number.

What matters is the change between two runs. Fragmentation rising from 3.0 to 3.5 is a
regression without knowing the absolute quality, and that is the failure this was
built to catch: tuning one threshold, improving one class while degrading
another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ball_path import BallPath
from .trajectory_data import TrajectoryData

#: metric name -> (weight, higher_is_better)
WEIGHTS: Dict[str, Tuple[float, bool]] = {
    "identity_fragmentation": (0.30, False),
    "detection_recall": (0.20, True),
    "trajectory_jitter": (0.20, False),
    "ball_continuity": (0.15, True),
    "track_longevity": (0.15, True),
}


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "E"


def caveats() -> List[str]:
    """The standing limitations of this scorecard.

    Exposed as a function because the text is written into every report. Keeping
    one copy means a re-rendered report and a freshly computed one cannot
    disagree.
    """
    return [
        "These are label-free proxies. IDF1 and HOTA remain the reference. They cannot see a "
        "consistent bias: a tracker that always lags three frames scores well here.",
        "Sub-score mappings are simple and monotonic, intended for comparing versions of this "
        "pipeline against each other.",
        "A high overall score means the output is internally consistent. It says nothing about "
        "whether the identities are correct.",
        "detection_recall can stay high while identity is broken: the same players found under "
        "four different ids still count as found.",
    ]


@dataclass
class Metric:
    name: str
    value: float
    unit: str
    score: float
    grade: str
    meaning: str

    def as_dict(self) -> dict:
        return {
            "value": round(self.value, 4),
            "unit": self.unit,
            "score": round(self.score, 1),
            "grade": self.grade,
            "meaning": self.meaning,
        }


class SelfCheck:
    """Compute label-free quality proxies from a trajectory CSV."""

    def __init__(
        self,
        data: TrajectoryData,
        ball_path: Optional[BallPath] = None,
        long_track_min_frames: int = 10,
        longevity_fraction: float = 0.25,
    ) -> None:
        self.data = data
        self.ball = ball_path
        self.long_track_min_frames = int(long_track_min_frames)
        self.longevity_frames = int(longevity_fraction * data.frame_span())

    # --------------------------------------------------------------- measures
    def frame_coverage(self) -> float:
        """Fraction of frames in which at least one player was tracked.

        Reported as context and left unscored, because it is almost always near 1.0 even
        when tracking is poor, because finding one player is enough to pass.
        """
        if not self.data.frames:
            return 0.0
        span = list(range(self.data.first_frame, self.data.last_frame + 1))
        if not span:
            return 0.0
        found = sum(1 for f in span if any(r.is_person for r in self.data.by_frame.get(f, ())))
        return found / len(span)

    def expected_people(self) -> float:
        """Typical peak number of players simultaneously visible (p90)."""
        return self.data.peak_people_per_frame()

    def detection_recall(self) -> float:
        """Mean players found per frame, over the typical peak count.

        Answers "how much of the group is usually found", which the usual
        frame-coverage figure cannot: that one saturates as soon as a single
        player is detected.
        """
        peak = self.expected_people()
        if peak <= 0:
            return 0.0
        return min(1.0, self.data.mean_people_per_frame() / peak)

    def identity_fragmentation(self) -> float:
        """Track ids per player. ``1.0`` is perfect; higher means re-issued ids."""
        expected = self.expected_people()
        if expected <= 0:
            return float("inf")
        tracks = [tid for tid, rows in self.data.person_tracks().items()
                  if len(rows) >= self.long_track_min_frames]
        return len(tracks) / expected

    def track_longevity(self) -> float:
        """Share of player rows belonging to ids that live a while."""
        total = 0
        long_rows = 0
        for tid, rows in self.data.person_tracks().items():
            total += len(rows)
            if len(rows) >= max(self.longevity_frames, 2):
                long_rows += len(rows)
        return long_rows / total if total else 0.0

    def trajectory_jitter(self) -> float:
        """Mean second difference of position, divided by mean step size.

        Dimensionless, so it does not depend on frame size or clip speed. A
        smoothed path has a small value; a path that snaps between detections
        has a large one.
        """
        ratios: List[float] = []
        for rows in self.data.person_tracks().values():
            if len(rows) < 5:
                continue
            pts = np.array([[r.x, r.y] for r in rows], dtype=float)
            steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if steps.mean() < 1e-6:
                continue
            accel = np.linalg.norm(np.diff(pts, n=2, axis=0), axis=1)
            ratios.append(float(accel.mean() / steps.mean()))
        return float(np.median(ratios)) if ratios else 0.0

    def ball_continuity(self) -> float:
        """Share of the stitched ball path that was actually observed."""
        if self.ball is None or len(self.ball) == 0:
            return 0.0
        return 1.0 - float(self.ball.interpolated.sum()) / len(self.ball)

    def speed_tail_ratio(self) -> float:
        """p99 / median speed. A large tail suggests id switches."""
        speeds = [r.speed_px_s for r in self.data.person_rows if r.speed_px_s > 0]
        if len(speeds) < 10:
            return 0.0
        median = float(np.median(speeds))
        return float(np.percentile(speeds, 99) / median) if median > 0 else 0.0

    # ---------------------------------------------------------------- scoring
    def _score_metric(self, name: str, value: float) -> float:
        if name == "detection_recall":
            # 0.95 of the typical peak is excellent; 0.40 is unusable
            return 100.0 * _clamp01((value - 0.40) / 0.55)
        if name == "identity_fragmentation":
            # 1.0 segments per player is perfect; 3.0 is unusable
            return 100.0 * _clamp01((3.0 - value) / 2.0)
        if name == "trajectory_jitter":
            return 100.0 * _clamp01((0.50 - value) / 0.45)
        if name == "ball_continuity":
            return 100.0 * _clamp01((value - 0.30) / 0.60)
        if name == "track_longevity":
            return 100.0 * _clamp01(value / 0.70)
        return 50.0

    def report(self) -> dict:
        values = {
            "detection_recall": self.detection_recall(),
            "identity_fragmentation": self.identity_fragmentation(),
            "trajectory_jitter": self.trajectory_jitter(),
            "ball_continuity": self.ball_continuity(),
            "track_longevity": self.track_longevity(),
        }
        meanings = {
            "detection_recall": "players found per frame, over the typical peak count",
            "identity_fragmentation": "track ids issued per player (1.0 is perfect)",
            "trajectory_jitter": "path wobble relative to step size (lower is smoother)",
            "ball_continuity": "share of the ball path observed directly, without interpolation",
            "track_longevity": "share of rows belonging to long-lived track ids",
        }
        units = {
            "detection_recall": "ratio",
            "identity_fragmentation": "ids per player",
            "trajectory_jitter": "ratio",
            "ball_continuity": "ratio",
            "track_longevity": "ratio",
        }

        metrics = []
        weighted = 0.0
        for name, (weight, _higher_better) in WEIGHTS.items():
            value = values[name]
            score = self._score_metric(name, value)
            metrics.append(
                Metric(name, value, units[name], score, _grade(score), meanings[name])
            )
            weighted += weight * score

        overall = round(weighted, 1)
        return {
            "overall_score": overall,
            "overall_grade": _grade(overall),
            "metrics": {m.name: m.as_dict() for m in metrics},
            "context": {
                "frames": self.data.frame_span(),
                "frames_with_any_player": round(self.frame_coverage(), 4),
                "person_tracks_total": len(self.data.person_tracks()),
                "person_tracks_above_min_frames": sum(
                    1 for rows in self.data.person_tracks().values()
                    if len(rows) >= self.long_track_min_frames
                ),
                "typical_peak_people_per_frame": round(self.expected_people(), 2),
                "mean_people_per_frame": round(self.data.mean_people_per_frame(), 2),
                "median_people_per_frame": round(self.data.median_people_per_frame(), 2),
                "speed_p99_over_median": round(self.speed_tail_ratio(), 2),
                "ball_points": len(self.ball) if self.ball is not None else 0,
                "ball_remaining_gaps": len(self.ball.gaps()) if self.ball is not None else 0,
                # An id should identify one object. Ids that appear as both a
                # person and a ball mean the tracker recycled them, and any
                # analysis keyed on the raw id blends the two silently.
                "track_ids_spanning_classes": self.data.ids_spanning_classes(),
            },
            "caveats": caveats(),
        }
