"""Per-track trajectory smoothing and kinematic measurement.

The original version stored a filter per track and reported "speed" as
pixels-per-frame divided by ``dt``, that is, pixels per second, while the README
advertised metres. It also had no notion of distance travelled or of a track
ending.

Metric conversion is done by projecting consecutive positions into court space
and measuring the distance there, rather than multiplying a pixel step by a
scalar "metres per pixel". That distinction matters: a perspective homography
scales differently along x and y, and differently again at each location, so a
single scalar scale would silently bias every speed. Measuring the step in
court coordinates is exact by construction.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .kalman import KalmanFilterCV


class TrailBuffer:
    """Fixed-length, per-track position history for faded polyline rendering."""

    def __init__(self, maxlen: int = 90) -> None:
        self.maxlen = int(maxlen)
        self._trails: Dict[int, Deque[Tuple[float, float]]] = {}

    def push(self, track_id: int, point: Tuple[float, float]) -> None:
        trail = self._trails.get(track_id)
        if trail is None:
            trail = deque(maxlen=self.maxlen)
            self._trails[track_id] = trail
        trail.append(point)

    def get(self, track_id: int) -> List[Tuple[float, float]]:
        return list(self._trails.get(track_id, ()))

    def items(self):
        return self._trails.items()

    def prune(self, keep_ids) -> None:
        """Shrink trails belonging to tracks that no longer exist."""
        keep = set(keep_ids)
        for tid in [t for t in self._trails if t not in keep]:
            self._trails[tid].popleft()
            if not self._trails[tid]:
                del self._trails[tid]


class TrackState:
    """Everything known about one tracked object."""

    __slots__ = (
        "track_id",
        "cls",
        "kf",
        "smoothed",
        "raw",
        "distance_px",
        "distance_m",
        "speed_px_s",
        "max_speed_px_s",
        "speed_m_s",
        "max_speed_m_s",
        "first_frame",
        "last_frame",
        "frames_seen",
    )

    def __init__(self, track_id: int, cls: int, kf: KalmanFilterCV, frame_id: int) -> None:
        self.track_id = track_id
        self.cls = cls
        self.kf = kf
        self.smoothed: List[Tuple[float, float]] = []
        self.raw: List[Tuple[float, float]] = []
        self.distance_px = 0.0
        self.distance_m = 0.0
        self.speed_px_s = 0.0
        self.max_speed_px_s = 0.0
        self.speed_m_s = 0.0
        self.max_speed_m_s = 0.0
        self.first_frame = frame_id
        self.last_frame = frame_id
        self.frames_seen = 0


class TrajectoryAnalyzer:
    """Maintains one Kalman filter per track and derives kinematic metrics.

    ``pixel2court`` may be supplied up front or attached later, since court
    calibration often only becomes possible once the first frame has been
    inspected. When it is present, metric distance and speed are produced too.
    """

    def __init__(
        self,
        fps: float = 25.0,
        smoothing: bool = True,
        pixel2court: Optional[Callable[[float, float], Optional[Tuple[float, float]]]] = None,
        max_jump_ratio: float = 2.5,
    ) -> None:
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.dt = 1.0 / self.fps
        self.smoothing = smoothing
        self.pixel2court = pixel2court
        self.max_jump_ratio = float(max_jump_ratio)
        self.tracks: Dict[int, TrackState] = {}

    @property
    def calibrated(self) -> bool:
        return self.pixel2court is not None

    # ------------------------------------------------------------------ input
    def add_point(
        self,
        tid: int,
        cx: float,
        cy: float,
        cls: int = 0,
        frame_id: int = 0,
    ) -> Tuple[float, float]:
        """Feed a raw observation; returns the smoothed position."""
        state = self.tracks.get(tid)
        if state is None:
            state = TrackState(tid, cls, KalmanFilterCV(dt=self.dt), frame_id)
            state.kf.reset((cx, cy))
            state.smoothed.append((float(cx), float(cy)))
            state.raw.append((float(cx), float(cy)))
            state.frames_seen = 1
            state.last_frame = frame_id
            self.tracks[tid] = state
            return float(cx), float(cy)

        if self.smoothing:
            state.kf.predict()
            sx, sy = state.kf.update((cx, cy))
        else:
            sx, sy = float(cx), float(cy)

        state.raw.append((float(cx), float(cy)))
        state.smoothed.append((float(sx), float(sy)))
        state.frames_seen += 1
        state.last_frame = frame_id

        prev_x, prev_y = state.smoothed[-2]
        step_px = float(np.hypot(sx - prev_x, sy - prev_y))

        # Reject implausible jumps (the signature of an ID switch after a long
        # occlusion). The gate is relative to the track's own recent average
        # step, so it scales with how fast the object actually moves instead of
        # relying on an absolute pixel threshold.
        if step_px > self.max_jump_ratio * self._recent_step_scale(state):
            sx, sy = prev_x, prev_y
            state.smoothed[-1] = (sx, sy)
            step_px = 0.0
            state.speed_px_s = 0.0
        else:
            state.distance_px += step_px
            state.speed_px_s = step_px * self.fps
            state.max_speed_px_s = max(state.max_speed_px_s, state.speed_px_s)

        if self.pixel2court is not None:
            step_m = self._step_in_meters((prev_x, prev_y), (sx, sy))
            if step_m is not None:
                state.distance_m += step_m
                state.speed_m_s = step_m * self.fps
                state.max_speed_m_s = max(state.max_speed_m_s, state.speed_m_s)

        return sx, sy

    # -------------------------------------------------------------- internals
    @staticmethod
    def _recent_step_scale(state: TrackState) -> float:
        recent = state.smoothed[-11:]
        if len(recent) < 2:
            return 1e9  # no history yet: never gate the second point
        arr = np.asarray(recent, dtype=float)
        return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).mean() + 0.5)

    def _step_in_meters(
        self, p0: Tuple[float, float], p1: Tuple[float, float]
    ) -> Optional[float]:
        """Exact court-space distance between two image points."""
        c0 = self.pixel2court(*p0)
        c1 = self.pixel2court(*p1)
        if c0 is None or c1 is None:
            return None
        return float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))

    # ----------------------------------------------------------------- output
    def get(self, tid: int) -> Optional[TrackState]:
        return self.tracks.get(tid)

    def summary(
        self, class_names: Optional[Dict[int, str]] = None, calibrated: Optional[bool] = None
    ) -> List[dict]:
        """One row of per-track aggregate metrics, sorted by distance."""
        class_names = class_names or {}
        if calibrated is None:
            calibrated = self.calibrated

        rows = []
        for state in self.tracks.values():
            duration = max(state.frames_seen * self.dt, 1e-6)
            rows.append(
                {
                    "track_id": state.track_id,
                    "class": class_names.get(state.cls, str(state.cls)),
                    "frames_seen": state.frames_seen,
                    "first_frame": state.first_frame,
                    "last_frame": state.last_frame,
                    "duration_s": round(duration, 3),
                    "distance_px": round(state.distance_px, 2),
                    "avg_speed_px_s": round(state.distance_px / duration, 2),
                    "max_speed_px_s": round(state.max_speed_px_s, 2),
                    "distance_m": round(state.distance_m, 2) if calibrated else None,
                    "avg_speed_m_s": round(state.distance_m / duration, 3) if calibrated else None,
                    "max_speed_m_s": round(state.max_speed_m_s, 3) if calibrated else None,
                }
            )
        rows.sort(key=lambda r: r["distance_px"], reverse=True)
        return rows

    @property
    def active_count(self) -> int:
        return len(self.tracks)
