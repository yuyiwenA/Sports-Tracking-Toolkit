"""Stitch fragmented ball detections into one usable ball path.

The need for this module comes from measurement. On the reference clip the ball
detector produced 68 unbroken runs across 26 track ids and covered 59% of the
frames. Questions about a match, such as who passed to whom or who held
possession, all depend on knowing where the ball was continuously, so the
fragments have to be rejoined before any of them can be asked.

The rejoining rule is a step function of the time gap rather than a single
weighted cost, because the tolerance depends on how long the ball went
unobserved. Across four frames the ball cannot have travelled far; after two
seconds it may have been kicked sixty metres.

Gaps that survive are interpolated only when short, and every interpolated point
is flagged so downstream analysis can decide whether to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .trajectory_data import TrackRow


@dataclass
class Tracklet:
    """One unbroken run of ball detections sharing a track id."""

    track_id: int
    frames: np.ndarray
    xy: np.ndarray

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def __len__(self) -> int:
        return int(self.frames.size)

    def tail_velocity(self, window: int = 5) -> np.ndarray:
        """Pixels per frame, estimated over the last few points."""
        n = len(self)
        if n < 2:
            return np.zeros(2)
        k = min(n - 1, window)
        dt = float(self.frames[-1] - self.frames[-1 - k])
        if dt <= 0:
            return np.zeros(2)
        return (self.xy[-1] - self.xy[-1 - k]) / dt


class BallPath:
    """A single continuous ball trajectory plus how it was assembled."""

    def __init__(
        self,
        frames: np.ndarray,
        xy: np.ndarray,
        interpolated: np.ndarray,
        chains: List[List[int]],
        dropped: List[int],
        gates_hit: List[str],
    ) -> None:
        self.frames = frames
        self.xy = xy
        self.interpolated = interpolated
        self.chains = chains
        self.dropped = dropped
        self.gates_hit = gates_hit
        self._index = {int(f): i for i, f in enumerate(frames)}

    # ------------------------------------------------------------------ views
    def __len__(self) -> int:
        return int(self.frames.size)

    def has(self, frame: int) -> bool:
        return int(frame) in self._index

    def position_at(self, frame: int) -> Optional[Tuple[float, float]]:
        idx = self._index.get(int(frame))
        if idx is None:
            return None
        return float(self.xy[idx, 0]), float(self.xy[idx, 1])

    def is_interpolated(self, frame: int) -> bool:
        idx = self._index.get(int(frame))
        return bool(self.interpolated[idx]) if idx is not None else False

    def window(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        """Frames and positions inside ``[start, end]``."""
        mask = (self.frames >= start) & (self.frames <= end)
        return self.frames[mask], self.xy[mask]

    def gaps(self, min_length: int = 2) -> List[Tuple[int, int, int]]:
        """Unobserved runs, as ``(frame_before, frame_after, length)``."""
        out = []
        for i in range(len(self.frames) - 1):
            delta = int(self.frames[i + 1] - self.frames[i])
            if delta > min_length:
                out.append((int(self.frames[i]), int(self.frames[i + 1]), delta - 1))
        return out

    def speeds(self, fps: float) -> np.ndarray:
        """Per-sample speed in pixels/second (0 where a gap was bridged)."""
        speed = np.zeros(len(self), dtype=float)
        for i in range(1, len(self)):
            step = int(self.frames[i] - self.frames[i - 1])
            if step <= 0:
                continue
            dist = float(np.linalg.norm(self.xy[i] - self.xy[i - 1]))
            speed[i] = dist / (step / fps)
        return speed

    def coverage(self) -> float:
        if len(self.frames) == 0:
            return 0.0
        span = int(self.frames[-1] - self.frames[0]) + 1
        return len(self.frames) / max(span, 1)

    # ---------------------------------------------------------------- reporting
    def summary(self, dropped_total: int, before: Dict[str, float], fps: float) -> dict:
        speeds = self.speeds(fps)
        real = speeds[~self.interpolated]
        return {
            "tracklets_before": int(before.get("tracklets", 0)),
            "longest_tracklet_before": int(before.get("longest", 0)),
            "coverage_before": round(float(before.get("coverage", 0.0)), 4),
            "tracklets_dropped": dropped_total,
            "chains_after": len(self.chains),
            "primary_chain_tracklets": len(self.chains[0]) if self.chains else 0,
            "points_after": len(self),
            "interpolated_points": int(self.interpolated.sum()),
            "coverage_after": round(self.coverage(), 4),
            "remaining_gaps": len(self.gaps()),
            "median_speed_px_s": round(float(np.median(real)) if real.size else 0.0, 1),
            "p95_speed_px_s": round(float(np.percentile(real, 95)) if real.size else 0.0, 1),
            "gates": self.gates_hit,
        }


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_tracklets(ball_tracks: Dict[int, List[TrackRow]], min_len: int = 2):
    """Split ball rows into unbroken tracklets, dropping singletons.

    A one-frame ball detection is usually grass texture, a boot or a line
    intersection. Keeping such detections would let the stitcher chain unrelated
    blips into a plausible-looking path.
    """
    tracklets: List[Tracklet] = []
    dropped: List[int] = []
    for tid, rows in ball_tracks.items():
        rows = sorted(rows, key=lambda r: r.frame)
        run_frames: List[int] = []
        run_xy: List[Tuple[float, float]] = []
        prev = None
        for row in rows:
            if prev is not None and row.frame != prev + 1:
                _flush(tracklets, dropped, tid, run_frames, run_xy, min_len)
                run_frames, run_xy = [], []
            run_frames.append(row.frame)
            run_xy.append((row.x, row.y))
            prev = row.frame
        _flush(tracklets, dropped, tid, run_frames, run_xy, min_len)
    tracklets.sort(key=lambda t: t.start)
    return tracklets, dropped


def _flush(tracklets, dropped, tid, run_frames, run_xy, min_len) -> None:
    if not run_frames:
        return
    if len(run_frames) < min_len:
        dropped.extend([tid] * len(run_frames))
        return
    tracklets.append(
        Tracklet(tid, np.asarray(run_frames, dtype=int), np.asarray(run_xy, dtype=float))
    )


#: (max_gap_frames, max_jump_px) gates, applied in order. The jump allowance
#: grows with the gap because an unobserved ball may have been kicked.
DEFAULT_GATES: Sequence[Tuple[int, float]] = ((4, 140.0), (10, 260.0), (22, 380.0))


def stitch(
    tracklets: List[Tracklet],
    gates: Sequence[Tuple[int, float]] = DEFAULT_GATES,
    velocity_horizon: int = 4,
) -> Tuple[List[List[int]], List[str]]:
    """Chain tracklets into as few continuous paths as the gates allow.

    Chains hold *indices* into ``tracklets``, never track ids. A single track id
    can yield several tracklets, because the tracker loses and re-acquires the
    ball under the same id. A chain keyed on ids would look up the wrong fragment
    and duplicate points. That mistake inflated a path to 2300% of its own span
    before it was caught.
    """
    order = sorted(range(len(tracklets)), key=lambda i: tracklets[i].start)
    chains: List[List[int]] = []
    gate_log: List[str] = []

    for idx in order:
        tracklet = tracklets[idx]
        best = None
        best_cost = None
        for chain in chains:
            last = tracklets[chain[-1]]
            gap = tracklet.start - last.end
            if gap <= 0:
                continue  # overlapping in time: cannot be the same ball notion
            allowed = None
            for max_gap, max_jump in gates:
                if gap <= max_gap:
                    allowed = max_jump
                    break
            if allowed is None:
                continue
            # predict where the ball would be, capped so a long gap does not
            # extrapolate a stale velocity across the whole pitch
            predicted = last.xy[-1] + last.tail_velocity() * min(gap, velocity_horizon)
            jump = float(np.linalg.norm(predicted - tracklet.xy[0]))
            if jump > allowed:
                continue
            cost = jump + 6.0 * gap
            if best_cost is None or cost < best_cost:
                best, best_cost = chain, cost
                gate_log.append(
                    f"tracklet {chain[-1]}->{idx} gap={gap} jump={jump:.0f} gate={allowed:.0f}"
                )
        if best is None:
            chains.append([idx])
        else:
            best.append(idx)

    chains.sort(key=lambda c: sum(len(tracklets[i]) for i in c), reverse=True)
    return chains, gate_log


def build_ball_path(
    ball_tracks: Dict[int, List[TrackRow]],
    max_interpolate_gap: int = 12,
    min_tracklet_len: int = 2,
    gates: Sequence[Tuple[int, float]] = DEFAULT_GATES,
    fps: float = 25.0,
) -> Tuple[Optional[BallPath], dict]:
    """Full pipeline: tracklets -> stitch -> primary path -> interpolate.

    Returns ``(path_or_None, diagnostics)``.
    """
    tracklets, dropped = build_tracklets(ball_tracks, min_len=min_tracklet_len)
    if tracklets:
        start = min(t.start for t in tracklets)
        end = max(t.end for t in tracklets)
        span = end - start + 1
        # count distinct frames. Tracklets overlap in time, so summing their
        # lengths can exceed the span and report a coverage above 100%, which is
        # meaningless
        seen = set()
        for t in tracklets:
            seen.update(int(f) for f in t.frames)
        before = {
            "tracklets": len(tracklets),
            "distinct_track_ids": len({t.track_id for t in tracklets}),
            "longest": max((len(t) for t in tracklets), default=0),
            "coverage": len(seen) / max(span, 1),
            "span_frames": span,
        }
    else:
        before = {"tracklets": 0, "distinct_track_ids": 0, "longest": 0, "coverage": 0.0,
                  "span_frames": 0}
    if not tracklets:
        return None, {"error": "no ball tracklets after filtering", "before": before}

    chains, gate_log = stitch(tracklets, gates=gates)
    primary = chains[0]

    frames: List[int] = []
    xy: List[Tuple[float, float]] = []
    interpolated: List[bool] = []
    for i, index in enumerate(primary):
        tracklet = tracklets[index]
        if i > 0:
            prev = tracklets[primary[i - 1]]
            gap = tracklet.start - prev.end - 1
            if 0 < gap <= max_interpolate_gap:
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    point = prev.xy[-1] * (1 - frac) + tracklet.xy[0] * frac
                    frames.append(prev.end + k)
                    xy.append((float(point[0]), float(point[1])))
                    interpolated.append(True)
        for f, p in zip(tracklet.frames, tracklet.xy):
            frames.append(int(f))
            xy.append((float(p[0]), float(p[1])))
            interpolated.append(False)

    order = np.argsort(np.asarray(frames, dtype=int), kind="stable")
    unique: List[int] = []
    seen_frames = set()
    for i in order.tolist():
        if frames[i] in seen_frames:
            continue  # duplicate frames would corrupt every speed computation
        seen_frames.add(frames[i])
        unique.append(i)

    path = BallPath(
        frames=np.asarray([frames[i] for i in unique], dtype=int),
        xy=np.asarray([xy[i] for i in unique], dtype=float),
        interpolated=np.asarray([interpolated[i] for i in unique], dtype=bool),
        chains=[[tracklets[i].track_id for i in chain] for chain in chains],
        dropped=dropped,
        gates_hit=gate_log,
    )
    return path, {"before": before, "dropped": len(dropped), "fps": fps}
