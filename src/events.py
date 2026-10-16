"""The semantic layer: turn positions into things that happened.

The pipeline up to this point answers who was where. This module answers what
happened.

Every event is derived from relationships between objects: ball to player
distance, ball to previous-possessor distance, and each player's own speed
history. None of them use an absolute position in the frame, because the footage
moves. In the reference clip the penalty area travels from x≈160-800 to x≈80-650
across the sequence, so any rule of the form "the ball entered this region" would
be measuring the camera. Rules built on relative geometry survive camera motion, so
passes, possession and sprints work on this clip while zone-based rules do not.

Each event carries the evidence that produced it. There is no ground truth for
these clips, so inspection is the only available verification and it has to be
possible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ball_path import BallPath
from .teams import TeamResult
from .trajectory_data import TrackRow, TrajectoryData

#: Event kinds, in rough order of how interesting they are.
PASS = "pass"
TURNOVER = "turnover"
POSSESSION = "possession"
SPRINT = "sprint"
CHANGE_OF_DIRECTION = "change_of_direction"
BALL_SPEED_PEAK = "ball_speed_peak"
SHOT = "shot"
LOOSE_BALL = "loose_ball"


@dataclass
class Event:
    frame: int
    time_s: float
    kind: str
    team: str = "unknown"
    track_id: Optional[int] = None
    partner_id: Optional[int] = None
    partner_team: str = "unknown"
    magnitude: Optional[float] = None
    confidence: float = 1.0
    detail: str = ""

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    """A stretch of frames during which one player was in control of the ball."""

    track_id: int
    start: int
    end: int
    team: str = "unknown"

    @property
    def frames(self) -> int:
        return self.end - self.start + 1

    def duration(self, fps: float) -> float:
        return self.frames / fps


def _percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    return float(np.percentile(values, q)) if len(values) else default


class EventDetector:
    """Detect match events from a stitched ball path plus player trajectories.

    Thresholds are expressed in pixels per second, because the
    reference clip is uncalibrated. Where a threshold has a natural physical
    counterpart, since a sprint is about 5.5 m/s, the code prefers the metric value
    whenever a homography made metres available, and otherwise falls back to a
    population-relative rule and says so in the output.
    """

    def __init__(
        self,
        data: TrajectoryData,
        ball_path: BallPath,
        teams: Optional[TeamResult] = None,
        possession_radius_px: Optional[float] = None,
        min_possession_frames: int = 3,
        pass_min_travel_factor: float = 1.4,
        pass_max_gap_s: float = 3.5,
        min_sprint_frames: int = 8,
        sprint_threshold_px_s: Optional[float] = None,
        cod_angle_deg: float = 60.0,
        cod_min_gap_frames: int = 10,
        attacking_direction: Optional[str] = None,
        shot_speed_px_s: Optional[float] = None,
    ) -> None:
        self.data = data
        self.ball = ball_path
        self.teams = teams
        self.fps = data.fps
        self.dt = data.dt

        self.min_possession_frames = int(min_possession_frames)
        self.pass_min_travel_factor = float(pass_min_travel_factor)
        self.pass_max_gap_frames = int(round(pass_max_gap_s * self.fps))
        self.min_sprint_frames = int(min_sprint_frames)
        self.cod_angle_deg = float(cod_angle_deg)
        self.cod_min_gap_frames = int(cod_min_gap_frames)
        self.attacking_direction = attacking_direction

        # A player is "in possession" when the ball is within roughly one body
        # height of them. Derived from the data rather than hard-coded, because
        # the apparent size of a player changes with the camera's zoom.
        heights = [r.h for r in data.person_rows if r.h > 0]
        self.median_person_height = float(np.median(heights)) if heights else 0.0
        if possession_radius_px is not None:
            self.possession_radius = float(possession_radius_px)
        elif self.median_person_height > 0:
            self.possession_radius = 1.6 * self.median_person_height
        else:
            width, _ = data.frame_size()
            self.possession_radius = 0.06 * width

        # Sprint threshold: prefer a real speed limit when the clip is metric.
        metric_speeds = [r.speed_m_s for r in data.person_rows if r.speed_m_s]
        if sprint_threshold_px_s is not None:
            self.sprint_threshold = float(sprint_threshold_px_s)
            self.sprint_basis = "explicit px/s"
        elif metric_speeds:
            median_m = float(np.median(metric_speeds))
            self.sprint_threshold = median_m * 2.6 / max(self._px_per_metre_estimate(), 1e-9)
            self.sprint_basis = "derived from metric speeds"
        else:
            speeds = [r.speed_px_s for r in data.person_rows if r.speed_px_s > 0]
            median_speed = float(np.median(speeds)) if speeds else 1.0
            self.sprint_threshold = 2.6 * median_speed
            self.sprint_basis = f"2.6x the clip median speed ({median_speed:.0f} px/s)"

        ball_speeds = ball_path.speeds(self.fps)
        self.shot_speed_threshold = (
            float(shot_speed_px_s)
            if shot_speed_px_s is not None
            else _percentile(ball_speeds[ball_speeds > 0], 85, 1.0)
        )

    def _px_per_metre_estimate(self) -> float:
        """Rough pixels-per-metre, only used to reconcile metric thresholds."""
        for row in self.data.person_rows:
            if row.speed_m_s and row.speed_px_s:
                return row.speed_px_s / row.speed_m_s
        return 1.0

    # --------------------------------------------------------------- utilities
    def nearest_person(self, frame: int, point: Tuple[float, float]):
        """Closest tracked person to ``point`` in ``frame``, with its distance."""
        best_row, best_dist = None, None
        for row in self.data.by_frame.get(frame, ()):  # type: ignore[arg-type]
            if not row.is_person:
                continue
            dist = float(np.hypot(row.x - point[0], row.y - point[1]))
            if best_dist is None or dist < best_dist:
                best_row, best_dist = row, dist
        return best_row, (best_dist if best_dist is not None else float("inf"))

    def team_of(self, track_id: Optional[int]) -> str:
        if track_id is None or self.teams is None:
            return "unknown"
        return self.teams.team_of(track_id)

    # ------------------------------------------------------------- possession
    def possession_segments(self) -> List[Segment]:
        raw: List[Segment] = []
        for frame in self.ball.frames:
            point = self.ball.position_at(int(frame))
            if point is None:
                continue
            row, dist = self.nearest_person(int(frame), point)
            if row is None:
                continue
            if dist > self.possession_radius:
                continue  # loose ball: nobody is in control
            if raw and raw[-1].track_id == row.track_id and raw[-1].end == frame - 1:
                raw[-1].end = int(frame)
            else:
                raw.append(Segment(row.track_id, int(frame), int(frame), self.team_of(row.track_id)))

        # Treat one- and two-frame blips as tracking artefacts. Drop short
        # segments, then merge neighbours that became adjacent, so a brief loss
        # of the ball does not split one possession into three.
        kept = [s for s in raw if s.frames >= self.min_possession_frames]
        merged: List[Segment] = []
        for seg in kept:
            if merged and merged[-1].track_id == seg.track_id:
                merged[-1].end = seg.end
            elif merged and seg.start - merged[-1].end <= self.min_possession_frames:
                merged[-1].end = seg.end
            else:
                merged.append(seg)
        return merged

    # ------------------------------------------------------------------- pass
    def possession_diagnostics(self) -> dict:
        """Why possession came out the way it did.

        When the possession analysis yields nothing, printing zero is useless:
        the reader cannot tell whether the ball was never near anyone, whether
        the radius is too tight, or whether the ball was barely detected. These
        numbers answer that, and they cost one pass over the data.
        """
        distances: List[float] = []
        ball_frames = 0
        for frame in self.ball.frames:
            point = self.ball.position_at(int(frame))
            if point is None:
                continue
            ball_frames += 1
            _, dist = self.nearest_person(int(frame), point)
            if dist != float("inf"):
                distances.append(dist)
        if not distances:
            return {
                "ball_frames": ball_frames,
                "ball_frames_with_a_player_in_frame": 0,
                "note": "the ball and at least one player never coexisted in a frame",
            }
        arr = np.asarray(distances)
        radius = self.possession_radius
        return {
            "ball_frames": ball_frames,
            "ball_frames_with_a_player_in_frame": len(distances),
            "possession_radius_px": round(radius, 1),
            "distance_to_nearest_player_px": {
                "p05": round(float(np.percentile(arr, 5)), 1),
                "p25": round(float(np.percentile(arr, 25)), 1),
                "p50": round(float(np.percentile(arr, 50)), 1),
                "p75": round(float(np.percentile(arr, 75)), 1),
                "p95": round(float(np.percentile(arr, 95)), 1),
            },
            "frames_within_radius": int((arr <= radius).sum()),
            "frames_within_radius_ratio": round(float((arr <= radius).mean()), 4),
            "note": (
                "possession needs the ball to sit within the radius of a player for "
                "several consecutive frames; scattered ball false positives sit far "
                "from everyone and cannot produce a possession segment"
                if (arr <= radius).mean() < 0.15
                else "distribution looks usable for possession analysis"
            ),
        }
    def _ball_observed_ratio(self, start: int, end: int) -> float:
        frames = np.arange(start, end + 1)
        if frames.size == 0:
            return 1.0
        present = np.array([self.ball.has(int(f)) for f in frames])
        if not present.any():
            return 0.0
        real = np.array(
            [self.ball.has(int(f)) and not self.ball.is_interpolated(int(f)) for f in frames]
        )
        return float(real.sum() / present.sum())

    def _passes_and_turnovers(self, segments: List[Segment], events: List[Event]) -> None:
        for prev, nxt in zip(segments, segments[1:]):
            if prev.track_id == nxt.track_id:
                continue
            gap = nxt.start - prev.end
            if gap > self.pass_max_gap_frames:
                continue

            a = self.ball.position_at(prev.end)
            b = self.ball.position_at(nxt.start)
            if a is None or b is None:
                continue
            travel = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            # A genuine transfer moves the ball clearly away from the passer.
            # Without this, jitter in which of two nearby players is "closest"
            # would be reported as a pass dozens of times a second.
            if travel < self.pass_min_travel_factor * self.possession_radius:
                continue

            observed = self._ball_observed_ratio(prev.end, nxt.start)
            frame = nxt.start
            row = self.data.by_track.get(nxt.track_id, [])
            conf = None
            for r in row:
                if r.frame == frame:
                    conf = r.conf
                    break
            confidence = round(min(1.0, observed * (conf if conf is not None else 1.0)), 3)

            receiver_team = nxt.team
            same_team = (
                receiver_team != "unknown" and prev.team != "unknown" and receiver_team == prev.team
            )
            other_team = (
                receiver_team != "unknown" and prev.team != "unknown" and receiver_team != prev.team
            )

            if other_team:
                events.append(
                    Event(
                        frame=frame,
                        time_s=round(frame / self.fps, 3),
                        kind=TURNOVER,
                        team=prev.team,
                        track_id=prev.track_id,
                        partner_id=nxt.track_id,
                        partner_team=receiver_team,
                        magnitude=round(travel, 1),
                        confidence=confidence,
                        detail=(
                            f"ball travelled {travel:.0f} px over {gap} frames; "
                            f"possession left team {prev.team} for team {receiver_team}; "
                            f"ball observed in {observed * 100:.0f}% of the interval"
                        ),
                    )
                )
            elif same_team or receiver_team == "unknown":
                # Without team labels every transfer is still reported, but the
                # detail says the same-side reading is an assumption rather
                # than a measurement.
                note = (
                    "teams unknown, so a same-side transfer is assumed"
                    if receiver_team == "unknown" or prev.team == "unknown"
                    else f"same team ({prev.team})"
                )
                events.append(
                    Event(
                        frame=frame,
                        time_s=round(frame / self.fps, 3),
                        kind=PASS,
                        team=prev.team,
                        track_id=prev.track_id,
                        partner_id=nxt.track_id,
                        partner_team=receiver_team,
                        magnitude=round(travel, 1),
                        confidence=confidence,
                        detail=(
                            f"ball travelled {travel:.0f} px over {gap} frames ({note}); "
                            f"observed in {observed * 100:.0f}% of the interval"
                        ),
                    )
                )

    # ----------------------------------------------------------------- sprint
    def _sprints_and_turns(self, events: List[Event]) -> Dict[int, dict]:
        stats: Dict[int, dict] = {}
        for tid, rows in self.data.person_tracks().items():
            if len(rows) < 3:
                continue
            team = self.team_of(tid)
            speeds = [(r.frame, r.x, r.y, r.speed_px_s) for r in rows]

            run_start = None
            run_peak = 0.0
            sprints = 0
            for i, (frame, _, _, speed) in enumerate(speeds):
                if speed >= self.sprint_threshold:
                    if run_start is None:
                        run_start = frame
                    run_peak = max(run_peak, speed)
                    continue
                if run_start is not None:
                    if frame - run_start >= self.min_sprint_frames:
                        sprints += 1
                        events.append(
                            Event(
                                frame=run_start,
                                time_s=round(run_start / self.fps, 3),
                                kind=SPRINT,
                                team=team,
                                track_id=tid,
                                magnitude=round(run_peak, 1),
                                confidence=1.0,
                                detail=(
                                    f"{frame - run_start} frames above {self.sprint_threshold:.0f} px/s, "
                                    f"peak {run_peak:.0f} px/s"
                                ),
                            )
                        )
                    run_start, run_peak = None, 0.0
            if run_start is not None and (speeds[-1][0] - run_start) >= self.min_sprint_frames:
                sprints += 1
                events.append(
                    Event(
                        frame=run_start,
                        time_s=round(run_start / self.fps, 3),
                        kind=SPRINT,
                        team=team,
                        track_id=tid,
                        magnitude=round(run_peak, 1),
                        confidence=0.9,
                        detail=f"speed run truncated by track end, peak {run_peak:.0f} px/s",
                    )
                )

            turns = self._change_of_direction(speeds, tid, team, events)
            stats[tid] = {"team": team, "sprints": sprints, "turns": turns}
        return stats

    def _change_of_direction(self, speeds, tid: int, team: str, events: List[Event]) -> int:
        window = 5
        points = np.array([[x, y] for _, x, y, _ in speeds], dtype=float)
        frames = [f for f, _, _, _ in speeds]
        if len(points) < window + 2:
            return 0
        count = 0
        last_frame = -10 ** 9
        for i in range(window, len(points) - window):
            v1 = points[i] - points[i - window]
            v2 = points[i + window] - points[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 3 or n2 < 3:
                continue
            cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosang)))
            if angle < self.cod_angle_deg:
                continue
            if speeds[i][3] < 0.45 * self.sprint_threshold:
                continue  # standing still and wobbling is not a turn
            if frames[i] - last_frame < self.cod_min_gap_frames:
                continue
            last_frame = frames[i]
            count += 1
            events.append(
                Event(
                    frame=frames[i],
                    time_s=round(frames[i] / self.fps, 3),
                    kind=CHANGE_OF_DIRECTION,
                    team=team,
                    track_id=tid,
                    magnitude=round(angle, 1),
                    confidence=0.8,
                    detail=f"heading changed {angle:.0f} deg over {2 * window} frames",
                )
            )
        return count

    # -------------------------------------------------------------- ball peaks
    def _ball_peaks(self, top_n: int = 8) -> List[Event]:
        """Local maxima of ball speed above the threshold, strongest first."""
        speeds = self.ball.speeds(self.fps)
        peaks: List[Event] = []
        for i in range(1, len(speeds) - 1):
            if speeds[i] <= 0:
                continue
            if speeds[i] >= speeds[i - 1] and speeds[i] >= speeds[i + 1]:
                if speeds[i] < self.shot_speed_threshold:
                    continue
                frame = int(self.ball.frames[i])
                peaks.append(
                    Event(
                        frame=frame,
                        time_s=round(frame / self.fps, 3),
                        kind=BALL_SPEED_PEAK,
                        magnitude=round(float(speeds[i]), 1),
                        confidence=1.0 if not self.ball.interpolated[i] else 0.5,
                        detail=(
                            f"local speed maximum {speeds[i]:.0f} px/s "
                            f"(threshold {self.shot_speed_threshold:.0f} px/s)"
                        ),
                    )
                )
        peaks.sort(key=lambda e: -(e.magnitude or 0.0))
        return peaks[:top_n]

    # ------------------------------------------------------------------ shots
    def _shots(self, events: List[Event]) -> None:
        """Ball travelling fast toward the attacking goal.

        Off by default. It needs a fixed idea of where the goal is in the frame,
        and on a panning camera that idea drifts: the penalty area in the
        reference clip shifts by roughly a hundred pixels over the sequence. The
        option exists because on a static broadcast camera it is sound, and the
        caller must supply ``attacking_direction`` to acknowledge the assumption.
        """
        if not self.attacking_direction:
            return
        sign = {"up": -1, "down": 1}.get(self.attacking_direction)
        if sign is None:
            return
        speeds = self.ball.speeds(self.fps)
        last_frame = -10 ** 9
        for i in range(1, len(self.ball.frames)):
            if speeds[i] < self.shot_speed_threshold:
                continue
            dy = self.ball.xy[i, 1] - self.ball.xy[i - 1, 1]
            dx = self.ball.xy[i, 0] - self.ball.xy[i - 1, 0]
            if abs(dy) < 1e-6 or np.sign(dy) != sign:
                continue
            if abs(dy) < 0.5 * abs(dx):
                continue  # more sideways than forward
            frame = int(self.ball.frames[i])
            if frame - last_frame < self.fps:
                continue
            last_frame = frame
            events.append(
                Event(
                    frame=frame,
                    time_s=round(frame / self.fps, 3),
                    kind=SHOT,
                    magnitude=round(float(speeds[i]), 1),
                    confidence=0.5,
                    detail=(
                        f"ball at {speeds[i]:.0f} px/s heading {self.attacking_direction}; "
                        "image-space direction, valid only for a static camera"
                    ),
                )
            )

    # ------------------------------------------------------------------ driver
    def detect(self) -> Tuple[List[Event], List[Segment], Dict[int, dict]]:
        segments = self.possession_segments()
        events: List[Event] = []
        self._passes_and_turnovers(segments, events)
        per_player = self._sprints_and_turns(events)
        events.extend(self._ball_peaks())
        self._shots(events)
        events.sort(key=lambda e: (e.frame, e.kind))
        return events, segments, per_player

    # ----------------------------------------------------------------- summary
    def summary(
        self, events: Sequence[Event], segments: Sequence[Segment], per_player: Dict[int, dict]
    ) -> dict:
        counts: Dict[str, int] = {}
        for event in events:
            counts[event.kind] = counts.get(event.kind, 0) + 1

        by_team: Dict[str, float] = {"A": 0.0, "B": 0.0, "unknown": 0.0}
        for seg in segments:
            by_team[seg.team if seg.team in by_team else "unknown"] += seg.duration(self.fps)
        total_possession = sum(by_team.values())

        passes = [e for e in events if e.kind == PASS and (e.magnitude or 0) > 0]
        longest_pass = max(passes, key=lambda e: e.magnitude or 0, default=None)

        return {
            "frames_analysed": int(self.data.frame_span()),
            "possession_radius_px": round(self.possession_radius, 1),
            "median_person_height_px": round(self.median_person_height, 1),
            "sprint_threshold_px_s": round(self.sprint_threshold, 1),
            "sprint_threshold_basis": self.sprint_basis,
            "shot_speed_threshold_px_s": round(self.shot_speed_threshold, 1),
            "attacking_direction": self.attacking_direction,
            "possession_segments": len(segments),
            "possession_seconds_by_team": {k: round(v, 2) for k, v in by_team.items()},
            "possession_share": {
                k: round(v / total_possession, 3) if total_possession > 0 else 0.0
                for k, v in by_team.items()
            },
            "possession_switches": sum(
                1 for a, b in zip(segments, segments[1:]) if a.track_id != b.track_id
            ),
            "longest_pass": (
                {
                    "frame": longest_pass.frame,
                    "from": longest_pass.track_id,
                    "to": longest_pass.partner_id,
                    "distance_px": longest_pass.magnitude,
                }
                if longest_pass
                else None
            ),
            "per_player": per_player,
            "event_counts": counts,
        }
