"""Offline event analysis: turn a trajectory CSV into match events and a scorecard.

This runs as a post-processing tool instead of another stage inside the
run loop. Inference is the expensive part and presentation is not, so keeping
them apart means the event rules, the thresholds and the charts can all be
re-tuned in seconds instead of re-processing the whole clip. The same reasoning
already applies to ``render_report.py``.

Usage::

    python tools/analyze.py --trajectory outputs/trajectory.csv \\
                            --video 13386302_3840_2160_24fps.mp4 \\
                            --out outputs

Writes ``events.csv``, ``events.json``, ``selfcheck.json`` and ``events.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ball_path import build_ball_path  # noqa: E402
from src.events import EventDetector  # noqa: E402
from src.selfcheck import SelfCheck  # noqa: E402
from src.teams import TeamAssigner, team_colour_hex  # noqa: E402
from src.trajectory_data import (  # noqa: E402
    load_report,
    load_trajectory,
    sibling_report,
    transform_from_report,
)

PLOT_INK = "#1d232b"
PLOT_MUTED = "#5b6673"
PLOT_LINE = "#e6eaef"
TEAM_FALLBACK = {"A": "#2f6fed", "B": "#e0533d", "official": "#c9821a", "unknown": "#b4b2a9"}

KIND_COLORS = {
    "pass": "#2f6fed",
    "turnover": "#e0533d",
    "sprint": "#12a37a",
    "change_of_direction": "#c9821a",
    "ball_speed_peak": "#8a5cd6",
    "shot": "#d0453a",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tools/analyze.py",
        description="Detect match events and score output quality from a trajectory CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectory", default="outputs/trajectory.csv")
    p.add_argument("--video", default=None,
                   help="source video; needed only for jersey-colour team assignment")
    p.add_argument("--out", default=None, help="output directory (default: alongside the CSV)")
    p.add_argument("--no-teams", action="store_true", help="skip jersey-colour team assignment")
    p.add_argument("--teams", type=int, default=2,
                   help="number of teams to cluster for; <=0 searches for the best k instead")
    p.add_argument("--sprint-threshold", type=float, default=None,
                   help="sprint speed in px/s; by default 2.6x the clip median speed")
    p.add_argument("--attacking-direction", choices=("up", "down"), default=None,
                   help="enable shot detection toward this image direction (static camera only)")
    p.add_argument("--max-interpolate-gap", type=int, default=12,
                   help="longest ball gap to bridge by interpolation")
    p.add_argument("--no-charts", action="store_true")
    return p


def team_colours(team_result) -> Dict[str, str]:
    colours = dict(TEAM_FALLBACK)
    if team_result is not None and team_result.colours:
        for name, bgr in team_result.colours.items():
            colours[name] = team_colour_hex(bgr)
    return colours


def source_transform(report: dict, video_path: Optional[str]):
    """Rebuild the crop/resize transform so crops land on the right pixels."""
    import cv2

    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return transform_from_report(report, width, height)


def write_events_csv(path: str, events) -> None:
    columns = ["frame", "time_s", "kind", "team", "track_id", "partner_id",
               "partner_team", "magnitude", "confidence", "detail"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for event in events:
            writer.writerow(event.as_row())


def render_board(
    path: str,
    data,
    ball_path,
    events,
    segments,
    per_player,
    colours: Dict[str, str],
    fps: float,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return None

    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d8dde3",
        "text.color": PLOT_INK,
        "axes.labelcolor": PLOT_MUTED,
        "xtick.color": PLOT_MUTED,
        "ytick.color": PLOT_MUTED,
        "axes.grid": True,
        "grid.color": PLOT_LINE,
        "grid.linewidth": 0.8,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4))
    fig.suptitle("Match events: semantic layer", fontsize=13, fontweight="bold",
                 color="#12171d", x=0.02, ha="left")

    # 1. ball speed with events marked
    ax = axes[0][0]
    speeds = ball_path.speeds(fps)
    times = ball_path.frames / fps
    ax.plot(times, speeds, color="#2f6fed", linewidth=1.0, label="ball speed")
    ax.fill_between(times, speeds, color="#2f6fed", alpha=0.12)
    for kind in ("pass", "ball_speed_peak"):
        pts = [(e.time_s, e.magnitude or 0.0) for e in events if e.kind == kind]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16,
                       color=KIND_COLORS[kind], zorder=3, label=kind.replace("_", " "))
    ax.set_title("Ball speed and events")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("px/s")
    ax.legend(frameon=False, fontsize=8)

    # 2. possession timeline
    ax = axes[0][1]
    if segments:
        for seg in segments:
            colour = colours.get(seg.team, TEAM_FALLBACK["unknown"])
            ax.barh(0, seg.duration(fps), left=seg.start / fps, height=0.5,
                    color=colour, edgecolor="none")
        ax.set_ylim(-0.6, 0.6)
        share = {}
        total = sum(s.duration(fps) for s in segments)
        for seg in segments:
            share[seg.team] = share.get(seg.team, 0.0) + seg.duration(fps)
        ax.legend(
            handles=[Patch(facecolor=colours.get(k, "#b4b2a9"),
                           label=f"{k}  {v / max(total, 1e-9) * 100:.0f}%")
                     for k, v in sorted(share.items(), key=lambda kv: -kv[1])],
            frameon=False, fontsize=8, loc="upper right",
        )
    ax.set_title(f"Possession timeline ({len(segments)} segments)")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)

    # 3. per-player event counts
    ax = axes[1][0]
    rows = sorted(per_player.items(), key=lambda kv: -(kv[1]["sprints"] + kv[1]["turns"]))[:10]
    if rows:
        labels = [f"#{tid}" for tid, _ in rows][::-1]
        sprints = [v["sprints"] for _, v in rows][::-1]
        turns = [v["turns"] for _, v in rows][::-1]
        ax.barh(labels, sprints, color=KIND_COLORS["sprint"], height=0.62, label="sprints")
        ax.barh(labels, turns, left=sprints, color=KIND_COLORS["change_of_direction"],
                height=0.62, label="changes of direction")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Movement events per player")
    ax.set_xlabel("count")
    ax.grid(axis="y", visible=False)

    # 4. event timeline
    ax = axes[1][1]
    kinds = ["pass", "turnover", "sprint", "change_of_direction", "ball_speed_peak"]
    present = [k for k in kinds if any(e.kind == k for e in events)]
    for i, kind in enumerate(present):
        xs = [e.time_s for e in events if e.kind == kind]
        ax.scatter(xs, [i] * len(xs), s=18, color=KIND_COLORS[kind], alpha=0.85)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([k.replace("_", " ") for k in present])
    ax.set_title(f"Event timeline ({len(events)} events)")
    ax.set_xlabel("time (s)")
    ax.grid(axis="y", visible=False)

    for row in axes:
        for ax_ in row:
            ax_.spines["top"].set_visible(False)
            ax_.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = args.trajectory
    if not os.path.exists(csv_path):
        print(f"[error] not found: {csv_path}", file=sys.stderr)
        return 2
    out_dir = args.out or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    report = load_report(sibling_report(csv_path))
    fps = report.get("video", {}).get("fps")
    data = load_trajectory(csv_path, fps=fps)
    print(f"[info] {len(data.rows)} rows, {len(data.person_tracks())} player tracks, "
          f"{len(data.ball_tracks())} ball tracklets, {data.frame_span()} frames @ {data.fps:.2f} fps")

    # --- ball path -------------------------------------------------------
    ball_path, diagnostics = build_ball_path(
        data.ball_tracks(), max_interpolate_gap=args.max_interpolate_gap, fps=data.fps
    )
    if ball_path is None:
        print("[error] could not assemble a ball path; nothing to analyse", file=sys.stderr)
        return 1
    before = diagnostics["before"]
    print(f"[info] ball path: {before['tracklets']} tracklets (longest {before['longest']}) "
          f"-> {len(ball_path.chains)} chains, primary uses "
          f"{len(ball_path.chains[0])} tracklets")
    print(f"[info] ball coverage {before['coverage'] * 100:.1f}% -> {ball_path.coverage() * 100:.1f}%, "
          f"{int(ball_path.interpolated.sum())} interpolated points, "
          f"{len(ball_path.gaps())} gaps left")

    # --- teams -----------------------------------------------------------
    team_result = None
    if args.video and not args.no_teams and os.path.exists(args.video):
        transform = source_transform(report, args.video)
        if transform is not None and not transform.is_identity():
            print(f"[info] source transform: {transform.describe()}")
        team_result = TeamAssigner(n_teams=args.teams).fit(
            args.video, data.person_tracks(), transform=transform
        )
        print(f"[info] teams: separation {team_result.strength.upper()}, {team_result.reason}")
        if team_result.colours:
            print("       kit colours: " + ", ".join(
                f"{name} {team_colour_hex(bgr)}" for name, bgr in sorted(team_result.colours.items())))
        print("       players per label: " + ", ".join(
            f"{k}={v}" for k, v in sorted(_count_teams(team_result).items())))
        if not team_result.reliable:
            print("       NOTE: team labels are unreliable; team-based reasoning is disabled "
                  "and events fall back to team-agnostic rules")
        elif team_result.strength == "weak":
            print("       NOTE: separation is weak; treat team-dependent events as indicative")
    else:
        print("[info] teams: skipped (no --video), events fall back to team-agnostic rules")

    # --- events ----------------------------------------------------------
    detector = EventDetector(
        data,
        ball_path,
        teams=team_result,
        sprint_threshold_px_s=args.sprint_threshold,
        attacking_direction=args.attacking_direction,
    )
    events, segments, per_player = detector.detect()
    summary = detector.summary(events, segments, per_player)
    possession_diag = detector.possession_diagnostics()

    # --- self-check ------------------------------------------------------
    scorecard = SelfCheck(data, ball_path).report()

    colours = team_colours(team_result)
    events_csv = os.path.join(out_dir, "events.csv")
    write_events_csv(events_csv, events)

    payload = {
        "source": os.path.basename(csv_path),
        "ball_path": ball_path.summary(len(ball_path.dropped), before, data.fps),
        "teams": (
            {
                "reliable": team_result.reliable,
                "strength": team_result.strength,
                "reason": team_result.reason,
                "silhouette": round(team_result.silhouette, 4),
                "clusters": team_result.n_clusters,
                "colours": {k: team_colour_hex(v) for k, v in team_result.colours.items()},
                "players_per_team": _count_teams(team_result),
            }
            if team_result
            else None
        ),
        "events": summary,
        "possession_diagnostics": possession_diag,
        "selfcheck": scorecard,
        "possession_segments": [
            {"track_id": s.track_id, "team": s.team, "start": s.start, "end": s.end,
             "duration_s": round(s.duration(data.fps), 2)}
            for s in segments
        ],
    }
    events_json = os.path.join(out_dir, "events.json")
    with open(events_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    score_json = os.path.join(out_dir, "selfcheck.json")
    with open(score_json, "w", encoding="utf-8") as handle:
        json.dump(scorecard, handle, indent=2, ensure_ascii=False)

    artifacts = [events_csv, events_json, score_json]
    if not args.no_charts:
        board = os.path.join(out_dir, "events.png")
        if render_board(board, data, ball_path, events, segments, per_player, colours, data.fps):
            artifacts.append(board)

    # --- console report --------------------------------------------------
    print("\n" + "=" * 66)
    print("  MATCH EVENTS")
    print("=" * 66)
    for kind, count in sorted(summary["event_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<24} {count}")
    print(f"\n  possession segments   {summary['possession_segments']}")
    print(f"  possession switches   {summary['possession_switches']}")
    if team_result and team_result.reliable:
        share = summary["possession_share"]
        print("  possession share      " + ", ".join(
            f"{k} {v * 100:.0f}%" for k, v in share.items() if v > 0))
    else:
        print("  possession share      team labels unavailable, share not reported")
    if not segments:
        dist = possession_diag.get("distance_to_nearest_player_px", {})
        print(f"\n  [why no possession] ball was within "
              f"{possession_diag.get('possession_radius_px', 0):.0f} px of a player in only "
              f"{possession_diag.get('frames_within_radius', 0)} of "
              f"{possession_diag.get('ball_frames', 0)} ball frames")
        if dist:
            print(f"                      distance to nearest player: "
                  f"p25 {dist['p25']:.0f}, p50 {dist['p50']:.0f}, p75 {dist['p75']:.0f} px")
        print("                      -> the ball detections are mostly far from any player, "
              "i.e. these are false positives and not the real ball")
    print(f"  sprint threshold      {summary['sprint_threshold_px_s']:.0f} px/s "
          f"({summary['sprint_threshold_basis']})")

    print("\n" + "=" * 66)
    print("  OUTPUT QUALITY SELF-CHECK (no ground truth used)")
    print("=" * 66)
    print(f"  overall  {scorecard['overall_score']}/100   grade {scorecard['overall_grade']}")
    for name, metric in scorecard["metrics"].items():
        print(f"    {name:<24} {metric['value']:>8.3f} {metric['unit']:<14} "
              f"{metric['score']:>5.1f}  {metric['grade']}")
    print("\n  these are proxies, not IDF1/HOTA; see selfcheck.json for caveats")
    print("\n  artifacts")
    for art in artifacts:
        print(f"    - {art}")
    print("=" * 66)
    return 0


def _count_teams(team_result) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in team_result.teams.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    sys.exit(main())
