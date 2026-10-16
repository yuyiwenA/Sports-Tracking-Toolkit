"""Session-level analytics and reporting.

Accumulates per-frame statistics during the run and, at the end, writes:

* ``session_report.json``: machine-readable summary (calibration, throughput,
  per-track kinematics, per-class counts);
* ``heatmap.png``: spatial occupancy of tracked objects;
* ``summary.png``: a four-panel analysis board (speed profiles, distance
  ranking, per-class counts, occupancy density).

Charts follow one rule: light background, hairline grid, no chartjunk.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Palette shared with the frame annotation.
PLOT_COLORS = ["#2f6fed", "#e0533d", "#12a37a", "#c9821a", "#8a5cd6", "#3d9be0"]

#: Chart colours for the conventional classes, mirroring Visualizer.CLASS_COLORS
#: so that a series in the report is the same colour as the box in the video.
#: Declared as literal hex instead of imported, because the visualiser works
#: in BGR and charts need RGB; the two are easy to mistranslate silently.
CLASS_CHART_COLORS = {
    "person": "#5ac85a",
    "sports ball": "#ffb43c",
}


class SessionAnalytics:
    """Runtime statistics collector + report writer."""
    def __init__(self, fps: float, frame_size: Tuple[int, int], class_names: Optional[Dict[int, str]] = None):
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.frame_size = (int(frame_size[1]), int(frame_size[0]))  # width, height
        self.class_names = class_names or {}

        self.frames_processed = 0
        self.elapsed_s = 0.0
        #: Time between the first and last processed frame. Throughput is
        #: derived from this instead of ``elapsed_s``, because total wall
        #: time includes model loading and first-inference warm-up, which can
        #: understate the steady-state rate by a factor of five or more.
        self.processing_span_s = 0.0
        self.per_class_counts: Counter = Counter()
        self.track_class: Dict[int, int] = {}
        self.speed_history: Dict[int, List[float]] = defaultdict(list)
        self.occupancy: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    # ------------------------------------------------------------------ ingest
    def update(
        self,
        tracks: Sequence,
        speeds: Optional[Dict[int, float]] = None,
        centers: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> None:
        speeds = speeds or {}
        centers = centers or {}
        self.per_class_counts.update(t.cls for t in tracks)
        for track in tracks:
            self.track_class[track.track_id] = track.cls
            if track.track_id in speeds:
                self.speed_history[track.track_id].append(float(speeds[track.track_id]))
            point = centers.get(track.track_id) or track.center
            self.occupancy[track.track_id].append((float(point[0]), float(point[1])))

    # ------------------------------------------------------------------ report
    def build_report(self, extras: Optional[dict] = None) -> dict:
        span = self.processing_span_s or self.elapsed_s
        frames = self.frames_processed
        rate = (frames - 1) / span if frames > 1 and span > 0 else None

        report = {
            "video": {
                "fps": round(self.fps, 3),
                "frame_width": self.frame_size[0],
                "frame_height": self.frame_size[1],
                "frames_processed": frames,
                "duration_s": round(frames / self.fps, 2),
            },
            "throughput": {
                "elapsed_s": round(self.elapsed_s, 3),
                "processing_span_s": round(span, 3),
                "processing_fps": round(rate, 2) if rate else None,
                "realtime_factor": round(rate / self.fps, 3) if rate else None,
                "warmup_excluded": bool(self.processing_span_s),
            },
            "detections": {
                self.class_names.get(cls, str(cls)): int(count)
                for cls, count in sorted(self.per_class_counts.items())
            },
            "tracked_objects": len(self.track_class),
        }
        report["tracked_objects_by_class"] = {
            self.class_names.get(cls, str(cls)): sum(
                1 for c in self.track_class.values() if c == cls
            )
            for cls in sorted(set(self.track_class.values()))
        }
        if extras:
            report.update(extras)
        return report

    def write_report(self, path: str, extras: Optional[dict] = None) -> dict:
        report = self.build_report(extras)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        return report

    # ------------------------------------------------------------------ charts
    def _class_color(self, cls: int) -> str:
        """Stable colour per class.

        Keyed by class *name*, never by class id. Keying by id looks fine until
        the ids change: a report rebuilt offline from a CSV assigns ids by
        sorted order, which would silently repaint every series and contradict
        the video it is describing.
        """
        name = self.class_names.get(cls, str(cls))
        if name in CLASS_CHART_COLORS:
            return CLASS_CHART_COLORS[name]
        names = sorted(self.class_names.values())
        index = names.index(name) if name in names else 0
        return PLOT_COLORS[index % len(PLOT_COLORS)]

    def write_heatmap(self, path: str, blur: int = 0, gamma: float = 0.7) -> Optional[str]:
        """Write a spatial occupancy plate.

        The colour ramp is composited with an alpha tied to local density instead
        of being blended uniformly onto a pale background. A uniform blend keeps
        the page light but flattens the hotspots the image exists to show.

        The blur radius scales with frame height. With a small fixed radius a
        sparse multi-object scene renders as thin streaks, which shows individual
        paths instead of activity zones.
        """
        if not self.occupancy:
            return None
        import cv2

        width, height = self.frame_size
        canvas = np.zeros((height, width), dtype=np.float32)
        for points in self.occupancy.values():
            for x, y in points:
                if 0 <= int(y) < height and 0 <= int(x) < width:
                    canvas[int(y), int(x)] += 1.0

        if blur <= 0:
            blur = max(31, int(round(height * 0.16)))
        if blur % 2 == 0:
            blur += 1
        canvas = cv2.GaussianBlur(canvas, (blur, blur), 0)
        if canvas.max() > 0:
            canvas = canvas / canvas.max()

        colored = cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
        alpha = np.clip(canvas ** gamma, 0.0, 1.0)[..., None]
        background = np.full_like(colored, 247.0)
        plate = (colored * alpha + background * (1.0 - alpha)).astype(np.uint8)

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        cv2.imwrite(path, plate)
        return path

    def write_summary_board(
        self,
        path: str,
        trajectory_summary: Sequence[dict],
        speed_unit: str = "px/s",
    ) -> Optional[str]:
        if not trajectory_summary:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None

        plt.rcParams.update(
            {
                "figure.facecolor": "#ffffff",
                "axes.facecolor": "#ffffff",
                "axes.edgecolor": "#d8dde3",
                "axes.labelcolor": "#333a44",
                "text.color": "#1d232b",
                "xtick.color": "#5b6673",
                "ytick.color": "#5b6673",
                "axes.grid": True,
                "grid.color": "#eceff3",
                "grid.linewidth": 0.8,
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.titleweight": "bold",
            }
        )

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
        fig.suptitle(
            "Sports Tracking: session analysis",
            fontsize=13,
            fontweight="bold",
            color="#12171d",
            x=0.02,
            ha="left",
        )

        # 1. speed profiles for the most-tracked objects
        ax = axes[0][0]
        top_ids = [
            row["track_id"]
            for row in sorted(trajectory_summary, key=lambda r: r["frames_seen"], reverse=True)[:4]
        ]
        for i, tid in enumerate(top_ids):
            series = self.speed_history.get(tid)
            if not series:
                continue
            ax.plot(
                np.arange(len(series)) / self.fps,
                series,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
                linewidth=1.4,
                label=f"#{tid}",
            )
        ax.set_title("Speed profile (top tracks)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"speed ({speed_unit})")
        ax.legend(frameon=False, fontsize=8, ncol=2)

        # 2. distance ranking
        ax = axes[0][1]
        rows = sorted(trajectory_summary, key=lambda r: r["distance_px"], reverse=True)[:10]
        labels = [f"#{r['track_id']}" for r in rows]
        values = [r["distance_px"] for r in rows]
        colors = [
            self._class_color(self.track_class.get(r["track_id"], 0)) for r in rows
        ] if rows else []
        ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
        ax.set_title("Distance travelled (px)")
        ax.set_xlabel("path length (px)")
        ax.grid(axis="y", visible=False)

        # 3. per-class detections
        ax = axes[1][0]
        if self.per_class_counts:
            names = [self.class_names.get(c, str(c)) for c in self.per_class_counts]
            counts = list(self.per_class_counts.values())
            colors = [self._class_color(c) for c in self.per_class_counts]
            ax.bar(names, counts, color=colors, width=0.5)
            ax.set_title("Detections per class")
            ax.set_ylabel("boxes")
            ax.grid(axis="x", visible=False)

        # 4. occupancy density (1-D marginal of the heatmap)
        ax = axes[1][1]
        width, height = self.frame_size
        acc = np.zeros((height, width), dtype=np.float32)
        for points in self.occupancy.values():
            for x, y in points:
                if 0 <= int(y) < height and 0 <= int(x) < width:
                    acc[int(y), int(x)] += 1.0
        if acc.sum() > 0:
            density = self._smooth(acc.sum(axis=1), window=max(5, height // 40))
            density = density / max(density.max(), 1e-6)
            ax.fill_between(np.arange(height), density, color="#2f6fed", alpha=0.22)
            ax.plot(np.arange(height), density, color="#2f6fed", linewidth=1.3)
        ax.set_title("Vertical occupancy density")
        ax.set_xlabel("image row")
        ax.set_ylabel("normalised density")
        ax.set_ylim(0, 1.05)
        ax.invert_yaxis()

        for row in axes:
            for ax_ in row:
                ax_.spines["top"].set_visible(False)
                ax_.spines["right"].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        """Moving-average smoothing.

        Occupancy is a histogram of a few thousand discrete points spread over
        a thousand rows, so the raw marginal is mostly zeros with isolated
        spikes. Smoothing turns it into a readable profile of where activity
        concentrates, which is the question the panel is meant to answer.
        """
        window = max(int(window), 1)
        if window <= 1 or values.size < window:
            return values
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(values, kernel, mode="same")
