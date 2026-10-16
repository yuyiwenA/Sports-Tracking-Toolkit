"""Frame annotation.

Renders, in order: court quadrilateral (if calibrated), faded motion trails,
bounding boxes with class-aware colours, and a heads-up panel carrying session
statistics. Also provides the court mini-map, which projects smoothed positions
back onto a clean top-down court diagram.

Note on colours: HSV hues are used and converted on the fly, so every track
gets a stable, visually distinct colour that does not depend on a hard-coded
table of ten values.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

#: BGR colours per conventional class id (person / sports ball).
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (90, 200, 90),    # person - green
    32: (60, 180, 255),  # sports ball - amber
}
FALLBACK_COLOR = (200, 200, 200)

TRAIL_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (60, 110, 240),   # person - red-ish
    32: (0, 170, 255),   # ball - orange
}


class Visualizer:
    """Stateless drawing helpers."""

    # ------------------------------------------------------------------ boxes
    @staticmethod
    def color_for(track_id: int, cls: int) -> Tuple[int, int, int]:
        if cls in CLASS_COLORS:
            return CLASS_COLORS[cls]
        # deterministic hue per track id
        hue = (int(track_id) * 47) % 180
        hsv = np.uint8([[[hue, 200, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracks: Iterable,
        class_names: Optional[Dict[int, str]] = None,
        speeds: Optional[Dict[int, float]] = None,
        speed_unit: str = "px/s",
    ) -> np.ndarray:
        """Draw boxes + labels for the given tracks in place."""
        class_names = class_names or {}
        speeds = speeds or {}

        for track in tracks:
            color = Visualizer.color_for(track.track_id, track.cls)
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # corner accents distinguish a tracker from a detector
            length = max(8, min(x2 - x1, y2 - y1) // 4)
            thickness = 3
            for (px, py, dx, dy) in (
                (x1, y1, 1, 1),
                (x2, y1, -1, 1),
                (x1, y2, 1, -1),
                (x2, y2, -1, -1),
            ):
                cv2.line(frame, (px, py), (px + dx * length, py), color, thickness)
                cv2.line(frame, (px, py), (px, py + dy * length), color, thickness)

            # Compact labels on small boxes. On a wide shot every player box is
            # only a few dozen pixels tall, and a full "#id class speed" chip
            # would cover more of the pitch than it explains. Colour already
            # encodes the class, and the HUD carries the per-class counts.
            compact = min(x2 - x1, y2 - y1) < 34
            label = f"#{track.track_id}"
            if not compact:
                name = class_names.get(track.cls)
                if name:
                    label += f" {name}"
                if track.track_id in speeds:
                    label += f" {speeds[track.track_id]:.1f}{speed_unit}"

            Visualizer._label(frame, label, x1, y1, color, scale=0.42 if compact else 0.55)
        return frame

    @staticmethod
    def _label(frame: np.ndarray, text: str, x: int, y: int, color, scale: float = 0.55) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        top = max(0, y - th - baseline - 6)
        cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + baseline + 6), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 5, top + th + 2),
            font,
            scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------------------- trails
    @staticmethod
    def draw_trail(
        frame: np.ndarray,
        points: Sequence[Tuple[float, float]],
        cls: int = 0,
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw a fading polyline so recent motion is emphasised."""
        if len(points) < 2:
            return frame

        base = TRAIL_COLORS.get(cls)
        n = len(points)
        for i in range(1, n):
            alpha = i / n
            if base is not None:
                color = tuple(int(c * (0.35 + 0.65 * alpha)) for c in base)
            else:
                color = (int(120 * (0.35 + 0.65 * alpha)),) * 3
            p0 = (int(points[i - 1][0]), int(points[i - 1][1]))
            p1 = (int(points[i][0]), int(points[i][1]))
            cv2.line(frame, p0, p1, color, max(1, int(thickness * (0.4 + 0.6 * alpha))), cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------- HUD
    @staticmethod
    def draw_hud(frame: np.ndarray, lines: Sequence[str], origin: Tuple[int, int] = (12, 12)) -> np.ndarray:
        """Translucent statistics panel in the top-left corner."""
        if not lines:
            return frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.5, 1
        padding = 10
        line_h = 20
        width = max(cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines) + padding * 2
        height = line_h * len(lines) + padding * 2 - 4

        x, y = origin
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (28, 32, 38), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (110, 160, 220), 1)

        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x + padding, y + padding + line_h * i + 12),
                font,
                scale,
                (235, 238, 242),
                thickness,
                cv2.LINE_AA,
            )
        return frame

    # -------------------------------------------------------------- mini-map
    @staticmethod
    def draw_court_overlay(
        frame: np.ndarray, court, color=(120, 200, 255), thickness: int = 2
    ) -> np.ndarray:
        """Outline the calibrated playing area on the frame."""
        if not court.calibrated:
            return frame
        corners = [court.court2pixel(*c) for c in court.real_court]
        if any(c is None for c in corners):
            return frame
        pts = np.array([[int(p[0]), int(p[1])] for p in corners], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_minimap(
        frame: np.ndarray,
        court,
        positions: Dict[int, Tuple[float, float]],
        class_of: Optional[Dict[int, int]] = None,
        size: Tuple[int, int] = (220, 132),
        margin: int = 14,
    ) -> np.ndarray:
        """Top-down court inset with each track projected onto it."""
        if not court.calibrated:
            return frame

        class_of = class_of or {}
        map_w, map_h = size
        fh, fw = frame.shape[:2]
        x0, y0 = fw - map_w - margin, fh - map_h - margin
        if x0 < 0 or y0 < 0:
            return frame

        panel = frame.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + map_w, y0 + map_h), (26, 30, 36), -1)
        cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + map_w, y0 + map_h), (110, 160, 220), 1)

        inner_x, inner_y = x0 + 10, y0 + 10
        inner_w, inner_h = map_w - 20, map_h - 20
        cv2.rectangle(
            frame, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h), (90, 120, 150), 1
        )
        cv2.line(
            frame,
            (inner_x + inner_w // 2, inner_y),
            (inner_x + inner_w // 2, inner_y + inner_h),
            (70, 95, 120),
            1,
        )

        for tid, (cx, cy) in positions.items():
            u = cx / max(court.court_length, 1e-6)
            v = cy / max(court.court_width, 1e-6)
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                continue
            px = int(inner_x + u * inner_w)
            py = int(inner_y + v * inner_h)
            color = TRAIL_COLORS.get(class_of.get(tid, -1), (0, 200, 255))
            cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, (245, 245, 245), 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "court map",
            (x0 + 10, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )
        return frame

    # -------------------------------------------------- backward-compatible API
    @staticmethod
    def draw(frame: np.ndarray, tracks) -> np.ndarray:
        """Legacy entry point kept for compatibility (boxes + ids only)."""
        normalised = []
        for t in tracks:
            if hasattr(t, "bbox"):
                normalised.append(t)
            else:  # [x1, y1, x2, y2, tid]
                from .tracker import Track

                x1, y1, x2, y2, tid = t
                normalised.append(Track(int(tid), (int(x1), int(y1), int(x2), int(y2))))
        return Visualizer.draw_tracks(frame, normalised)
