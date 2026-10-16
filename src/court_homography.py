"""Court homography: map image pixels to real-world court coordinates.

The original implementation defined ``real_court`` (28x15 basketball) but never
provided a way to fill in the image-side points, so ``calibrate_manual`` was
never called and ``pixel2court`` silently returned pixels. Every distance and
speed figure in the CSV was therefore mislabelled: pixels were written into
columns named ``court_x`` / ``court_y``.

This module fixes that by making the calibration state explicit. Three modes:

``manual``   four image points supplied by the caller (or read from JSON);
``auto``     court lines detected with a Hough transform and reduced to a
             quadrilateral, which works on clean broadcast or top-down views;
``default``  a documented heuristic that maps the lower portion of the frame
             onto the court. Approximate by construction, but it keeps the
             geometric outputs in metres instead of silently leaking pixels.

Consumers can always ask :attr:`calibrated` and :attr:`mode` to know which
situation they are in.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: Playable-area dimensions in metres, ``(length, width)``.
SPORT_PRESETS = {
    "basketball": (28.0, 15.0),
    "soccer": (105.0, 68.0),
    "football": (105.0, 68.0),
    "tennis": (23.77, 10.97),
    "volleyball": (18.0, 9.0),
    "badminton": (13.4, 6.1),
    "handball": (40.0, 20.0),
    "hockey": (91.4, 55.0),
}


class CourtHomography:
    """Perspective mapping between the image plane and the court plane."""

    def __init__(self, sport: str = "basketball") -> None:
        if sport not in SPORT_PRESETS:
            raise ValueError(
                f"unknown sport {sport!r}; choose from {sorted(SPORT_PRESETS)}"
            )
        self.sport = sport
        length, width = SPORT_PRESETS[sport]
        self.court_length = float(length)
        self.court_width = float(width)

        # Court-plane corners in metres, ordered TL, TR, BR, BL.
        self.real_court = np.array(
            [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]],
            dtype=np.float32,
        )

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.mode: str = "none"  # none | manual | auto | default

    # ---------------------------------------------------------------- status
    @property
    def calibrated(self) -> bool:
        return self.H is not None

    # ------------------------------------------------------------ calibration
    def calibrate_manual(self, image_points: Sequence[Sequence[float]]) -> None:
        """Fit the homography from four image points ordered TL, TR, BR, BL."""
        img_pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        if img_pts.shape[0] != 4:
            raise ValueError("expected exactly 4 image points, ordered TL, TR, BR, BL")
        if not self._is_convex(img_pts):
            raise ValueError("the four image points do not form a convex quadrilateral")
        self._fit(img_pts)
        self.mode = "manual"

    def calibrate_from_string(self, spec: str) -> None:
        """Parse ``"x1,y1;x2,y2;x3,y3;x4,y4"`` and calibrate."""
        points = []
        for chunk in spec.strip().split(";"):
            if not chunk.strip():
                continue
            x_str, y_str = chunk.split(",")
            points.append((float(x_str), float(y_str)))
        self.calibrate_manual(points)

    def calibrate_from_json(self, path: str) -> None:
        """Read ``{"points": [[x, y], ...]}`` (optionally with ``"sport"``)."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "sport" in payload:
            self.set_sport(payload["sport"])
        points = payload["points"] if isinstance(payload, dict) else payload
        self.calibrate_manual(points)

    def set_sport(self, sport: str) -> None:
        if sport not in SPORT_PRESETS:
            raise ValueError(f"unknown sport {sport!r}")
        self.sport = sport
        length, width = SPORT_PRESETS[sport]
        self.court_length, self.court_width = float(length), float(width)
        self.real_court = np.array(
            [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]], dtype=np.float32
        )
        self.H = self.H_inv = None
        self.mode = "none"

    def calibrate_default(self, frame_shape: Sequence[int], top_fraction: float = 0.25) -> None:
        """Heuristic fallback: treat the lower part of the frame as a trapezoid.

        Assumes a wide/medium shot looking slightly down at the court: the
        visible playing surface is modelled as the region below
        ``top_fraction`` of the frame height, narrower at the top than at the
        bottom. This is an approximation and is recorded as such via
        :attr:`mode`; it exists so that downstream metrics stay in metres
        instead of silently reporting pixels.
        """
        height, width = int(frame_shape[0]), int(frame_shape[1])
        top_y = height * top_fraction
        bottom_y = float(height)
        top_inset = width * 0.18
        bottom_inset = width * 0.02

        img_pts = np.array(
            [
                [top_inset, top_y],
                [width - top_inset, top_y],
                [width - bottom_inset, bottom_y],
                [bottom_inset, bottom_y],
            ],
            dtype=np.float32,
        )
        self._fit(img_pts)
        self.mode = "default"

    def auto_calibrate(self, frame: np.ndarray, min_area_ratio: float = 0.06) -> bool:
        """Detect court markings and fit a quadrilateral. Returns success."""
        try:
            quad = self._detect_court_quad(frame)
        except Exception:
            quad = None

        if quad is None:
            return False

        area_ratio = abs(cv2.contourArea(quad.astype(np.float32))) / float(
            frame.shape[0] * frame.shape[1]
        )
        if area_ratio < min_area_ratio:
            return False

        self._fit(quad)
        self.mode = "auto"
        return True

    # ------------------------------------------------------------ transforms
    def pixel2court(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Image pixel -> court metres, or ``None`` when uncalibrated."""
        if self.H is None:
            return None
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(res[0]), float(res[1])

    def court2pixel(self, cx: float, cy: float) -> Optional[Tuple[float, float]]:
        """Court metres -> image pixel, or ``None`` when uncalibrated."""
        if self.H_inv is None:
            return None
        pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H_inv)[0][0]
        return float(res[0]), float(res[1])

    def meters_per_pixel(self, px: float, py: float, probe: float = 5.0) -> float:
        """Local scale at a pixel, estimated by finite differences."""
        if self.H is None:
            return 1.0
        a = self.pixel2court(px, py)
        b = self.pixel2court(px + probe, py)
        c = self.pixel2court(px, py + probe)
        if a is None or b is None or c is None:
            return 1.0
        dx = np.hypot(b[0] - a[0], b[1] - a[1]) / probe
        dy = np.hypot(c[0] - a[0], c[1] - a[1]) / probe
        return float((dx + dy) / 2.0)

    # -------------------------------------------------------------- internals
    def _fit(self, img_pts: np.ndarray) -> None:
        H, _ = cv2.findHomography(img_pts, self.real_court)
        if H is None:
            raise ValueError("homography estimation failed; check the point order")
        self.H = H
        self.H_inv = np.linalg.inv(H)

    @staticmethod
    def _is_convex(points: np.ndarray) -> bool:
        hull = cv2.convexHull(points.astype(np.float32))
        return len(hull) == len(points)

    @staticmethod
    def _detect_court_quad(frame: np.ndarray) -> Optional[np.ndarray]:
        """Find the playing surface as a quadrilateral.

        Two complementary strategies, because neither is sufficient alone:

        ``surface``  segment the coloured playing surface (green pitch, blue
                     court) and take the largest contour. Works whenever the
                     whole surface is inside the frame, even if markings are
                     faint or partly obscured by players.
        ``lines``    Hough-transform the bright painted lines and reduce their
                     convex hull to four corners. Works when the surface colour
                     is ambiguous but the markings are crisp.

        A quad whose corners hug the frame border is rejected either way: that
        shape means the surface is cropped, so its extent is unknown and the
        resulting metres-per-pixel would be meaningless.
        """
        height, width = frame.shape[:2]
        candidates = []
        for finder in (CourtHomography._quad_from_surface, CourtHomography._quad_from_lines):
            try:
                quad = finder(frame)
            except Exception:
                quad = None
            if quad is not None and CourtHomography._is_convex(quad):
                candidates.append(quad)

        if not candidates:
            return None

        def score(quad: np.ndarray) -> float:
            area = abs(cv2.contourArea(quad.astype(np.float32))) / float(height * width)
            margin = 0.015 * max(height, width)
            border_penalty = 1.0
            for x, y in quad:
                if min(x, y, width - 1 - x, height - 1 - y) < margin:
                    border_penalty = 0.25
                    break
            return area * border_penalty

        best = max(candidates, key=score)
        return best

    @staticmethod
    def _quad_from_surface(frame: np.ndarray) -> Optional[np.ndarray]:
        """Largest coloured playing surface, approximated to four corners."""
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Green (grass) plus blue (hard court) in one mask.
        mask = cv2.inRange(hsv, (25, 35, 35), (100, 255, 255))
        mask |= cv2.inRange(hsv, (95, 45, 45), (135, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.06 * float(height * width):
            return None

        hull = cv2.convexHull(largest)
        peri = cv2.arcLength(hull, True)
        for factor in np.linspace(0.01, 0.08, 15):
            approx = cv2.approxPolyDP(hull, factor * peri, True)
            if len(approx) == 4:
                quad = CourtHomography._order_quad(approx.reshape(4, 2).astype(np.float32))
                # a surface touching the frame edge is cropped, so its extent is unknown
                margin = 0.015 * max(height, width)
                if any(
                    min(x, y, width - 1 - x, height - 1 - y) < margin for x, y in quad
                ):
                    return None
                return quad
        return None

    @staticmethod
    def _quad_from_lines(frame: np.ndarray) -> Optional[np.ndarray]:
        """Bright painted markings reduced to four corners."""
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Court paint is bright and desaturated.
        mask = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
        )
        edges = cv2.Canny(mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(60, width // 16),
            minLineLength=width // 8,
            maxLineGap=max(10, width // 60),
        )
        if lines is None or len(lines) < 6:
            return None

        endpoints = np.concatenate([seg[0] for seg in lines], axis=0).astype(np.float32)
        hull = cv2.convexHull(endpoints.astype(np.int32))
        peri = cv2.arcLength(hull, True)

        for factor in np.linspace(0.01, 0.10, 19):
            approx = cv2.approxPolyDP(hull, factor * peri, True)
            if len(approx) != 4:
                continue
            quad = CourtHomography._order_quad(approx.reshape(4, 2).astype(np.float32))
            margin = 0.015 * max(height, width)
            if any(min(x, y, width - 1 - x, height - 1 - y) < margin for x, y in quad):
                return None
            return quad
        return None

    @staticmethod
    def _order_quad(points: np.ndarray) -> np.ndarray:
        """Order four points as TL, TR, BR, BL."""
        pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        ordered[0] = pts[np.argmin(s)]   # top-left  = smallest x + y
        ordered[2] = pts[np.argmax(s)]   # bottom-right = largest x + y
        ordered[1] = pts[np.argmin(diff)]  # top-right = smallest y - x
        ordered[3] = pts[np.argmax(diff)]  # bottom-left = largest y - x
        return ordered

    # ------------------------------------------------------------- reporting
    def describe(self) -> dict:
        return {
            "sport": self.sport,
            "mode": self.mode,
            "calibrated": self.calibrated,
            "court_length_m": self.court_length,
            "court_width_m": self.court_width,
        }
