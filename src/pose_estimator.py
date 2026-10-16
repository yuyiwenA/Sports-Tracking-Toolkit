"""Optional pose keypoint extraction (MediaPipe).

MediaPipe is an optional dependency: it is a large wheel with narrow Python
version support, and the tracking pipeline is fully functional without it. This
module therefore never imports MediaPipe at module import time, and
:class:`PoseEstimator.available` lets callers skip the stage cleanly instead of
crashing at ``import``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

#: Bone pairs used when drawing a skeleton, in MediaPipe's landmark indexing.
SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


class PoseEstimator:
    """Thin wrapper over ``mediapipe.solutions.pose``."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._pose = None
        try:  # pragma: no cover - depends on the environment
            import mediapipe as mp

            self._mp_pose = mp.solutions.pose
            self._mp_drawing = mp.solutions.drawing_utils
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        except Exception:
            self._pose = None

    @property
    def available(self) -> bool:
        return self._pose is not None

    @staticmethod
    def install_hint() -> str:
        return (
            "Pose estimation needs MediaPipe, which is not installed. "
            "Install it with `pip install mediapipe` (Python 3.9-3.12 recommended). "
            "The tracking pipeline runs fine without it."
        )

    def estimate(self, frame: np.ndarray) -> Optional[List[tuple]]:
        """Return landmark pixel coordinates, or ``None`` if nothing detected."""
        if self._pose is None:
            return None
        rgb = frame[:, :, ::-1]
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        height, width = frame.shape[:2]
        return [
            (lm.x * width, lm.y * height, lm.visibility)
            for lm in result.pose_landmarks.landmark
        ]

    def draw(self, frame: np.ndarray, landmarks) -> np.ndarray:
        if landmarks is None:
            return frame
        import cv2

        for a, b in SKELETON:
            if a < len(landmarks) and b < len(landmarks):
                pa = (int(landmarks[a][0]), int(landmarks[a][1]))
                pb = (int(landmarks[b][0]), int(landmarks[b][1]))
                cv2.line(frame, pa, pb, (240, 200, 90), 2, cv2.LINE_AA)
        for x, y, vis in landmarks:
            if vis > 0.4:
                cv2.circle(frame, (int(x), int(y)), 3, (90, 240, 200), -1, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
