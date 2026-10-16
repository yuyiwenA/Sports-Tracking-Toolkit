"""A tiny, dependency-free constant-velocity Kalman filter.

The original toolkit relied on ``filterpy`` for a 4-state Kalman filter. That
pulled in a heavy, largely unmaintained dependency for ~40 lines of linear
algebra. This module keeps the same behaviour (constant velocity in the image
plane) with pure NumPy, which also removes the ``filterpy``/``numpy>=2``
compatibility hazard.

State vector: ``x = [px, py, vx, vy]^T``.
"""

from __future__ import annotations

import numpy as np


class KalmanFilterCV:
    """Constant-velocity Kalman filter over a 2-D position measurement."""

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 1.0,
        measurement_noise: float = 4.0,
        initial_uncertainty: float = 100.0,
    ) -> None:
        self.dt = float(dt)

        # State transition: position += velocity * dt
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        # Measurement: we observe position only.
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

        self.Q = np.eye(4) * float(process_noise)
        self.R = np.eye(2) * float(measurement_noise)
        self._P0 = np.eye(4) * float(initial_uncertainty)

        self.x = np.zeros((4, 1), dtype=float)
        self.P = self._P0.copy()
        self.initialized = False

    # ------------------------------------------------------------------ API
    def reset(self, z) -> None:
        """Seed the filter with a first measurement ``z = (x, y)``."""
        self.x = np.array([float(z[0]), float(z[1]), 0.0, 0.0], dtype=float).reshape(4, 1)
        self.P = self._P0.copy()
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Advance the state one time step, returning the predicted position."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.position

    def update(self, z) -> np.ndarray:
        """Correct the state with measurement ``z = (x, y)``."""
        if not self.initialized:
            self.reset(z)
            return self.position

        z_col = np.asarray(z, dtype=float).reshape(2, 1)
        innovation = z_col - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.position

    def innovation_distance(self, z) -> float:
        """Mahalanobis distance of a measurement, useful for gating outliers."""
        if not self.initialized:
            return 0.0
        z_col = np.asarray(z, dtype=float).reshape(2, 1)
        innovation = z_col - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation))

    # ------------------------------------------------------------- read-only
    @property
    def position(self) -> np.ndarray:
        return self.x[:2].ravel().copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].ravel().copy()

    @property
    def speed(self) -> float:
        """Filtered translational speed in units/second."""
        return float(np.hypot(*self.velocity))
