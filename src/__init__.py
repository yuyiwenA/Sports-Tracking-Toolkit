"""Sports-Tracking-Toolkit.

A modular computer-vision toolkit for sports video analytics: detection,
multi-object tracking, court homography, trajectory smoothing and
kinematic measurement.

Submodules are imported lazily (``from src.detector import Detector``) so that
lightweight pieces such as :mod:`src.court_homography` can be used without
pulling in the deep-learning stack.
"""

__version__ = "0.3.0"

__all__ = [
    "Detector",
    "Tracker",
    "Track",
    "CourtHomography",
    "TrajectoryAnalyzer",
    "Visualizer",
    "SessionAnalytics",
    "BallPath",
    "TeamAssigner",
    "EventDetector",
    "SelfCheck",
]


def __getattr__(name):  # PEP 562 lazy re-export
    if name in ("Detector",):
        from .detector import Detector
        return Detector
    if name in ("Tracker", "Track"):
        from .tracker import Track, Tracker
        return {"Tracker": Tracker, "Track": Track}[name]
    if name == "CourtHomography":
        from .court_homography import CourtHomography
        return CourtHomography
    if name == "TrajectoryAnalyzer":
        from .trajectory import TrajectoryAnalyzer
        return TrajectoryAnalyzer
    if name == "Visualizer":
        from .visualizer import Visualizer
        return Visualizer
    if name == "SessionAnalytics":
        from .analytics import SessionAnalytics
        return SessionAnalytics
    if name == "BallPath":
        from .ball_path import BallPath
        return BallPath
    if name == "TeamAssigner":
        from .teams import TeamAssigner
        return TeamAssigner
    if name == "EventDetector":
        from .events import EventDetector
        return EventDetector
    if name == "SelfCheck":
        from .selfcheck import SelfCheck
        return SelfCheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
