"""Backward-compatibility shim.

The canonical implementation lives in :mod:`src.detector`; this module exists
so that older scripts importing ``src.detect`` keep working.
"""

from .detector import DEFAULT_CLASSES, Detector

__all__ = ["Detector", "DEFAULT_CLASSES"]
