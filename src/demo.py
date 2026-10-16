"""Compatibility entry point.

Historically this file contained a full copy of ``demo.py``, which is how the
two copies drifted out of sync (this one even imported ``src.detector``, a
module that did not exist). It is now a thin shim that delegates to the single
implementation in :mod:`src.pipeline`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
