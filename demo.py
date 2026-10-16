"""Sports-Tracking-Toolkit: command-line entry point.

Run ``python demo.py --help`` for the full option list.

Quick start::

    python demo.py --video test.mp4
    python demo.py --video test.mp4 --calibrate manual --court-points "10,20;900,10;920,500;5,480"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
