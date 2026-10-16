"""Extract annotated frames and build a contact sheet from a result video.

Usage::

    python tools/screenshot.py outputs/result.mp4 -n 6 -o outputs/screenshots

Writes ``frame_0001.png`` ... plus a ``contact_sheet.png`` montage, which is
handy for reports, README images and pull-request reviews.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import cv2
import numpy as np


def sample_frames(video_path: str, count: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        total = len(frames)
        if total == 0:
            raise SystemExit("video contains no readable frames")
        idxs = np.linspace(0, total - 1, min(count, total)).astype(int)
        return [frames[i] for i in idxs]

    # skip the very first/last frames: they are often black or mid-fade
    idxs = np.linspace(int(total * 0.03), int(total * 0.97), count).astype(int)
    picked = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            picked.append(frame)
    cap.release()
    return picked


def contact_sheet(frames: List[np.ndarray], cols: int = 3, cell_w: int = 640, gap: int = 16) -> np.ndarray:
    if not frames:
        raise SystemExit("no frames to lay out")

    cell_h = int(frames[0].shape[0] * cell_w / frames[0].shape[1])
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.full((rows * cell_h + (rows + 1) * gap, cols * cell_w + (cols + 1) * gap, 3), 246, np.uint8)

    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        y = gap + r * (cell_h + gap)
        x = gap + c * (cell_w + gap)
        sheet[y:y + cell_h, x:x + cell_w] = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(sheet, (x - 1, y - 1), (x + cell_w, y + cell_h), (214, 220, 226), 1)
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description="Screenshot / contact-sheet extractor for result videos")
    parser.add_argument("video", help="annotated result video")
    parser.add_argument("-n", "--count", type=int, default=6, help="how many frames to sample")
    parser.add_argument("-o", "--out", default="outputs/screenshots", help="output directory")
    parser.add_argument("--cols", type=int, default=3, help="contact-sheet columns")
    parser.add_argument("--no-sheet", action="store_true", help="only write individual frames")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = sample_frames(args.video, args.count)
    if not frames:
        raise SystemExit("no frames sampled")

    stem = os.path.splitext(os.path.basename(args.video))[0]
    written = []
    for i, frame in enumerate(frames, 1):
        path = os.path.join(args.out, f"{stem}_frame_{i:02d}.png")
        cv2.imwrite(path, frame)
        written.append(path)

    if not args.no_sheet:
        sheet = contact_sheet(frames, cols=args.cols)
        sheet_path = os.path.join(args.out, "contact_sheet.png")
        cv2.imwrite(sheet_path, sheet)
        written.append(sheet_path)

    print(f"[ok] wrote {len(written)} file(s) to {args.out}")
    for path in written:
        print(f"     - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
