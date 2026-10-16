"""Multi-object tracking.

Two back-ends sit behind one interface:

* ``bytetrack``, Ultralytics' ByteTrack implementation. It keeps ids through
  short occlusions, which is the common case in sports footage: players crossing,
  the ball hidden behind a body.
* ``iou``, a greedy IoU and Kalman tracker implemented in this module.

The second back-end exists because Ultralytics has changed ``BYTETracker``'s
constructor and ``update()`` signature several times, and its constructors read
attributes off a namespace-like object. Instead of pinning a version,
:class:`Tracker` inspects the installed implementation at runtime and falls back
if the interface does not match.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Track:
    """One confirmed object track for the current frame."""

    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    cls: int = 0
    conf: float = 1.0
    age: int = 0

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N, 4)`` xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class _Namespace:
    """Duck-typed stand-in for the argparse namespace Ultralytics expects."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _BoxesLike:
    """Minimal duck-type of Ultralytics' ``Boxes`` for the tracker input.

    Ultralytics 8.4 rewrote ``BYTETracker``: its ``update()`` no longer takes a
    NumPy array of detections, it takes a Results/Boxes-like object exposing
    ``conf``, ``cls`` and ``xywh``, and supporting boolean-mask indexing (it
    splits detections into high- and low-confidence subsets with
    ``results[mask]``).

    Rather than construct a full ``ultralytics.engine.results.Boxes`` object
    (whose constructor demands ``orig_shape``, an optional ``data`` tensor and a
    task hint), this small adapter supplies the surface the tracker reads.
    It keeps the project's Detector/Tracker separation intact.
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.xywh.shape[0])

    def __getitem__(self, mask):
        mask = np.asarray(mask)
        return _BoxesLike(self.xywh[mask], self.conf[mask], self.cls[mask])


def _dets_to_boxes_like(dets: np.ndarray) -> _BoxesLike:
    """Convert ``[x1, y1, x2, y2, conf, cls]`` rows into a Boxes-like object."""
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    xywh = np.stack(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=1
    )
    return _BoxesLike(xywh, dets[:, 4], dets[:, 5])



class _IoUKalmanTracker:
    """Greedy IoU association with per-track Kalman smoothing.

    Deliberately simple: predict, associate by IoU (class-constrained), correct
    matched tracks, spawn tracks for unmatched detections, retire tracks that
    go unseen for ``max_age`` frames.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_age: int = 30,
        min_hits: int = 2,
        dt: float = 1.0,
    ) -> None:
        from .kalman import KalmanFilterCV

        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.dt = float(dt)
        self._kf_factory = KalmanFilterCV
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}

    def update(self, dets: np.ndarray, frame_shape: Optional[Sequence[int]] = None) -> List[Track]:
        # 1. predict all existing tracks forward
        for state in self._tracks.values():
            state["kf"].predict()
            state["time_since_update"] += 1

        dets = np.asarray(dets, dtype=float).reshape(-1, 6) if len(dets) else np.zeros((0, 6))

        results: List[Track] = []
        if len(self._tracks) == 0 or len(dets) == 0:
            matched_pairs: List[Tuple[int, int]] = []
        else:
            track_ids = list(self._tracks.keys())
            pred_boxes = np.array(
                [self._tracks[tid]["box"] for tid in track_ids], dtype=float
            )
            det_boxes = dets[:, :4]
            ious = iou_matrix(pred_boxes, det_boxes)

            # class gate: person must not be matched to ball
            track_cls = np.array([self._tracks[tid]["cls"] for tid in track_ids])
            det_cls = dets[:, 5].astype(int)
            ious = np.where(track_cls[:, None] == det_cls[None, :], ious, 0.0)

            matched_pairs = []
            used_tracks, used_dets = set(), set()
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            for ti, di in order:
                if ious[ti, di] < self.iou_threshold:
                    break
                if ti in used_tracks or di in used_dets:
                    continue
                used_tracks.add(int(ti))
                used_dets.add(int(di))
                matched_pairs.append((int(ti), int(di)))

            unmatched_dets = [d for d in range(len(dets)) if d not in used_dets]
        if len(self._tracks) == 0 or len(dets) == 0:
            unmatched_dets = list(range(len(dets)))

        # 2. correct matched tracks
        for ti, di in matched_pairs:
            tid = track_ids[ti]
            state = self._tracks[tid]
            x1, y1, x2, y2 = dets[di, :4]
            state["kf"].update(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            state["box"] = self._compose_box(state, x1, y1, x2, y2)
            state["cls"] = int(dets[di, 5])
            state["conf"] = float(dets[di, 4])
            state["time_since_update"] = 0
            state["hits"] += 1
            state["age"] += 1

        # 3. spawn new tracks for leftover detections
        for di in unmatched_dets:
            x1, y1, x2, y2 = dets[di, :4]
            kf = self._kf_factory(dt=self.dt)
            kf.reset(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            self._tracks[self._next_id] = {
                "kf": kf,
                "box": (x1, y1, x2, y2),
                "cls": int(dets[di, 5]),
                "conf": float(dets[di, 4]),
                "size": (x2 - x1, y2 - y1),
                "hits": 1,
                "age": 0,
                "time_since_update": 0,
            }
            self._next_id += 1

        # 4. retire stale tracks, emit confirmed ones
        for tid in list(self._tracks.keys()):
            state = self._tracks[tid]
            if state["time_since_update"] > self.max_age:
                del self._tracks[tid]
                continue
            if state["hits"] >= self.min_hits and state["time_since_update"] == 0:
                results.append(
                    Track(
                        track_id=tid,
                        bbox=tuple(int(v) for v in state["box"]),
                        cls=state["cls"],
                        conf=state["conf"],
                        age=state["age"],
                    )
                )

        # 5. predict visualisation boxes for unmatched tracks so trails stay smooth
        for tid in list(self._tracks.keys()):
            state = self._tracks[tid]
            if state["time_since_update"] > 0:
                cx, cy = state["kf"].position
                w, h = state["size"]
                state["box"] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

        return results

    @staticmethod
    def _compose_box(state: dict, x1, y1, x2, y2):
        """Blend measured box size with its history, centred on the filtered position.

        Box dimensions fluctuate frame to frame far more than a player's true
        size does, so an exponential moving average on ``(w, h)`` removes visible
        breathing of the drawn rectangle while the Kalman filter handles the
        centre.
        """
        w, h = x2 - x1, y2 - y1
        prev_w, prev_h = state.get("size", (w, h))
        bw = 0.7 * w + 0.3 * prev_w
        bh = 0.7 * h + 0.3 * prev_h
        state["size"] = (bw, bh)

        cx, cy = state["kf"].position
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


class Tracker:
    """Uniform tracker façade: ByteTrack when available, IoU+Kalman otherwise."""

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: float = 25.0,
        backend: str = "bytetrack",
        iou_threshold: float = 0.25,
    ) -> None:
        self.backend = backend
        self._bt = None
        self._bt_in_kind = "array"
        self._fallback = _IoUKalmanTracker(
            iou_threshold=iou_threshold,
            max_age=int(track_buffer),
            dt=1.0 / max(float(frame_rate), 1e-6),
        )

        if backend == "bytetrack":
            built = self._try_build_bytetrack(
                track_thresh, track_buffer, match_thresh, frame_rate
            )
            if built is None:
                # Degrade gracefully instead of crashing the whole pipeline.
                self.backend = "iou"
            else:
                self._bt, self._bt_in_kind = built

    # ------------------------------------------------------------------ build
    @staticmethod
    def _try_build_bytetrack(track_thresh, track_buffer, match_thresh, frame_rate):
        """Instantiate Ultralytics' ``BYTETracker`` across API generations.

        Ultralytics changed this class twice: the constructor used to take
        ``(args, frame_rate)`` and read ``track_thresh``; since 8.4 it takes
        ``args`` alone and reads ``track_high_thresh`` / ``track_low_thresh`` /
        ``new_track_thresh``. Both shapes are attempted, so the backend keeps
        working across versions instead of silently degrading.

        Returns ``(tracker, input_kind)``, or ``None`` if no variant fits.
        """
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except Exception:
            return None

        modern = {  # ultralytics >= 8.4
            "track_high_thresh": track_thresh,
            "track_low_thresh": max(0.05, track_thresh * 0.4),
            "new_track_thresh": track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "fuse_score": False,
        }
        legacy = {  # ultralytics < 8.4
            "track_thresh": track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "mot20": False,
            "fuse_score": False,
        }

        for cfg in (modern, legacy):
            args = _Namespace(**cfg)
            for kwargs in ({"frame_rate": frame_rate}, {}):
                try:
                    tracker = BYTETracker(args, **kwargs)
                except Exception:
                    continue
                return tracker, Tracker._input_kind(tracker)
        return None

    @staticmethod
    def _input_kind(tracker) -> str:
        """``"array"`` for the pre-8.4 signature, ``"boxes"`` for 8.4+."""
        try:
            params = list(inspect.signature(tracker.update).parameters)
        except (TypeError, ValueError):
            return "array"
        return "array" if "output_results" in params else "boxes"

    # ----------------------------------------------------------------- update
    def update(self, dets: Sequence[Sequence[float]], frame_shape: Sequence[int]) -> List[Track]:
        """Advance the tracker by one frame.

        ``dets``: sequence of ``[x1, y1, x2, y2, conf, cls]``.
        ``frame_shape``: ``(height, width)`` of the frame.
        """
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6) if len(dets) else np.zeros((0, 6), np.float32)

        if self._bt is not None:
            try:
                return self._bytetrack_update(dets, frame_shape)
            except Exception:
                # Interface drifted at call time. Switch back-ends and continue.
                self._bt = None
                self.backend = "iou"

        return self._fallback.update(dets, frame_shape)

    def _bytetrack_update(self, dets: np.ndarray, frame_shape) -> List[Track]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        bt = self._bt

        if self._bt_in_kind == "array":
            raw = bt.update(dets, (h, w), (h, w))
        else:
            payload = (
                _dets_to_boxes_like(dets)
                if len(dets)
                else _BoxesLike(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
            )
            raw = bt.update(payload, None)

        tracks: List[Track] = []
        for item in raw:
            if hasattr(item, "tlbr"):  # STrack object (pre-8.4)
                x1, y1, x2, y2 = item.tlbr
                tid = int(item.track_id)
                cls = int(getattr(item, "cls", 0) or 0)
                conf = float(getattr(item, "score", 1.0) or 1.0)
            else:  # ndarray row [x1, y1, x2, y2, track_id, score, cls, idx]
                row = np.asarray(item, dtype=float).ravel()
                if row.size < 5:
                    continue
                x1, y1, x2, y2 = row[:4]
                tid = int(row[4])
                conf = float(row[5]) if row.size > 5 else 1.0
                cls = int(row[6]) if row.size > 6 else 0
            tracks.append(
                Track(
                    track_id=tid,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    cls=cls,
                    conf=conf,
                )
            )
        return tracks
