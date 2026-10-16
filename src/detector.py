"""YOLOv8 detection front-end.

Wraps ``ultralytics.YOLO`` and normalises its output into plain
``[x1, y1, x2, y2, conf, cls]`` rows so the rest of the pipeline stays
framework-agnostic.

The original repository shipped this file as ``detect.py`` while
``demo.py`` imported ``src.detector``, so the project could not start.
This module is the canonical implementation; ``src/detect.py`` remains as a
one-line backward-compatibility shim.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

#: COCO class ids that are meaningful for sports footage.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32
DEFAULT_CLASSES: Sequence[str] = ("person", "sports ball")


class Detector:
    """A thin, opinionated wrapper around a YOLO model."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thres: float = 0.3,
        iou_thres: float = 0.5,
        imgsz: int = 640,
        classes: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        class_conf: Optional[dict] = None,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: keeps light modules light

        self.model = YOLO(model_path)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes else None
        self.device = device
        #: Per-class confidence overrides, ``{class_id: threshold}``.
        #:
        #: One global threshold cannot serve both a player and a ball. The ball
        #: is a handful of pixels and scores erratically, so it needs a much
        #: stricter gate than a person. On the reference aerial clip, a 0.18
        #: global threshold produced ~4 ball detections per frame on a pitch
        #: that contains exactly one ball; raising the ball threshold to 0.45
        #: reduced that to exactly one, while leaving person recall untouched.
        self.class_conf = {int(k): float(v) for k, v in (class_conf or {}).items()}

    # ------------------------------------------------------------- thresholds
    @property
    def min_conf(self) -> float:
        """Lowest threshold in play, used as the detector's own gate.

        Inference runs at this value so that a lenient per-class threshold is
        never pre-empted by a stricter global one; the per-class decision is
        then applied afterwards.
        """
        if not self.class_conf:
            return self.conf_thres
        return min(self.conf_thres, *self.class_conf.values())

    def threshold_for(self, cls: int) -> float:
        return self.class_conf.get(int(cls), self.conf_thres)

    # ------------------------------------------------------------- metadata
    @property
    def names(self) -> dict:
        """id -> class name mapping advertised by the model."""
        names = getattr(self.model, "names", None) or {}
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return dict(names)

    def resolve_classes(self, spec) -> Optional[List[int]]:
        """Turn ``"person,sports ball"`` or ``"0,32"`` into a list of ids.

        Returns ``None`` when ``spec`` is empty or contains ``all``, meaning
        "do not filter".
        """
        if spec is None:
            return None
        names = self.names
        lowered = {str(v).lower(): k for k, v in names.items()}

        ids: List[int] = []
        for raw in spec:
            token = str(raw).strip()
            if not token:
                continue
            if token.lower() in ("all", "*"):
                return None
            if token.lstrip("-").isdigit():
                ids.append(int(token))
            elif token.lower() in lowered:
                ids.append(lowered[token.lower()])
            else:
                raise ValueError(
                    f"unknown class {token!r}; known classes: {sorted(names.values())}"
                )
        return sorted(set(ids)) or None

    # ------------------------------------------------------------ inference
    def detect(self, frame: np.ndarray) -> List[list]:
        """Run inference on a BGR frame.

        Returns a list of ``[x1, y1, x2, y2, conf, cls]`` with integer box
        coordinates, clipped to the frame.
        """
        height, width = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.min_conf,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: List[list] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf < self.threshold_for(cls):
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                x1 = int(max(0, min(x1, width - 1)))
                y1 = int(max(0, min(y1, height - 1)))
                x2 = int(max(0, min(x2, width - 1)))
                y2 = int(max(0, min(y2, height - 1)))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append([x1, y1, x2, y2, conf, cls])
        return detections
