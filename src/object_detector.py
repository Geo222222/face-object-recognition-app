"""
Wrapper around Ultralytics YOLO models for object detection.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


class ObjectDetector:
    """
    Lightweight interface for performing object detection with YOLOv8.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        confidence: float = 0.35,
        iou: float = 0.45,
        allowed_classes: Optional[Iterable[str]] = None,
    ) -> None:
        self.model = YOLO(model_name)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.allowed_classes = set(allowed_classes) if allowed_classes else None

    def predict(self, frame: np.ndarray) -> List[dict]:
        """
        Run object detection on a single frame.
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )
        detections: List[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                if self.allowed_classes and cls_name not in self.allowed_classes:
                    continue

                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                confidence = float(box.conf.cpu().numpy()[0])
                bbox = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": confidence,
                    "box": xyxy,
                }
                detections.append(bbox)
        LOGGER.debug("Detected %d objects", len(detections))
        return detections


