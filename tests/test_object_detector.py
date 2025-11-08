"""
Unit tests for the ObjectDetector wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from src.object_detector import ObjectDetector


class DummyTensor:
    """
    Minimal tensor shim that mimics `.cpu().numpy()` returning a numpy array.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def cpu(self) -> "DummyTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


@dataclass
class DummyBox:
    cls: DummyTensor
    xyxy: DummyTensor
    conf: DummyTensor


class DummyResult:
    def __init__(self, boxes: List[DummyBox]) -> None:
        self.boxes = boxes


class DummyModel:
    def __init__(self) -> None:
        self.names = {0: "person", 1: "cell phone"}

    def predict(self, **kwargs) -> List[DummyResult]:
        return kwargs["__dummy_results__"]


@pytest.fixture(autouse=True)
def patch_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace the YOLO class with a dummy implementation to avoid real model loads.
    """

    def fake_yolo(*args, **kwargs):  # noqa: ANN001
        return DummyModel()

    monkeypatch.setattr("src.object_detector.YOLO", fake_yolo)


def make_box(x1: float, y1: float, x2: float, y2: float, cls_id: int, conf: float) -> DummyBox:
    return DummyBox(
        cls=DummyTensor(np.array([cls_id], dtype=np.float32)),
        xyxy=DummyTensor(np.array([[x1, y1, x2, y2]], dtype=np.float32)),
        conf=DummyTensor(np.array([conf], dtype=np.float32)),
    )


def test_predict_filters_allowed_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ObjectDetector should drop detections outside the allowed class set.
    """
    detector = ObjectDetector(
        model_name="yolov8n.pt",
        allowed_classes={"person"},
    )

    dummy_results = [
        DummyResult(
            [
                make_box(0, 0, 10, 10, cls_id=0, conf=0.9),
                make_box(5, 5, 15, 15, cls_id=1, conf=0.8),
            ]
        )
    ]

    def fake_predict(**kwargs):  # noqa: ANN001
        kwargs["__dummy_results__"] = dummy_results
        return DummyModel().predict(**kwargs)

    monkeypatch.setattr(detector.model, "predict", fake_predict)
    outputs = detector.predict(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(outputs) == 1
    assert outputs[0]["class_name"] == "person"
    assert outputs[0]["box"] == [0.0, 0.0, 10.0, 10.0]


def test_predict_returns_empty_when_no_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    No detections should return an empty list.
    """
    detector = ObjectDetector(model_name="yolov8n.pt", allowed_classes=None)

    dummy_results = [DummyResult(boxes=[])]

    def fake_predict(**kwargs):  # noqa: ANN001
        kwargs["__dummy_results__"] = dummy_results
        return DummyModel().predict(**kwargs)

    monkeypatch.setattr(detector.model, "predict", fake_predict)
    outputs = detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    assert outputs == []

