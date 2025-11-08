"""
Tests for the high-level RecognitionApp pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.app import RecognitionApp
from src.config import AppConfig


class StubFaceDatabase:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002
        pass

    def load(self) -> None:
        self.loaded = True

    def match(self, encoding):  # noqa: ANN001
        return ("Test User", 0.32)


class StubObjectDetector:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002
        self._predictions = []

    def set_predictions(self, detections):  # noqa: ANN001
        self._predictions = detections

    def predict(self, frame):  # noqa: ANN001
        return list(self._predictions)


@pytest.fixture
def recognition_app(monkeypatch: pytest.MonkeyPatch, tmp_path) -> RecognitionApp:  # noqa: ANN001
    """
    Provide a RecognitionApp instance with stubbed dependencies.
    """
    monkeypatch.setattr("src.app.FaceDatabase", StubFaceDatabase)
    monkeypatch.setattr("src.app.ObjectDetector", StubObjectDetector)

    config = AppConfig()
    config.face_dataset_dir = tmp_path / "faces"
    config.ensure_directories()

    app = RecognitionApp(config)
    detections = [
        {"class_id": 0, "class_name": "person", "confidence": 0.95, "box": [0, 0, 10, 10]},
        {"class_id": 1, "class_name": "cell phone", "confidence": 0.87, "box": [15, 15, 30, 30]},
    ]
    app.detector.set_predictions(detections)
    return app


def test_process_frame_runs_detection_and_face_flow(
    recognition_app: RecognitionApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure process_frame integrates detector predictions and face matching.
    """

    face_locations = [(5, 20, 20, 5)]
    face_encodings = [[0.1, 0.2, 0.3]]

    monkeypatch.setattr("src.app.face_recognition.face_locations", lambda img: face_locations)
    monkeypatch.setattr(
        "src.app.face_recognition.face_encodings",
        lambda img, locs: face_encodings,
    )

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    annotated = recognition_app.process_frame(frame)

    assert annotated.shape == frame.shape
    # Expect annotation colors to alter pixels at bounding box edges.
    assert not np.array_equal(annotated, frame)
    assert (annotated[1, 1] != 0).any()
    assert (annotated[15, 15] != 0).any()


def test_annotate_frame_draws_boxes(recognition_app: RecognitionApp) -> None:
    """
    Verify annotate_frame applies expected colors for objects and faces.
    """
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    detections = [
        {"class_id": 0, "class_name": "person", "confidence": 0.9, "box": [2, 2, 20, 20]},
    ]
    faces = [((4, 15, 18, 4), "Test User", 0.25)]

    annotated = recognition_app.annotate_frame(frame, detections, faces)

    # Green-ish object rectangle
    assert tuple(annotated[2, 2]) == (52, 235, 180)
    # Orange face rectangle
    assert tuple(annotated[4, 4]) == (235, 168, 52)

