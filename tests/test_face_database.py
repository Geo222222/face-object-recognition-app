"""
Unit tests for the FaceDatabase helper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest

from src.face_database import FaceDatabase


@pytest.fixture
def dummy_face_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Create a temporary dataset layout and monkeypatch face_recognition helpers.
    """
    encodings: Iterator[List[float]] = iter(
        [
            [0.1, 0.2, 0.3],  # Alice
            [0.8, 0.7, 0.6],  # Bob
        ]
    )

    def fake_load_image(path: Path) -> np.ndarray:  # noqa: ANN001
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_face_encodings(image: np.ndarray) -> List[List[float]]:  # noqa: ANN001
        try:
            return [next(encodings)]
        except StopIteration:
            return []

    def fake_face_distance(known: List[List[float]], query: List[float]) -> np.ndarray:
        known_np = np.asarray(known)
        query_np = np.asarray(query)
        distances = np.linalg.norm(known_np - query_np, axis=1)
        return distances

    monkeypatch.setattr("src.face_database.face_recognition.load_image_file", fake_load_image)
    monkeypatch.setattr("src.face_database.face_recognition.face_encodings", fake_face_encodings)
    monkeypatch.setattr("src.face_database.face_recognition.face_distance", fake_face_distance)

    dataset = tmp_path / "faces"
    (dataset / "Alice").mkdir(parents=True)
    (dataset / "Bob").mkdir()

    for subdir in ("Alice", "Bob"):
        img_path = dataset / subdir / "img1.jpg"
        img_path.write_bytes(b"stub")

    return dataset


def test_load_populates_records(dummy_face_dataset: Path) -> None:
    """
    Ensure load discovers encodings for each person directory.
    """
    db = FaceDatabase(dummy_face_dataset, tolerance=0.5)
    db.load()

    assert db.labels == {"Alice": 1, "Bob": 1}


def test_match_returns_best_label(dummy_face_dataset: Path) -> None:
    """
    Matching should return the closest label within tolerance.
    """
    db = FaceDatabase(dummy_face_dataset, tolerance=0.5)
    db.load()

    name, distance = db.match([0.12, 0.19, 0.28])
    assert name == "Alice"
    assert distance < 0.5


def test_match_unknown_when_distance_exceeds_tolerance(dummy_face_dataset: Path) -> None:
    """
    If no known embedding is below tolerance, return Unknown.
    """
    db = FaceDatabase(dummy_face_dataset, tolerance=0.2)
    db.load()

    name, distance = db.match([1.5, 1.5, 1.5])
    assert name == "Unknown"
    assert distance >= 0.2

