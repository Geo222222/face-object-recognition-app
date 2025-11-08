"""
Utilities for loading and managing known face encodings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from itertools import chain
from typing import Dict, List, Tuple

import face_recognition
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    """
    Container representing a known person's face encoding.
    """

    name: str
    encoding: List[float]


class FaceDatabase:
    """
    Loads encoded faces from an on-disk dataset organized by identity.
    """

    def __init__(self, dataset_dir: Path, tolerance: float = 0.45) -> None:
        self.dataset_dir = dataset_dir
        self.tolerance = tolerance
        self._faces: List[FaceRecord] = []

    def load(self) -> None:
        """
        Discover and encode faces stored on disk.
        """
        if not self.dataset_dir.exists():
            LOGGER.warning("Face dataset directory does not exist: %s", self.dataset_dir)
            return

        self._faces.clear()
        for person_dir in sorted(p for p in self.dataset_dir.iterdir() if p.is_dir()):
            encodings = self._encode_person(person_dir)
            self._faces.extend(encodings)
        LOGGER.info("Loaded %d known faces", len(self._faces))

    def _encode_person(self, person_dir: Path) -> List[FaceRecord]:
        """
        Encode all images for a single person.
        """
        records: List[FaceRecord] = []
        image_candidates = chain(
            person_dir.glob("*.jpg"),
            person_dir.glob("*.jpeg"),
            person_dir.glob("*.png"),
        )
        for image_path in sorted(image_candidates):
            LOGGER.debug("Encoding face from %s", image_path)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                LOGGER.warning("No face found in image %s", image_path)
                continue
            record = FaceRecord(name=person_dir.name, encoding=encodings[0])
            records.append(record)
        return records

    def match(self, face_encoding: List[float]) -> Tuple[str, float]:
        """
        Find the best matching known face for the provided encoding.
        Returns the name and the distance score.
        """
        if not self._faces:
            return ("Unknown", 1.0)

        known_encodings = [record.encoding for record in self._faces]
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) == 0:
            return ("Unknown", 1.0)

        distances_np = np.asarray(distances)
        best_index = int(distances_np.argmin())
        best_distance = float(distances_np[best_index])
        best_record = self._faces[best_index]

        if best_distance <= self.tolerance:
            return (best_record.name, best_distance)
        return ("Unknown", best_distance)

    @property
    def labels(self) -> Dict[str, int]:
        """
        Return label statistics for the loaded dataset.
        """
        counts: Dict[str, int] = {}
        for record in self._faces:
            counts[record.name] = counts.get(record.name, 0) + 1
        return counts


