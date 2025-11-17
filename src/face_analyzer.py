"""
Face analysis using DeepFace library (age, gender, emotion).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


@dataclass
class FaceAnalysisResult:
    """Container for face analysis results."""

    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None


class FaceAnalyzer:
    """
    Analyzes faces for age, gender, and emotion using DeepFace library.
    """

    def __init__(self) -> None:
        """
        Initialize face analyzer with DeepFace.

        Raises:
            ImportError: If DeepFace is not installed.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not installed. Install with: pip install deepface"
            )
        self._DeepFace = DeepFace
        LOGGER.info("Face analyzer initialized with DeepFace")

    def analyze(
        self,
        face_image: np.ndarray,
        face_box: Optional[tuple[int, int, int, int]] = None,
        analyze_age: bool = True,
        analyze_gender: bool = True,
        analyze_emotion: bool = True,
    ) -> FaceAnalysisResult:
        """
        Analyze face for age, gender, and emotion.

        Args:
            face_image: RGB image containing a face (can be full image or cropped face).
            face_box: Optional (top, right, bottom, left) bounding box if face_image is full frame.
            analyze_age: If True, estimate age.
            analyze_gender: If True, estimate gender.
            analyze_emotion: If True, estimate emotion.

        Returns:
            FaceAnalysisResult with age, gender, and emotion (or None if analysis fails).
        """
        try:
            # Crop face if bounding box provided
            if face_box:
                top, right, bottom, left = face_box
                face_image = face_image[top:bottom, left:right]

            # Build actions list based on what's requested
            actions = []
            if analyze_age:
                actions.append("age")
            if analyze_gender:
                actions.append("gender")
            if analyze_emotion:
                actions.append("emotion")

            if not actions:
                return FaceAnalysisResult()

            # DeepFace expects RGB format (which we already have)
            result = self._DeepFace.analyze(
                face_image,
                actions=actions,
                enforce_detection=False,
                silent=True,
            )

            # DeepFace returns a list, get first result
            analysis = result[0] if isinstance(result, list) else result

            age = int(analysis.get("age", 0)) if analyze_age and "age" in analysis else None
            gender = analysis.get("dominant_gender", None) if analyze_gender and "dominant_gender" in analysis else None
            emotion = analysis.get("dominant_emotion", None) if analyze_emotion and "dominant_emotion" in analysis else None

            return FaceAnalysisResult(age=age, gender=gender, emotion=emotion)
        except Exception as exc:
            LOGGER.warning("Face analysis failed: %s", exc)
            return FaceAnalysisResult()

    def analyze_batch(
        self,
        face_images: list[np.ndarray],
        face_boxes: Optional[list[tuple[int, int, int, int]]] = None,
        analyze_age: bool = True,
        analyze_gender: bool = True,
        analyze_emotion: bool = True,
    ) -> list[FaceAnalysisResult]:
        """
        Analyze multiple faces.

        Args:
            face_images: List of face images.
            face_boxes: Optional list of bounding boxes (one per image).
            analyze_age: If True, estimate age.
            analyze_gender: If True, estimate gender.
            analyze_emotion: If True, estimate emotion.

        Returns:
            List of FaceAnalysisResult objects.
        """
        if face_boxes is None:
            face_boxes = [None] * len(face_images)

        return [
            self.analyze(img, box, analyze_age, analyze_gender, analyze_emotion)
            for img, box in zip(face_images, face_boxes)
        ]


# Backward compatibility alias
AgeEstimator = FaceAnalyzer

