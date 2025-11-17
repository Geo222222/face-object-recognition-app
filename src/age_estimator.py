"""
Age estimation using DeepFace library.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class AgeEstimator:
    """
    Estimates age from face regions using DeepFace library.
    """

    def __init__(self) -> None:
        """
        Initialize age estimator with DeepFace.
        
        Raises:
            ImportError: If DeepFace is not installed.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not installed. Install with: pip install deepface"
            )
        self._DeepFace = DeepFace
        LOGGER.info("Age estimator initialized with DeepFace")

    def estimate(
        self, face_image: np.ndarray, face_box: Optional[tuple[int, int, int, int]] = None
    ) -> Optional[int]:
        """
        Estimate age from a face image.

        Args:
            face_image: RGB image containing a face (can be full image or cropped face).
            face_box: Optional (top, right, bottom, left) bounding box if face_image is full frame.

        Returns:
            Estimated age as an integer, or None if estimation fails.
        """
        try:
            # Crop face if bounding box provided
            if face_box:
                top, right, bottom, left = face_box
                face_image = face_image[top:bottom, left:right]

            # DeepFace expects RGB format (which we already have)
            result = self._DeepFace.analyze(
                face_image,
                actions=["age"],
                enforce_detection=False,
                silent=True,
            )
            age = result[0]["age"]
            return int(age)
        except Exception as exc:
            LOGGER.warning("Age estimation failed: %s", exc)
            return None

    def estimate_batch(
        self,
        face_images: list[np.ndarray],
        face_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    ) -> list[Optional[int]]:
        """
        Estimate age for multiple faces.

        Args:
            face_images: List of face images.
            face_boxes: Optional list of bounding boxes (one per image).

        Returns:
            List of estimated ages (None if estimation failed).
        """
        if face_boxes is None:
            face_boxes = [None] * len(face_images)

        return [self.estimate(img, box) for img, box in zip(face_images, face_boxes)]
