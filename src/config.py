"""
Configuration utilities for the recognition application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class AppConfig:
    """
    Centralized configuration options for the recognition pipeline.
    """

    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    face_dataset_dir: Path = Path("data/faces")
    output_dir: Path = Path("output")
    enable_age_estimation: bool = True
    analysis_frame_skip: int = 5  # Analyze every Nth frame (1 = every frame, higher = less frequent)
    analysis_cache_frames: int = 10  # Cache analysis results for N frames
    recognition_stability_frames: int = 5  # Minimum consecutive frames with same match before showing name
    unknown_stability_frames: int = 5  # Minimum consecutive frames with Unknown before switching from recognized name
    allowed_object_classes: List[str] = field(
        default_factory=lambda: [
            "person",
            "cell phone",
            "backpack",
            "laptop",
            "book",
            "chair",
            "cup",
            "bottle",
            "traffic light",
            "stop sign",
            "bus",
            "truck",
            "car",
            "school bus",
        ]
    )

    def ensure_directories(self) -> None:
        """
        Create all required directories if they do not already exist.
        """
        self.face_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    """
    Factory helper to create the application configuration.
    """
    config = AppConfig()
    config.ensure_directories()
    return config


