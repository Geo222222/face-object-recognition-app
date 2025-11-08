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
            "school bus",  # Custom label variants
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


