"""
Command-line application for real-time face and object recognition.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from .config import AppConfig, load_config
from .face_database import FaceDatabase
from .object_detector import ObjectDetector

LOGGER = logging.getLogger("recognition_app")


def configure_logging(verbose: bool = False) -> None:
    """
    Configure the root logger.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


class RecognitionApp:
    """
    Coordinate face and object detection on image streams.
    """

    def __init__(self, config: AppConfig, verbose: bool = False) -> None:
        configure_logging(verbose)

        self.config = config
        self.face_db = FaceDatabase(config.face_dataset_dir)
        self.face_db.load()

        self.detector = ObjectDetector(
            model_name=config.model_name,
            device=config.device,
            confidence=config.confidence_threshold,
            iou=config.iou_threshold,
            allowed_classes=config.allowed_object_classes,
        )

    def _resize_for_display(self, frame: np.ndarray, width: int = 1280) -> np.ndarray:
        """
        Resize the frame to improve readability of overlays.
        """
        h, w = frame.shape[:2]
        if w <= width:
            return frame
        scale = width / w
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

    def annotate_frame(
        self,
        frame: np.ndarray,
        object_detections: Iterable[dict],
        face_matches: Iterable[Tuple[Tuple[int, int, int, int], str, float]],
    ) -> np.ndarray:
        """
        Draw annotations for detected objects and faces onto the frame.
        """
        annotated = frame.copy()
        for detection in object_detections:
            x1, y1, x2, y2 = map(int, detection["box"])
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (52, 235, 180), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (52, 235, 180),
                2,
            )

        for (top, right, bottom, left), name, distance in face_matches:
            cv2.rectangle(annotated, (left, top), (right, bottom), (235, 168, 52), 2)
            label = f"{name} ({distance:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(
                annotated,
                label,
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (235, 168, 52),
                2,
            )
        return annotated

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Execute detection and annotation on a single frame.
        """
        object_detections = self.detector.predict(frame)

        # Face recognition expects contiguous RGB array
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        matches = [
            (location, *self.face_db.match(face_encoding))
            for location, face_encoding in zip(face_locations, face_encodings)
        ]
        annotated = self.annotate_frame(frame, object_detections, matches)
        return annotated

    def _open_video_source(self, source: str) -> cv2.VideoCapture:
        """
        Create a cv2.VideoCapture based on the provided source.
        """
        if source == "camera":
            return cv2.VideoCapture(0)
        if Path(source).exists():
            return cv2.VideoCapture(str(source))
        raise FileNotFoundError(f"Video source not found: {source}")

    def run(self, source: str = "camera", snapshot: Optional[Path] = None) -> None:
        """
        Run the recognition loop over the specified source.
        """
        capture = self._open_video_source(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open source {source}")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    LOGGER.info("End of stream or camera read failure.")
                    break

                annotated = self.process_frame(frame)
                display = self._resize_for_display(annotated)
                cv2.imshow("RecognitionApp", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    LOGGER.info("Received quit signal from keyboard.")
                    break
                if key == ord("s"):
                    self._save_snapshot(annotated, snapshot)
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _save_snapshot(self, frame: np.ndarray, snapshot: Optional[Path]) -> None:
        """
        Save annotated frame to disk.
        """
        target_dir = snapshot.parent if snapshot else self.config.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = snapshot or target_dir / "snapshot.png"
        cv2.imwrite(str(filename), frame)
        LOGGER.info("Saved snapshot to %s", filename)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Face and object recognition using YOLOv8 and face_recognition."
    )
    parser.add_argument(
        "--source",
        default="camera",
        help="Video source. Use 'camera' for webcam or provide a video/image path.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Optional path to save a snapshot when pressing 's'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    Entry point for the CLI.
    """
    args = parse_args(argv)
    config = load_config()
    app = RecognitionApp(config, verbose=args.verbose)
    try:
        app.run(source=args.source, snapshot=args.snapshot)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Application failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


