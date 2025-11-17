"""
Command-line application for real-time face and object recognition.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from .config import AppConfig, load_config
from .face_database import FaceDatabase
from .object_detector import ObjectDetector
from .face_analyzer import FaceAnalyzer, FaceAnalysisResult

try:
    from .control_panel import ControlPanel, ControlState
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    ControlPanel = None
    ControlState = None

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

        # Initialize face analyzer if enabled
        self.face_analyzer: Optional[FaceAnalyzer] = None
        if config.enable_age_estimation:
            try:
                self.face_analyzer = FaceAnalyzer()
                LOGGER.info("Face analyzer initialized (age, gender, emotion)")
            except ImportError as exc:
                LOGGER.warning(
                    "Face analysis requires DeepFace. Install with: pip install deepface"
                )
                self.face_analyzer = None
            except Exception as exc:
                LOGGER.warning("Failed to initialize face analyzer: %s", exc)
                self.face_analyzer = None

        # Control state for GUI checkboxes
        self.control_state: Optional[ControlState] = None
        if GUI_AVAILABLE:
            self.control_state = ControlState()

        # Performance optimization: frame skipping and caching for analysis
        self._frame_counter = 0
        # Cache format: {face_encoding_hash: (FaceAnalysisResult, frame_number)}
        # Use face encoding hash to track same person across frames
        self._cached_analyses: dict[int, tuple[FaceAnalysisResult, int]] = {}
        
        # Temporal smoothing for face recognition labels
        # Format: {face_id: {"name": str, "distance": float, "count": int, "unknown_count": int, 
        #                    "last_frame": int, "location": tuple, "encoding": np.ndarray}}
        # Tracks recent matches to avoid flickering between "Unknown" and recognized names
        # Uses face location + name to track across frames (more stable than hash alone)
        self._recognition_tracker: dict[int, dict[str, any]] = {}
        self._next_face_id = 0
        
        # Track what we've logged to avoid duplicate console output
        # Format: {face_id: {"last_logged_name": str, "last_logged_analysis": tuple}}
        self._logged_records: dict[int, dict[str, any]] = {}

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
        face_analyses: Optional[Iterable[FaceAnalysisResult]] = None,
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

        matches_list = list(face_matches)
        face_analyses_list = list(face_analyses) if face_analyses else [FaceAnalysisResult()] * len(matches_list)
        
        for idx, ((top, right, bottom, left), name, distance) in enumerate(matches_list):
            cv2.rectangle(annotated, (left, top), (right, bottom), (235, 168, 52), 2)
            
            # Get analysis results for this face
            analysis = face_analyses_list[idx] if idx < len(face_analyses_list) else FaceAnalysisResult()
            
            # Build label parts
            label_parts = []
            
            # Add name
            if name != "Unknown" and name != "Face":
                label_parts.append(name)
            
            # Add age, gender, emotion if available
            if analysis.age is not None:
                label_parts.append(f"Age: {analysis.age}")
            if analysis.gender is not None:
                label_parts.append(analysis.gender)
            if analysis.emotion is not None:
                label_parts.append(analysis.emotion)
            
            # Add distance for recognized faces
            if name != "Unknown" and name != "Face":
                label_parts.append(f"({distance:.2f})")
            elif name == "Unknown":
                label_parts.insert(0, "Unknown")
            
            # Build final label
            if label_parts:
                label = " | ".join(label_parts)
            else:
                label = name if name else "Face"
            
            cv2.putText(
                annotated,
                label,
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Slightly smaller font to fit more info
                (235, 168, 52),
                2,
            )
        return annotated

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Execute detection and annotation on a single frame.
        """
        object_detections = self.detector.predict(frame)

        # Check if face recognition is enabled via control panel
        face_recognition_enabled = (
            self.control_state.face_recognition_enabled if self.control_state else True
        )

        # Check if analysis features are enabled via control panel
        age_estimation_enabled = (
            self.control_state.age_estimation_enabled if self.control_state else True
        )
        gender_detection_enabled = (
            self.control_state.gender_detection_enabled if self.control_state else True
        )
        emotion_detection_enabled = (
            self.control_state.emotion_detection_enabled if self.control_state else True
        )

        # Convert to RGB for face recognition
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        matches: list[Tuple[Tuple[int, int, int, int], str, float]] = []
        face_analyses: Optional[list[FaceAnalysisResult]] = None
        face_locations = []

        # Only detect faces if face recognition is enabled
        # If face recognition is disabled, don't detect/show faces at all
        if face_recognition_enabled:
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)

            # Perform face recognition and matching
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Apply temporal smoothing to reduce flickering
            # Only show recognized names after they've been consistent for multiple frames
            self._frame_counter += 1
            stable_matches = []
            
            # Match current faces to tracked faces by location and encoding similarity
            matched_tracker_ids: set[int] = set()
            
            for idx, (location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Get current match result
                current_name, current_distance = self.face_db.match(face_encoding)
                top, right, bottom, left = location
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                # Find closest matching tracked face by location and name
                best_tracker_id = None
                best_distance = float('inf')
                
                for tracker_id, tracker in self._recognition_tracker.items():
                    if tracker_id in matched_tracker_ids:
                        continue
                    
                    # Check location proximity (within 50 pixels of center)
                    last_location = tracker.get("location")
                    if last_location:
                        last_top, last_right, last_bottom, last_left = last_location
                        last_center_x = (last_left + last_right) // 2
                        last_center_y = (last_top + last_bottom) // 2
                        
                        loc_distance = ((center_x - last_center_x) ** 2 + (center_y - last_center_y) ** 2) ** 0.5
                        
                        # Prefer same name or close location
                        if loc_distance < 100:  # Within 100 pixels
                            score = loc_distance
                            if tracker["name"] == current_name and current_name != "Unknown":
                                score *= 0.5  # Prefer same name
                            
                            if score < best_distance:
                                best_distance = score
                                best_tracker_id = tracker_id
                
                # Use existing tracker or create new one
                if best_tracker_id is not None and best_distance < 100:
                    tracker_id = best_tracker_id
                    tracker = self._recognition_tracker[tracker_id]
                    matched_tracker_ids.add(tracker_id)
                else:
                    # Create new tracker
                    tracker_id = self._next_face_id
                    self._next_face_id += 1
                    self._recognition_tracker[tracker_id] = {
                        "name": current_name,
                        "distance": current_distance,
                        "count": 1 if current_name != "Unknown" else 0,
                        "unknown_count": 0,
                        "last_frame": self._frame_counter,
                        "location": location,
                    }
                    tracker = self._recognition_tracker[tracker_id]
                
                # Update tracker based on current match
                tracker["location"] = location  # Update location
                tracker["last_frame"] = self._frame_counter
                
                if current_name != "Unknown":
                    # We have a recognized name
                    if tracker["name"] == current_name:
                        # Same recognized name - increment stability count
                        tracker["count"] += 1
                        tracker["distance"] = current_distance
                        tracker["unknown_count"] = 0  # Reset unknown count
                    else:
                        # New recognized name - reset count
                        tracker["name"] = current_name
                        tracker["distance"] = current_distance
                        tracker["count"] = 1
                        tracker["unknown_count"] = 0
                else:
                    # Current match is Unknown
                    tracker["unknown_count"] += 1
                    
                    # Only switch to Unknown if it's been Unknown for enough frames
                    # AND the previous name wasn't stable enough to persist
                    if tracker["unknown_count"] >= self.config.unknown_stability_frames:
                        if tracker["count"] < self.config.recognition_stability_frames:
                            # Previous name wasn't stable anyway, switch to Unknown
                            tracker["name"] = "Unknown"
                            tracker["distance"] = 1.0
                            tracker["count"] = 0
                    # If unknown_count < threshold and we had a stable match, keep showing the old name
                
                # Determine what name to display
                display_name = tracker["name"]
                display_distance = tracker["distance"]
                
                # Only show recognized name if it's been stable for enough frames
                if display_name != "Unknown" and tracker["count"] < self.config.recognition_stability_frames:
                    display_name = "Unknown"
                    display_distance = 1.0
                
                # Don't show Unknown if we recently had a stable match and Unknown hasn't persisted long enough
                if display_name == "Unknown" and tracker["count"] >= self.config.recognition_stability_frames:
                    # We had a stable match recently, but current match is Unknown
                    # Only switch if Unknown has persisted long enough
                    if tracker["unknown_count"] < self.config.unknown_stability_frames:
                        # Keep showing the last stable name (persist through temporary failures)
                        display_name = tracker["name"]
                        display_distance = tracker["distance"]
                
                stable_matches.append((location, display_name, display_distance))
                
                # Store tracker_id in matches for later logging
                # We'll match this with analysis results
            
            # Clean up old tracker entries (not seen recently)
            current_frame = self._frame_counter
            to_remove = [
                tracker_id for tracker_id, tracker in self._recognition_tracker.items()
                if (current_frame - tracker["last_frame"]) > self.config.analysis_cache_frames
            ]
            for tracker_id in to_remove:
                del self._recognition_tracker[tracker_id]
            
            matches = stable_matches
            face_analyses: Optional[list[FaceAnalysisResult]] = None
            
            # Analyze faces if analyzer is available and any analysis is enabled
            if self.face_analyzer and face_locations and (
                age_estimation_enabled or gender_detection_enabled or emotion_detection_enabled
            ):
                face_analyses = []
                
                # Only run expensive DeepFace analysis every Nth frame (performance optimization)
                should_analyze = (self._frame_counter % self.config.analysis_frame_skip) == 0
                
                # Clean old cache entries (older than cache_frames)
                cache_threshold = self._frame_counter - self.config.analysis_cache_frames
                self._cached_analyses = {
                    face_hash: (result, frame_num)
                    for face_hash, (result, frame_num) in self._cached_analyses.items()
                    if frame_num >= cache_threshold
                }
                
                # Analyze each face with caching
                for idx, location in enumerate(face_locations):
                    top, right, bottom, left = location
                    analysis = FaceAnalysisResult()
                    
                    # Use face encoding as cache key (same person = same encoding)
                    # Hash a subset of the encoding for efficiency
                    face_hash = None
                    if idx < len(face_encodings):
                        encoding_sample = tuple(face_encodings[idx].flatten()[:16].tolist())
                        face_hash = hash(encoding_sample)
                    
                    # Check cache first
                    if face_hash is not None and face_hash in self._cached_analyses:
                        cached_result, _ = self._cached_analyses[face_hash]
                        analysis = cached_result
                        LOGGER.debug("Using cached analysis for face at %s", location)
                    elif should_analyze:
                        # Perform new analysis (expensive operation)
                        analysis = self.face_analyzer.analyze(
                            rgb_frame,
                            (top, right, bottom, left),
                            analyze_age=age_estimation_enabled,
                            analyze_gender=gender_detection_enabled,
                            analyze_emotion=emotion_detection_enabled,
                        )
                        # Cache the result using face hash
                        if face_hash is not None:
                            self._cached_analyses[face_hash] = (analysis, self._frame_counter)
                        if analysis.age is not None or analysis.gender or analysis.emotion:
                            LOGGER.debug("Face analysis: age=%s, gender=%s, emotion=%s", 
                                       analysis.age, analysis.gender, analysis.emotion)
                    
                    face_analyses.append(analysis)
            elif (age_estimation_enabled or gender_detection_enabled or emotion_detection_enabled) and not self.face_analyzer:
                LOGGER.warning("Face analysis enabled but face analyzer not initialized")
                face_analyses = None

        # Log face recognition and predictions to console (only for faces, not objects)
        if face_recognition_enabled:
            self._log_face_recognition_and_predictions(matches, face_analyses)

        annotated = self.annotate_frame(frame, object_detections, matches, face_analyses)
        return annotated
    
    def _log_face_recognition_and_predictions(
        self, 
        matches: list[Tuple[Tuple[int, int, int, int], str, float]], 
        face_analyses: Optional[list[FaceAnalysisResult]]
    ) -> None:
        """
        Log face recognition and prediction results to console.
        Only logs when results change or are new.
        """
        if not matches:
            return
        
        face_analyses_list = face_analyses if face_analyses else [FaceAnalysisResult()] * len(matches)
        
        for idx, ((location, name, distance), analysis) in enumerate(zip(matches, face_analyses_list)):
            # Find the tracker_id for this face (match by location)
            tracker_id = None
            top, right, bottom, left = location
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            for tid, tracker in self._recognition_tracker.items():
                if tracker.get("location"):
                    t_top, t_right, t_bottom, t_left = tracker["location"]
                    t_center_x = (t_left + t_right) // 2
                    t_center_y = (t_top + t_bottom) // 2
                    loc_distance = ((center_x - t_center_x) ** 2 + (center_y - t_center_y) ** 2) ** 0.5
                    if loc_distance < 50:  # Same face location
                        tracker_id = tid
                        break
            
            if tracker_id is None:
                continue
            
            # Get what we last logged for this face
            last_logged = self._logged_records.get(tracker_id, {})
            last_name = last_logged.get("name", "")
            last_analysis = last_logged.get("analysis", (None, None, None))
            
            # Build current analysis tuple for comparison
            current_analysis = (
                analysis.age if analysis.age is not None else None,
                analysis.gender if analysis.gender else None,
                analysis.emotion if analysis.emotion else None
            )
            
            # Get tracker to check stability
            tracker = self._recognition_tracker.get(tracker_id)
            if tracker is None:
                continue
            
            # Only log if name changed or analysis changed
            name_changed = name != last_name and name != "Unknown"
            analysis_changed = current_analysis != last_analysis and any(current_analysis)
            is_new_recognition = name != "Unknown" and tracker.get("count", 0) == self.config.recognition_stability_frames
            
            if name_changed or analysis_changed or is_new_recognition:
                # Build log message
                parts = []
                if name != "Unknown":
                    parts.append(name)
                
                if analysis.age is not None:
                    parts.append(f"Age: {analysis.age}")
                
                if analysis.gender:
                    parts.append(analysis.gender)
                
                if analysis.emotion:
                    parts.append(analysis.emotion)
                
                if name != "Unknown":
                    parts.append(f"({distance:.2f})")
                
                log_message = " | ".join(parts)
                
                if name != "Unknown":
                    print(f"👤 Face recognized: {log_message}")
                elif any(current_analysis):
                    print(f"👤 Face analysis: {log_message}")
                
                # Update logged record
                self._logged_records[tracker_id] = {
                    "name": name,
                    "analysis": current_analysis,
                }
            
            # Clean up old logged records
            if tracker_id in self._recognition_tracker:
                if (self._frame_counter - self._recognition_tracker[tracker_id]["last_frame"]) > self.config.analysis_cache_frames:
                    if tracker_id in self._logged_records:
                        del self._logged_records[tracker_id]

    def _open_video_source(self, source: str) -> cv2.VideoCapture:
        """
        Create a cv2.VideoCapture based on the provided source.
        """
        if source == "camera":
            return cv2.VideoCapture(0)
        if Path(source).exists():
            return cv2.VideoCapture(str(source))
        raise FileNotFoundError(f"Video source not found: {source}")

    def run(self, source: str = "camera", snapshot: Optional[Path] = None, show_controls: bool = True) -> None:
        """
        Run the recognition loop over the specified source.

        Args:
            source: Video source (camera or file path).
            snapshot: Optional path for saving snapshots.
            show_controls: If True, show GUI control panel (requires tkinter).
        """
        # Start control panel in separate thread if available and requested
        control_panel: Optional[ControlPanel] = None
        panel_thread: Optional[threading.Thread] = None

        if show_controls and GUI_AVAILABLE and self.control_state:
            try:
                # Create control panel in a separate thread
                # Use a lambda to ensure proper thread context
                control_panel = None
                
                def create_and_run_panel():
                    nonlocal control_panel
                    try:
                        control_panel = ControlPanel(self.control_state)
                        control_panel.run()
                    except Exception as exc:
                        LOGGER.error("Control panel thread error: %s", exc)
                        traceback.print_exc()
                
                panel_thread = threading.Thread(target=create_and_run_panel, daemon=False, name="ControlPanel")
                panel_thread.start()
                
                # Give the window time to initialize and appear
                for _ in range(10):  # Wait up to 1 second
                    time.sleep(0.1)
                    if control_panel is not None:
                        break
                
                if control_panel is not None:
                    LOGGER.info("Control panel started")
                else:
                    LOGGER.warning("Control panel creation timed out")
            except Exception as exc:
                LOGGER.warning("Failed to start control panel: %s", exc)
                traceback.print_exc()
                control_panel = None

        capture = self._open_video_source(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open source {source}")

        try:
            while True:
                # Check if control panel thread is still alive
                if panel_thread and not panel_thread.is_alive():
                    LOGGER.info("Control panel closed, exiting...")
                    break

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
            # Stop control panel by closing the window (it will handle cleanup)
            if control_panel and hasattr(control_panel, 'root'):
                try:
                    # Schedule window close in tkinter thread
                    control_panel.root.after(0, control_panel.root.quit)
                except Exception:
                    pass

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
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Disable GUI control panel (checkboxes).",
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
        app.run(source=args.source, snapshot=args.snapshot, show_controls=not args.no_controls)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Application failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


