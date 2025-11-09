# Face & Object Recognition Pipeline

## System Overview
- **Config bootstrap (`AppConfig`)** – loads defaults, ensures `data/faces/` and `output/` exist, and wires selected classes/device.  
  `23:57:src/config.py`
- **Runtime coordinator (`RecognitionApp`)** – manages the capture loop, orchestrates detections, and handles UI + snapshots.  
  `52:214:src/app.py`
- **Object detector wrapper** – lightweight facade over Ultralytics YOLOv8 with class filtering and confidence/I0U thresholds.  
  `8:73:src/object_detector.py`
- **Face database** – scans enrollment folders, encodes faces, and performs nearest-neighbor search with tolerance filtering.  
  `13:117:src/face_database.py`

## Frame Processing Flow
1. **Capture input**  
   `capture.read()` pulls a frame from webcam/video. On success the raw BGR frame is passed to `process_frame()`.  
   `130:152:src/app.py`

2. **Detect objects (YOLOv8)**  
   `ObjectDetector.predict()` runs YOLO once per frame, returning bounding boxes, COCO class names, and confidence scores.  
   `38:68:src/object_detector.py`

3. **Prepare frame for faces**  
   Convert BGR → RGB and enforce contiguous memory (`np.ascontiguousarray`) so dlib’s encoder accepts the image.  
   `106:108:src/app.py`

4. **Localize faces**  
   `face_recognition.face_locations()` locates face bounding boxes using HOG/CNN cascades.  
   `109:109:src/app.py`

5. **Generate embeddings**  
   `face_recognition.face_encodings()` produces 128-D vectors aligned with the detected face locations.  
   `110:110:src/app.py`

6. **Match with enrollment**  
   For each embedding, `FaceDatabase.match()` computes Euclidean distances to known encodings and returns the best label if within tolerance.  
   `70:104:src/face_database.py`

7. **Annotate detections**  
   `annotate_frame()` draws green YOLO boxes with confidences and orange dashed boxes for face matches (name + distance).  
   `63:101:src/app.py`

8. **Display & interact**  
   The annotated frame is resized for readability, shown via `cv2.imshow`, and keyboard shortcuts (`q`, `s`) control quit/snapshot behavior.  
   `132:151:src/app.py`

## Testing Strategy
- **Unit tests**  
  - `tests/test_face_database.py` mocks `face_recognition` to validate dataset loading and tolerance logic.  
  - `tests/test_object_detector.py` swaps YOLO with dummy tensors to check filtering and empty-output paths.
- **Integration test**  
  - `tests/test_app.py` stubs dependencies and asserts `process_frame()` annotates pixels as expected, covering the full pipeline.

## Future Enhancements
- Persist detection events to disk or message queues for analytics.
- Add configuration file / CLI flags for thresholds and classes.
- Provide Docker setup and REST/gRPC endpoints for multi-client access.

