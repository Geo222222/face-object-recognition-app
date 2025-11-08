# How the Recognition App Works

1. **Capture a frame**  
   `RecognitionApp.run()` reads from the webcam with `capture.read()`, then hands each frame to `process_frame()`.  
   `130:148:src/app.py`

2. **Process the frame**  
   `process_frame()` orchestrates both object detection and face recognition for each frame.  
   `98:126:src/app.py`

3. **Load configuration and helpers**  
   At startup the app imports `AppConfig`, `FaceDatabase`, and `ObjectDetector`. `AppConfig` supplies defaults (model name, thresholds, directories), `FaceDatabase` manages known embeddings, and `ObjectDetector` wraps the Ultralytics YOLOv8 model.  
   `23:42:src/app.py`

4. **Detect objects with YOLOv8**  
   `object_detections = self.detector.predict(frame)` runs the Ultralytics YOLOv8 nano checkpoint via `ObjectDetector`. YOLO’s convolutional network scans the frame once (“You Only Look Once”) to propose boxes, class labels (e.g., person, cell phone), and confidences.  
   `45:61:src/object_detector.py` and `100:106:src/app.py`

5. **Prepare for face recognition**  
   BGR frames from OpenCV are converted to contiguous RGB arrays with `rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])`.  
   `107:108:src/app.py`

6. **Locate faces**  
   `face_locations = face_recognition.face_locations(rgb_frame)` finds face bounding boxes in the current frame.  
   `109:109:src/app.py`

7. **Encode faces**  
   `face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)` turns each detected face into a 128‑D embedding.  
   `110:110:src/app.py`

8. **Match faces**  
   For each `(location, face_encoding)` pair, the app calls `self.face_db.match(face_encoding)` to compare against known embeddings and returns the best label plus the distance.  
   `111:116:src/app.py`

9. **Annotate objects**  
   Inside `annotate_frame()`, the app loops over `object_detections`, drawing green YOLO boxes and labels such as `cell phone 0.86`.  
   `63:86:src/app.py`

10. **Annotate faces**  
   The same method draws orange boxes for recognized faces, adding the person’s name and match score.  
   `88:101:src/app.py`

11. **Display the result**  
    The annotated frame is resized if needed and rendered to the GUI with `cv2.imshow("RecognitionApp", display)`.  
    `136:152:src/app.py`


