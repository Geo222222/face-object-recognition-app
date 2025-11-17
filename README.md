# Face and Object Recognition Application

![CI](https://github.com/Geo222222/face-object-recognition-app/actions/workflows/tests.yml/badge.svg)

This project shows how to pair modern computer vision models to identify people and contextual objects in real time. The app streams webcam frames through Ultralytics YOLOv8 for object detection and the `face_recognition` embedding pipeline for known-face lookup, then overlays both results live.

## Highlights
- Real-time webcam or video-file processing with optimized performance
- YOLOv8 detections filtered to campus-relevant COCO classes (person, cell phone, bus, etc.)
- Local face database with tolerance-based matching and interpretability scores
- **Age, Gender, and Emotion Detection** using DeepFace library
- Real-time GUI control panel with toggles for all features
- Performance optimizations: frame skipping and result caching for smooth operation
- Keyboard-triggered snapshots for audit evidence and paper figures
- Fully tested Python package with modular architecture and IEEE-style report

## Getting Started

### 1. Set Up the Environment
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Enroll Known Faces
Add at least one clear JPEG/PNG per person under `data/faces/<PersonName>/`. Multiple angles improve matching confidence.

### 2a. Install Face Analysis Support
Age, gender, and emotion detection use DeepFace (automatically installed via requirements.txt). If you need to install manually:

```powershell
pip install deepface tf-keras
```

DeepFace will automatically download its models on first use (this happens automatically when you first run the app).

### 3. Launch Real-Time Recognition

**Simple way (recommended):**
```powershell
python start.py
```

**Options:**
```powershell
python start.py                    # Use camera (default)
python start.py --camera           # Explicitly use camera
python start.py --video video.mp4  # Process a video file
python start.py --image photo.jpg  # Process a single image
python start.py --no-controls      # Disable GUI control panel
python start.py --verbose          # Enable verbose logging
```

**GUI Control Panel:** A separate window with checkboxes will appear to enable/disable:
- **Enable Face Recognition** - Toggle face recognition on/off
- **Enable Age Prediction** - Toggle age estimation on/off
- **Enable Gender Detection** - Toggle gender detection on/off
- **Enable Emotion Detection** - Toggle emotion detection on/off

Controls take effect immediately in the video feed.

**Keyboard Controls:**
- Press `s` to save the current annotated frame to `output/snapshot.png`.
- Press `q` to quit the application.

**Note:** Detected faces will display analysis results alongside the person's name when enabled. For example:
- "John Doe | Age: 28 | Man | happy | (0.35)" - shows name, age, gender, emotion, and recognition confidence
- "Unknown | Age: 35 | Woman | neutral" - shows analysis for unrecognized faces

### 4. Capture Evidence & Figures
Press `s` during a session to store an annotated frame in `output/`. Use these images in reports or replace the sample figure under `paper/figures/`.

### 5. Verify with Automated Tests

```powershell
python -m pytest
```

The suite covers the face database loader, YOLO wrapper, and `RecognitionApp` frame-processing integration using stubs/mocks.

## Demo

- Recorded demo (YouTube/GIF): _coming soon_
- Annotated architecture overview: `docs/HOWTHINGSWORK.md`

## Project Structure
- `src/` – application modules (`config`, `face_database`, `object_detector`, `face_analyzer`, `control_panel`, `app`)
- `data/faces/` – user-supplied enrollment images (ignored by Git)
- `output/` – captured snapshots (ignored by Git)
- `paper/` – IEEE-style report source (ignored by Git, generate locally)
- `resume/` – resume files (ignored by Git)
- `docs/` – supplementary explanations and diagrams
- `tests/` – pytest suite with unit + integration coverage
- `scripts/` – utility scripts (`download_faces`, `convert_to_docx`)
- `requirements.txt` – dependency manifest (includes DeepFace for age/gender/emotion detection)

## Performance Optimizations

The app includes several performance optimizations to reduce lag and improve stability:

- **Frame Skipping**: DeepFace analysis runs every 5 frames by default (configurable via `config.analysis_frame_skip`)
- **Result Caching**: Analysis results are cached per face and reused for 10 frames (configurable via `config.analysis_cache_frames`)
- **Smart Caching**: Uses face encodings to track the same person across frames, so moving faces don't trigger unnecessary re-analysis
- **Temporal Smoothing**: Face recognition labels require consistent matches across multiple frames before displaying, preventing flickering between "Unknown" and recognized names

**To adjust performance settings**, modify `src/config.py`:
```python
analysis_frame_skip: int = 5      # Lower = more frequent (slower), Higher = less frequent (faster)
analysis_cache_frames: int = 10   # How many frames to cache results
recognition_stability_frames: int = 3  # Minimum consecutive frames with same match before showing name
```

**Note**: The `recognition_stability_frames` setting controls how stable face recognition must be before displaying the name. Higher values mean more stability (less flickering) but slower updates when a person first appears.

## Tech Stack & References
- Ultralytics YOLOv8 <https://github.com/ultralytics/ultralytics>
- face_recognition / dlib embeddings <https://github.com/ageitgey/face_recognition>
- DeepFace <https://github.com/serengil/deepface> for age, gender, and emotion detection
- OpenCV | NumPy | python-docx | pytest

_Developed for Full Sail University COS570 — Face & Object Recognition Project_


