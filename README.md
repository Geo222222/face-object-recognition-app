# Face and Object Recognition Application

![CI](https://github.com/Geo222222/face-object-recognition-app/actions/workflows/tests.yml/badge.svg)

This project shows how to pair modern computer vision models to identify people and contextual objects in real time. The app streams webcam frames through Ultralytics YOLOv8 for object detection and the `face_recognition` embedding pipeline for known-face lookup, then overlays both results live.

## Highlights
- Real-time webcam or video-file processing (18–21 FPS on CPU)
- YOLOv8 detections filtered to campus-relevant COCO classes (person, cell phone, bus, etc.)
- Local face database with tolerance-based matching and interpretability scores
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

### 3. Launch Real-Time Recognition

```powershell
python -m src.app --source camera
```

- Press `s` to save the current annotated frame to `output/snapshot.png`.
- Press `q` to quit.

To analyze a video file:

```powershell
python -m src.app --source path\to\video.mp4
```

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
- `src/` – application modules (`config`, `face_database`, `object_detector`, `app`)
- `data/faces/` – user-supplied enrollment images (ignored by Git)
- `output/` – captured snapshots (ignored by Git)
- `paper/` – IEEE-style report source (ignored by Git, generate locally)
- `docs/` – supplementary explanations and diagrams
- `tests/` – pytest suite with unit + integration coverage
- `scripts/` – utility scripts (`download_faces`, `convert_to_docx`)
- `requirements.txt` – dependency manifest

## Tech Stack & References
- Ultralytics YOLOv8 <https://github.com/ultralytics/ultralytics>
- face_recognition / dlib embeddings <https://github.com/ageitgey/face_recognition>
- OpenCV | NumPy | python-docx | pytest

_Developed for Full Sail University COS570 — Face & Object Recognition Project_


