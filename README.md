# Face and Object Recognition Application

This project provides a Python application capable of real-time object detection (e.g., school buses, cell phones) and facial recognition. It combines Ultralytics YOLOv8 for object detection with `face_recognition` for identifying known individuals.

## Features
- Real-time detection from a webcam or video file
- Configurable list of object classes to track
- Face recognition against a local dataset of labeled face images
- Snapshot capture with on-frame annotations

## Getting Started

### 1. Install Dependencies
Create a virtual environment and install requirements:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Known Faces
Add images to `data/faces/<PersonName>/`. Multiple JPEG or PNG images per person improve accuracy.

### 3. Run the Application

```powershell
python -m src.app --source camera
```

- Press `s` to save the current annotated frame to `output/snapshot.png`.
- Press `q` to quit.

To analyze a video file:

```powershell
python -m src.app --source path\to\video.mp4
```

### 4. Generate Evidence for the Report
Use the snapshot feature to capture frames demonstrating detected faces and objects. Place selected images under `paper/figures/` for referencing in the explanatory paper.

### 5. Run the Automated Tests

```powershell
python -m pytest
```

Tests live under `tests/` and cover the face database, YOLO wrapper, and frame-processing pipeline.

## Project Structure
- `src/`: Application source code
- `data/faces/`: Known face dataset (user-provided)
- `paper/`: Explanatory paper and figures
- `docs/`: Supplemental explanations such as `HOWTHINGSWORK.md`
- `tests/`: Pytest suite
- `requirements.txt`: Python dependencies

## References
- Ultralytics YOLO: <https://github.com/ultralytics/ultralytics>
- face_recognition: <https://github.com/ageitgey/face_recognition>


