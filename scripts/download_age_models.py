"""
Download age estimation model files for OpenCV DNN.

Downloads the required Caffe model files from OpenCV's test data repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

# Model files - these need to be downloaded manually from OpenCV or use DeepFace
# Alternative: Use DeepFace instead (easier): pip install deepface
# 
# For OpenCV DNN models, download from:
# https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/
# 
# Files needed:
# - age_deploy.prototxt
# - age_net.caffemodel

# Try multiple sources for the models
MODEL_FILES = {
    "age_deploy.prototxt": [
        "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/age_deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_deploy.prototxt",
    ],
    "age_net.caffemodel": [
        "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/age_net.caffemodel",
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/age_net.caffemodel",
    ],
}


def download_file(urls: list[str], output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to the specified path, trying multiple URLs if needed.

    Args:
        urls: List of URLs to try (in order).
        output_path: Path where file should be saved.
        chunk_size: Size of chunks to read during download.

    Returns:
        True if download succeeded, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {output_path.name}...")
    
    for url in urls:
        try:
            response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

                print(f"\n  Saved to {output_path}")
                return True
        except requests.exceptions.RequestException:
            continue
    
    print(f"  ERROR: Failed to download from all sources")
    return False


def main() -> int:
    """
    Download age estimation model files.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Determine model directory (relative to project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "models" / "age"

    print(f"Downloading age estimation models to {model_dir}")
    print("Note: If download fails, consider using DeepFace instead: pip install deepface\n")

    all_success = True
    for filename, urls in MODEL_FILES.items():
        output_path = model_dir / filename
        if output_path.exists():
            print(f"{filename} already exists, skipping...")
            continue

        if not download_file(urls, output_path):
            all_success = False
    
    if not all_success:
        print("\nWARNING: Some models failed to download.")
        print("Alternative: Install DeepFace for easier setup:")
        print("  pip install deepface")
        print("\nOr download manually from:")
        print("  https://github.com/opencv/opencv_extra/tree/master/testdata/dnn")
        return 1

    print("\n✓ Age estimation models downloaded successfully!")
    print(f"\nModels location: {model_dir}")
    print("You can now use age estimation in the face recognition app.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

