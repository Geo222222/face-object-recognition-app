#!/usr/bin/env python3
"""
Simple launcher script for the face and object recognition application.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure we can import from src
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app import RecognitionApp, load_config


def main() -> int:
    """
    Main entry point for the application launcher.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Face and Object Recognition Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                    # Run with camera (default)
  python start.py --camera           # Explicitly use camera
  python start.py --video video.mp4  # Process a video file
  python start.py --image photo.jpg  # Process a single image
  python start.py --no-controls      # Run without GUI control panel
  python start.py --verbose          # Enable verbose logging
        """,
    )
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--camera",
        action="store_true",
        default=True,
        help="Use webcam as video source (default)",
    )
    source_group.add_argument(
        "--video",
        type=str,
        metavar="PATH",
        help="Path to video file to process",
    )
    source_group.add_argument(
        "--image",
        type=str,
        metavar="PATH",
        help="Path to image file to process",
    )
    
    # Other options
    parser.add_argument(
        "--snapshot",
        type=Path,
        metavar="PATH",
        help="Optional path to save a snapshot when pressing 's'",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Disable GUI control panel (checkboxes)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    
    args = parser.parse_args()
    
    # Determine video source
    if args.video:
        source = args.video
    elif args.image:
        source = args.image
    else:
        source = "camera"
    
    # Load configuration and initialize app
    try:
        config = load_config()
        app = RecognitionApp(config, verbose=args.verbose)
        
        print(f"Starting recognition application...")
        print(f"Source: {source}")
        print(f"Controls: {'Disabled' if args.no_controls else 'Enabled'}")
        print(f"Press 'q' to quit, 's' to save snapshot")
        print()
        
        app.run(
            source=source,
            snapshot=args.snapshot,
            show_controls=not args.no_controls,
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

