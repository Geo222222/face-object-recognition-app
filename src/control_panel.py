"""
Tkinter control panel for real-time toggling of face recognition and age estimation.
"""
from __future__ import annotations

import logging
import threading
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tkinter import Tk

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None
    logging.warning("tkinter not available. GUI controls will be disabled.")


LOGGER = logging.getLogger(__name__)


class ControlState:
    """
    Thread-safe state container for control panel settings.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._face_recognition_enabled = True
        self._age_estimation_enabled = True
        self._gender_detection_enabled = True
        self._emotion_detection_enabled = True

    @property
    def face_recognition_enabled(self) -> bool:
        """Get face recognition enabled state."""
        with self._lock:
            return self._face_recognition_enabled

    @face_recognition_enabled.setter
    def face_recognition_enabled(self, value: bool) -> None:
        """Set face recognition enabled state."""
        with self._lock:
            self._face_recognition_enabled = value
            LOGGER.info("Face recognition %s", "enabled" if value else "disabled")

    @property
    def age_estimation_enabled(self) -> bool:
        """Get age estimation enabled state."""
        with self._lock:
            return self._age_estimation_enabled

    @age_estimation_enabled.setter
    def age_estimation_enabled(self, value: bool) -> None:
        """Set age estimation enabled state."""
        with self._lock:
            self._age_estimation_enabled = value
            LOGGER.info("Age estimation %s", "enabled" if value else "disabled")

    @property
    def gender_detection_enabled(self) -> bool:
        """Get gender detection enabled state."""
        with self._lock:
            return self._gender_detection_enabled

    @gender_detection_enabled.setter
    def gender_detection_enabled(self, value: bool) -> None:
        """Set gender detection enabled state."""
        with self._lock:
            self._gender_detection_enabled = value
            LOGGER.info("Gender detection %s", "enabled" if value else "disabled")

    @property
    def emotion_detection_enabled(self) -> bool:
        """Get emotion detection enabled state."""
        with self._lock:
            return self._emotion_detection_enabled

    @emotion_detection_enabled.setter
    def emotion_detection_enabled(self, value: bool) -> None:
        """Set emotion detection enabled state."""
        with self._lock:
            self._emotion_detection_enabled = value
            LOGGER.info("Emotion detection %s", "enabled" if value else "disabled")


class ControlPanel:
    """
    Tkinter GUI panel for controlling recognition features.
    """

    def __init__(self, state: ControlState, title: str = "Recognition Controls") -> None:
        """
        Initialize the control panel.

        Args:
            state: Shared state object for thread-safe communication.
            title: Window title.
        """
        if tk is None:
            raise ImportError("tkinter is not available. Cannot create control panel.")

        self.state = state
        self.root: Tk = tk.Tk()
        self.root.title(title)
        self.root.geometry("300x240")
        self.root.resizable(False, False)
        
        # Position window in top-right corner
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = self.root.winfo_screenwidth() - width - 20
        y = 20
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Create checkboxes
        self._create_widgets()
        
        # Make sure window is on top and visible
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after_idle(lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass

        # Start the GUI update loop
        self._running = True

    def _create_widgets(self) -> None:
        """Create and layout GUI widgets."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Face recognition checkbox
        self.face_recognition_var = tk.BooleanVar(value=self.state.face_recognition_enabled)
        face_checkbox = ttk.Checkbutton(
            main_frame,
            text="Enable Face Recognition",
            variable=self.face_recognition_var,
            command=self._on_face_recognition_toggle,
        )
        face_checkbox.grid(row=0, column=0, sticky=tk.W, pady=5)

        # Age estimation checkbox
        self.age_estimation_var = tk.BooleanVar(value=self.state.age_estimation_enabled)
        age_checkbox = ttk.Checkbutton(
            main_frame,
            text="Enable Age Prediction",
            variable=self.age_estimation_var,
            command=self._on_age_estimation_toggle,
        )
        age_checkbox.grid(row=1, column=0, sticky=tk.W, pady=5)

        # Gender detection checkbox
        self.gender_detection_var = tk.BooleanVar(value=self.state.gender_detection_enabled)
        gender_checkbox = ttk.Checkbutton(
            main_frame,
            text="Enable Gender Detection",
            variable=self.gender_detection_var,
            command=self._on_gender_detection_toggle,
        )
        gender_checkbox.grid(row=2, column=0, sticky=tk.W, pady=5)

        # Emotion detection checkbox
        self.emotion_detection_var = tk.BooleanVar(value=self.state.emotion_detection_enabled)
        emotion_checkbox = ttk.Checkbutton(
            main_frame,
            text="Enable Emotion Detection",
            variable=self.emotion_detection_var,
            command=self._on_emotion_detection_toggle,
        )
        emotion_checkbox.grid(row=3, column=0, sticky=tk.W, pady=5)

        # Status label
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, pady=(10, 0), sticky=(tk.W, tk.E))

        ttk.Label(status_frame, text="Status:", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W
        )
        self.status_label = ttk.Label(status_frame, text="Active", foreground="green")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Controls take effect immediately.\nPress 'q' in video window to quit.",
            font=("TkDefaultFont", 8),
            foreground="gray",
        )
        instructions.grid(row=5, column=0, pady=(10, 0), sticky=tk.W)

    def _on_face_recognition_toggle(self) -> None:
        """Handle face recognition checkbox toggle."""
        enabled = self.face_recognition_var.get()
        self.state.face_recognition_enabled = enabled

    def _on_age_estimation_toggle(self) -> None:
        """Handle age estimation checkbox toggle."""
        enabled = self.age_estimation_var.get()
        self.state.age_estimation_enabled = enabled

    def _on_gender_detection_toggle(self) -> None:
        """Handle gender detection checkbox toggle."""
        enabled = self.gender_detection_var.get()
        self.state.gender_detection_enabled = enabled

    def _on_emotion_detection_toggle(self) -> None:
        """Handle emotion detection checkbox toggle."""
        enabled = self.emotion_detection_var.get()
        self.state.emotion_detection_enabled = enabled

    def _on_closing(self) -> None:
        """Handle window close event."""
        self._running = False
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        """Start the GUI event loop."""
        try:
            # Initialize tkinter in this thread
            self.root.update_idletasks()
            self.root.deiconify()  # Show window if it was hidden
            
            # Start the main loop
            self.root.after(100, self._check_state)  # Periodic state check
            self.root.mainloop()
        except Exception as exc:
            LOGGER.error("Control panel error: %s", exc)
            traceback.print_exc()
            self._running = False
    
    def _check_state(self) -> None:
        """Periodic check to ensure window stays alive."""
        if not self._running:
            return
        try:
            self.root.after(100, self._check_state)
        except Exception:
            self._running = False

    def stop(self) -> None:
        """Stop the GUI and close the window."""
        self._running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        """Check if the panel is still running."""
        return self._running

