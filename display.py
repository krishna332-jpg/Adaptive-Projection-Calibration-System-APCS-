"""
Display Manager
----------------
v4.0 CHANGE: APCS no longer owns the projector output continuously —
TouchDesigner does (fed via video_output.py + osc_bridge.py). This file
now only manages:

  1. The camera debug window (always local, always ours — shows floor
     quad, tracked objects, FPS, etc. for the operator).
  2. A brief, temporary fullscreen window used ONLY during the black/
     white structured-light calibration flash. It's opened right before
     calibration and closed right after, so TouchDesigner's own output
     window can occupy the projector monitor the rest of the time.
"""

import time
import cv2
import numpy as np

from config import PROJECTION_MONITOR
from calibration import PROJECTION_WINDOW

CAMERA_WINDOW = "APCS Camera View"


def setup_windows():
    """Creates the camera debug window. Call once at startup."""
    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)


def open_calibration_window():
    """
    Creates and fullscreens the calibration flash window on
    PROJECTION_MONITOR. Call this right before calibrate(), and pair it
    with osc_bridge.send_calibration_start() so TouchDesigner hides its
    own output first (otherwise TD's window may sit on top of this one).
    """
    cv2.namedWindow(PROJECTION_WINDOW, cv2.WINDOW_NORMAL)
    _fullscreen_on_monitor(PROJECTION_WINDOW, PROJECTION_MONITOR)


def close_calibration_window():
    """
    Destroys the calibration flash window. Call this right after
    calibrate() returns, paired with osc_bridge.send_calibration_done()
    so TouchDesigner can safely restore its output on that monitor.
    """
    try:
        cv2.destroyWindow(PROJECTION_WINDOW)
    except cv2.error:
        pass  # window was never created / already closed — fine


def _fullscreen_on_monitor(window_name: str, monitor_index: int):
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if 0 <= monitor_index < len(monitors):
            m = monitors[monitor_index]
            cv2.moveWindow(window_name, m.x, m.y)
            cv2.resizeWindow(window_name, m.width, m.height)
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            print(f"[DISPLAY] Calibration window -> monitor {monitor_index}: "
                  f"{m.width}x{m.height} at ({m.x},{m.y})")
            return
        else:
            print(f"[DISPLAY] WARN: monitor {monitor_index} not found "
                  f"({len(monitors)} monitor(s) detected). "
                  f"Using primary display.")
    except ImportError:
        print("[DISPLAY] WARN: 'screeninfo' not installed. "
              "Install it for reliable projector targeting: pip install screeninfo")
    except Exception as e:
        print(f"[DISPLAY] WARN: Could not query monitors ({e}).")

    if monitor_index > 0:
        cv2.moveWindow(window_name, 1920 * monitor_index, 0)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )


class FPSCounter:
    """Rolling average FPS counter."""

    def __init__(self, window=30):
        self._times = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def draw_overlay(cam_frame: np.ndarray,
                 floor_corners,
                 tracked: dict,
                 object_mode: str,
                 fps: float,
                 calibrated: bool,
                 debug: bool) -> np.ndarray:
    """
    Draws all camera-view overlays:
      - Floor quad outline (green)
      - Tracked object contours + ID labels + height readings (orange)
      - Calibration status, FPS, object mode (text)
      - Debug: object masks highlighted in red
    """
    display = cam_frame.copy()
    h, w = display.shape[:2]

    # Debug: highlight tracked object areas in red
    if debug and tracked:
        red_overlay = np.zeros_like(display)
        for data in tracked.values():
            cv2.drawContours(
                red_overlay, [data["contour"]], -1, (0, 0, 200), thickness=cv2.FILLED
            )
        display = cv2.addWeighted(display, 1.0, red_overlay, 0.4, 0)

    # Floor quad
    if floor_corners is not None:
        cv2.polylines(display, [floor_corners.astype(np.int32)], True, (0, 255, 0), 2)
        for i, pt in enumerate(floor_corners.astype(np.int32)):
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(display, ["TL", "TR", "BR", "BL"][i],
                        (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Tracked objects
    for obj_id, data in tracked.items():
        cv2.drawContours(display, [data["contour"]], -1, (255, 150, 0), 2)
        cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
        h_cm = data.get("height_cm", 0.0)
        correction = data.get("correction", 1.0)
        label = f"ID{obj_id}  h={h_cm:.1f}cm  x{correction:.2f}"
        cv2.circle(display, (cx, cy), 4, (255, 150, 0), -1)
        cv2.putText(display, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

    # Status text
    status = "CALIBRATED" if calibrated else "NOT CALIBRATED - press C"
    status_color = (0, 255, 0) if calibrated else (0, 0, 255)
    cv2.putText(display, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    info = (f"FPS: {fps:.1f}  |  Mode: {object_mode.upper()}  |  "
            f"Objects: {len(tracked)}  |  D=debug  O=mode  C=recal  Q=quit")
    cv2.putText(display, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    return display


def show_camera(cam_overlay: np.ndarray):
    """Shows the local camera debug view only. The actual content output
    now goes to TouchDesigner via video_output.py, not through here."""
    cv2.imshow(CAMERA_WINDOW, cam_overlay)


def read_key() -> int:
    return cv2.waitKey(1) & 0xFF


def destroy():
    cv2.destroyAllWindows()