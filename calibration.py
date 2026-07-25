"""
Projector Calibration
----------------------
Detects the exact floor area the projector is throwing light onto,
using structured light (black then white projector frames, camera diff).
Returns the 4 corner points of the projection quad in camera-pixel space.

This is still needed even with a depth sensor, because the depth sensor
tells you WHERE objects are in 3D space, but you still need to know
WHICH PIXELS in the camera view correspond to which pixels in the
projector output. That mapping comes from this calibration step.
"""

import time
import cv2
import numpy as np

from config import CALIBRATION_SETTLE_SEC, CALIBRATION_DIFF_THRESH

PROJECTION_WINDOW = "APCS Projection Output"


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def _largest_quad(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = mask.shape[0] * mask.shape[1]
    best, best_area = None, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.01:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best_area, best = area, approx
        elif area > best_area and best is None:
            best_area, best = area, cnt

    if best is None:
        return None
    if len(best) == 4:
        return order_corners(best)
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    return order_corners(np.array(box, dtype=np.float32))


def _show_solid(color: tuple, duration: float, size: tuple):
    w, h = size
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.imshow(PROJECTION_WINDOW, frame)
    cv2.waitKey(1)
    time.sleep(duration)


def _grab_stable(sensor, settle_frames: int = 6, retries: int = 30):
    frame = None
    good, attempts = 0, 0
    while good < settle_frames and attempts < retries:
        try:
            color, _ = sensor.get_frames()
            frame = color
            good += 1
        except Exception:
            time.sleep(0.03)
        attempts += 1
    return frame


def calibrate(sensor, output_size: tuple, save_debug: bool = False) -> np.ndarray | None:
    """
    Runs the black/white structured-light calibration sequence.
    Returns 4 ordered corner points (TL, TR, BR, BL) in camera-pixel
    space, or None if calibration failed.
    """
    print("[CALIBRATE] Projecting BLACK - capturing ambient baseline...")
    _show_solid((0, 0, 0), CALIBRATION_SETTLE_SEC, output_size)
    off_frame = _grab_stable(sensor)

    print("[CALIBRATE] Projecting WHITE - capturing lit baseline...")
    _show_solid((255, 255, 255), CALIBRATION_SETTLE_SEC, output_size)
    on_frame = _grab_stable(sensor)

    if off_frame is None or on_frame is None:
        print("[CALIBRATE] ERROR: Could not read frames from sensor during calibration.")
        return None

    off_gray = cv2.cvtColor(off_frame, cv2.COLOR_BGR2GRAY)
    on_gray = cv2.cvtColor(on_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(cv2.subtract(on_gray, off_gray), (5, 5), 0)

    _, mask = cv2.threshold(diff, CALIBRATION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    if save_debug:
        cv2.imwrite("debug_off.png", off_frame)
        cv2.imwrite("debug_on.png", on_frame)
        cv2.imwrite("debug_diff.png", diff)
        cv2.imwrite("debug_mask.png", mask)
        print(f"[CALIBRATE] Debug images saved. Max diff seen: {diff.max()}")

    corners = _largest_quad(mask)

    if corners is None:
        print(f"[CALIBRATE] ERROR: Could not find projector throw region.")
        print(f"            Max brightness difference: {diff.max()} "
              f"(threshold: {CALIBRATION_DIFF_THRESH})")
        if diff.max() < CALIBRATION_DIFF_THRESH:
            print("            HINT: The projection window may not be fullscreen on "
                  "the projector. Set PROJECTION_MONITOR in config.py.")
        else:
            print("            HINT: Try lowering CALIBRATION_DIFF_THRESH in config.py.")
        return None

    print("[CALIBRATE] Done. Projection quad locked in.")
    return corners
