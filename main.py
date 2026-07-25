"""
APCS - Adaptive Projection Calibration System
===============================================
Entry point. Run this file to start the system.

v4.0 ARCHITECTURE CHANGE:
  APCS itself no longer warps/composites the final projected image.
  It now does the sensing + math (camera, depth, YOLO detection,
  tracking, size-correction physics, floor calibration) and streams
  the results to TouchDesigner, which performs the actual geometry
  warp, edge blending, and projector output:

    - Live content frame  -> video_output.py -> Spout/NDI -> TouchDesigner
    - Tracked object data -> osc_bridge.py   -> OSC        -> TouchDesigner
    - Floor calibration corners -> OSC (once per calibration run)

HOW TO RUN:
    python main.py --video skull_snake.mp4
    python main.py --image some_image.png

    Save calibration debug images if calibration fails:
        python main.py --video skull_snake.mp4 --save-debug

CONTROLS (while running):
    Q / ESC   quit
    SPACE     pause / resume video
    C         recalibrate (re-run black/white projector scan)
    O         toggle object mode: AVOID <-> INVOLVE
    D         toggle debug overlay (shows object masks in red)

BEFORE RUNNING:
    Make sure TouchDesigner is open with:
      - An OSC In CHOP/DAT listening on config.OSC_PORT
      - A Spout In TOP (or NDI In TOP) named config.VIDEO_SENDER_NAME
    See README.md for the full TouchDesigner-side setup.
"""

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np

import config
from config import MIN_DEPTH_MM, MAX_DEPTH_MM
from depth import get_depth_sensor
from calibration import calibrate
from detection import YoloDetector, CentroidTracker
from projection import compute_size_correction
from display import (setup_windows, open_calibration_window,
                      close_calibration_window, draw_overlay, show_camera,
                      read_key, destroy, FPSCounter)
from osc_bridge import OSCBridge
from video_output import VideoOutput

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("apcs.log"),
    ],
)
logger = logging.getLogger("apcs.main")


# ── Phase 3: Depth Clipping ───────────────────────────────────────────────────

def apply_depth_clipping(depth_frame):
    """
    Phase 3: Zeros out all depth values outside the valid exhibit zone.
    Prevents YOLO and size correction from reacting to ceiling reflections,
    background walls, or people walking far outside the exhibit area.
    depth_frame is in CM, config values are in MM — converted here.
    """
    if depth_frame is None:
        return depth_frame
    min_cm = MIN_DEPTH_MM / 10.0
    max_cm = MAX_DEPTH_MM / 10.0
    valid = (depth_frame >= min_cm) & (depth_frame <= max_cm)
    return np.where(valid, depth_frame, 0.0).astype(np.float32)


# ── Content source ────────────────────────────────────────────────────────────

class ContentSource:
    def __init__(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            self._type = "image"
            self._frame = cv2.imread(path)
            if self._frame is None:
                sys.exit(f"[ERROR] Could not load image: {path}")
        elif ext in VIDEO_EXTS:
            self._type = "video"
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                sys.exit(f"[ERROR] Could not open video: {path}")
            self._frame = None
            self._paused = False
        else:
            sys.exit(f"[ERROR] Unsupported file type '{ext}'.")

    def next_frame(self):
        if self._type == "image":
            return self._frame.copy()
        if not self._paused:
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if ret:
                self._frame = frame
        return self._frame.copy() if self._frame is not None else None

    def toggle_pause(self):
        if self._type == "video":
            self._paused = not self._paused

    def release(self):
        if self._type == "video" and hasattr(self, "_cap"):
            self._cap.release()


# ── Floor depth estimation ────────────────────────────────────────────────────

def estimate_floor_depth(depth_frame, floor_corners):
    """
    Estimates the floor baseline depth by taking the median of all valid
    depth readings inside the calibrated projection quad.
    Returns depth in centimeters.
    """
    if depth_frame is None:
        return 150.0
    h, w = depth_frame.shape
    if floor_corners is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, floor_corners.astype(np.int32), 255)
        vals = depth_frame[mask == 255]
    else:
        vals = depth_frame.flatten()
    vals = vals[vals > 0]
    if vals.size == 0:
        return 150.0
    return float(np.median(vals))


# ── Calibration (with TouchDesigner handshake) ────────────────────────────────

def run_calibration(sensor, output_size, osc: OSCBridge, save_debug: bool):
    """
    Wraps calibrate() with the TouchDesigner handshake:
      1. Tell TD to hide/blank its output (so our flash is visible).
      2. Open our own temporary fullscreen window and run the black/
         white scan.
      3. Close our window, tell TD it can show its output again, and
         send the resulting floor corners to TD over OSC.
    """
    osc.send_calibration_start()
    open_calibration_window()
    try:
        # Give TD a brief moment to actually hide its window before we
        # start flashing black/white on the same monitor.
        time.sleep(0.3)
        floor_corners = calibrate(sensor, output_size, save_debug=save_debug)
    finally:
        close_calibration_window()
        osc.send_calibration_done()

    if floor_corners is not None:
        osc.send_floor_corners(floor_corners)
    return floor_corners


# ── Sensor read with auto-recovery ───────────────────────────────────────────

def safe_get_frames(sensor, max_retries=5):
    """
    Wraps sensor.get_frames() so a single dropped/glitched frame (or a
    brief hardware hiccup) doesn't crash a long-running, unattended
    exhibit. Returns (None, None) if the sensor can't recover.
    """
    for attempt in range(max_retries):
        try:
            return sensor.get_frames()
        except Exception as e:
            logger.warning(
                "Sensor read failed (attempt %d/%d): %s",
                attempt + 1, max_retries, e,
            )
            time.sleep(0.1)
    logger.error("Sensor unresponsive after %d attempts.", max_retries)
    return None, None


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(content_path: str, save_debug: bool = False):
    content = ContentSource(content_path)
    sensor = get_depth_sensor()
    sensor.start()

    osc = OSCBridge()
    video_out = VideoOutput()

    color_frame, depth_frame = safe_get_frames(sensor)
    if color_frame is None:
        logger.error("Could not get an initial frame from the sensor. Exiting.")
        sensor.stop()
        return

    frame_h, frame_w = color_frame.shape[:2]
    output_size = (frame_w, frame_h)

    setup_windows()
    floor_corners = run_calibration(sensor, output_size, osc, save_debug)

    detector = YoloDetector()
    tracker = CentroidTracker()
    fps_counter = FPSCounter()

    object_mode = config.OBJECT_MODE
    osc.send_mode(object_mode)
    debug = False
    floor_depth_cm = 150.0

    logger.info("=" * 60)
    logger.info("APCS - Adaptive Projection Calibration System")
    logger.info("Content  : %s", content_path)
    logger.info("Depth    : %s", config.DEPTH_BACKEND)
    logger.info("Mode     : %s", object_mode.upper())
    logger.info("Clipping : %.0fcm - %.0fcm", MIN_DEPTH_MM / 10, MAX_DEPTH_MM / 10)
    logger.info("Smoothing: %s frames", config.DEPTH_SMOOTHING_FRAMES)
    logger.info("OSC      : %s -> %s:%s",
                "enabled" if osc.enabled else "disabled",
                config.OSC_HOST, config.OSC_PORT)
    logger.info("Video out: %s", config.VIDEO_OUTPUT_MODE)
    logger.info("Controls : Q=quit | SPACE=pause | C=recalibrate | O=mode | D=debug")
    logger.info("=" * 60)

    try:
        while True:
            # ── Get frames ───────────────────────────────────────────────────
            color_frame, depth_frame = safe_get_frames(sensor)
            if color_frame is None:
                # Sensor is down; keep the loop alive so the operator can
                # still quit/recalibrate, but skip this iteration's work.
                key = read_key()
                if key in (ord("q"), 27):
                    break
                continue

            # Phase 3: clip depth to valid exhibit zone before anything else
            depth_frame = apply_depth_clipping(depth_frame)

            content_frame = content.next_frame()
            if content_frame is None:
                continue

            fps = fps_counter.tick()

            # ── Update floor depth estimate ──────────────────────────────────
            if floor_corners is not None:
                floor_depth_cm = estimate_floor_depth(depth_frame, floor_corners)

            # ── Detect + track objects ───────────────────────────────────────
            detections = detector.detect(color_frame, floor_corners)
            # Phase 4 smoothing happens inside tracker.update() automatically
            tracked = tracker.update(detections, depth_frame, floor_depth_cm)

            # ── Compute per-object size correction (sent to TD, not applied
            #    to any image here — TD does the actual resizing) ────────────
            if object_mode == "involve":
                for data in tracked.values():
                    data["correction"] = compute_size_correction(
                        data["height_cm"], floor_depth_cm
                    )
            else:
                for data in tracked.values():
                    data["correction"] = 1.0

            # ── Send everything to TouchDesigner ─────────────────────────────
            osc.send_tracked_objects(tracked)
            video_out.send(content_frame)

            # ── Draw camera overlay (local debug view only) ──────────────────
            overlay = draw_overlay(
                color_frame, floor_corners, tracked,
                object_mode, fps, floor_corners is not None, debug
            )
            show_camera(overlay)

            # ── Key handling ─────────────────────────────────────────────────
            key = read_key()
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                content.toggle_pause()
            elif key == ord("c"):
                logger.info("Recalibrating...")
                floor_corners = run_calibration(sensor, output_size, osc, save_debug)
            elif key == ord("o"):
                object_mode = "involve" if object_mode == "avoid" else "avoid"
                osc.send_mode(object_mode)
                logger.info("Object mode: %s", object_mode.upper())
            elif key == ord("d"):
                debug = not debug
                logger.info("Debug overlay: %s", "ON" if debug else "OFF")

    finally:
        content.release()
        sensor.stop()
        video_out.close()
        destroy()
        logger.info("APCS stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="APCS - Adaptive Projection Calibration System"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to video file")
    group.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument(
        "--save-debug", action="store_true",
        help="Save calibration debug images if calibration fails."
    )
    args = parser.parse_args()
    content_path = args.video if args.video else args.image
    run(content_path, save_debug=args.save_debug)


if __name__ == "__main__":
    main()