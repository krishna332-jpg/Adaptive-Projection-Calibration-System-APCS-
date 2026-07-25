"""
APCS Configuration
------------------
All tunable parameters in one place. Change values here rather than
hunting through code. Comments explain what each setting does.

v4.0 CHANGE: APCS no longer renders the final projected output itself.
It now sends live tracking data (OSC) and the raw content video
(Spout/NDI) to a professional mapping tool (TouchDesigner recommended),
which performs the actual geometry warp, edge blending, and output.
See the new "Output Bridge" section at the bottom of this file.
"""

# ── Calibration ────────────────────────────────────────────────────────────

# Seconds to wait after projector switches black/white before camera reads.
# Increase if calibration keeps failing (slow projector warm-up).
CALIBRATION_SETTLE_SEC = 1.2

# Brightness difference threshold for detecting the projector's light throw.
# Lower = more sensitive (use in bright rooms). Higher = less noise (dark rooms).
CALIBRATION_DIFF_THRESH = 25

# ── Depth sensor ────────────────────────────────────────────────────────────

# Which depth sensor backend to use.
# Options:
#   "simulated"  — no hardware needed, fake depth for testing
#   "kinect_v2"  — Microsoft Kinect for Windows v2 (2014 USB model)
#                  Also works with Kinect Studio v2.0 .xef playback
#   "realsense"  — Intel RealSense (D415/D435/D455 etc.)
#   "kinect"     — Microsoft Azure Kinect DK (2019 model, different SDK)
DEPTH_BACKEND = "kinect_v2"

# For simulated backend: how many fake "raised" objects to generate each frame.
SIMULATED_NUM_OBJECTS = 2

# ── YOLO object detection ────────────────────────────────────────────────────

# YOLO model to use. yolov8n = smallest/fastest (recommended).
# Options: yolov8n, yolov8s, yolov8m (bigger = more accurate but slower).
YOLO_MODEL = "yolov8n.pt"

# Minimum confidence for a YOLO detection to count (0-1).
# Lower = more detections but more false positives.
YOLO_CONFIDENCE = 0.35

# ── Object tracking ──────────────────────────────────────────────────────────

# Max pixel distance an object can move between frames and still be
# considered the same object. Increase for fast-moving objects.
TRACKER_MAX_DISTANCE = 100

# How many consecutive frames an object can be missing before it's dropped.
# Increase if objects flicker in/out due to detection instability.
TRACKER_MAX_MISSING_FRAMES = 8

# Position smoothing factor (0 = no smoothing, 0.9 = heavy smoothing).
# Higher = smoother movement but slightly slower to respond.
TRACKER_SMOOTHING = 0.4

# ── Projection / size correction ────────────────────────────────────────────

# Floor distance from depth sensor (cm). Anything within this range
# of the floor reading is treated as floor, not an object.
FLOOR_TOLERANCE_CM = 3.0

# Size correction is clamped to this range to prevent extreme values
# from a single noisy depth frame making the image vanish or balloon.
MIN_SIZE_CORRECTION = 0.2
MAX_SIZE_CORRECTION = 3.0

# ── Display ──────────────────────────────────────────────────────────────────

# Which monitor is used for the calibration black/white flash
# (0 = primary/laptop screen, 1 = first extended display / projector).
# NOTE: This is ONLY used during the brief calibration sequence now.
# The actual live content output is handled by TouchDesigner, not here.
PROJECTION_MONITOR = 1

# Camera device index (0 = default/built-in, 1+ for external cameras).
CAMERA_INDEX = 0

# Object mode: "avoid" (video skips over objects) or "involve" (projects over them).
# This value is sent to TouchDesigner over OSC; TD is responsible for
# actually applying the mode to the output.
OBJECT_MODE = "avoid"

# ── Phase 3: Depth Clipping Planes ──────────────────────────────────────────
# Only track objects within this Z-window. Values in MILLIMETERS.
# Kinect v2 valid range: 500mm (0.5m) to 4500mm (4.5m).
# Adjust MAX_DEPTH_MM to match your actual room/exhibit depth.
MIN_DEPTH_MM = 500.0    # 0.5m: Kinect v2 minimum reliable reading
MAX_DEPTH_MM = 4500.0   # 4.5m: Kinect v2 maximum reliable reading

# ── Phase 4: Temporal Depth Filtering ───────────────────────────────────────
# Number of frames to average depth over per object (smooths jitter).
DEPTH_SMOOTHING_FRAMES = 5

# ── v4.0: Output Bridge (OSC + Spout/NDI to TouchDesigner) ─────────────────

# Master switch. If False, OSC sending is skipped entirely (useful for
# testing the detection/tracking pipeline alone, with no TD running).
OSC_ENABLED = True

# TouchDesigner is normally on the SAME machine, so 127.0.0.1 is correct.
# Only change this if TouchDesigner is running on a different computer
# on the same network.
OSC_HOST = "127.0.0.1"

# Must match the port set on the OSC In CHOP/DAT inside TouchDesigner.
OSC_PORT = 7000

# How many times per second to send tracking data over OSC.
# Doesn't need to match camera FPS exactly; 30-60 is plenty smooth.
OSC_SEND_RATE = 30

# Which method to use for sending the live content video frame to
# TouchDesigner. Options:
#   "spout"  — Windows only. Requires: pip install SpoutGL
#              Recommended when APCS and TouchDesigner run on the same PC.
#   "ndi"    — Cross-platform, works over network too.
#              Requires: pip install ndi-python (and the NDI Runtime installed)
#   "window" — Fallback/testing only. Opens a plain OpenCV preview window
#              instead of sending anywhere. Use this if Spout/NDI aren't
#              installed yet, so the rest of the pipeline still runs.
VIDEO_OUTPUT_MODE = "spout"

# Name TouchDesigner will look for in its Spout In TOP / NDI source list.
VIDEO_SENDER_NAME = "APCS_Output"