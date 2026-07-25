# APCS — Adaptive Projection Calibration System

A real-time floor projection mapping system that automatically calibrates
to the projector's throw area, detects and tracks physical objects placed
on the floor using YOLO, and corrects the projected image size per-object
using depth sensor data — so the video looks the same size everywhere,
even on raised surfaces.

---

## What It Does

| Feature | Detail |
|---|---|
| Auto floor calibration | Projects black then white, camera finds the lit area — no markers needed |
| Full-frame projection | Video/image plays across the entire calibrated floor area |
| Live object tracking | YOLO detects objects, centroid tracker gives each a persistent ID |
| Depth-correct sizing | Per-object image resize based on real height above floor |
| AVOID / INVOLVE mode | Video skips over objects (AVOID) or projects over them (INVOLVE) |
| Any depth sensor | Swap sensors by changing one line in `config.py` |
| FPS counter | Live performance readout in the camera view |

---

## Project Structure

```
APCS/
├── main.py           Entry point — run this
├── config.py         All settings in one place — edit this first
├── calibration.py    Structured-light projector throw detection
├── detection.py      YOLO object detection + centroid tracker
├── projection.py     Warping, depth-based size correction, compositing
├── display.py        Window management, fullscreen, overlays, FPS
├── depth/
│   ├── __init__.py   Sensor factory (loads backend from config.py)
│   ├── base.py       Abstract interface all sensors must implement
│   ├── simulated.py  Fake depth — runs without hardware for testing
│   ├── realsense.py  Intel RealSense backend
│   └── kinect.py     Azure Kinect backend
└── README.md
```

---

## Requirements

### Python packages
```
pip install opencv-python numpy ultralytics screeninfo
```

### For real depth sensors (install only what you need)
```
pip install pyrealsense2    # Intel RealSense
pip install pyk4a           # Azure Kinect
```

### Hardware
- A projector connected as an extended display (not mirrored)
- A depth camera **OR** nothing (simulated mode works without hardware)
- A webcam (built-in or external) — used by simulated mode for the color image

---

## Quick Start

**1. Clone / copy the APCS folder to your machine.**

**2. Install requirements:**
```
pip install opencv-python numpy ultralytics screeninfo
```

**3. Set your depth sensor in `config.py`:**
```python
DEPTH_BACKEND = "simulated"   # no hardware needed, for testing
# DEPTH_BACKEND = "realsense"  # Intel RealSense
# DEPTH_BACKEND = "kinect"     # Azure Kinect
```

**4. Set which monitor is your projector in `config.py`:**
```python
PROJECTION_MONITOR = 1   # 0 = laptop screen, 1 = projector (most common)
```

**5. Run:**
```
python main.py --video skull_snake.mp4
python main.py --image some_image.png
```

**6. At startup**, the projector will briefly flash black then white —
   this is the calibration scan. Keep the floor clear of objects during
   this step. Wait for `[CALIBRATE] Done.` in the terminal.

**7. Video starts playing.** Place objects on the floor — they'll be
   detected, tracked, and the projected image on them will auto-resize
   to appear the same size as the surrounding floor.

---

## Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `SPACE` | Pause / resume video |
| `C` | Recalibrate (re-run black/white scan) |
| `O` | Toggle AVOID / INVOLVE mode |
| `D` | Toggle debug overlay (tracked objects highlighted in red) |

---

## Configuration Reference (`config.py`)

| Setting | Default | What it does |
|---|---|---|
| `DEPTH_BACKEND` | `"simulated"` | Depth sensor to use |
| `PROJECTION_MONITOR` | `1` | Which display is the projector |
| `CAMERA_INDEX` | `0` | Webcam device number |
| `OBJECT_MODE` | `"avoid"` | Starting mode (avoid/involve) |
| `CALIBRATION_DIFF_THRESH` | `25` | Brightness diff threshold for calibration. Lower in bright rooms. |
| `CALIBRATION_SETTLE_SEC` | `1.2` | Wait time after projector switches frame |
| `YOLO_MODEL` | `"yolov8n.pt"` | YOLO model size (n=fastest, s/m=more accurate) |
| `YOLO_CONFIDENCE` | `0.35` | Detection confidence threshold |
| `TRACKER_MAX_DISTANCE` | `100` | Max px an object can move between frames and keep its ID |
| `TRACKER_SMOOTHING` | `0.4` | Position smoothing (0=none, 0.9=heavy) |
| `FLOOR_TOLERANCE_CM` | `3.0` | Depth within this range of floor = treated as floor |
| `MIN_SIZE_CORRECTION` | `0.2` | Minimum allowed size correction factor |
| `MAX_SIZE_CORRECTION` | `3.0` | Maximum allowed size correction factor |

---

## Swapping in a Real Depth Sensor

**Intel RealSense:**
1. `pip install pyrealsense2`
2. In `config.py`: `DEPTH_BACKEND = "realsense"`
3. Run normally — no other code changes needed.

**Azure Kinect:**
1. Install [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/releases)
2. `pip install pyk4a`
3. In `config.py`: `DEPTH_BACKEND = "kinect"`
4. Run normally.

**Any other sensor:**
1. Create a new file in `depth/` (e.g. `depth/mysensor.py`)
2. Inherit from `DepthSensor` in `depth/base.py`
3. Implement `start()`, `get_frames()`, `stop()`
4. Add it to the factory in `depth/__init__.py`
5. Set `DEPTH_BACKEND = "mysensor"` in `config.py`

---

## Troubleshooting

**Calibration fails / "Could not find projector throw region"**
- Make sure the projection window is fullscreen on the projector monitor (`PROJECTION_MONITOR` in config.py)
- Dim the room — ambient light drowns out the projector's black/white difference
- Lower `CALIBRATION_DIFF_THRESH` (e.g. `15`) if the room can't be fully dimmed
- Run with `--save-debug` to get `debug_diff.png` showing what the camera actually captured

**Video not fullscreen on projector**
- Install screeninfo: `pip install screeninfo`
- Check `PROJECTION_MONITOR` in config.py matches your projector's display number

**YOLO not detecting objects**
- Lower `YOLO_CONFIDENCE` in config.py (e.g. `0.25`)
- YOLO is trained on common objects — very unusual props may not be recognized
- Objects must be fully inside the calibrated floor area

**Objects detected but size correction looks wrong**
- Check `DEPTH_BACKEND` is set to your actual sensor, not `"simulated"`
- In AVOID mode, size correction is not applied (video is just masked out)
- Size correction only activates in INVOLVE mode

---

## Known Limitations

- **AVOID mode** does not resize — it masks objects out entirely, so the size distortion problem doesn't appear at all. This is intentional.
- **INVOLVE mode** applies per-object depth correction — requires a real depth sensor for accuracy. With `simulated` backend, correction is based on fake depth data.
- **YOLO** is trained on common everyday objects. Museum-specific props (skulls, snake sculptures, etc.) may need a custom-trained model for reliable detection.
- A single RGB camera without a depth sensor cannot measure real object height — this is a hardware physics limit, not a software gap.

---

## Credits

Built for museum floor projection installation.
Uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.
