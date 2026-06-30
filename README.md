# Adaptive Projection Calibration System (APCS)

A real-time computer vision software that automatically calibrates projector output, maps digital content onto a floor surface, detects objects, and performs localized A4 projection size correction using a single camera.

## Overview

Adaptive Projection Calibration System (APCS) is a Python-based projection mapping application that combines computer vision and image processing to create an interactive projection environment.

The system automatically detects the projector's projection area using structured-light calibration (black/white projection), eliminating the need for manual markers. Once calibrated, images or videos are accurately projected onto the detected floor region.

A unique feature of this project is localized A4 surface correction. When an A4 sheet is moved closer to or farther from the projector, the projected content on that sheet is automatically resized to maintain a consistent physical appearance, while the rest of the projected scene remains unchanged.

The software also detects and tracks objects placed on the floor in real time using either traditional OpenCV image processing or YOLO object detection.

---

# Features

* Automatic projector calibration using structured-light scanning
* Markerless floor projection mapping
* Real-time perspective correction using homography
* Localized A4 projection size correction
* Real-time object detection and tracking
* Persistent multi-object tracking with unique IDs
* Two object detection modes:

  * Brightness Difference Detection (OpenCV)
  * YOLOv8 AI Object Detection
* Object Avoid Mode
* Object Involve Mode
* Supports both images and videos
* Fullscreen projector output
* Runtime recalibration support
* Adjustable detection sensitivity

---

# Project Workflow

1. Open camera
2. Detect projector area using black/white structured-light calibration
3. Compute floor homography
4. Load image or video
5. Detect A4 sheet
6. Capture A4 reference size
7. Detect and track floor objects
8. Warp content onto the calibrated floor
9. Apply localized A4 size correction
10. Display the final projected output

---

# Technology Stack

* Python 3.11+
* OpenCV
* NumPy
* Ultralytics YOLOv8 (Optional)
* ScreenInfo

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/adaptive-projection-calibration-system.git
cd adaptive-projection-calibration-system
```

Install dependencies:

```bash
pip install opencv-python numpy screeninfo ultralytics
```

---

# Usage

Project a video:

```bash
python floor_a4_combo_mapper.py --video sample.mp4
```

Project an image:

```bash
python floor_a4_combo_mapper.py --image image.png
```

Using YOLO object detection:

```bash
python floor_a4_combo_mapper.py --video sample.mp4 --detector yolo
```

---

# Keyboard Controls

| Key     | Function                             |
| ------- | ------------------------------------ |
| Q / ESC | Quit                                 |
| SPACE   | Pause / Resume                       |
| D       | Toggle Debug View                    |
| O       | Toggle Object Mode (Avoid / Involve) |
| C       | Recalibrate Projection               |
| R       | Capture New A4 Reference             |
| [ ]     | Adjust Detection Sensitivity         |

---

# Project Structure

```
floor_a4_combo_mapper.py
│
├── Camera Initialization
├── Projection Calibration
├── Geometry Utilities
├── A4 Detection
├── Object Detection
├── YOLO Integration
├── Object Tracking
├── Image/Video Source
├── Perspective Warping
├── Localized A4 Correction
└── Main Application Loop
```

---

# Applications

* Interactive Museum Displays
* Smart Classrooms
* Interactive Floors
* Exhibition Installations
* Projection Mapping
* Educational Demonstrations
* Research in Computer Vision

---

# Future Improvements

* Multi-camera support
* Depth camera integration
* Automatic A4 recognition using fiducial markers
* GPU acceleration
* Graphical User Interface (GUI)
* Multi-surface calibration
* Automatic lighting compensation

---

# Limitations

* Requires a camera and projector setup.
* Localized size correction is currently supported only for A4 sheets because their real-world dimensions are known.
* Brightness-difference detection may be affected by significant lighting changes or strong shadows.
* YOLO mode requires downloading the model during the first run.

---

# Author

**Athul Krishna K S**

BCA Student | Full Stack Developer | Computer Vision Enthusiast

---

# License

This project was developed as part of an internship project for educational and demonstration purposes.
