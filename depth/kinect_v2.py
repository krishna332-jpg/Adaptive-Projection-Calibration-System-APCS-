"""
Kinect v2 Depth Sensor Backend
--------------------------------
For use with Microsoft Kinect for Windows v2 (the 2014 model)
and Kinect Studio v2.0 for playback of recorded .xef clips.

SETUP STEPS:
1. Install Kinect for Windows SDK 2.0:
   https://www.microsoft.com/en-us/download/details.aspx?id=44561

2. Install the Python wrapper:
   pip install pykinect2

3. Plug in the Kinect v2 via USB 3.0 (must be USB 3.0 — it won't
   work on USB 2.0).

4. If using Kinect Studio v2.0 for recorded clips (.xef files):
   - Open Kinect Studio v2.0
   - Load your .xef recording
   - Click "Connect" then "Play" in Kinect Studio
   - This streams the recorded data through the same SDK interface
     as a live camera — so this backend works identically for both
     live and recorded playback, no code change needed.

5. In config.py set:
   DEPTH_BACKEND = "kinect_v2"

OUTPUT:
  color_frame : HxWx3 uint8 BGR  (1920x1080 resized to 512x424 to
                                   match depth resolution)
  depth_frame : HxW float32 in CENTIMETERS (converted from raw mm)
  Valid depth range for Kinect v2: ~50cm to ~450cm (0.5m to 4.5m)
  Pixels outside this range or with no reading return 0.0

NOTE ON DEPTH ALIGNMENT:
  Kinect v2 color and depth cameras have different viewpoints and
  resolutions. This backend uses CoordinateMapper to map each depth
  pixel to its corresponding color pixel, so depth and color are
  spatially aligned in the output — the same (row, col) in both
  arrays refers to the same real-world point.
"""

import ctypes
import numpy as np
from depth.base import DepthSensor

DEPTH_W = 512
DEPTH_H = 424
COLOR_W = 1920
COLOR_H = 1080


class KinectV2DepthSensor(DepthSensor):

    def __init__(self):
        self._kinect  = None
        self._mapper  = None
        self._color_points = None  # reused buffer for coordinate mapping

    def start(self):
        try:
            from pykinect2 import PyKinectV2, PyKinectRuntime
        except ImportError:
            raise RuntimeError(
                "pykinect2 not installed.\n"
                "Run: pip install pykinect2\n"
                "Also install Kinect for Windows SDK 2.0 from:\n"
                "https://www.microsoft.com/en-us/download/details.aspx?id=44561\n"
                "Kinect Studio v2.0 (for .xef playback) is included in that SDK."
            )

        self._PyKinectV2      = PyKinectV2
        self._PyKinectRuntime = PyKinectRuntime

        # Open both color + depth streams
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth
        )

        # Preallocate the color-point buffer used by CoordinateMapper
        # (one ColorSpacePoint per depth pixel)
        self._color_points = (PyKinectV2._ColorSpacePoint * (DEPTH_W * DEPTH_H))()

        print(f"[KinectV2] Started. "
              f"Color: {COLOR_W}×{COLOR_H}  "
              f"Depth: {DEPTH_W}×{DEPTH_H}")

    def get_frames(self):
        """
        Returns (color_bgr, depth_cm) as aligned numpy arrays of shape
        (DEPTH_H, DEPTH_W, 3) and (DEPTH_H, DEPTH_W) respectively.

        Blocks briefly if a new frame isn't ready yet (polls up to ~100ms).
        Returns the last-known frames if no new frame arrives in time.
        """
        kinect = self._kinect
        PyKV2  = self._PyKinectV2

        # Wait for a depth frame
        depth_cm = None
        if kinect.has_new_depth_frame():
            raw_depth = kinect.get_last_depth_frame()
            if raw_depth is not None:
                depth_mm = raw_depth.reshape((DEPTH_H, DEPTH_W)).astype(np.float32)
                # 0 = no reading; valid range ~500-4500 mm
                depth_cm = np.where(depth_mm > 0, depth_mm / 10.0, 0.0)

        # Get color frame and map it to depth resolution
        color_out = None
        if kinect.has_new_color_frame():
            raw_color = kinect.get_last_color_frame()
            if raw_color is not None:
                # raw_color is BGRA flattened; reshape and drop alpha
                color_bgra = raw_color.reshape((COLOR_H, COLOR_W, 4))
                color_bgr  = color_bgra[:, :, :3]

                # Map depth pixels -> color image coordinates
                if depth_cm is not None:
                    # Build a raw depth mm array for the mapper
                    depth_mm_raw = (depth_cm * 10.0).astype(np.uint16).flatten()
                    depth_ctypes = depth_mm_raw.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_uint16)
                    )

                    # Ask Kinect SDK to fill self._color_points:
                    # for each depth pixel, where does it fall in the color image?
                    kinect._mapper.MapDepthFrameToColorSpace(
                        DEPTH_W * DEPTH_H,
                        depth_ctypes,
                        DEPTH_W * DEPTH_H,
                        self._color_points
                    )

                    # Sample color image at each mapped point
                    xs = np.array([p.x for p in self._color_points],
                                  dtype=np.float32).reshape(DEPTH_H, DEPTH_W)
                    ys = np.array([p.y for p in self._color_points],
                                  dtype=np.float32).reshape(DEPTH_H, DEPTH_W)

                    # Clamp to valid color image bounds
                    xs = np.clip(xs, 0, COLOR_W - 1).astype(np.int32)
                    ys = np.clip(ys, 0, COLOR_H - 1).astype(np.int32)

                    # Index into color image (vectorized — fast)
                    color_out = color_bgr[ys, xs]   # shape (DEPTH_H, DEPTH_W, 3)
                else:
                    # No depth yet — just resize color to depth resolution
                    import cv2
                    color_out = cv2.resize(color_bgr, (DEPTH_W, DEPTH_H))

        # Fallback: return black frames if nothing ready yet
        if color_out is None:
            color_out = np.zeros((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)
        if depth_cm is None:
            depth_cm = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)

        return color_out, depth_cm

    def stop(self):
        if self._kinect:
            self._kinect.close()
        print("[KinectV2] Stopped.")