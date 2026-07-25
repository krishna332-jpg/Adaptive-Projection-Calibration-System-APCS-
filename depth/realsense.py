"""
Intel RealSense Depth Sensor Backend
--------------------------------------
Swap this in by setting DEPTH_BACKEND = "realsense" in config.py.

Requires: pip install pyrealsense2
Hardware: Any Intel RealSense depth camera (D415, D435, D455, etc.)

The RealSense outputs depth in millimeters by default - this backend
converts to centimeters so the rest of the system stays consistent.
"""

import numpy as np
from depth.base import DepthSensor


class RealSenseDepthSensor(DepthSensor):

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None
        self._align = None

    def start(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise RuntimeError(
                "pyrealsense2 not installed. Run: pip install pyrealsense2\n"
                "Also ensure Intel RealSense SDK is installed from:\n"
                "https://github.com/IntelRealSense/librealsense/releases"
            )

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        try:
            self._pipeline.start(config)
        except Exception as e:
            raise RuntimeError(f"Could not start RealSense camera: {e}\n"
                               f"Check it's plugged in and not in use by another app.")

        self._align = rs.align(rs.stream.color)
        print("[RealSense] Camera started.")

    def get_frames(self):
        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            h, w = self.height, self.width
            return (np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32))

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # RealSense depth is in millimeters - convert to centimeters
        depth_cm = depth_raw / 10.0

        return color, depth_cm

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
        print("[RealSense] Camera stopped.")
