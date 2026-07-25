"""
Azure Kinect Depth Sensor Backend
------------------------------------
Swap this in by setting DEPTH_BACKEND = "kinect" in config.py.

Requires: pip install pyk4a
Hardware: Microsoft Azure Kinect DK

Depth is converted from millimeters to centimeters for consistency
with the rest of the system.
"""

import numpy as np
from depth.base import DepthSensor


class KinectDepthSensor(DepthSensor):

    def __init__(self, width=640, height=576, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self._device = None

    def start(self):
        try:
            import pyk4a
            from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
        except ImportError:
            raise RuntimeError(
                "pyk4a not installed. Run: pip install pyk4a\n"
                "Also ensure Azure Kinect SDK is installed from:\n"
                "https://github.com/microsoft/Azure-Kinect-Sensor-SDK/releases"
            )

        fps_map = {15: FPS.FPS_15, 30: FPS.FPS_30}
        self._device = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=fps_map.get(self.fps, FPS.FPS_30),
            synchronized_images_only=True,
        ))
        self._device.start()
        print("[Kinect] Camera started.")

    def get_frames(self):
        capture = self._device.get_capture()

        color = capture.color[:, :, :3]  # drop alpha channel
        depth_raw = capture.transformed_depth.astype(np.float32)

        # Kinect depth is in millimeters - convert to centimeters
        depth_cm = depth_raw / 10.0

        return color, depth_cm

    def stop(self):
        if self._device:
            self._device.stop()
        print("[Kinect] Camera stopped.")
