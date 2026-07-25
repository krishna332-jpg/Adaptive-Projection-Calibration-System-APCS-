"""
Simulated Depth Sensor
-----------------------
Generates fake depth data so the full pipeline can be tested without
any real depth sensor hardware connected. Simulates a flat floor with
a configurable number of randomly placed "raised" objects that move
around slowly, so live tracking and size correction can be tested visually.
"""

import cv2
import numpy as np
from depth.base import DepthSensor


class SimulatedDepthSensor(DepthSensor):

    def __init__(self, width=640, height=480, floor_depth_cm=150.0, num_objects=2):
        self.width = width
        self.height = height
        self.floor_depth_cm = floor_depth_cm
        self.num_objects = num_objects

        # Each simulated object: [cx, cy, radius, height_cm, vx, vy]
        rng = np.random.default_rng(42)
        self._objects = []
        for _ in range(num_objects):
            self._objects.append({
                "cx": float(rng.integers(100, width - 100)),
                "cy": float(rng.integers(100, height - 100)),
                "radius": int(rng.integers(40, 80)),
                "height_cm": float(rng.uniform(10, 40)),
                "vx": float(rng.uniform(-1.5, 1.5)),
                "vy": float(rng.uniform(-1.5, 1.5)),
            })

        self._color = None
        self._started = False
        self._cap = None

    def start(self):
        # Try to open the real webcam for the color image, fall back to a
        # solid grey synthetic image if no camera is available.
        self._cap = cv2.VideoCapture(0)
        self._started = True
        print("[SIM] Simulated depth sensor started. No real depth hardware needed.")

    def get_frames(self):
        if not self._started:
            raise RuntimeError("Call start() before get_frames()")

        # Color frame: from webcam if available, otherwise synthetic
        if self._cap and self._cap.isOpened():
            ret, color = self._cap.read()
            if not ret or color is None:
                color = np.full((self.height, self.width, 3), 80, dtype=np.uint8)
            else:
                color = cv2.resize(color, (self.width, self.height))
        else:
            color = np.full((self.height, self.width, 3), 80, dtype=np.uint8)

        # Depth frame: flat floor with raised blobs for simulated objects
        depth = np.full((self.height, self.width), self.floor_depth_cm, dtype=np.float32)

        for obj in self._objects:
            cx, cy, r = int(obj["cx"]), int(obj["cy"]), obj["radius"]
            h_cm = obj["height_cm"]
            y_coords, x_coords = np.ogrid[:self.height, :self.width]
            dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            inside = dist <= r
            # Object is closer to sensor (raised), so its depth value is lower
            depth[inside] = self.floor_depth_cm - h_cm

            # Move object for next frame (bounce off edges)
            obj["cx"] += obj["vx"]
            obj["cy"] += obj["vy"]
            if obj["cx"] - r < 0 or obj["cx"] + r > self.width:
                obj["vx"] *= -1
            if obj["cy"] - r < 0 or obj["cy"] + r > self.height:
                obj["vy"] *= -1

        return color, depth

    def stop(self):
        if self._cap:
            self._cap.release()
        self._started = False
        print("[SIM] Simulated depth sensor stopped.")
