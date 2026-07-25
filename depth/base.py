"""
Depth Sensor Base Interface
----------------------------
Abstract base class that every depth sensor backend must implement.
This is the "swap layer" - swap one sensor for another by changing
DEPTH_BACKEND in config.py, without touching any other code.
"""

from abc import ABC, abstractmethod
import numpy as np


class DepthSensor(ABC):
    """
    All depth sensor backends inherit from this class and implement
    these three methods. The rest of the system only ever calls these
    three - it never talks to sensor hardware directly.
    """

    @abstractmethod
    def start(self):
        """
        Initialize and start the sensor.
        Called once at startup. Raises RuntimeError if sensor not found.
        """

    @abstractmethod
    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (color_frame, depth_frame) as numpy arrays.

        color_frame: HxWx3 uint8 BGR image (same format as cv2 camera frame)
        depth_frame: HxW float32 array of distances in centimeters.
                     0.0 means "no reading" (out of range or occluded).

        This is called every frame in the main loop - must be fast.
        """

    @abstractmethod
    def stop(self):
        """
        Stop the sensor and release all hardware resources.
        Called once at shutdown.
        """
