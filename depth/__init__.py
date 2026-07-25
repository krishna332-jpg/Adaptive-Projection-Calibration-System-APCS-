"""
Depth sensor factory.
Returns the right backend based on DEPTH_BACKEND in config.py.

Options:
  "simulated"  — fake depth, no hardware needed (default, for testing)
  "realsense"  — Intel RealSense (D415, D435, D455 etc.)
  "kinect_v2"  — Microsoft Kinect for Windows v2 (2014 model)
                 Works with live camera AND Kinect Studio v2.0 playback.
  "kinect"     — Microsoft Azure Kinect DK (newer 2019 model, different SDK)
"""

from config import DEPTH_BACKEND, SIMULATED_NUM_OBJECTS


def get_depth_sensor():
    backend = DEPTH_BACKEND.lower().strip()

    if backend == "realsense":
        from depth.realsense import RealSenseDepthSensor
        return RealSenseDepthSensor()

    elif backend == "kinect_v2":
        from depth.kinect_v2 import KinectV2DepthSensor
        return KinectV2DepthSensor()

    elif backend == "kinect":
        from depth.kinect import KinectDepthSensor
        return KinectDepthSensor()

    elif backend == "simulated":
        from depth.simulated import SimulatedDepthSensor
        return SimulatedDepthSensor(num_objects=SIMULATED_NUM_OBJECTS)

    else:
        raise ValueError(
            f"Unknown DEPTH_BACKEND '{backend}' in config.py.\n"
            f"Valid options: 'simulated', 'realsense', 'kinect_v2', 'kinect'"
        )
