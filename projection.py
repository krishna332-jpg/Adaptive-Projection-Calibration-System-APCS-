"""
Projection Math
-----------------
v4.0 CHANGE: this module used to also warp the content frame onto the
floor quad and composite depth-corrected patches per object
(`warp_full_frame`, `composite_object_patch`, `build_avoid_mask`).
That work now happens inside TouchDesigner (Corner Pin TOP + per-object
Transform TOPs), which does it with GPU acceleration, live-adjustable
corner handles, and proper edge blending — all things a hand-rolled
OpenCV homography can't easily give you.

What stays here is the one piece of physics that's still APCS's job:
figuring out, for a given object, HOW MUCH to shrink/enlarge the content
so it reads as the correct size once projected onto a raised surface.
That single number (`size_correction`) gets sent to TouchDesigner over
OSC (see osc_bridge.py) for each tracked object, and TD applies it.

  A projector throws light in a cone. Surfaces closer to the projector
  intercept rays that haven't spread as far yet, so the same rays cover
  a SMALLER real-world area on a raised surface than on the floor. This
  makes the projected image look BIGGER on raised surfaces. The fix is
  to pre-shrink the content for that specific patch, by the exact ratio
  of (object_distance / floor_distance), so after projection the image
  lands at the same apparent size as it would on the flat floor.
"""

import numpy as np

from config import MIN_SIZE_CORRECTION, MAX_SIZE_CORRECTION


def compute_size_correction(height_cm: float, floor_depth_cm: float) -> float:
    """
    Computes the size correction factor for an object raised `height_cm`
    above the floor, where the floor is `floor_depth_cm` from the sensor.

    Physics: projector image size scales linearly with distance from lens.
    An object at (floor_depth - height) is closer by `height` cm, so its
    image is magnified by (floor_depth / (floor_depth - height)).
    We invert that to get the pre-shrink factor needed to cancel it out.

    Clamped to [MIN_SIZE_CORRECTION, MAX_SIZE_CORRECTION] from config.py
    to prevent a single noisy depth frame from causing extreme values.

    TouchDesigner side: apply this as a uniform scale on that object's
    Transform TOP, centered on the object.
    """
    if floor_depth_cm <= 0 or height_cm <= 0:
        return 1.0
    object_depth = floor_depth_cm - height_cm
    if object_depth <= 0:
        return MIN_SIZE_CORRECTION
    correction = object_depth / floor_depth_cm
    return float(np.clip(correction, MIN_SIZE_CORRECTION, MAX_SIZE_CORRECTION))