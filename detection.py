"""
Object Detection and Tracking
-------------------------------
Combines YOLO object detection with a centroid tracker to give each
detected floor object a persistent ID that survives frame-to-frame,
even as objects move across the projection area.

Phase 4 addition: per-object rolling depth history (temporal filtering)
so size correction uses a smoothed depth value instead of a raw
single-frame reading, preventing the projected image from jittering.
"""

from collections import deque
import cv2
import numpy as np

from config import (YOLO_MODEL, YOLO_CONFIDENCE, TRACKER_MAX_DISTANCE,
                    TRACKER_MAX_MISSING_FRAMES, TRACKER_SMOOTHING,
                    DEPTH_SMOOTHING_FRAMES)


# ── YOLO detector ────────────────────────────────────────────────────────────

class YoloDetector:
    """
    Detects objects in a camera frame using a locally-run YOLO model.
    Filters to keep only detections whose center falls inside the
    calibrated projection quad.
    """

    def __init__(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        print(f"[YOLO] Loading model '{YOLO_MODEL}' "
              f"(first run downloads it automatically)...")
        self._model = YOLO(YOLO_MODEL)
        print("[YOLO] Model ready.")

    def detect(self, color_frame: np.ndarray,
               floor_corners) -> list:
        """
        Returns list of (centroid_xy, contour) tuples for detections
        inside the floor quad.
        """
        results = self._model.predict(
            color_frame, conf=YOLO_CONFIDENCE, verbose=False
        )
        if not results:
            return []

        h, w = color_frame.shape[:2]
        quad_mask = None
        if floor_corners is not None:
            quad_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(quad_mask, floor_corners.astype(np.int32), 255)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if quad_mask is not None and quad_mask[cy, cx] == 0:
                continue

            contour = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.int32
            ).reshape(-1, 1, 2)
            detections.append(((float(cx), float(cy)), contour))

        return detections


# ── Centroid tracker ──────────────────────────────────────────────────────────

class CentroidTracker:
    """
    Gives each detected object a persistent integer ID that stays the
    same across frames as the object moves.

    Phase 4: each object maintains a rolling deque of recent depth
    readings. Smoothed depth (moving average) is used for size
    correction instead of raw single-frame depth, preventing jitter.
    """

    def __init__(self):
        self._next_id = 0
        self._objects = {}        # id -> {centroid, contour, missing, height_cm}
        self._depth_histories = {}  # Phase 4: id -> deque of recent height readings

    def update(self, detections: list,
               depth_frame,
               floor_depth_cm: float) -> dict:
        """
        detections   : list of (centroid_xy, contour) from YoloDetector
        depth_frame  : HxW float32 depth map in cm (or None)
        floor_depth_cm: baseline floor depth in cm

        Returns dict of id -> {centroid, contour, height_cm}
        """
        if not detections:
            self._age_all()
            return self._alive()

        if not self._objects:
            for centroid, contour in detections:
                h_cm = self._read_height(
                    self._next_id, centroid, depth_frame, floor_depth_cm
                )
                self._objects[self._next_id] = {
                    "centroid": centroid,
                    "contour": contour,
                    "missing": 0,
                    "height_cm": h_cm,
                }
                self._next_id += 1
            return self._alive()

        existing_ids = list(self._objects.keys())
        existing_centroids = np.array(
            [self._objects[i]["centroid"] for i in existing_ids]
        )
        new_centroids = np.array([d[0] for d in detections])

        dists = np.linalg.norm(
            existing_centroids[:, None, :] - new_centroids[None, :, :], axis=2
        )

        matched_e, matched_n = set(), set()
        for flat_idx in np.argsort(dists, axis=None):
            ei, ni = np.unravel_index(flat_idx, dists.shape)
            if ei in matched_e or ni in matched_n:
                continue
            if dists[ei, ni] > TRACKER_MAX_DISTANCE:
                continue
            obj_id = existing_ids[ei]
            new_centroid, new_contour = detections[ni]
            old = self._objects[obj_id]["centroid"]
            s = TRACKER_SMOOTHING
            smoothed = (
                old[0] * s + new_centroid[0] * (1 - s),
                old[1] * s + new_centroid[1] * (1 - s),
            )
            h_cm = self._read_height(
                obj_id, smoothed, depth_frame, floor_depth_cm
            )
            self._objects[obj_id].update({
                "centroid": smoothed,
                "contour": new_contour,
                "missing": 0,
                "height_cm": h_cm,
            })
            matched_e.add(ei)
            matched_n.add(ni)

        for ei, obj_id in enumerate(existing_ids):
            if ei not in matched_e:
                self._objects[obj_id]["missing"] += 1
                if self._objects[obj_id]["missing"] > TRACKER_MAX_MISSING_FRAMES:
                    del self._objects[obj_id]
                    self._depth_histories.pop(obj_id, None)  # Phase 4: clean up

        for ni, (centroid, contour) in enumerate(detections):
            if ni not in matched_n:
                h_cm = self._read_height(
                    self._next_id, centroid, depth_frame, floor_depth_cm
                )
                self._objects[self._next_id] = {
                    "centroid": centroid,
                    "contour": contour,
                    "missing": 0,
                    "height_cm": h_cm,
                }
                self._next_id += 1

        return self._alive()

    def _read_height(self, obj_id, centroid, depth_frame, floor_depth_cm):
        """
        Reads raw depth at centroid, computes height above floor,
        then applies Phase 4 temporal smoothing via rolling average.
        """
        if depth_frame is None:
            return self._get_smoothed_height(obj_id)

        cx, cy = int(centroid[0]), int(centroid[1])
        h, w = depth_frame.shape
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        r = 5
        patch = depth_frame[
            max(0, cy - r):min(h, cy + r),
            max(0, cx - r):min(w, cx + r)
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            return self._get_smoothed_height(obj_id)

        raw_depth = float(np.median(valid))
        raw_height = max(0.0, floor_depth_cm - raw_depth)

        # Phase 4: add to rolling history, return moving average
        if obj_id not in self._depth_histories:
            self._depth_histories[obj_id] = deque(maxlen=DEPTH_SMOOTHING_FRAMES)
        self._depth_histories[obj_id].append(raw_height)
        return float(np.mean(self._depth_histories[obj_id]))

    def _get_smoothed_height(self, obj_id):
        """Returns last known smoothed height if sensor drops a frame."""
        if obj_id in self._depth_histories and self._depth_histories[obj_id]:
            return float(np.mean(self._depth_histories[obj_id]))
        return 0.0

    def _age_all(self):
        for obj_id in list(self._objects.keys()):
            self._objects[obj_id]["missing"] += 1
            if self._objects[obj_id]["missing"] > TRACKER_MAX_MISSING_FRAMES:
                del self._objects[obj_id]
                self._depth_histories.pop(obj_id, None)  # Phase 4: clean up

    def _alive(self):
        return {
            oid: {
                "centroid": d["centroid"],
                "contour": d["contour"],
                "height_cm": d["height_cm"],
            }
            for oid, d in self._objects.items()
        }