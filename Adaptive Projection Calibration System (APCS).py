"""
Floor Projection Mapper with A4 Local Size-Correction
---------------------------------------------------------
Projects ONE video across the full calibrated floor area (same
structured-light calibration as the floor mapper - no marker needed).

Within that full frame, an A4 sheet is tracked as a special object:
the patch of video landing ON the A4 sheet is automatically shrunk or
enlarged to compensate for the sheet being raised closer to (or moved
farther from) the camera/projector - WITHOUT affecting the rest of the
video frame around it. The sheet's local size never grows past covering
its own detected area (it doesn't bleed outside its own quad).

Any OTHER object placed on the floor is still detected and tracked live
(same persistent-ID tracker as before) but, since we don't know its real
size, it can't get the same size-correction - it just gets masked out
(AVOID mode) or projected over normally (INVOLVE mode), same as before.

OBJECT DETECTION METHOD - two options, pick with --detector:
    --detector diff (default) - brightness comparison against the
        calibrated lit-floor baseline. No extra install. Sensitive to
        shadows and lighting changes (something getting darker/brighter
        without a real object there can register as a false detection).
    --detector yolo - free, local AI object detection (pip install
        ultralytics). More robust to lighting/shadows since it recognizes
        actual object shapes rather than just brightness changes. First
        run needs internet to download the small model file (~6MB), then
        works fully offline. Still only gives a 2D box - no height/depth,
        same single-camera limitation as everything else in this file.

WHY ONLY THE A4 SHEET GETS SIZE-CORRECTED (read this before asking "why
not other objects too"):
    Size-correction requires knowing the object's TRUE real-world size in
    advance, so "looks bigger in camera" can be confidently interpreted as
    "is closer," rather than "is just a bigger object." A4 paper is
    always 21cm wide, everywhere, so that ambiguity doesn't exist for it.
    A random object has no such guarantee - a big object far away and a
    small object close up can look pixel-for-pixel identical to a single
    camera. This is a hardware/optics limitation, not a missing feature -
    it would need a depth sensor or stereo camera to resolve for
    arbitrary objects.

HOW THE A4 SIZE-CORRECTION WORKS (no manual measuring):
    1. First time the sheet is detected, its pixel width is stored as the
       "reference" (assumed normal/ground) size.
    2. Every frame: current pixel width vs. reference width gives a
       size_correction ratio.
         - Sheet closer to lens (bigger in camera) -> ratio < 1 -> the
           video patch on the sheet is shrunk before being composited in.
         - Sheet farther (smaller in camera) -> ratio > 1 -> enlarged.
    3. This is a pure pixel-ratio trick - not a depth/distance measurement
       in real units. It only works because A4's real width is fixed.
    4. Press R to manually re-capture the reference size if the
       auto-capture grabbed a bad first frame (e.g. sheet was already
       raised when the script started).

HOW TO RUN:
    py -3.11 floor_a4_combo_mapper.py --video skull_snake.mp4
    py -3.11 floor_a4_combo_mapper.py --image some_image.png

CONTROLS:
    Q or ESC  = quit
    SPACE     = pause/resume video
    D         = toggle debug view
    O         = toggle non-A4 object mode: AVOID <-> INVOLVE
    C         = recalibrate floor projection quad
    R         = re-capture A4 reference size
    [ and ]   = decrease / increase object-detection sensitivity
"""

import argparse
import platform
import sys
import time
import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# ─────────────────────────────────────────────────────────────────────────
# Camera / window setup
# ─────────────────────────────────────────────────────────────────────────

def open_camera(index: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def setup_projection_window(window_name: str, monitor_index: int):
    """Moves the projection output window onto the requested monitor and
    makes it borderless fullscreen there. See floor_projection_mapper.py
    for the original version of this function and more detailed notes."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if 0 <= monitor_index < len(monitors):
            m = monitors[monitor_index]
            cv2.moveWindow(window_name, m.x, m.y)
            cv2.resizeWindow(window_name, m.width, m.height)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(f"[INFO] Projection window placed on monitor {monitor_index}: "
                  f"{m.width}x{m.height} at ({m.x},{m.y})")
            return
        else:
            print(f"[WARN] --monitor {monitor_index} out of range ({len(monitors)} found).")
    except ImportError:
        print("[WARN] 'screeninfo' not installed - install for reliable monitor targeting: "
              "pip install screeninfo")
    except Exception as e:
        print(f"[WARN] Could not query monitors ({e}).")

    if monitor_index > 0:
        cv2.moveWindow(window_name, 1920 * monitor_index, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ─────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────

def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def largest_quad_from_mask(mask: np.ndarray):
    """Finds the largest 4-corner-ish quad in a binary mask, falling back
    to a min-area rectangle of the biggest blob if needed."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = mask.shape[0] * mask.shape[1]
    best, best_area = None, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.01:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area, best = area, approx
        elif area > best_area:
            best_area, best = area, cnt

    if best is None:
        return None
    if len(best) == 4:
        return order_corners(best)
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    return order_corners(np.array(box, dtype=np.float32))


def corner_apparent_width_px(corners: np.ndarray) -> float:
    """Average of top-edge and bottom-edge length, in pixels."""
    tl, tr, br, bl = corners
    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    return float((top_w + bottom_w) / 2.0)


# ─────────────────────────────────────────────────────────────────────────
# Floor calibration (structured light: black/white projector throw scan)
# ─────────────────────────────────────────────────────────────────────────

def show_solid_frame(window_name, size, color, duration_sec):
    w, h = size
    solid = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.imshow(window_name, solid)
    cv2.waitKey(1)
    time.sleep(duration_sec)


def grab_stable_frame(cap, settle_frames=5, retries=30):
    frame = None
    good_reads, attempts = 0, 0
    while good_reads < settle_frames and attempts < retries:
        ret, f = cap.read()
        attempts += 1
        if ret:
            frame = f
            good_reads += 1
        else:
            time.sleep(0.03)
    return frame


def calibrate_projection(cap, output_window, output_size, settle_sec=1.0,
                          diff_thresh=25, save_debug=False):
    print("[CALIBRATE] Projecting BLACK, capturing ambient baseline...")
    show_solid_frame(output_window, output_size, (0, 0, 0), settle_sec)
    off_frame = grab_stable_frame(cap)

    print("[CALIBRATE] Projecting WHITE, capturing lit baseline...")
    show_solid_frame(output_window, output_size, (255, 255, 255), settle_sec)
    on_frame = grab_stable_frame(cap)

    if off_frame is None or on_frame is None:
        print("[CALIBRATE] Failed to read camera frames during calibration.")
        return None, None

    off_gray = cv2.cvtColor(off_frame, cv2.COLOR_BGR2GRAY)
    on_gray = cv2.cvtColor(on_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(cv2.subtract(on_gray, off_gray), (5, 5), 0)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    if save_debug:
        cv2.imwrite("calibrate_off.png", off_frame)
        cv2.imwrite("calibrate_on.png", on_frame)
        cv2.imwrite("calibrate_diff.png", diff)
        cv2.imwrite("calibrate_mask.png", mask)
        print(f"[CALIBRATE] Saved debug images (max diff seen: {diff.max()})")

    corners = largest_quad_from_mask(mask)
    if corners is None:
        print("[CALIBRATE] Could not find a projector throw region.")
        print(f"           Max brightness difference: {diff.max()} (threshold {diff_thresh})")
        if diff.max() < diff_thresh:
            print("           Likely cause: 'Projection Output' window isn't fullscreen "
                  "on the projector display. Use --monitor to fix this.")
        return None, None

    print("[CALIBRATE] Done. Projection quad locked in.")
    return corners, on_frame


# ─────────────────────────────────────────────────────────────────────────
# A4 sheet detection (separate from generic floor-object detection)
# ─────────────────────────────────────────────────────────────────────────

def find_a4_corners(frame: np.ndarray):
    """Detects a white A4 sheet via HSV color masking - matte white only,
    capped brightness to exclude glowing screens."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 50, 240])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    best, best_area = None, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.005 or area > frame_area * 0.5:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        long_side, short_side = max(w, h), min(w, h)
        ratio = long_side / max(short_side, 1)
        if not (1.1 < ratio < 1.8):
            continue
        if area > best_area:
            best_area, best = area, approx

    if best is None:
        return None
    return order_corners(best)


# ─────────────────────────────────────────────────────────────────────────
# Generic floor-object detection + live tracking (no size correction)
# ─────────────────────────────────────────────────────────────────────────

def find_objects_on_floor(cam_frame, lit_baseline, corners, brightness_thresh=35, min_area_frac=0.0015):
    h, w = cam_frame.shape[:2]
    quad_mask = np.zeros((h, w), dtype=np.uint8)
    if corners is None:
        return [], np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, corners.astype(np.int32), 255)

    cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(lit_baseline, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(cv2.absdiff(cam_gray, base_gray), (5, 5), 0)
    _, obj_mask = cv2.threshold(diff, brightness_thresh, 255, cv2.THRESH_BINARY)
    objects_mask = cv2.bitwise_and(obj_mask, quad_mask)

    kernel = np.ones((5, 5), np.uint8)
    objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = h * w
    object_contours = [c for c in contours if cv2.contourArea(c) > frame_area * min_area_frac]
    return object_contours, objects_mask


class YoloFloorDetector:
    """
    Drop-in alternative to the brightness-diff detector, using a free,
    locally-run YOLO model (ultralytics) for object detection instead of
    comparing against a lit-floor baseline.

    WHY THIS HELPS: brightness-diff detection breaks down with shadows,
    lighting changes, or anything that changes the floor's brightness
    without a real object being there. YOLO instead recognizes actual
    object shapes/classes it was trained on, which is far more robust to
    those conditions - but it still only outputs a 2D bounding box, NOT
    height/depth, for exactly the single-camera reasons covered earlier.

    Detections are converted into the SAME (centroid, contour) format the
    existing CentroidTracker already expects, so nothing downstream
    (tracking, masking, AVOID/INVOLVE) needs to change.

    NOTE: requires `pip install ultralytics` and an internet connection
    the FIRST time it runs (auto-downloads the small 'yolov8n.pt' model,
    a few MB, then it's cached locally and works fully offline after).
    """

    def __init__(self, model_name="yolov8n.pt", confidence=0.35):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        print(f"[INFO] Loading YOLO model '{model_name}' (first run downloads it automatically)...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        print("[INFO] YOLO model loaded.")

    def detect(self, cam_frame, corners):
        """
        Runs YOLO on the frame, keeps only detections whose center falls
        inside the calibrated floor quad, and returns (contours, mask) in
        the same shape find_objects_on_floor() returns, so it's a true
        drop-in replacement at the call site.
        """
        h, w = cam_frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        contours = []

        quad_mask = None
        if corners is not None:
            quad_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(quad_mask, corners.astype(np.int32), 255)

        results = self.model.predict(cam_frame, conf=self.confidence, verbose=False)
        if not results:
            return contours, mask

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if quad_mask is not None and quad_mask[min(cy, h - 1), min(cx, w - 1)] == 0:
                continue  # detection center is outside the floor area - ignore

            cnt = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32).reshape(-1, 1, 2)
            contours.append(cnt)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

        return contours, mask


class CentroidTracker:
    """Persistent multi-object tracker - same logic as in
    floor_projection_mapper.py. See that file for detailed comments."""

    def __init__(self, max_distance=80, max_missing_frames=8, smoothing=0.5):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.smoothing = smoothing

    def update(self, detections):
        if not detections:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]["missing"] += 1
                if self.objects[obj_id]["missing"] > self.max_missing_frames:
                    del self.objects[obj_id]
            return self._alive_view()

        if not self.objects:
            for centroid, contour in detections:
                self.objects[self.next_id] = {"centroid": centroid, "contour": contour, "missing": 0}
                self.next_id += 1
            return self._alive_view()

        existing_ids = list(self.objects.keys())
        existing_centroids = np.array([self.objects[i]["centroid"] for i in existing_ids])
        new_centroids = np.array([d[0] for d in detections])
        dists = np.linalg.norm(existing_centroids[:, None, :] - new_centroids[None, :, :], axis=2)

        matched_existing, matched_new = set(), set()
        flat_indices = np.argsort(dists, axis=None)
        for flat_idx in flat_indices:
            ei, ni = np.unravel_index(flat_idx, dists.shape)
            if ei in matched_existing or ni in matched_new:
                continue
            if dists[ei, ni] > self.max_distance:
                continue
            obj_id = existing_ids[ei]
            new_centroid, new_contour = detections[ni]
            old_centroid = self.objects[obj_id]["centroid"]
            s = self.smoothing
            smoothed = (old_centroid[0] * s + new_centroid[0] * (1 - s),
                        old_centroid[1] * s + new_centroid[1] * (1 - s))
            self.objects[obj_id]["centroid"] = smoothed
            self.objects[obj_id]["contour"] = new_contour
            self.objects[obj_id]["missing"] = 0
            matched_existing.add(ei)
            matched_new.add(ni)

        for ei, obj_id in enumerate(existing_ids):
            if ei not in matched_existing:
                self.objects[obj_id]["missing"] += 1
                if self.objects[obj_id]["missing"] > self.max_missing_frames:
                    del self.objects[obj_id]

        for ni, (centroid, contour) in enumerate(detections):
            if ni not in matched_new:
                self.objects[self.next_id] = {"centroid": centroid, "contour": contour, "missing": 0}
                self.next_id += 1

        return self._alive_view()

    def _alive_view(self):
        return {oid: {"centroid": d["centroid"], "contour": d["contour"]} for oid, d in self.objects.items()}


def detections_from_contours(contours):
    detections = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        detections.append(((M["m10"] / M["m00"], M["m01"] / M["m00"]), cnt))
    return detections


def mask_from_tracked_objects(tracked, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    contours = [d["contour"] for d in tracked.values()]
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


# ─────────────────────────────────────────────────────────────────────────
# Content source (image or video)
# ─────────────────────────────────────────────────────────────────────────

class ContentSource:
    def __init__(self, path: str):
        import os
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            self._type = "image"
            self._frame = cv2.imread(path)
            if self._frame is None:
                sys.exit(f"[ERROR] Could not load image: {path}")
        elif ext in VIDEO_EXTS:
            self._type = "video"
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                sys.exit(f"[ERROR] Could not open video: {path}")
            self._frame = None
            self._paused = False
        else:
            sys.exit(f"[ERROR] Unsupported file type: {ext}")

    def next_frame(self):
        if self._type == "image":
            return self._frame.copy()
        if not self._paused:
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if ret:
                self._frame = frame
        return self._frame.copy() if self._frame is not None else None

    def toggle_pause(self):
        if self._type == "video":
            self._paused = not self._paused

    def release(self):
        if self._type == "video":
            self._cap.release()


# ─────────────────────────────────────────────────────────────────────────
# Warping
# ─────────────────────────────────────────────────────────────────────────

def warp_full_frame(content_frame, corners, output_size):
    """Warps the full video onto the floor quad - this is the base layer,
    covering the whole projection area normally."""
    h_content, w_content = content_frame.shape[:2]
    src_pts = np.float32([[0, 0], [w_content, 0], [w_content, h_content], [0, h_content]])
    H, _ = cv2.findHomography(src_pts, corners.astype(np.float32))
    if H is None:
        return None
    return cv2.warpPerspective(content_frame, H, output_size)


def sample_content_patch(content_frame, floor_corners, a4_corners_cam, output_frame_shape):
    """
    Figures out which patch of the CONTENT FRAME (the source video/image,
    not the camera or output) lands on the A4 sheet's current position,
    given where the A4 sheet is within the calibrated floor quad.

    This works by inverse-mapping the A4 sheet's camera-space corners back
    into content-frame coordinates, using the same homography that maps
    content -> floor. That tells us exactly which rectangular region of
    the source content would normally be displayed on that patch of floor.
    """
    h_content, w_content = content_frame.shape[:2]
    src_pts = np.float32([[0, 0], [w_content, 0], [w_content, h_content], [0, h_content]])
    H, _ = cv2.findHomography(src_pts, floor_corners.astype(np.float32))
    if H is None:
        return None, None
    H_inv = np.linalg.inv(H)

    a4_pts_cam = a4_corners_cam.astype(np.float32).reshape(-1, 1, 2)
    a4_pts_content = cv2.perspectiveTransform(a4_pts_cam, H_inv).reshape(4, 2)
    return a4_pts_content, H


def composite_a4_patch(output, content_frame, floor_corners, a4_corners_cam,
                        size_correction, output_size):
    """
    Renders the locally size-corrected video patch for the A4 sheet
    directly into `output` (the full warped frame), so only that small
    region changes and everything else in `output` stays untouched.
    """
    a4_pts_content, H = sample_content_patch(content_frame, floor_corners, a4_corners_cam, output.shape)
    if a4_pts_content is None:
        return output

    h_content, w_content = content_frame.shape[:2]

    # Bounding box of the patch in content-frame coords, clamped to valid range.
    xs = np.clip(a4_pts_content[:, 0], 0, w_content - 1)
    ys = np.clip(a4_pts_content[:, 1], 0, h_content - 1)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if x1 <= x0 or y1 <= y0:
        return output

    patch = content_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return output

    if size_correction < 1.0:
        ph, pw = patch.shape[:2]
        new_w = max(1, int(pw * size_correction))
        new_h = max(1, int(ph * size_correction))
        shrunk = cv2.resize(patch, (new_w, new_h))
        padded = np.zeros_like(patch)
        x_off = (pw - new_w) // 2
        y_off = (ph - new_h) // 2
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = shrunk
        patch = padded
    elif size_correction > 1.0:
        # Enlarge then center-crop back to original patch size, so it still
        # fits exactly the A4 quad's footprint (a "zoomed in" look) rather
        # than overflowing past the sheet's own boundary.
        ph, pw = patch.shape[:2]
        big_w = max(1, int(pw * size_correction))
        big_h = max(1, int(ph * size_correction))
        enlarged = cv2.resize(patch, (big_w, big_h))
        x_off = (big_w - pw) // 2
        y_off = (big_h - ph) // 2
        patch = enlarged[y_off:y_off + ph, x_off:x_off + pw]

    # Warp this (possibly resized) patch onto the A4 sheet's actual camera
    # quad position within the output frame, and composite it in, masked
    # to the sheet's own corners so it never bleeds outside its own area.
    patch_h, patch_w = patch.shape[:2]
    src_pts = np.float32([[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]])
    H_patch, _ = cv2.findHomography(src_pts, a4_corners_cam.astype(np.float32))
    if H_patch is None:
        return output

    warped_patch = cv2.warpPerspective(patch, H_patch, output_size)

    patch_mask = np.zeros(output.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(patch_mask, a4_corners_cam.astype(np.int32), 255)

    keep_rest = cv2.bitwise_not(patch_mask)
    output_bg = cv2.bitwise_and(output, output, mask=keep_rest)
    patch_fg = cv2.bitwise_and(warped_patch, warped_patch, mask=patch_mask)
    return cv2.add(output_bg, patch_fg)


# ─────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────

def run(content_path, camera_index, obj_thresh, object_mode, monitor_index,
        diff_thresh, settle_sec, save_debug, min_correction, max_correction,
        detector_type="diff", yolo_confidence=0.35):
    source = ContentSource(content_path)
    cap = open_camera(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open camera {camera_index}. Try --camera 1")

    ret, test = cap.read()
    if not ret:
        sys.exit("[ERROR] Camera opened but couldn't read a frame.")
    frame_h, frame_w = test.shape[:2]
    debug_mode = False

    setup_projection_window("Projection Output", monitor_index)
    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)

    print("=" * 64)
    print("Floor Projection Mapper with A4 Local Size-Correction")
    print(f"Object detector: {detector_type.upper()}")
    print("Calibrating against the projector's own light throw...")
    print("=" * 64)

    floor_corners, lit_baseline = calibrate_projection(
        cap, "Projection Output", (frame_w, frame_h),
        settle_sec=settle_sec, diff_thresh=diff_thresh, save_debug=save_debug
    )
    tracker = CentroidTracker(max_distance=80, max_missing_frames=8, smoothing=0.5)
    a4_reference_width_px = None

    yolo_detector = None
    if detector_type == "yolo":
        yolo_detector = YoloFloorDetector(confidence=yolo_confidence)

    print("Controls: Q=quit | SPACE=pause | D=debug | O=avoid/involve | C=recalibrate")
    print("          R=recapture A4 reference size | [ / ] = object sensitivity")
    print("=" * 64)

    while True:
        ret, cam_frame = cap.read()
        if not ret:
            continue
        content_frame = source.next_frame()
        if content_frame is None:
            continue

        # --- A4 sheet: detect + compute its own size-correction ---
        a4_corners = find_a4_corners(cam_frame)
        a4_size_correction = 1.0
        a4_current_width = None
        if a4_corners is not None:
            a4_current_width = corner_apparent_width_px(a4_corners)
            if a4_reference_width_px is None:
                a4_reference_width_px = a4_current_width
                print(f"[INFO] A4 reference width captured: {a4_reference_width_px:.1f}px")
            if a4_reference_width_px:
                a4_size_correction = float(np.clip(
                    a4_reference_width_px / a4_current_width, min_correction, max_correction
                ))

        # --- Generic floor objects: detect + track (no size correction) ---
        object_contours, objects_mask = [], None
        if floor_corners is not None:
            if detector_type == "yolo" and yolo_detector is not None:
                object_contours, objects_mask = yolo_detector.detect(cam_frame, floor_corners)
            elif lit_baseline is not None:
                object_contours, objects_mask = find_objects_on_floor(
                    cam_frame, lit_baseline, floor_corners, brightness_thresh=obj_thresh
                )
        detections = detections_from_contours(object_contours)
        tracked = tracker.update(detections)
        tracked_mask = mask_from_tracked_objects(tracked, cam_frame.shape) if tracked else None

        # --- Compose output: full frame, then locally correct the A4 patch ---
        output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        if floor_corners is not None:
            warped = warp_full_frame(content_frame, floor_corners, (frame_w, frame_h))
            if warped is not None:
                output = warped

                if object_mode == "avoid" and tracked_mask is not None and tracked:
                    keep_mask = cv2.bitwise_not(tracked_mask)
                    output = cv2.bitwise_and(output, output, mask=keep_mask)

                if a4_corners is not None and abs(a4_size_correction - 1.0) > 0.02:
                    output = composite_a4_patch(
                        output, content_frame, floor_corners, a4_corners,
                        a4_size_correction, (frame_w, frame_h)
                    )

        # --- Camera view / debug overlay ---
        cam_display = cam_frame.copy()
        if debug_mode and tracked_mask is not None:
            obj_color = np.zeros_like(cam_display)
            obj_color[:, :, 2] = tracked_mask
            cam_display = cv2.addWeighted(cam_display, 1.0, obj_color, 0.5, 0)

        if floor_corners is not None:
            cv2.polylines(cam_display, [floor_corners.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(cam_display, "FLOOR CALIBRATED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(cam_display, "NOT CALIBRATED - press C", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if a4_corners is not None:
            cv2.polylines(cam_display, [a4_corners.astype(np.int32)], True, (255, 0, 255), 2)
            cv2.putText(cam_display, f"A4: {a4_current_width:.0f}px "
                        f"(ref {a4_reference_width_px:.0f}px) scale={a4_size_correction:.2f}x",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)

        if tracked:
            for obj_id, data in tracked.items():
                cv2.drawContours(cam_display, [data["contour"]], -1, (255, 150, 0), 2)
                cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
                cv2.putText(cam_display, f"ID {obj_id}", (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

        cv2.putText(cam_display, "Q=quit SPACE=pause D=debug O=mode C=recal R=A4ref [ ]=sens",
                    (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        cv2.imshow("Projection Output", output)
        cv2.imshow("Camera View", cam_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            source.toggle_pause()
        elif key == ord("d"):
            debug_mode = not debug_mode
        elif key == ord("o"):
            object_mode = "involve" if object_mode == "avoid" else "avoid"
            print(f"[INFO] Object mode: {object_mode.upper()}")
        elif key == ord("c"):
            print("[INFO] Recalibrating floor...")
            floor_corners, lit_baseline = calibrate_projection(
                cap, "Projection Output", (frame_w, frame_h),
                settle_sec=settle_sec, diff_thresh=diff_thresh, save_debug=save_debug
            )
        elif key == ord("r"):
            if a4_corners is not None:
                a4_reference_width_px = corner_apparent_width_px(a4_corners)
                print(f"[INFO] A4 reference re-captured: {a4_reference_width_px:.1f}px")
            else:
                print("[INFO] No A4 sheet currently detected - can't recapture.")
        elif key == ord("["):
            obj_thresh = max(5, obj_thresh - 5)
        elif key == ord("]"):
            obj_thresh = min(255, obj_thresh + 5)

    cap.release()
    source.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--video", type=str)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--obj-thresh", type=int, default=35,
                         help="Brightness-diff sensitivity (only used with --detector diff). Default 35.")
    parser.add_argument("--diff-thresh", type=int, default=25)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--object-mode", choices=["avoid", "involve"], default="avoid")
    parser.add_argument("--min-correction", type=float, default=0.3)
    parser.add_argument("--max-correction", type=float, default=2.0)
    parser.add_argument("--detector", choices=["diff", "yolo"], default="diff",
                         help="Object detection method for non-A4 floor objects. "
                              "'diff' = brightness comparison vs. calibrated lit floor (no extra "
                              "install, but sensitive to shadows/lighting changes). "
                              "'yolo' = free local AI object detection (pip install ultralytics, "
                              "more robust, needs internet on first run to download the model). "
                              "Default 'diff'.")
    parser.add_argument("--yolo-confidence", type=float, default=0.35,
                         help="Minimum confidence (0-1) for a YOLO detection to count. "
                              "Lower = more detections but more false positives. Default 0.35.")
    args = parser.parse_args()
    content = args.image if args.image else args.video
    run(content, args.camera, args.obj_thresh, args.object_mode, args.monitor,
        args.diff_thresh, args.settle, args.save_debug, args.min_correction, args.max_correction,
        detector_type=args.detector, yolo_confidence=args.yolo_confidence)


if __name__ == "__main__":
    main()
