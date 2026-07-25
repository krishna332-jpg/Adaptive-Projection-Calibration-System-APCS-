"""
OSC Bridge to TouchDesigner
----------------------------
Sends everything TouchDesigner needs to do the actual projection mapping:

  /apcs/calibration/start          (no args)  — tell TD to hide/blank its
                                                 output so the black/white
                                                 calibration flash is visible
                                                 on the projector
  /apcs/calibration/done           (no args)  — tell TD to restore its output
  /apcs/calibration/corners         f f f f f f f f
                                    — the 4 floor corners (TL,TR,BR,BL) as
                                      8 floats (x1,y1,x2,y2,x3,y3,x4,y4), in
                                      camera-pixel space. TD uses these as
                                      the starting points for its Corner Pin.

  /apcs/mode                        s          — "avoid" or "involve"

  /apcs/object/<id>                  f f f f f f
                                    — x, y, w, h (camera-pixel bounding box),
                                      height_cm, size_correction_factor
  /apcs/objects/active               s          — comma-separated list of
                                                    currently-alive object ids,
                                                    e.g. "0,2,3". TD uses this
                                                    to know which objects to
                                                    remove from its scene.

Requires: pip install python-osc

This module never raises on send failure (a dropped OSC packet during a
live exhibit shouldn't crash the exhibit) — errors are logged instead.
"""

import logging

import config

logger = logging.getLogger("apcs.osc")

try:
    from pythonosc.udp_client import SimpleUDPClient
except ImportError:
    SimpleUDPClient = None


class OSCBridge:
    """
    Thin wrapper around python-osc for talking to TouchDesigner.
    If OSC_ENABLED is False in config.py, or python-osc isn't installed,
    every method silently becomes a no-op instead of crashing the app.
    """

    def __init__(self):
        self._enabled = bool(config.OSC_ENABLED)
        self._client = None

        if not self._enabled:
            logger.info("OSC disabled in config.py (OSC_ENABLED = False).")
            return

        if SimpleUDPClient is None:
            logger.warning(
                "python-osc not installed — OSC output disabled. "
                "Run: pip install python-osc"
            )
            self._enabled = False
            return

        try:
            self._client = SimpleUDPClient(config.OSC_HOST, config.OSC_PORT)
            logger.info(
                "OSC bridge ready -> %s:%s", config.OSC_HOST, config.OSC_PORT
            )
        except Exception as e:
            logger.warning("Could not create OSC client: %s", e)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Calibration handshake ────────────────────────────────────────────

    def send_calibration_start(self):
        """Tell TD to hide/blank its output so the projector shows our
        black/white calibration flash instead of TD's last frame."""
        self._send("/apcs/calibration/start", [])

    def send_calibration_done(self):
        """Tell TD it's safe to show its output again."""
        self._send("/apcs/calibration/done", [])

    def send_floor_corners(self, corners):
        """
        corners: 4x2 array-like (TL, TR, BR, BL), in camera-pixel space.
        Sent as 8 flat floats so it works with any OSC receiver.
        """
        if corners is None:
            return
        flat = [float(v) for pt in corners for v in pt]
        if len(flat) != 8:
            logger.warning(
                "send_floor_corners: expected 8 values, got %d — skipping.",
                len(flat),
            )
            return
        self._send("/apcs/calibration/corners", flat)

    # ── Live state ────────────────────────────────────────────────────────

    def send_mode(self, mode: str):
        self._send("/apcs/mode", [mode])

    def send_tracked_objects(self, tracked: dict):
        """
        tracked: dict of id -> {centroid, contour, height_cm, correction}
                 (correction is optional; defaults to 1.0 if not present)
        Sends one message per object plus a summary of active ids so
        TouchDesigner can clean up objects that disappeared.
        """
        if not self._enabled:
            return

        active_ids = []
        for obj_id, data in tracked.items():
            contour = data["contour"].reshape(-1, 2)
            x0, y0 = contour[:, 0].min(), contour[:, 1].min()
            x1, y1 = contour[:, 0].max(), contour[:, 1].max()
            w, h = x1 - x0, y1 - y0
            height_cm = float(data.get("height_cm", 0.0))
            correction = float(data.get("correction", 1.0))

            self._send(
                f"/apcs/object/{obj_id}",
                [float(x0), float(y0), float(w), float(h), height_cm, correction],
            )
            active_ids.append(str(obj_id))

        self._send("/apcs/objects/active", [",".join(active_ids)])

    # ── Internal ──────────────────────────────────────────────────────────

    def _send(self, address: str, args: list):
        if not self._enabled or self._client is None:
            return
        try:
            self._client.send_message(address, args)
        except Exception as e:
            logger.debug("OSC send failed on %s: %s", address, e)