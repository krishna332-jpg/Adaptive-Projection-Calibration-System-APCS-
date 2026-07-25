"""
Video Output Bridge
---------------------
Sends the live content frame (video/image) to TouchDesigner instead of
drawing it into an OpenCV window ourselves. TouchDesigner receives this
as a texture and does the actual floor-warp + per-object correction +
final output to the projector.

Three modes, set via VIDEO_OUTPUT_MODE in config.py:

  "spout"   Windows only. Same-machine, GPU texture sharing — the
            standard way to feed TouchDesigner/Resolume/MadMapper from
            a Python/OpenCV app. Requires: pip install SpoutGL
            In TouchDesigner: add a "Spout In" TOP, set its sender name
            to match VIDEO_SENDER_NAME in config.py.

  "ndi"     Cross-platform, also works over a network. Requires:
            pip install ndi-python, and the NDI Runtime installed
            (https://ndi.video/tools/). In TouchDesigner: add an
            "NDI In" TOP and pick VIDEO_SENDER_NAME from its source list.

  "window"  Fallback for testing without Spout/NDI installed yet. Just
            opens a normal OpenCV window showing the raw content frame,
            same as the old behavior. Nothing goes to TouchDesigner.

If the library for "spout" or "ndi" isn't installed, or fails to
initialize (e.g. no GPU context, wrong OS), this module automatically
falls back to "window" mode and logs a warning — it will never crash
the main loop because the output bridge failed.
"""

import logging

import cv2
import numpy as np

import config

logger = logging.getLogger("apcs.video_output")

PREVIEW_WINDOW = "APCS Output (fallback preview - not sent to TD)"


class VideoOutput:
    """
    Call send(frame) once per loop iteration. Internally picks the
    Spout / NDI / window backend based on config.VIDEO_OUTPUT_MODE and
    falls back safely if the chosen backend can't be initialized.
    """

    def __init__(self):
        self._mode = config.VIDEO_OUTPUT_MODE.lower().strip()
        self._impl = None
        self._w, self._h = None, None

        if self._mode == "spout":
            self._impl = self._try_init_spout()
        elif self._mode == "ndi":
            self._impl = self._try_init_ndi()
        elif self._mode == "window":
            self._impl = None  # handled directly in send()
        else:
            logger.warning(
                "Unknown VIDEO_OUTPUT_MODE '%s' — falling back to 'window'.",
                self._mode,
            )
            self._mode = "window"

        if self._mode != "window" and self._impl is None:
            logger.warning(
                "Falling back to local preview window (nothing is being "
                "sent to TouchDesigner)."
            )
            self._mode = "window"

        if self._mode == "window":
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

    # ── Backend init ──────────────────────────────────────────────────────

    def _try_init_spout(self):
        try:
            import SpoutGL  # noqa: F401
        except ImportError:
            logger.warning(
                "SpoutGL not installed — run: pip install SpoutGL "
                "(Windows only)."
            )
            return None
        try:
            sender = SpoutGL.SpoutSender()
            sender.setSenderName(config.VIDEO_SENDER_NAME)
            logger.info(
                "Spout sender '%s' ready. In TouchDesigner, add a "
                "Spout In TOP and select this sender.",
                config.VIDEO_SENDER_NAME,
            )
            return ("spout", sender)
        except Exception as e:
            logger.warning("Could not initialize Spout sender: %s", e)
            return None

    def _try_init_ndi(self):
        try:
            import NDIlib as ndi  # from the ndi-python package
        except ImportError:
            logger.warning(
                "ndi-python not installed — run: pip install ndi-python "
                "(also requires the NDI Runtime)."
            )
            return None
        try:
            if not ndi.initialize():
                raise RuntimeError("NDIlib.initialize() failed")
            send_settings = ndi.SendCreate()
            send_settings.ndi_name = config.VIDEO_SENDER_NAME
            sender = ndi.send_create(send_settings)
            video_frame = ndi.VideoFrameV2()
            logger.info(
                "NDI sender '%s' ready. In TouchDesigner, add an NDI In "
                "TOP and pick this source.",
                config.VIDEO_SENDER_NAME,
            )
            return ("ndi", ndi, sender, video_frame)
        except Exception as e:
            logger.warning("Could not initialize NDI sender: %s", e)
            return None

    # ── Public API ────────────────────────────────────────────────────────

    def send(self, frame: np.ndarray):
        """Send one BGR frame. Safe to call every loop iteration."""
        if frame is None:
            return

        if self._mode == "window":
            cv2.imshow(PREVIEW_WINDOW, frame)
            return

        try:
            if self._mode == "spout":
                self._send_spout(frame)
            elif self._mode == "ndi":
                self._send_ndi(frame)
        except Exception as e:
            logger.warning(
                "Video output send failed (%s) — switching to fallback "
                "preview window for the rest of this run.",
                e,
            )
            self._mode = "window"
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.imshow(PREVIEW_WINDOW, frame)

    def _send_spout(self, frame: np.ndarray):
        _, sender = self._impl
        h, w = frame.shape[:2]
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # SpoutGL expects the image top-to-bottom flipped relative to
        # OpenCV's row order in most TouchDesigner setups — invert=True
        # handles this for you.
        sender.sendImage(rgba.tobytes(), w, h, SpoutGLFormat(), True, 0)

    def _send_ndi(self, frame: np.ndarray):
        _, ndi, sender, video_frame = self._impl
        h, w = frame.shape[:2]
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        video_frame.data = rgba
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
        video_frame.xres = w
        video_frame.yres = h
        ndi.send_send_video_v2(sender, video_frame)

    def close(self):
        if self._mode == "window":
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except Exception:
                pass
        elif self._mode == "ndi" and self._impl is not None:
            try:
                _, ndi, sender, _ = self._impl
                ndi.send_destroy(sender)
                ndi.destroy()
            except Exception:
                pass


def SpoutGLFormat():
    """
    GL_RGBA constant used by SpoutGL.sendImage(). Defined locally so this
    file doesn't hard-fail importing SpoutGL/PyOpenGL at module load time
    on machines that don't have Spout set up yet (e.g. during testing in
    'window' mode on Linux/Mac).
    """
    GL_RGBA = 0x1908
    return GL_RGBA