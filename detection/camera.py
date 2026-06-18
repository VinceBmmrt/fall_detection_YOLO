"""Camera backends: OpenCVCamera, RealSenseCamera, and CameraFactory."""
import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseCamera(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return a BGR frame or None on failure."""

    @abstractmethod
    def reconnect(self) -> None:
        """Attempt to re-open the camera source."""

    @abstractmethod
    def release(self) -> None:
        """Release hardware resources."""


class OpenCVCamera(BaseCamera):
    """USB, file, or RTSP camera via cv2.VideoCapture."""

    def __init__(self, source: int | str, width: int, height: int, fps: int) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def reconnect(self) -> None:
        logger.warning("OpenCVCamera: reconnecting to source %s", self._source)
        if self._cap:
            self._cap.release()
        self._open()

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


class RealSenseCamera(BaseCamera):
    """Intel RealSense D435i via pyrealsense2 (lazy import)."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        import pyrealsense2 as rs  # lazy — not available on non-Jetson machines

        self._rs = rs
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline: object | None = None
        self._open()

    def _open(self) -> None:
        pipeline = self._rs.pipeline()
        cfg = self._rs.config()
        cfg.enable_stream(self._rs.stream.color, self._width, self._height, self._rs.format.bgr8, self._fps)
        pipeline.start(cfg)
        self._pipeline = pipeline

    def read(self) -> np.ndarray | None:
        if self._pipeline is None:
            return None
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def reconnect(self) -> None:
        logger.warning("RealSenseCamera: reconnecting")
        self.release()
        self._open()

    def release(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None


class CameraFactory:
    @staticmethod
    def create(cfg: dict) -> BaseCamera:
        """Instantiate the correct camera from config['camera']."""
        cam = cfg["camera"]
        backend = cam["backend"]
        width, height, fps = cam["width"], cam["height"], cam["fps"]

        if backend == "realsense":
            return RealSenseCamera(width, height, fps)
        if backend == "opencv":
            return OpenCVCamera(cam["source"], width, height, fps)
        raise ValueError(f"Unknown camera backend: {backend!r}")
