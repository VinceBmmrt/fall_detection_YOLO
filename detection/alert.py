"""Alert system: handlers and AlertManager with sliding window + cooldown."""
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AlertHandler(ABC):
    @abstractmethod
    def send(self, frame: np.ndarray) -> None:
        """Fire an alert, optionally using the triggering frame."""


class ConsoleAlertHandler(AlertHandler):
    def __init__(self, save_image: bool, save_dir: str) -> None:
        self._save_image = save_image
        self._save_dir = Path(save_dir)

    def send(self, frame: np.ndarray) -> None:
        logger.warning("FALL DETECTED")
        print("[ALERT] Fall detected!")
        if self._save_image:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            filename = self._save_dir / f"fall_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), frame)
            logger.info("Saved alert image to %s", filename)


class WebhookAlertHandler(AlertHandler):
    def send(self, frame: np.ndarray) -> None:
        raise NotImplementedError("WebhookAlertHandler not yet implemented")


class AgentToolHandler(AlertHandler):
    """Future integration point for AI agent tool call — wire up here only."""
    def send(self, frame: np.ndarray) -> None:
        raise NotImplementedError("AgentToolHandler not yet implemented")


class AlertManager:
    """Sliding window over recent frames; fires handler when fall is confirmed."""

    def __init__(self, handler: AlertHandler, min_fall_frames: int, cooldown_seconds: float) -> None:
        self._handler = handler
        self._min_fall_frames = min_fall_frames
        self._cooldown = cooldown_seconds
        self._window: deque[bool] = deque(maxlen=min_fall_frames)
        self._last_alert: float = 0.0

    def update(self, is_fallen: bool, frame: np.ndarray) -> bool:
        """Push latest detection; returns True if alert fired this frame."""
        self._window.append(is_fallen)
        if (
            len(self._window) == self._min_fall_frames
            and all(self._window)
            and (time.monotonic() - self._last_alert) >= self._cooldown
        ):
            self._last_alert = time.monotonic()
            self._handler.send(frame)
            return True
        return False


def build_alert_manager(cfg: dict) -> AlertManager:
    a = cfg["alert"]
    handler_name = a["handler"]

    if handler_name == "console":
        handler = ConsoleAlertHandler(a["save_image"], a["save_dir"])
    elif handler_name == "webhook":
        handler = WebhookAlertHandler()
    elif handler_name == "agent":
        handler = AgentToolHandler()
    else:
        raise ValueError(f"Unknown alert handler: {handler_name!r}")

    return AlertManager(handler, a["min_fall_frames"], a["cooldown_seconds"])
