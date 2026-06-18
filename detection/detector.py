"""Main detection loop."""
import logging

import cv2

from .alert import build_alert_manager
from .camera import CameraFactory
from .model import FALLEN, build_model

logger = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    camera = CameraFactory.create(cfg)
    model = build_model(cfg)
    alert_mgr = build_alert_manager(cfg)
    show = cfg["display"]["show_window"]
    title = cfg["display"]["window_title"]

    logger.info("Detection loop started")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                logger.warning("Failed to read frame, reconnecting...")
                camera.reconnect()
                continue

            detections = model.predict(frame)
            is_fallen = any(d["label"] == FALLEN for d in detections)

            for d in detections:
                logger.info("[%s] conf=%.2f", d["label"], d["confidence"])

            alerted = alert_mgr.update(is_fallen, frame)
            if alerted:
                logger.warning("FALL ALERT FIRED")

            if show:
                annotated = model.annotate(frame, detections)
                cv2.imshow(title, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if show:
            cv2.destroyAllWindows()
        logger.info("Detection loop stopped")
