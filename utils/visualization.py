"""Visualization utilities for drawing detection results on images."""

import cv2
import numpy as np

from models.yolo_model import Detection

# Color map per damage class (BGR format for OpenCV)
COLOR_MAP = {
    "scratch": (0, 255, 255),  # Yellow
    "dent": (0, 165, 255),  # Orange
    "crack": (0, 0, 255),  # Red
}
DEFAULT_COLOR = (0, 255, 0)  # Green fallback

# Drawing constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
LABEL_PADDING = 4


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    severities: list[float],
) -> np.ndarray:
    """Draw bounding boxes with labels and severity on image.

    Args:
        image: Input RGB image (H, W, 3).
        detections: List of Detection objects from YOLO.
        severities: Corresponding severity scores (0-100) per detection.

    Returns:
        Annotated image copy in BGR format (ready for cv2.imshow/imwrite).
    """
    # Convert RGB to BGR for OpenCV drawing
    annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for det, severity in zip(detections, severities):
        x1, y1, x2, y2 = det.bbox
        color = COLOR_MAP.get(det.class_name, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Build label text
        sev_text = f"{severity:.1f}%" if severity >= 0 else "N/A"
        label = f"{det.class_name} {det.confidence:.0%} | Sev: {sev_text}"

        # Calculate label background size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        label_y = max(y1 - LABEL_PADDING, text_h + LABEL_PADDING)

        # Draw filled background for label readability
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - LABEL_PADDING),
            (x1 + text_w + LABEL_PADDING, label_y + LABEL_PADDING),
            color,
            cv2.FILLED,
        )

        # Draw label text (black on colored background)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y),
            FONT,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICKNESS,
        )

    return annotated
