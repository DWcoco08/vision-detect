"""YOLOv8 damage detection wrapper module."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single damage detection result."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str  # scratch, dent, crack
    confidence: float  # 0.0 - 1.0


class DamageDetector:
    """Wrapper around YOLOv8 for vehicle damage detection.

    Loads a custom-trained YOLO model and runs inference
    on images to detect scratch, dent, and crack regions.
    """

    def __init__(self, model_path: str, confidence: float = 0.25):
        """Initialize detector with model weights.

        Args:
            model_path: Path to YOLO best.pt weights file.
            confidence: Minimum confidence threshold for detections.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}. "
                "Train a model or provide a valid best.pt path."
            )
        self.model = YOLO(str(path))
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run damage detection on an image.

        Args:
            image: Input image as RGB numpy array (H, W, 3).

        Returns:
            List of Detection objects with bbox, class, confidence.
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Extract bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Extract class name and confidence
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                class_name = result.names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_name=class_name,
                        confidence=conf,
                    )
                )

        return detections
