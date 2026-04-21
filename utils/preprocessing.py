"""Image preprocessing utilities for loading and cropping."""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load image from file path and convert to RGB.

    Args:
        path: Path to image file.

    Returns:
        Image as RGB numpy array (H, W, 3).

    Raises:
        FileNotFoundError: If image file does not exist or cannot be read.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_detection(
    image: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop a detection region from image.

    Clamps bounding box coordinates to image boundaries
    to prevent index-out-of-range errors.

    Args:
        image: Source image as numpy array (H, W, 3).
        bbox: Bounding box as (x1, y1, x2, y2).

    Returns:
        Cropped image region as numpy array.
    """
    h, w = image.shape[:2]
    x1 = max(0, min(bbox[0], w))
    y1 = max(0, min(bbox[1], h))
    x2 = max(0, min(bbox[2], w))
    y2 = max(0, min(bbox[3], h))
    return image[y1:y2, x1:x2]
