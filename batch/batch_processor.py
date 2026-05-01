"""Batch image processing for vehicle damage detection.

Processes all images in a directory, saves annotated results,
optionally generates PDF reports, and exports a CSV summary.
"""

import logging
from pathlib import Path

import cv2
import pandas as pd

from models.yolo_model import DamageDetector
from models.severity_model import SeverityPredictor
from reports.pdf_report import REPAIR_COST, DamageReport
from utils.preprocessing import crop_detection, load_image
from utils.visualization import draw_detections

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_batch(
    input_dir: str,
    output_dir: str,
    detector: DamageDetector,
    predictor: SeverityPredictor,
    generate_pdf: bool = False,
) -> pd.DataFrame:
    """Process all images in a directory for damage detection.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory to save results.
        detector: Loaded DamageDetector instance.
        predictor: Loaded SeverityPredictor instance.
        generate_pdf: Whether to generate PDF reports per image.

    Returns:
        DataFrame with summary of all detections.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output subdirectories
    annotated_dir = output_path / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    if generate_pdf:
        reports_dir = output_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.warning("No images found in %s", input_dir)
        return pd.DataFrame()

    logger.info("Found %d images to process", len(image_files))

    # Process each image
    rows = []
    for img_file in image_files:
        row = _process_single(
            img_file, annotated_dir, detector, predictor
        )

        # Generate PDF report if requested
        if generate_pdf and row["num_detections"] > 0:
            _generate_report(
                img_file, annotated_dir, detector, predictor,
                reports_dir / f"{img_file.stem}_report.pdf"
            )

        rows.append(row)
        logger.info("Processed: %s (%d detections)", img_file.name, row["num_detections"])

    # Build summary DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = output_path / "summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Summary saved to %s", csv_path)

    return df


def _process_single(
    img_file: Path,
    annotated_dir: Path,
    detector: DamageDetector,
    predictor: SeverityPredictor,
) -> dict:
    """Process a single image and return summary row."""
    image = load_image(str(img_file))
    detections = detector.detect(image)

    if not detections:
        return {
            "image": img_file.name,
            "num_detections": 0,
            "damage_types": "",
            "avg_severity": 0.0,
            "max_severity": 0.0,
            "min_cost": 0,
            "max_cost": 0,
        }

    severities = []
    for det in detections:
        crop = crop_detection(image, det.bbox)
        severity = predictor.predict(crop)
        severities.append(severity)

    annotated = draw_detections(image, detections, severities)
    out_path = annotated_dir / f"{img_file.stem}_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)

    total_min, total_max = 0, 0
    for det, sev in zip(detections, severities):
        cost = REPAIR_COST.get(det.class_name, {"min": 100, "max": 400})
        factor = max(sev, 10) / 100
        total_min += int(cost["min"] * factor)
        total_max += int(cost["max"] * factor)

    valid_sevs = [s for s in severities if s >= 0]
    damage_types = ";".join(d.class_name for d in detections)

    return {
        "image": img_file.name,
        "num_detections": len(detections),
        "damage_types": damage_types,
        "avg_severity": round(sum(valid_sevs) / len(valid_sevs), 1) if valid_sevs else 0.0,
        "max_severity": round(max(valid_sevs), 1) if valid_sevs else 0.0,
        "min_cost": total_min,
        "max_cost": total_max,
    }


def _generate_report(
    img_file: Path,
    annotated_dir: Path,
    detector: DamageDetector,
    predictor: SeverityPredictor,
    pdf_path: Path,
) -> None:
    """Generate PDF report for a single image."""
    image = load_image(str(img_file))
    detections = detector.detect(image)

    severities = []
    for det in detections:
        crop = crop_detection(image, det.bbox)
        severities.append(predictor.predict(crop))

    annotated = draw_detections(image, detections, severities)

    report = DamageReport(
        image_path=str(img_file),
        original_image=image,
        annotated_image=annotated,
        detections=detections,
        severities=severities,
    )
    report.generate(str(pdf_path))
