"""PDF report generator for vehicle damage inspection results.

Generates professional PDF reports containing original and annotated
images, detection results table, and repair cost estimation.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

from models.yolo_model import Detection

# Estimated repair cost ranges per damage type (USD)
REPAIR_COST = {
    "scratch": {"min": 50, "max": 150},
    "dent": {"min": 150, "max": 500},
    "crack": {"min": 200, "max": 800},
}

# Severity thresholds for overall assessment
SEVERITY_LABELS = [
    (25, "LOW"),
    (50, "MODERATE"),
    (75, "HIGH"),
    (100, "CRITICAL"),
]


def _severity_label(avg_severity: float) -> str:
    """Map average severity to human-readable label."""
    for threshold, label in SEVERITY_LABELS:
        if avg_severity <= threshold:
            return label
    return "CRITICAL"


def _save_temp_image(image: np.ndarray, suffix: str = ".png") -> str:
    """Save numpy image to temp file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    # Ensure BGR for cv2.imwrite
    if len(image.shape) == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image
    cv2.imwrite(tmp.name, bgr)
    return tmp.name


class DamageReport:
    """Generates PDF inspection report for vehicle damage detection.

    Report includes: title, images (original + annotated),
    detection results table, cost estimation, and summary.
    """

    def __init__(
        self,
        image_path: str,
        original_image: np.ndarray,
        annotated_image: np.ndarray,
        detections: list[Detection],
        severities: list[float],
    ):
        """Initialize report data.

        Args:
            image_path: Path to original image file.
            original_image: Original RGB image array.
            annotated_image: Annotated BGR image array (from visualization).
            detections: List of Detection objects.
            severities: Corresponding severity scores per detection.
        """
        self.image_path = Path(image_path).name
        self.original_image = original_image
        self.annotated_image = annotated_image
        self.detections = detections
        self.severities = severities
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def generate(self, output_path: str) -> str:
        """Generate PDF report and save to file.

        Args:
            output_path: Destination path for PDF file.

        Returns:
            Absolute path to generated PDF.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        self._add_title(pdf)
        self._add_images(pdf)
        self._add_results_table(pdf)
        self._add_cost_estimation(pdf)
        self._add_summary(pdf)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pdf.output(output_path)
        return str(Path(output_path).resolve())

    def _add_title(self, pdf: FPDF) -> None:
        """Add report title and metadata."""
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "VEHICLE DAMAGE INSPECTION REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Date: {self.timestamp}  |  Image: {self.image_path}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # Divider line
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

    def _add_images(self, pdf: FPDF) -> None:
        """Add original and annotated images side by side."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Images", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Save images to temp files
        orig_path = _save_temp_image(self.original_image)
        annot_path = _save_temp_image(
            cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB)
        )

        img_width = 90  # Each image takes ~half page width
        y_pos = pdf.get_y()

        # Original image
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(img_width, 5, "Original", align="C")
        pdf.cell(img_width, 5, "Annotated", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.image(orig_path, x=10, w=img_width)
        annot_y = y_pos + 5
        pdf.image(annot_path, x=105, y=annot_y, w=img_width)

        # Move below images
        pdf.ln(6)

    def _add_results_table(self, pdf: FPDF) -> None:
        """Add detection results table."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Detection Results", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if not self.detections:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, "No damage detected.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
            return

        # Table header
        col_widths = [15, 40, 35, 35, 65]
        headers = ["#", "Type", "Confidence", "Severity", "Bounding Box"]

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 230, 230)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 9)
        for i, (det, sev) in enumerate(zip(self.detections, self.severities), 1):
            sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
            bbox_text = f"[{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]"
            row = [str(i), det.class_name, f"{det.confidence:.1%}", sev_text, bbox_text]
            for j, cell in enumerate(row):
                pdf.cell(col_widths[j], 7, cell, border=1, align="C")
            pdf.ln()

        pdf.ln(4)

    def _add_cost_estimation(self, pdf: FPDF) -> None:
        """Add repair cost estimation table."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Estimated Repair Cost", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if not self.detections:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, "No repairs needed.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
            return

        col_widths = [50, 45, 45, 50]
        headers = ["Damage Type", "Severity", "Min Cost ($)", "Max Cost ($)"]

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 230, 230)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
        pdf.ln()

        total_min, total_max = 0, 0
        pdf.set_font("Helvetica", "", 9)

        for det, sev in zip(self.detections, self.severities):
            cost = REPAIR_COST.get(det.class_name, {"min": 100, "max": 400})
            # Scale cost by severity
            severity_factor = max(sev, 10) / 100  # Minimum 10% factor
            min_cost = int(cost["min"] * severity_factor)
            max_cost = int(cost["max"] * severity_factor)
            total_min += min_cost
            total_max += max_cost

            sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
            row = [det.class_name, sev_text, str(min_cost), str(max_cost)]
            for j, cell in enumerate(row):
                pdf.cell(col_widths[j], 7, cell, border=1, align="C")
            pdf.ln()

        # Total row
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_widths[0] + col_widths[1], 7, "TOTAL", border=1, align="C")
        pdf.cell(col_widths[2], 7, str(total_min), border=1, align="C")
        pdf.cell(col_widths[3], 7, str(total_max), border=1, align="C")
        pdf.ln(8)

    def _add_summary(self, pdf: FPDF) -> None:
        """Add summary section."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        num = len(self.detections)
        pdf.set_font("Helvetica", "", 10)

        if num == 0:
            pdf.cell(0, 7, "No damage detected in this image.", new_x="LMARGIN", new_y="NEXT")
        else:
            valid_sevs = [s for s in self.severities if s >= 0]
            avg_sev = sum(valid_sevs) / len(valid_sevs) if valid_sevs else 0
            label = _severity_label(avg_sev)

            pdf.cell(0, 7, f"Total damages detected: {num}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 7, f"Average severity: {avg_sev:.1f}%", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 7, f"Overall assessment: {label}", new_x="LMARGIN", new_y="NEXT")

            # List damage types
            types = set(d.class_name for d in self.detections)
            pdf.cell(0, 7, f"Damage types found: {', '.join(types)}", new_x="LMARGIN", new_y="NEXT")
