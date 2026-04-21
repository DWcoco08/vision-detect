"""Streamlit Web UI for Vehicle Damage Detection.

Interactive web interface for uploading images, detecting damage,
viewing results, and downloading annotated images + PDF reports.

Usage:
    streamlit run app.py
"""

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from models.yolo_model import DamageDetector
from models.severity_model import SeverityPredictor
from reports.pdf_report import REPAIR_COST, DamageReport
from utils.preprocessing import crop_detection
from utils.visualization import draw_detections

# Page config
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_detector(model_path: str, confidence: float) -> DamageDetector:
    """Load and cache YOLO detector."""
    return DamageDetector(model_path, confidence=confidence)


@st.cache_resource
def load_predictor(model_path: str) -> SeverityPredictor:
    """Load and cache severity predictor."""
    return SeverityPredictor(model_path)


def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert BGR numpy image to PNG bytes for download."""
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


def main():
    st.title("🚗 Vehicle Damage Detection System")
    st.markdown("Upload an image to detect vehicle damage and estimate severity.")

    # --- Sidebar: Settings ---
    with st.sidebar:
        st.header("Settings")

        yolo_path = st.text_input("YOLO Weights", value="weights/best.pt")
        severity_path = st.text_input("Severity Weights", value="weights/severity.pth")
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

        st.divider()
        st.markdown("**How to use:**")
        st.markdown("1. Upload an image")
        st.markdown("2. Click **Detect Damage**")
        st.markdown("3. View results & download")

    # --- Main Area: Upload ---
    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"],
        help="Supported: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    # Read uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Show original image
    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    # --- Detect Button ---
    if not st.button("🔍 Detect Damage", type="primary", use_container_width=True):
        return

    # Load models
    try:
        detector = load_detector(yolo_path, confidence)
        predictor = load_predictor(severity_path)
    except FileNotFoundError as e:
        st.error(f"Model not found: {e}")
        return

    # Run detection
    with st.spinner("Detecting damage..."):
        detections = detector.detect(image_rgb)

    if not detections:
        st.warning("No damage detected in this image.")
        return

    # Predict severity for each detection
    severities = []
    for det in detections:
        crop = crop_detection(image_rgb, det.bbox)
        severity = predictor.predict(crop)
        severities.append(severity)

    # Draw annotations
    annotated = draw_detections(image_rgb, detections, severities)

    # --- Results Display ---
    st.success(f"Found **{len(detections)}** damage region(s)")

    # Side-by-side images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Detected")
        # annotated is BGR from draw_detections, convert for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

    # Results table
    st.subheader("Detection Results")
    table_data = []
    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
        table_data.append({
            "#": i,
            "Type": det.class_name,
            "Confidence": f"{det.confidence:.1%}",
            "Severity": sev_text,
            "Bbox": f"[{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]",
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Cost estimation
    st.subheader("Estimated Repair Cost")
    total_min, total_max = 0, 0
    cost_data = []
    for det, sev in zip(detections, severities):
        cost = REPAIR_COST.get(det.class_name, {"min": 100, "max": 400})
        factor = max(sev, 10) / 100
        min_c = int(cost["min"] * factor)
        max_c = int(cost["max"] * factor)
        total_min += min_c
        total_max += max_c
        cost_data.append({
            "Type": det.class_name,
            "Severity": f"{sev:.1f}%",
            "Min Cost ($)": min_c,
            "Max Cost ($)": max_c,
        })
    st.dataframe(cost_data, use_container_width=True, hide_index=True)
    st.metric("Total Estimated Cost", f"${total_min} - ${total_max}")

    # --- Downloads ---
    st.subheader("Downloads")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.download_button(
            label="📷 Download Annotated Image",
            data=image_to_bytes(annotated),
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl_col2:
        # Generate PDF on-the-fly
        report = DamageReport(
            image_path=uploaded_file.name,
            original_image=image_rgb,
            annotated_image=annotated,
            detections=detections,
            severities=severities,
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            report.generate(tmp.name)
            pdf_bytes = Path(tmp.name).read_bytes()

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
