"""Streamlit Web UI for Vehicle Damage Detection.

Usage:
    streamlit run app.py
"""

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

# Minimal CSS — only what Streamlit can't do natively
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 1000px; padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(model_path: str, confidence: float) -> DamageDetector:
    return DamageDetector(model_path, confidence=confidence)


@st.cache_resource
def load_predictor(model_path: str) -> SeverityPredictor:
    return SeverityPredictor(model_path)


def image_to_bytes(image: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", image)
    return buf.tobytes()


def main():
    # ── Header ──
    st.title("🚗 Vehicle Damage Detection")
    st.caption("Upload a photo · AI detects damage · Estimate repair cost")

    # ── Settings ──
    with st.expander("⚙️ Settings"):
        c1, c2, c3 = st.columns(3)
        yolo_path = c1.text_input("YOLO weights", value="weights/best.pt")
        severity_path = c2.text_input("Severity weights", value="weights/severity.pth")
        confidence = c3.slider("Confidence", 0.1, 1.0, 0.25, 0.05)

    # ── Upload ──
    uploaded = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"],
    )

    if not uploaded:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔍 Detect", "Scratch · Dent · Crack")
        c2.metric("📊 Severity", "0 – 100%")
        c3.metric("📄 Report", "PDF + Cost")
        return

    # Read image
    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    st.image(rgb, use_container_width=True)

    if not st.button("🔍 Detect Damage", type="primary", use_container_width=True):
        return

    # Load models
    try:
        detector = load_detector(yolo_path, confidence)
        predictor = load_predictor(severity_path)
    except FileNotFoundError as e:
        st.error(f"Model not found: {e}")
        return

    with st.spinner("Analyzing..."):
        detections = detector.detect(rgb)

    if not detections:
        st.info("No damage detected.")
        return

    # Predict severity
    severities = []
    for det in detections:
        crop = crop_detection(rgb, det.bbox)
        severities.append(predictor.predict(crop))

    annotated = draw_detections(rgb, detections, severities)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ── Metrics ──
    valid = [s for s in severities if s >= 0]
    avg = sum(valid) / len(valid) if valid else 0
    mx = max(valid) if valid else 0

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Damages", len(detections))
    m2.metric("Avg Severity", f"{avg:.1f}%")
    m3.metric("Max Severity", f"{mx:.1f}%")

    # ── Comparison ──
    st.subheader("Comparison")
    col1, col2 = st.columns(2)
    col1.image(rgb, caption="Original", use_container_width=True)
    col2.image(annotated_rgb, caption="Detected", use_container_width=True)

    # ── Detections table ──
    st.subheader("Detections")
    rows = []
    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
        rows.append({
            "#": i,
            "Type": det.class_name.capitalize(),
            "Confidence": f"{det.confidence:.0%}",
            "Severity": sev_text,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Cost ──
    st.subheader("Repair Cost")
    total_min, total_max = 0, 0
    cost_rows = []
    for det, sev in zip(detections, severities):
        cost = REPAIR_COST.get(det.class_name, {"min": 100, "max": 400})
        factor = max(sev, 10) / 100
        lo = int(cost["min"] * factor)
        hi = int(cost["max"] * factor)
        total_min += lo
        total_max += hi
        cost_rows.append({
            "Type": det.class_name.capitalize(),
            "Severity": f"{sev:.1f}%",
            "Min ($)": lo,
            "Max ($)": hi,
        })
    st.dataframe(cost_rows, use_container_width=True, hide_index=True)
    st.metric("Total Estimated Cost", f"${total_min} – ${total_max}")

    # ── Download ──
    st.subheader("Download")
    d1, d2 = st.columns(2)
    d1.download_button(
        "📷 Annotated Image",
        data=image_to_bytes(annotated),
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}_annotated.png",
        mime="image/png",
        use_container_width=True,
    )

    report = DamageReport(
        image_path=uploaded.name,
        original_image=rgb,
        annotated_image=annotated,
        detections=detections,
        severities=severities,
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        report.generate(tmp.name)
        pdf_bytes = Path(tmp.name).read_bytes()

    d2.download_button(
        "📄 PDF Report",
        data=pdf_bytes,
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
