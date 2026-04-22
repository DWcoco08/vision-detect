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

# -- Page config --
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Minimal CSS --
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 1100px; padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)


# -- Model loading (cached) --
@st.cache_resource
def load_detector(model_path: str, confidence: float) -> DamageDetector:
    return DamageDetector(model_path, confidence=confidence)


@st.cache_resource
def load_predictor(model_path: str) -> SeverityPredictor:
    return SeverityPredictor(model_path)


def image_to_bytes(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def severity_badge(sev: float) -> str:
    """Return severity level text."""
    if sev <= 25:
        return "Low"
    if sev <= 50:
        return "Moderate"
    if sev <= 75:
        return "High"
    return "Critical"


def severity_color(sev: float) -> str:
    """Return color for severity value."""
    if sev <= 25:
        return "#22c55e"
    if sev <= 50:
        return "#eab308"
    if sev <= 75:
        return "#f97316"
    return "#ef4444"


# =====================
#  SIDEBAR
# =====================
with st.sidebar:
    st.title("Vehicle Damage Detection")
    st.caption("Smart Transportation Project")

    st.divider()

    st.subheader("Model Settings")
    yolo_path = st.text_input("YOLO weights", value="weights/best.pt")
    severity_path = st.text_input("Severity weights", value="weights/severity.pth")
    confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

    st.divider()

    st.markdown(
        "**How to use**\n"
        "1. Upload a vehicle image\n"
        "2. Click **Detect Damage**\n"
        "3. Browse result tabs\n"
        "4. Download report"
    )

# =====================
#  MAIN AREA
# =====================
st.header("Upload & Analyze")

uploaded = st.file_uploader(
    "Drop a vehicle photo here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not uploaded:
    # Landing state — show capabilities
    st.info("Upload a vehicle image to detect damage, estimate severity, and generate PDF reports.")
    cols = st.columns(3)
    cols[0].metric("Detection", "3 classes")
    cols[1].metric("Severity", "0 - 100%")
    cols[2].metric("Export", "PDF + Image")
    st.stop()

# Read image
raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Preview uploaded image
with st.expander("Uploaded image preview", expanded=False):
    st.image(rgb, caption=uploaded.name, use_container_width=True)

# Detect button
if not st.button("Detect Damage", type="primary", use_container_width=True):
    st.stop()

# Load models
try:
    detector = load_detector(yolo_path, confidence)
    predictor = load_predictor(severity_path)
except FileNotFoundError as e:
    st.error(f"Model not found: {e}")
    st.stop()

# Run pipeline
with st.spinner("Analyzing image..."):
    detections = detector.detect(rgb)

if not detections:
    st.success("No damage detected — vehicle looks clean!")
    st.stop()

# Predict severity
severities = []
for det in detections:
    crop = crop_detection(rgb, det.bbox)
    severities.append(predictor.predict(crop))

annotated = draw_detections(rgb, detections, severities)
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Compute summary stats
valid = [s for s in severities if s >= 0]
avg = sum(valid) / len(valid) if valid else 0
mx = max(valid) if valid else 0

# =====================
#  RESULTS (Tabs)
# =====================
st.divider()
st.header("Results")

tab_overview, tab_details, tab_cost, tab_export = st.tabs(
    ["Overview", "Details", "Cost Estimate", "Export"]
)

# -- Tab: Overview --
with tab_overview:
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Damages Found", len(detections))
    m2.metric("Avg Severity", f"{avg:.1f}%")
    m3.metric("Max Severity", f"{mx:.1f}%")
    m4.metric("Assessment", severity_badge(avg))

    st.markdown("")  # spacer

    # Image comparison
    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.image(rgb, caption="Original", use_container_width=True)
    with col_annot:
        st.image(annotated_rgb, caption="Detected Damage", use_container_width=True)

# -- Tab: Details --
with tab_details:
    st.subheader(f"Detected Damages ({len(detections)})")

    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
        level = severity_badge(sev) if sev >= 0 else "N/A"
        color = severity_color(sev) if sev >= 0 else "#94a3b8"

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.markdown(f"**#{i} — {det.class_name.capitalize()}**")
            c2.markdown(f"Confidence: **{det.confidence:.0%}**")
            c3.markdown(f"Severity: **{sev_text}**")
            c4.markdown(
                f"<span style='color:{color}; font-weight:700'>{level}</span>",
                unsafe_allow_html=True,
            )

# -- Tab: Cost Estimate --
with tab_cost:
    st.subheader("Repair Cost Estimation")

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

    st.divider()

    tc1, tc2 = st.columns(2)
    tc1.metric("Total Min", f"${total_min:,}")
    tc2.metric("Total Max", f"${total_max:,}")

# -- Tab: Export --
with tab_export:
    st.subheader("Download Results")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Annotated Image**")
        st.download_button(
            "Download PNG",
            data=image_to_bytes(annotated),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

    with d2:
        st.markdown("**Inspection Report**")
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

        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
