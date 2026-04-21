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

# ── Page config ──
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS (only what theme can't handle) ──
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 1000px; padding-top: 1rem;}

/* Sidebar text readable on dark bg */
[data-testid="stSidebar"] {color: #e2e8f0;}
[data-testid="stSidebar"] label {color: #cbd5e1 !important;}
[data-testid="stSidebar"] .stSlider label {color: #cbd5e1 !important;}

/* Metric cards */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
}

/* Download buttons consistent */
.stDownloadButton button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ──
@st.cache_resource
def load_detector(model_path: str, confidence: float) -> DamageDetector:
    return DamageDetector(model_path, confidence=confidence)


@st.cache_resource
def load_predictor(model_path: str) -> SeverityPredictor:
    return SeverityPredictor(model_path)


def image_to_bytes(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def severity_level(sev: float) -> str:
    if sev <= 25:
        return "🟢 Low"
    if sev <= 50:
        return "🟡 Moderate"
    if sev <= 75:
        return "🟠 High"
    return "🔴 Critical"


# ══════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════
with st.sidebar:
    st.title("🚗 Damage Detection")
    st.caption("Smart Transportation Project")

    st.divider()

    st.subheader("⚙️ Model Settings")
    yolo_path = st.text_input("YOLO weights", value="weights/best.pt")
    severity_path = st.text_input("Severity weights", value="weights/severity.pth")
    confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

    st.divider()

    st.subheader("📖 How to use")
    st.markdown("""
1. Upload a vehicle image
2. Click **Detect Damage**
3. View results
4. Download report
""")


# ══════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════
st.title("Vehicle Damage Detection")
st.caption("Upload a vehicle photo to detect damage, estimate severity, and generate reports.")

# ── Upload ──
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔍 Detection", "3 classes")
    c2.metric("📊 Severity", "0 – 100%")
    c3.metric("📄 Export", "PDF + Image")
    st.stop()

# Read image
raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

st.image(rgb, caption=uploaded.name, use_container_width=True)

# ── Detect button ──
if not st.button("🔍 Detect Damage", type="primary", use_container_width=True):
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
    st.info("✅ No damage detected — vehicle looks clean!")
    st.stop()

# Predict severity
severities = []
for det in detections:
    crop = crop_detection(rgb, det.bbox)
    severities.append(predictor.predict(crop))

annotated = draw_detections(rgb, detections, severities)
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ── Results ──
st.divider()

# Metrics row
valid = [s for s in severities if s >= 0]
avg = sum(valid) / len(valid) if valid else 0
mx = max(valid) if valid else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Damages", len(detections))
m2.metric("Avg Severity", f"{avg:.1f}%")
m3.metric("Max Severity", f"{mx:.1f}%")
m4.metric("Assessment", severity_level(avg).split(" ")[1])

# Image comparison
st.subheader("📸 Comparison")
col1, col2 = st.columns(2)
col1.image(rgb, caption="Original", use_container_width=True)
col2.image(annotated_rgb, caption="Detected", use_container_width=True)

# Detection table
st.subheader("🔍 Detection Details")
table_data = []
for i, (det, sev) in enumerate(zip(detections, severities), 1):
    sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
    table_data.append({
        "#": i,
        "Type": det.class_name.capitalize(),
        "Confidence": f"{det.confidence:.0%}",
        "Severity": sev_text,
        "Level": severity_level(sev) if sev >= 0 else "—",
    })
st.dataframe(table_data, use_container_width=True, hide_index=True)

# Cost estimation
st.subheader("💰 Repair Cost Estimation")
total_min, total_max = 0, 0
cost_data = []
for det, sev in zip(detections, severities):
    cost = REPAIR_COST.get(det.class_name, {"min": 100, "max": 400})
    factor = max(sev, 10) / 100
    lo = int(cost["min"] * factor)
    hi = int(cost["max"] * factor)
    total_min += lo
    total_max += hi
    cost_data.append({
        "Type": det.class_name.capitalize(),
        "Severity": f"{sev:.1f}%",
        "Min Cost ($)": lo,
        "Max Cost ($)": hi,
    })

st.dataframe(cost_data, use_container_width=True, hide_index=True)

tc1, tc2 = st.columns(2)
tc1.metric("Min Total", f"${total_min}")
tc2.metric("Max Total", f"${total_max}")

# Downloads
st.subheader("📥 Download")
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
