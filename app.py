"""Streamlit Web UI for Vehicle Damage Detection.

Dark dashboard theme with custom CSS components.

Usage:
    streamlit run app.py
"""

import tempfile
from html import escape as html_escape
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


# -- CSS injection --
def inject_css():
    """Inject full custom CSS for dark dashboard theme."""
    st.markdown("""
    <style>
    /* ── Reset & base ── */
    #MainMenu, footer {visibility: hidden;}
    .block-container {max-width: 1100px; padding-top: 1.5rem;}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 60%, #1a1f2b 100%) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] span {
        color: #b0bec5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        color: #e8f0ff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ── Metric cards (glassmorphism) ── */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(0, 217, 255, 0.2);
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e8f0ff;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #8b95a5;
        font-weight: 500;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b95a5;
        font-weight: 500;
        padding: 12px 24px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #00d9ff !important;
        border-bottom: 2px solid #00d9ff !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c8d0dc;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #00d9ff !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.02);
        border: 2px dashed rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 8px;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,217,255,0.3);
        background: rgba(255,255,255,0.04);
    }

    /* ── Download buttons ── */
    .stDownloadButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #00d9ff 0%, #0097d9 100%) !important;
        color: #ffffff !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stDownloadButton button:hover {
        box-shadow: 0 4px 16px rgba(0, 217, 255, 0.3);
        transform: translateY(-1px);
    }

    /* ── Primary button ── */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #00d9ff 0%, #0097d9 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 4px 16px rgba(0, 217, 255, 0.3);
    }

    /* ── Containers with border ── */
    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stVerticalBlock"]) > [data-testid="stVerticalBlock"] {
        border-color: rgba(255,255,255,0.06) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        color: #b0bec5 !important;
        font-weight: 500;
    }

    /* ── Image captions ── */
    [data-testid="stImage"] > div > div > p {
        color: #8b95a5 !important;
        font-size: 0.85rem;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #9fa8b8;
        font-size: 1rem;
        font-weight: 300;
        margin: 0;
    }

    /* ── Feature cards (landing) ── */
    .feature-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        min-height: 140px;
    }
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(0, 217, 255, 0.15);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    .feature-title {
        color: #e8f0ff;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .feature-desc {
        color: #8b95a5;
        font-size: 0.85rem;
    }

    /* ── Detection card ── */
    .detection-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .detection-card.sev-low {
        border-left-color: #48c774;
        background: rgba(72, 199, 116, 0.05);
    }
    .detection-card.sev-moderate {
        border-left-color: #ffdd57;
        background: rgba(255, 221, 87, 0.05);
    }
    .detection-card.sev-high {
        border-left-color: #ff9500;
        background: rgba(255, 149, 0, 0.05);
    }
    .detection-card.sev-critical {
        border-left-color: #ff3860;
        background: rgba(255, 56, 96, 0.05);
    }
    .detection-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .detection-type {
        color: #e8f0ff;
        font-size: 1rem;
        font-weight: 600;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .severity-badge.sev-low { background: #48c774; color: #fff; }
    .severity-badge.sev-moderate { background: #ffdd57; color: #333; }
    .severity-badge.sev-high { background: #ff9500; color: #fff; }
    .severity-badge.sev-critical { background: #ff3860; color: #fff; }
    .detection-stats {
        display: flex;
        gap: 2rem;
        color: #8b95a5;
        font-size: 0.9rem;
    }
    .detection-stats strong {
        color: #c8d0dc;
    }

    /* ── Export card ── */
    .export-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .export-icon { font-size: 2rem; margin-bottom: 0.75rem; }
    .export-title { color: #e8f0ff; font-weight: 600; margin-bottom: 0.25rem; }
    .export-desc { color: #8b95a5; font-size: 0.85rem; margin-bottom: 1rem; }

    /* ── Section label ── */
    .section-label {
        color: #8b95a5;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()


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


def severity_level(sev: float) -> str:
    """Return severity level key for CSS class."""
    if sev <= 25:
        return "low"
    if sev <= 50:
        return "moderate"
    if sev <= 75:
        return "high"
    return "critical"


def severity_label(sev: float) -> str:
    """Return human-readable severity label."""
    if sev <= 25:
        return "Low"
    if sev <= 50:
        return "Moderate"
    if sev <= 75:
        return "High"
    return "Critical"


# =====================
#  SIDEBAR
# =====================
with st.sidebar:
    st.title("🚗 Damage Detection")
    st.caption("Smart Transportation Project")
    st.divider()

    st.subheader("Model Settings")
    yolo_path = st.text_input("YOLO weights", value="weights/best.pt")
    severity_path = st.text_input("Severity weights", value="weights/severity.pth")
    confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

    st.divider()

    st.markdown(
        "**How to use**\n\n"
        "1. Upload a vehicle image\n"
        "2. Click **Detect Damage**\n"
        "3. Browse result tabs\n"
        "4. Download report"
    )

    st.divider()
    st.caption("Vehicle Damage Detection v1.0")


# =====================
#  HERO BANNER
# =====================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Vehicle Damage Detection</div>
    <div class="hero-subtitle">
        AI-powered damage assessment — upload a photo, detect damage, estimate severity and repair costs
    </div>
</div>
""", unsafe_allow_html=True)


# =====================
#  UPLOAD
# =====================
uploaded = st.file_uploader(
    "Drop a vehicle photo here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not uploaded:
    # Landing — feature cards
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Damage Detection</div>
            <div class="feature-desc">YOLOv8 detects scratch, dent, crack with confidence scores</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Severity Estimation</div>
            <div class="feature-desc">ResNet18 CNN rates severity 0–100% per damage region</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">PDF Reports</div>
            <div class="feature-desc">Professional inspection reports with cost estimation</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# Read image
raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Preview uploaded image
with st.expander(f"📎 {uploaded.name}", expanded=False):
    st.image(rgb, use_container_width=True)

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
#  RESULTS
# =====================
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

tab_overview, tab_details, tab_cost, tab_export = st.tabs(
    ["Overview", "Details", "Cost Estimate", "Export"]
)

# -- Tab: Overview --
with tab_overview:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Damages Found", len(detections))
    m2.metric("Avg Severity", f"{avg:.1f}%")
    m3.metric("Max Severity", f"{mx:.1f}%")
    m4.metric("Assessment", severity_label(avg))

    st.markdown("")

    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.image(rgb, caption="Original", use_container_width=True)
    with col_annot:
        st.image(annotated_rgb, caption="Detected Damage", use_container_width=True)

# -- Tab: Details --
with tab_details:
    st.markdown(
        f'<div class="section-label">Detected Damages ({len(detections)})</div>',
        unsafe_allow_html=True,
    )

    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        sev_text = f"{sev:.1f}%" if sev >= 0 else "N/A"
        level = severity_level(sev) if sev >= 0 else "low"
        label = severity_label(sev) if sev >= 0 else "N/A"

        st.markdown(f"""
        <div class="detection-card sev-{level}">
            <div class="detection-header">
                <span class="detection-type">#{i} — {html_escape(det.class_name.capitalize())}</span>
                <span class="severity-badge sev-{level}">{label}</span>
            </div>
            <div class="detection-stats">
                <span>Confidence: <strong>{det.confidence:.0%}</strong></span>
                <span>Severity: <strong>{sev_text}</strong></span>
                <span>Region: <strong>[{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -- Tab: Cost Estimate --
with tab_cost:
    st.markdown(
        '<div class="section-label">Repair Cost Estimation</div>',
        unsafe_allow_html=True,
    )

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

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    tc1.metric("Total Min", f"${total_min:,}")
    tc2.metric("Total Max", f"${total_max:,}")

# -- Tab: Export --
with tab_export:
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("""
        <div class="export-card">
            <div class="export-icon">📷</div>
            <div class="export-title">Annotated Image</div>
            <div class="export-desc">Download image with damage annotations overlay</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "Download PNG",
            data=image_to_bytes(annotated),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

    with d2:
        st.markdown("""
        <div class="export-card">
            <div class="export-icon">📄</div>
            <div class="export-title">Inspection Report</div>
            <div class="export-desc">Full PDF report with detections, costs, and summary</div>
        </div>
        """, unsafe_allow_html=True)
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
            Path(tmp.name).unlink(missing_ok=True)

        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
