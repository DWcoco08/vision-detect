"""Streamlit Web UI for Vehicle Damage Detection.

Interactive web interface for uploading images, detecting damage,
viewing results, and downloading annotated images + PDF reports.

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

# --- Page Config ---
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hero header */
    .hero-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Upload area styling */
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Severity badge colors */
    .badge-low { background: #d1fae5; color: #065f46; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-moderate { background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-high { background: #fed7aa; color: #9a3412; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-critical { background: #fecaca; color: #991b1b; padding: 4px 12px; border-radius: 20px; font-weight: 600; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }

    /* Detection result cards */
    .detection-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .detection-type {
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: capitalize;
    }
    .scratch-color { color: #ca8a04; }
    .dent-color { color: #ea580c; }
    .crack-color { color: #dc2626; }

    /* Cost total */
    .cost-total {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    /* Download buttons */
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
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


def severity_badge(severity: float) -> str:
    """Return HTML badge for severity level."""
    if severity < 0:
        return '<span class="badge-low">N/A</span>'
    if severity <= 25:
        cls, label = "badge-low", "Low"
    elif severity <= 50:
        cls, label = "badge-moderate", "Moderate"
    elif severity <= 75:
        cls, label = "badge-high", "High"
    else:
        cls, label = "badge-critical", "Critical"
    return f'<span class="{cls}">{severity:.1f}% — {label}</span>'


def type_color_class(damage_type: str) -> str:
    """Return CSS class for damage type color."""
    return f"{damage_type}-color"


# --- Main App ---
def main():
    # Hero Header
    st.markdown('<div class="hero-title">Vehicle Damage Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">AI-powered vehicle inspection — Upload, Detect, Report</div>', unsafe_allow_html=True)

    # --- Sidebar: Settings ---
    with st.sidebar:
        st.markdown("### ⚙️ Model Settings")
        yolo_path = st.text_input("YOLO Weights", value="weights/best.pt")
        severity_path = st.text_input("Severity Weights", value="weights/severity.pth")
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

        st.divider()
        st.markdown("### 📖 How to use")
        st.markdown("""
        1. **Upload** a vehicle image
        2. Click **Detect Damage**
        3. View results & download report
        """)
        st.divider()
        st.markdown(
            '<div style="text-align:center;color:#9ca3af;font-size:0.8rem;">'
            'Vehicle Damage Detection v1.0<br>Smart Transportation Project'
            '</div>',
            unsafe_allow_html=True,
        )

    # --- Upload Section ---
    uploaded_file = st.file_uploader(
        "📸 Drop your vehicle image here",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        # Empty state
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value">🔍</div>'
                '<div class="metric-label">Detect Damage</div>'
                '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.5rem;">'
                'Scratch, Dent, Crack</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value">📊</div>'
                '<div class="metric-label">Estimate Severity</div>'
                '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.5rem;">'
                '0–100% per region</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value">📄</div>'
                '<div class="metric-label">Export Report</div>'
                '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.5rem;">'
                'PDF with cost estimation</div></div>',
                unsafe_allow_html=True,
            )
        return

    # Read uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Show uploaded image (smaller)
    st.image(image_rgb, caption=f"📸 {uploaded_file.name}", use_container_width=True)

    # --- Detect Button ---
    if not st.button("🔍 Detect Damage", type="primary", use_container_width=True):
        return

    # Load models
    try:
        detector = load_detector(yolo_path, confidence)
        predictor = load_predictor(severity_path)
    except FileNotFoundError as e:
        st.error(f"❌ Model not found: {e}")
        return

    # Run detection
    with st.spinner("🔄 Analyzing damage..."):
        detections = detector.detect(image_rgb)

    if not detections:
        st.success("✅ No damage detected — vehicle looks clean!")
        return

    # Predict severity
    severities = []
    for det in detections:
        crop = crop_detection(image_rgb, det.bbox)
        severity = predictor.predict(crop)
        severities.append(severity)

    # Draw annotations
    annotated = draw_detections(image_rgb, detections, severities)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # --- Summary Metrics ---
    valid_sevs = [s for s in severities if s >= 0]
    avg_sev = sum(valid_sevs) / len(valid_sevs) if valid_sevs else 0
    max_sev = max(valid_sevs) if valid_sevs else 0

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(detections)}</div>'
            f'<div class="metric-label">Damages Found</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{avg_sev:.1f}%</div>'
            f'<div class="metric-label">Avg Severity</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{max_sev:.1f}%</div>'
            f'<div class="metric-label">Max Severity</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        types_found = len(set(d.class_name for d in detections))
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{types_found}</div>'
            f'<div class="metric-label">Damage Types</div></div>',
            unsafe_allow_html=True,
        )

    # --- Side-by-side Images ---
    st.markdown('<div class="section-header">📸 Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated_rgb, caption="Detected Damage", use_container_width=True)

    # --- Detection Details ---
    st.markdown('<div class="section-header">🔍 Detection Details</div>', unsafe_allow_html=True)

    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        color_cls = type_color_class(det.class_name)
        badge = severity_badge(sev)
        st.markdown(
            f'<div class="detection-card">'
            f'<span style="color:#9ca3af;font-weight:600;">#{i}</span>&nbsp;&nbsp;'
            f'<span class="detection-type {color_cls}">{det.class_name}</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'Confidence: <strong>{det.confidence:.1%}</strong>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'Severity: {badge}'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<span style="color:#9ca3af;">bbox: [{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Cost Estimation ---
    st.markdown('<div class="section-header">💰 Repair Cost Estimation</div>', unsafe_allow_html=True)

    total_min, total_max = 0, 0
    cost_cols = st.columns(len(detections))
    for idx, (det, sev) in enumerate(zip(detections, severities)):
        cost = REPAIR_COST.get(det.class_name, {"min": 100, "max": 400})
        factor = max(sev, 10) / 100
        min_c = int(cost["min"] * factor)
        max_c = int(cost["max"] * factor)
        total_min += min_c
        total_max += max_c

        with cost_cols[idx]:
            color_cls = type_color_class(det.class_name)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="detection-type {color_cls}" style="font-size:0.95rem;">{det.class_name}</div>'
                f'<div class="metric-value" style="font-size:1.3rem;">${min_c} - ${max_c}</div>'
                f'<div class="metric-label">Sev: {sev:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="cost-total">Total Estimated Cost: ${total_min} — ${total_max}</div>',
        unsafe_allow_html=True,
    )

    # --- Downloads ---
    st.markdown('<div class="section-header">📥 Downloads</div>', unsafe_allow_html=True)
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

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#9ca3af;font-size:0.8rem;">'
        'Vehicle Damage Detection System — Smart Transportation Project'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
