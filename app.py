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
)

# --- Theme CSS ---
st.markdown("""
<style>
    /* Streamlit overrides */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 0;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ── HEADER BANNER (dark) ── */
    .banner {
        background: #0f172a;
        margin: -1rem -5rem 1.5rem -5rem;
        padding: 2.5rem 2rem 2rem 2rem;
        text-align: center;
    }
    .banner h1 {
        color: #f8fafc;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .banner p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }

    /* ── CARDS ── */
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .card-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    .card-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* ── SEVERITY PILLS ── */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pill-low { background: #dcfce7; color: #166534; }
    .pill-mod { background: #fef9c3; color: #854d0e; }
    .pill-high { background: #fed7aa; color: #9a3412; }
    .pill-crit { background: #fecaca; color: #991b1b; }

    /* ── DETECTION ROW ── */
    .det-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .det-num {
        font-weight: 800;
        color: #94a3b8;
        font-size: 0.9rem;
        min-width: 28px;
    }
    .det-type {
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: capitalize;
        min-width: 70px;
    }
    .det-info {
        color: #334155;
        font-size: 0.85rem;
    }
    .det-bbox {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-left: auto;
        font-family: monospace;
    }

    /* ── TYPE COLORS (high contrast) ── */
    .t-scratch { color: #a16207; }
    .t-dent { color: #c2410c; }
    .t-crack { color: #dc2626; }

    /* ── COST BAR ── */
    .cost-bar {
        background: #0f172a;
        color: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0.8rem;
        letter-spacing: -0.01em;
    }

    /* ── SECTION LABEL ── */
    .sec {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 2rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* ── FOOTER ── */
    .foot {
        text-align: center;
        color: #94a3b8;
        font-size: 0.75rem;
        padding: 2rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Cached Model Loading ---
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


def severity_pill(sev: float) -> str:
    """Return HTML pill for severity value."""
    if sev < 0:
        return '<span class="pill pill-low">N/A</span>'
    if sev <= 25:
        c = "pill-low"
    elif sev <= 50:
        c = "pill-mod"
    elif sev <= 75:
        c = "pill-high"
    else:
        c = "pill-crit"
    return f'<span class="pill {c}">{sev:.1f}%</span>'


def type_class(name: str) -> str:
    """CSS class for damage type."""
    return f"t-{name}"


# ========== MAIN ==========
def main():
    # --- Header Banner (dark) ---
    st.markdown(
        '<div class="banner">'
        '<h1>🚗 Vehicle Damage Detection</h1>'
        '<p>Upload a photo · AI detects damage · Estimate repair cost</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Settings (inline expander, not sidebar) ---
    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            yolo_path = st.text_input("YOLO weights", value="weights/best.pt")
        with c2:
            severity_path = st.text_input("Severity weights", value="weights/severity.pth")
        with c3:
            confidence = st.slider("Confidence", 0.1, 1.0, 0.25, 0.05)

    # --- Upload ---
    uploaded_file = st.file_uploader(
        "Drop vehicle image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        # Landing state — feature cards
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div class="card"><div class="card-value">🔍</div>'
                '<div class="card-label">Detect</div>'
                '<div class="card-sub">Scratch · Dent · Crack</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="card"><div class="card-value">📊</div>'
                '<div class="card-label">Severity</div>'
                '<div class="card-sub">0 – 100% per region</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="card"><div class="card-value">📄</div>'
                '<div class="card-label">Report</div>'
                '<div class="card-sub">PDF + cost estimation</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="foot">Smart Transportation Project</div>', unsafe_allow_html=True)
        return

    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preview uploaded image
    st.image(image_rgb, use_container_width=True)

    # Detect button
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
    with st.spinner("Analyzing..."):
        detections = detector.detect(image_rgb)

    if not detections:
        st.info("No damage detected — vehicle looks clean.")
        return

    # Severity prediction
    severities = []
    for det in detections:
        crop = crop_detection(image_rgb, det.bbox)
        severities.append(predictor.predict(crop))

    annotated = draw_detections(image_rgb, detections, severities)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ── METRICS ──
    valid_sevs = [s for s in severities if s >= 0]
    avg_sev = sum(valid_sevs) / len(valid_sevs) if valid_sevs else 0
    max_sev = max(valid_sevs) if valid_sevs else 0

    st.markdown('<div class="sec">Overview</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="card"><div class="card-value">{len(detections)}</div>'
            f'<div class="card-label">Damages</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="card"><div class="card-value">{avg_sev:.1f}%</div>'
            f'<div class="card-label">Avg Severity</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="card"><div class="card-value">{max_sev:.1f}%</div>'
            f'<div class="card-label">Max Severity</div></div>',
            unsafe_allow_html=True,
        )

    # ── IMAGES ──
    st.markdown('<div class="sec">Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated_rgb, caption="Detected", use_container_width=True)

    # ── DETECTIONS ──
    st.markdown('<div class="sec">Detections</div>', unsafe_allow_html=True)
    for i, (det, sev) in enumerate(zip(detections, severities), 1):
        tc = type_class(det.class_name)
        pill = severity_pill(sev)
        st.markdown(
            f'<div class="det-row">'
            f'<span class="det-num">#{i}</span>'
            f'<span class="det-type {tc}">{det.class_name}</span>'
            f'<span class="det-info">Conf: <b>{det.confidence:.0%}</b></span>'
            f'<span class="det-info">Severity: {pill}</span>'
            f'<span class="det-bbox">[{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── COST ──
    st.markdown('<div class="sec">Repair Cost</div>', unsafe_allow_html=True)
    total_min, total_max = 0, 0
    cost_cols = st.columns(min(len(detections), 4))
    for idx, (det, sev) in enumerate(zip(detections, severities)):
        cost = REPAIR_COST.get(det.class_name, {"min": 100, "max": 400})
        factor = max(sev, 10) / 100
        min_c = int(cost["min"] * factor)
        max_c = int(cost["max"] * factor)
        total_min += min_c
        total_max += max_c
        col_idx = idx % min(len(detections), 4)
        with cost_cols[col_idx]:
            tc = type_class(det.class_name)
            st.markdown(
                f'<div class="card">'
                f'<div class="det-type {tc}" style="font-size:0.85rem;">{det.class_name}</div>'
                f'<div class="card-value" style="font-size:1.2rem;">${min_c}–${max_c}</div>'
                f'<div class="card-sub">{sev:.1f}% severity</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="cost-bar">Total: ${total_min} – ${total_max}</div>',
        unsafe_allow_html=True,
    )

    # ── DOWNLOADS ──
    st.markdown('<div class="sec">Download</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📷  Annotated Image",
            data=image_to_bytes(annotated),
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_annotated.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
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
            "📄  PDF Report",
            data=pdf_bytes,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown('<div class="foot">Vehicle Damage Detection — Smart Transportation Project</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
