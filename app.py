"""
Helmet Detector — Streamlit dashboard
--------------------------------------
A dark, card-based dashboard UI for testing a YOLO helmet-detection model.

Run with:
    streamlit run app.py

Model classes:
    0: With Helmet
    1: Without Helmet
"""

import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Helmet Detector", page_icon="⛑️", layout="wide")

CLASS_NAMES = ["With Helmet", "Without Helmet"]
SHORT_LABEL = {"With Helmet": "Helmet", "Without Helmet": "No Helmet"}
GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#3b5bfd"
CLASS_COLOR_BGR = {"With Helmet": (86, 197, 34), "Without Helmet": (68, 68, 239)}

DEFAULT_MODEL_PATH = r"C:\Users\Nithin T\Desktop\CV_Projects\helmet detection\best__1_.pt"

# --------------------------------------------------------------------------
# Theme / CSS
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
:root {{
    --bg-card: #0d1424;
    --border: #1e2740;
    --text-primary: #e6e9f0;
    --text-secondary: #8a93a8;
    --green: {GREEN};
    --red: {RED};
    --blue: {BLUE};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at top left, #0d1428 0%, #070a13 60%);
    color: var(--text-primary);
}}

section[data-testid="stSidebar"] {{
    background: #080b15;
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1rem; }}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* ---------- Sidebar brand ---------- */
.brand-row {{
    display: flex; align-items: center; gap: 12px;
    padding: 4px 6px 18px 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 10px;
}}
.brand-icon {{
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; box-shadow: 0 4px 14px rgba(239,68,68,0.35);
}}
.brand-title {{ font-weight: 700; font-size: 1.05rem; line-height:1.1; color:#fff; }}
.brand-sub {{ font-size: 0.72rem; color: var(--text-secondary); }}

/* ---------- Nav buttons ---------- */
div[data-testid="stSidebar"] .stButton {{ margin-bottom: 3px; }}
div[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--text-secondary) !important;
    font-weight: 500;
    padding: 10px 14px;
    border-radius: 10px;
}}
div[data-testid="stSidebar"] .stButton > button:hover {{
    background: #131b30 !important;
    color: #fff !important;
}}
div[data-testid="stSidebar"] .stButton > button:focus {{ box-shadow: none !important; }}
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #3b5bfd, #2541c4) !important;
    color: #fff !important;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(59,91,253,0.35) !important;
}}
div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #3b5bfd, #2541c4) !important;
}}

.section-title {{
    display:flex; align-items:center; gap:8px;
    font-weight: 700; color:#fff; font-size: 0.95rem;
    margin: 18px 0 10px 4px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
}}
.field-label {{
    color: var(--text-secondary); font-size: 0.8rem; font-weight: 600;
    margin: 16px 0 8px 4px;
}}
.slider-row {{
    display:flex; justify-content:space-between; align-items:center;
    margin: 2px 4px -8px 4px;
}}
.slider-row .lbl {{ color: var(--text-secondary); font-size: 0.83rem; font-weight: 600; }}
.slider-row .val {{ color: #fff; font-size: 0.83rem; font-weight: 700; }}

/* ---------- Cards ---------- */
.card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 16px;
}}
.card-title {{
    font-weight: 700; font-size: 0.95rem; color: #fff;
    display:flex; align-items:center; gap:8px;
    margin-bottom: 6px;
}}
.badge {{ padding: 5px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }}
.badge-green {{ background: rgba(34,197,94,0.15); color: var(--green); border:1px solid rgba(34,197,94,0.35); }}
.badge-red   {{ background: rgba(239,68,68,0.15); color: var(--red); border:1px solid rgba(239,68,68,0.35); }}
.badge-blue  {{ background: rgba(59,91,253,0.15); color: var(--blue); border:1px solid rgba(59,91,253,0.35); }}
.badge-gray  {{ background: rgba(148,163,184,0.12); color: var(--text-secondary); border:1px solid var(--border); }}

/* ---------- Metric tiles ---------- */
.metric-tile {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 16px; padding: 16px 18px;
    display: flex; gap: 14px; align-items: flex-start;
}}
.metric-icon {{
    width: 42px; height: 42px; border-radius: 50%;
    display:flex; align-items:center; justify-content:center; font-size: 18px; flex-shrink: 0;
}}
.metric-label {{ color: var(--text-secondary); font-size: 0.8rem; font-weight: 600; }}
.metric-value {{ font-size: 1.8rem; font-weight: 800; color: #fff; line-height:1.2; }}
.metric-sub {{ font-size: 0.72rem; margin-top:2px; }}

/* ---------- Detection table ---------- */
.det-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
.det-table th {{
    text-align: left; color: var(--text-secondary); font-weight:600;
    padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.04em;
}}
.det-table td {{ padding: 9px 10px; border-bottom: 1px solid #131b30; color: var(--text-primary); }}
.det-table tr:hover td {{ background: #0f1730; }}

/* ---------- Top bar ---------- */
.topbar {{ display:flex; align-items:center; justify-content:flex-end; gap:12px; padding: 4px 0 20px 0; }}
.pill {{
    display:flex; align-items:center; gap:8px;
    background: var(--bg-card); border:1px solid var(--border);
    padding: 6px 14px; border-radius:999px; font-size:0.8rem; color: var(--text-secondary);
}}
.dot-green {{ width:7px; height:7px; border-radius:50%; background: var(--green); box-shadow: 0 0 6px var(--green); }}
.avatar {{
    width:30px; height:30px; border-radius:50%;
    background: linear-gradient(135deg,#3b82f6,#1e40af);
    display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:0.72rem; color:#fff; flex-shrink:0;
}}

/* ---------- Page title ---------- */
.title-row {{ display:flex; align-items:center; gap:14px; margin-bottom: 4px; }}
.title-icon {{
    width:46px; height:46px; border-radius:12px;
    background: linear-gradient(135deg,#1d2a4d,#131c36);
    border: 1px solid #23304f;
    display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0;
}}
.page-title {{ font-size: 1.9rem; font-weight: 800; color:#fff; line-height:1.15; }}
.page-sub {{ color: var(--text-secondary); margin: 2px 0 20px 60px; }}
.section-heading {{ font-size: 1.3rem; font-weight: 800; color:#fff; margin: 4px 0 14px 2px; }}

/* ---------- Sliders (blue accent) ---------- */
div[data-testid="stSlider"] [role="slider"] {{
    background-color: var(--blue) !important;
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 4px rgba(59,91,253,0.22) !important;
}}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
    background: var(--blue) !important;
}}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {{
    background: #1e2740 !important;
}}
div[data-testid="stTickBar"] {{ display:none; }}

/* Radio (input source) */
div[role="radiogroup"] label {{ color: var(--text-primary) !important; font-size:0.88rem; }}

/* Download button */
.stDownloadButton > button {{
    background: #101a30 !important; border: 1px solid var(--border) !important;
    color: #fff !important; border-radius: 9px !important;
    font-size: 0.78rem !important; font-weight: 600 !important; padding: 7px 14px !important;
}}
.stDownloadButton > button:hover {{ border-color: var(--blue) !important; color: var(--blue) !important; }}

/* Footer card */
.footer-card {{
    background: linear-gradient(160deg, #101a30, #0b1122);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 16px; margin-top: 24px; text-align:center;
}}
.footer-card .emoji {{ font-size: 26px; margin-bottom: 6px; }}
.footer-card .txt {{ font-size: 0.8rem; color: var(--text-secondary); font-weight:600; line-height:1.3; }}

.legend-row {{ display:flex; gap:16px; justify-content:flex-end; align-items:center; }}
.legend-item {{ display:flex; align-items:center; gap:6px; color: var(--text-secondary); font-size:0.78rem; }}
.legend-dot {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}

hr {{ border-color: var(--border); }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("history", [])
st.session_state.setdefault("prev_total", 0)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    return YOLO(model_path)


def annotate(image_bgr: np.ndarray, result, conf_thres: float):
    rows = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_thres:
                continue
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            color = CLASS_COLOR_BGR.get(label, (255, 255, 0))
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            text = f"{SHORT_LABEL.get(label, label)} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(image_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(image_bgr, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            rows.append({"class": label, "confidence": round(conf, 3),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    df = pd.DataFrame(rows, columns=["class", "confidence", "x1", "y1", "x2", "y2"])
    return image_bgr, df


def run_on_image(model, pil_image: Image.Image, conf_thres, iou_thres):
    image_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    results = model.predict(image_bgr, conf=conf_thres, iou=iou_thres, verbose=False)
    annotated_bgr, df = annotate(image_bgr.copy(), results[0], conf_thres)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), df


def log_history(source, label, df):
    with_h = int((df["class"] == "With Helmet").sum()) if not df.empty else 0
    without_h = int((df["class"] == "Without Helmet").sum()) if not df.empty else 0
    avg_conf = round(float(df["confidence"].mean()), 3) if not df.empty else 0.0
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source, "file": label, "total": int(len(df)),
        "with_helmet": with_h, "without_helmet": without_h, "avg_confidence": avg_conf,
    })


def metric_tile(col, icon, icon_bg, icon_fg, label, value, sub, sub_color):
    col.markdown(f"""
    <div class="metric-tile">
        <div class="metric-icon" style="background:{icon_bg}; color:{icon_fg};">{icon}</div>
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub" style="color:{sub_color};">{sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def detection_overview_chart(total, with_h, without_h):
    fig = go.Figure()
    fig.add_bar(x=["Total", "With Helmet", "Without Helmet"], y=[total, with_h, without_h],
                marker_color=[GREEN, GREEN, RED],
                text=[total, with_h, without_h], textposition="outside",
                textfont=dict(color="#e6e9f0", size=14), width=0.5)
    fig.update_layout(
        height=250, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a93a8"),
        yaxis=dict(gridcolor="#1e2740", zerolinecolor="#1e2740"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def detection_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p style='color:#8a93a8;'>No detections above the confidence threshold.</p>"
    rows_html = ""
    for i, r in df.reset_index(drop=True).iterrows():
        badge_cls = "badge-green" if r["class"] == "With Helmet" else "badge-red"
        rows_html += f"""
        <tr>
            <td>{i+1}</td>
            <td><span class="badge {badge_cls}">{r['class']}</span></td>
            <td>{r['confidence']:.3f}</td>
            <td>{r['x1']}</td><td>{r['y1']}</td><td>{r['x2']}</td><td>{r['y2']}</td>
        </tr>"""
    return f"""
    <table class="det-table">
        <thead><tr><th>#</th><th>Class</th><th>Confidence</th><th>X1</th><th>Y1</th><th>X2</th><th>Y2</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>"""


def slider_with_label(label, key, default):
    st.session_state.setdefault(key, default)
    st.markdown(f"""
    <div class="slider-row"><span class="lbl">{label}</span><span class="val">{st.session_state[key]:.2f}</span></div>
    """, unsafe_allow_html=True)
    return st.slider(label, 0.0, 1.0, step=0.05, key=key, label_visibility="collapsed")


def render_result_panels(pil_original, annotated_rgb, df):
    avg_conf = df["confidence"].mean() if not df.empty else 0.0
    total = len(df)
    with_h = int((df["class"] == "With Helmet").sum()) if total else 0
    without_h = int((df["class"] == "Without Helmet").sum()) if total else 0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🖼️ Original Image</div>', unsafe_allow_html=True)
        st.image(pil_original, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        hc1, hc2 = st.columns([2, 1.4])
        with hc1:
            st.markdown('<div class="card-title">🎯 Detection Result</div>', unsafe_allow_html=True)
        with hc2:
            badge = (f'<div style="text-align:right;"><span class="badge badge-green">✅ Helmet Detection: {avg_conf:.2f}</span></div>'
                     if total else '<div style="text-align:right;"><span class="badge badge-gray">No detections</span></div>')
            st.markdown(badge, unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    prev_total = st.session_state.get("prev_total", 0)
    delta = total - prev_total
    arrow = "↑" if delta >= 0 else "↓"
    st.session_state["prev_total"] = total

    if avg_conf >= 0.7:
        conf_sub, conf_arrow = "High accuracy", "↑"
    elif avg_conf >= 0.4:
        conf_sub, conf_arrow = "Moderate accuracy", "→"
    else:
        conf_sub, conf_arrow = "Low accuracy", "↓"

    m1, m2, m3, m4 = st.columns(4)
    metric_tile(m1, "🛡️", "#0f3d2c", GREEN, "Total Detections", total,
                f"{arrow} {abs(delta)} from last run", "#8a93a8")
    metric_tile(m2, "⛑️", "#0f3d2c", GREEN, "With Helmet", with_h,
                f"{(with_h/total*100 if total else 0):.0f}% of total", GREEN)
    metric_tile(m3, "🚫", "#3f1620", RED, "Without Helmet", without_h,
                f"{(without_h/total*100 if total else 0):.0f}% of total", RED)
    metric_tile(m4, "🎯", "#12233f", BLUE, "Average Confidence", f"{avg_conf:.2f}",
                f"{conf_arrow} {conf_sub}", BLUE)

    st.markdown('<div class="section-heading" style="margin-top:22px;">Results</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1.3])
    with r1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        hc1, hc2 = st.columns([1.3, 1.4])
        with hc1:
            st.markdown('<div class="card-title">📈 Detection Overview</div>', unsafe_allow_html=True)
        with hc2:
            st.markdown(f"""
            <div class="legend-row">
                <span class="legend-item"><span class="legend-dot" style="background:{GREEN};"></span>With Helmet</span>
                <span class="legend-item"><span class="legend-dot" style="background:{RED};"></span>Without Helmet</span>
            </div>""", unsafe_allow_html=True)
        st.plotly_chart(detection_overview_chart(total, with_h, without_h),
                         use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        hc1, hc2 = st.columns([2.2, 1])
        with hc1:
            st.markdown('<div class="card-title">🪖 Detection Details</div>', unsafe_allow_html=True)
        with hc2:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV", csv, "detections.csv", "text/csv", use_container_width=True)
        st.markdown(detection_table_html(df), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-row">
        <div class="brand-icon">⛑️</div>
        <div>
            <div class="brand-title">Helmet Detector</div>
            <div class="brand-sub">Safer People &bull; Smarter Sites</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [("🏠", "Home"), ("⬆️", "Upload"), ("🕑", "History"), ("⚙️", "Settings")]
    for icon, label in nav_items:
        is_active = st.session_state.page == label
        if st.button(f"{icon}   {label}", key=f"nav_{label}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = label
            st.rerun()

    st.markdown('<div class="section-title">⇄ Model Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="field-label">Model weights (.pt)</div>', unsafe_allow_html=True)
    model_file = st.file_uploader("Model weights", type=["pt"], label_visibility="collapsed")

    conf_thres = slider_with_label("Confidence threshold", "conf_thres_val", 0.25)
    iou_thres = slider_with_label("IoU threshold (NMS)", "iou_thres_val", 0.45)

    st.markdown('<div class="field-label" style="margin-top:20px;">Input source</div>', unsafe_allow_html=True)
    input_source = st.radio(
        "Input source",
        ["Image", "Upload Video", "Webcam Snapshot", "Live Webcam"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="footer-card">
        <div class="emoji">🛡️</div>
        <div class="txt">Better Safety<br>Through AI</div>
    </div>
    """, unsafe_allow_html=True)

# Resolve model path
if model_file is not None:
    tmp_model = Path(tempfile.gettempdir()) / model_file.name
    tmp_model.write_bytes(model_file.getbuffer())
    model_path_to_load = str(tmp_model)
elif Path(DEFAULT_MODEL_PATH).exists():
    model_path_to_load = DEFAULT_MODEL_PATH
else:
    model_path_to_load = None

# --------------------------------------------------------------------------
# Top bar (decorative) + page title
# --------------------------------------------------------------------------
st.markdown("""
<div class="topbar">
    <div class="pill">☀️ &nbsp;/&nbsp; 🌙</div>
    <div class="pill"><span class="dot-green"></span> Connected</div>
    <div class="pill" style="padding:5px 12px 5px 6px;">
        <div class="avatar">AI</div>
        <span style="color:#fff;font-weight:600;">Site Admin</span>
        <span style="color:#8a93a8;">⌄</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-row">
    <div class="title-icon">⛑️</div>
    <div class="page-title">Helmet Detection</div>
</div>
<div class="page-sub">Detect and monitor safety helmets using advanced AI models</div>
""", unsafe_allow_html=True)

if model_path_to_load is None:
    st.warning(
        f"No model found at `{DEFAULT_MODEL_PATH}` and none uploaded. "
        "Upload your `.pt` weights file in the sidebar to get started."
    )
    st.stop()

model = load_model(model_path_to_load)

# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
page = st.session_state.page

if page in ("Home", "Upload"):

    if input_source == "Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_up")
        if uploaded is not None:
            pil_image = Image.open(uploaded)
            with st.spinner("Running detection..."):
                annotated_rgb, df = run_on_image(model, pil_image, conf_thres, iou_thres)
            render_result_panels(pil_image, annotated_rgb, df)
            log_history("Image", uploaded.name, df)
        else:
            st.info("Upload an image to run detection.")

    elif input_source == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="vid_up")
        if uploaded_video is not None:
            tmp_video = Path(tempfile.gettempdir()) / uploaded_video.name
            tmp_video.write_bytes(uploaded_video.getbuffer())

            cap = cv2.VideoCapture(str(tmp_video))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            sample_every = max(1, int(fps // 5)) if fps else 1

            st.markdown('<div class="card"><div class="card-title">🎬 Live Video Detection</div></div>', unsafe_allow_html=True)
            frame_ph = st.empty()
            progress = st.progress(0)
            stop = st.button("Stop processing")

            frame_idx = 0
            all_dets = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop:
                    break
                frame_idx += 1
                if frame_idx % sample_every != 0:
                    continue
                results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)
                annotated_bgr, df = annotate(frame.copy(), results[0], conf_thres)
                frame_ph.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                if not df.empty:
                    all_dets.append(df)
                if total_frames:
                    progress.progress(min(frame_idx / total_frames, 1.0))
            cap.release()

            full_df = pd.concat(all_dets, ignore_index=True) if all_dets else pd.DataFrame(
                columns=["class", "confidence", "x1", "y1", "x2", "y2"])
            st.success(f"Processed {frame_idx} frames (sampled every {sample_every}).")

            total = len(full_df)
            with_h = int((full_df["class"] == "With Helmet").sum()) if total else 0
            without_h = int((full_df["class"] == "Without Helmet").sum()) if total else 0
            avg_conf = full_df["confidence"].mean() if total else 0.0

            m1, m2, m3, m4 = st.columns(4)
            metric_tile(m1, "🛡️", "#0f3d2c", GREEN, "Total Detections", total, "across sampled frames", "#8a93a8")
            metric_tile(m2, "⛑️", "#0f3d2c", GREEN, "With Helmet", with_h, f"{(with_h/total*100 if total else 0):.0f}% of total", GREEN)
            metric_tile(m3, "🚫", "#3f1620", RED, "Without Helmet", without_h, f"{(without_h/total*100 if total else 0):.0f}% of total", RED)
            metric_tile(m4, "🎯", "#12233f", BLUE, "Avg Confidence", f"{avg_conf:.2f}", "across frames", BLUE)

            log_history("Video", uploaded_video.name, full_df)
        else:
            st.info("Upload a video to run frame-by-frame detection.")

    elif input_source == "Webcam Snapshot":
        snapshot = st.camera_input("Take a photo")
        if snapshot is not None:
            pil_image = Image.open(snapshot)
            with st.spinner("Running detection..."):
                annotated_rgb, df = run_on_image(model, pil_image, conf_thres, iou_thres)
            render_result_panels(pil_image, annotated_rgb, df)
            log_history("Webcam Snapshot", "snapshot.jpg", df)
        else:
            st.info("Use your camera to capture a photo for detection.")

    elif input_source == "Live Webcam":
        st.markdown('<div class="card"><div class="card-title">📡 Live Footage Detection</div></div>', unsafe_allow_html=True)
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av

            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            class YOLOProcessor(VideoProcessorBase):
                def __init__(self):
                    self.model = model
                    self.conf = conf_thres
                    self.iou = iou_thres

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    results = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)
                    annotated_bgr, _ = annotate(img.copy(), results[0], self.conf)
                    return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

            webrtc_streamer(
                key="live-helmet-detection",
                video_processor_factory=YOLOProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.caption("Live detection runs entirely in your browser session via WebRTC. "
                       "Grant camera access when prompted.")
        except ImportError:
            st.error(
                "Live webcam mode needs the `streamlit-webrtc` and `av` packages.\n\n"
                "Install with:\n```\npip install streamlit-webrtc av\n```\nthen restart the app."
            )

elif page == "History":
    st.markdown('<div class="card"><div class="card-title">🕑 Run History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<p style='color:#8a93a8;'>No runs yet. Go to Home and run a detection.</p>", unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        rows_html = ""
        for i, r in hist_df.iterrows():
            rows_html += f"""<tr>
                <td>{r['time']}</td><td><span class="badge badge-blue">{r['source']}</span></td>
                <td>{r['file']}</td><td>{r['total']}</td>
                <td><span style="color:{GREEN};">{r['with_helmet']}</span></td>
                <td><span style="color:{RED};">{r['without_helmet']}</span></td>
                <td>{r['avg_confidence']}</td></tr>"""
        st.markdown(f"""
        <table class="det-table">
        <thead><tr><th>Time</th><th>Source</th><th>File</th><th>Total</th><th>With</th><th>Without</th><th>Avg Conf</th></tr></thead>
        <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
        st.download_button("⬇ Download history CSV", hist_df.to_csv(index=False).encode("utf-8"),
                            "history.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Settings":
    st.markdown('<div class="card"><div class="card-title">⚙️ App & Model Settings</div><br>', unsafe_allow_html=True)
    st.markdown(f"""
    - **Active model path:** `{model_path_to_load}`
    - **Classes:** `{CLASS_NAMES}`
    - **Confidence threshold:** `{conf_thres}`
    - **IoU threshold:** `{iou_thres}`
    - **Input source:** `{input_source}`
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Toggle light/dark mode from Streamlit's own menu (⋮ top-right of the browser window).")
