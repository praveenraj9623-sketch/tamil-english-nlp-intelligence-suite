"""
Tamil-English NLP Intelligence Suite — Streamlit Frontend v3.0
==============================================================
Production-grade portfolio UI demonstrating:
  • Tamil-English code-mixed sentiment classification
  • OpenAI Whisper ASR (speech-to-text)
  • gTTS Tamil TTS (text-to-speech)
  • FastAPI model serving  |  MLflow experiment tracking
  • Error analysis & iterative model evaluation

Designed to showcase regional-language AI engineering readiness.
"""

from __future__ import annotations

import base64
import html
import importlib.util
import json
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predict_sentiment          # noqa: E402
from src.speech_to_text import transcribe_audio      # noqa: E402
from src.text_to_speech import text_to_speech        # noqa: E402
from scripts.validate_artifacts import validate_artifacts  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
DEMO_DIR    = PROJECT_ROOT / "demo"
DEMO_TEXT_PATH  = DEMO_DIR / "sample_tamil_text.txt"
DEMO_AUDIO_PATH = DEMO_DIR / "sample_tamil_gtts.mp3"

DEFAULT_TAMIL_TEXT = (
    "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd. "
    "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd "
    "\u0b95\u0bcb\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc8 "
    "\u0baa\u0ba4\u0bbf\u0bb5\u0bc1 "
    "\u0b9a\u0bc6\u0baf\u0bcd\u0baf\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1. "
    "\u0ba8\u0bbe\u0b99\u0bcd\u0b95\u0bb3\u0bcd "
    "\u0bb5\u0bbf\u0bb0\u0bc8\u0bb5\u0bbf\u0bb2\u0bcd "
    "\u0b89\u0ba4\u0bb5\u0bc1\u0bb5\u0bcb\u0bae\u0bcd."
)

SAMPLE_SENTENCE = (
    "\u0b87\u0ba8\u0bcd\u0ba4 service romba nalla irukku, "
    "support team quick ah respond pannanga"
)

DEMO_SENTIMENT_EXAMPLES = [
    "Mass..... Thala acting Vera level Thala pola varuma..........",
    "Ena da viswasam bgm mariyae iruku",
    "Neruppuu banggg Sk mass",
]

SPEECH_LANGUAGE_OPTIONS = {
    "Tamil / Tanglish": "ta",
    "English": "en",
    "Auto detect": None,
}

SENTIMENT_EMOJIS = {
    "Positive": "🟢", "positive": "🟢",
    "Negative": "🔴", "negative": "🔴",
    "Neutral":  "🟡", "neutral":  "🟡",
    "Mixed":    "🟠", "mixed":    "🟠",
}

SENTIMENT_COLORS = {
    "Positive": "#34d399", "positive": "#34d399",
    "Negative": "#f87171", "negative": "#f87171",
    "Neutral":  "#fbbf24", "neutral":  "#fbbf24",
    "Mixed":    "#c084fc", "mixed":    "#c084fc",
}

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tamil-English NLP Suite | Praveen Raj",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System & CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Variables ── */
:root {
  --bg:          #09090f;
  --surface:     #0e0e1b;
  --card:        rgba(14, 14, 27, 0.96);
  --card-hover:  rgba(20, 20, 38, 0.98);
  --border:      rgba(99, 102, 241, 0.13);
  --border-hi:   rgba(99, 102, 241, 0.38);
  --primary:     #818cf8;
  --primary-dim: #6366f1;
  --success:     #34d399;
  --warning:     #fbbf24;
  --error:       #f87171;
  --purple:      #c084fc;
  --text:        #f1f5f9;
  --text-2:      #94a3b8;
  --text-muted:  #4b5563;
  --font:        'Inter', system-ui, -apple-system, sans-serif;
  --mono:        'JetBrains Mono', 'Fira Code', monospace;
  --r:           10px;
  --r-lg:        14px;
  --glow:        0 0 24px rgba(99, 102, 241, 0.18);
  --shadow:      0 4px 28px rgba(0, 0, 0, 0.45);
}

/* ── Keyframes ── */
@keyframes fadeUp   { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes gradText { 0%,100% { background-position:0% 50%; } 50% { background-position:100% 50%; } }
@keyframes pulse    { 0%,100% { opacity:1; } 50% { opacity:0.5; } }

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  font-family: var(--font) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

.stApp {
  background:
    radial-gradient(ellipse 75% 45% at 15% 0%, rgba(99,102,241,0.055) 0%, transparent 65%),
    radial-gradient(ellipse 55% 40% at 88% 95%, rgba(52,211,153,0.04) 0%, transparent 65%),
    var(--bg) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar          { width:5px; height:5px; }
::-webkit-scrollbar-track    { background:var(--bg); }
::-webkit-scrollbar-thumb    { background:rgba(99,102,241,0.4); border-radius:3px; }

/* ── Typography ── */
h1,h2,h3,h4,h5,h6 { font-family:var(--font) !important; color:var(--text) !important; }

h1 {
  font-size:2rem !important; font-weight:700 !important; letter-spacing:-0.03em !important;
  background: linear-gradient(130deg, #818cf8 0%, #c084fc 38%, #34d399 72%, #818cf8 100%);
  background-size: 220% 220%;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  animation: gradText 9s ease infinite, fadeUp 0.5s ease-out;
}
h2 { font-size:1.2rem !important; font-weight:600 !important; letter-spacing:-0.02em !important; }
h3 { font-size:1rem !important; font-weight:600 !important; }

/* ── Buttons ── */
.stButton > button {
  background: rgba(99,102,241,0.08) !important;
  border: 1px solid rgba(99,102,241,0.25) !important;
  color: var(--primary) !important;
  border-radius: var(--r) !important;
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  padding: 8px 18px !important;
  letter-spacing: 0.02em;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: rgba(99,102,241,0.18) !important;
  border-color: rgba(99,102,241,0.52) !important;
  box-shadow: 0 0 18px rgba(99,102,241,0.22) !important;
  transform: translateY(-1px);
  color: #fff !important;
}
.stButton > button:active { transform:translateY(0) scale(0.98) !important; }

.stFormSubmitButton > button {
  background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%) !important;
  border: 1px solid rgba(99,102,241,0.4) !important;
  color: #fff !important; font-weight:700 !important;
  font-size:0.85rem !important; letter-spacing:0.05em !important;
  text-transform:uppercase !important;
}
.stFormSubmitButton > button:hover {
  background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%) !important;
  box-shadow: 0 0 26px rgba(99,102,241,0.38) !important;
  transform: translateY(-2px) !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
  background: rgba(9,9,15,0.88) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  font-size: 0.92rem !important;
  line-height: 1.65 !important;
  transition: all 0.2s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: rgba(99,102,241,0.52) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.09), 0 0 14px rgba(99,102,241,0.13) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
  color: var(--text-muted) !important; font-style: italic;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 15px 18px !important;
  backdrop-filter: blur(16px) !important;
  transition: all 0.22s ease !important;
  animation: fadeUp 0.5s ease-out;
}
div[data-testid="stMetric"]:hover {
  border-color: var(--border-hi) !important;
  box-shadow: var(--glow) !important;
  transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
  color: var(--text-muted) !important;
  font-size: 0.68rem !important; font-weight:600 !important;
  text-transform:uppercase !important; letter-spacing:0.1em !important;
  font-family: var(--mono) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--text) !important; font-weight:700 !important;
  font-size: 1.5rem !important; letter-spacing:-0.02em !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding:0 !important; gap:0 !important;
}
div[data-testid="stTabs"] button {
  color: var(--text-muted) !important;
  font-family: var(--font) !important; font-weight:500 !important; font-size:0.86rem !important;
  border:none !important; border-bottom: 2px solid transparent !important;
  border-radius:0 !important; padding:10px 16px !important;
  transition: all 0.18s ease !important; margin-bottom:-1px;
}
div[data-testid="stTabs"] button:hover {
  color: var(--text-2) !important; background: rgba(99,102,241,0.04) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--primary) !important;
  border-bottom: 2px solid var(--primary) !important;
  background: rgba(99,102,241,0.05) !important;
}
div[data-testid="stTabs"] > div:last-child { padding-top:22px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 14px 12px !important; }

/* ── Form container ── */
div[data-testid="stForm"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: 20px 22px !important;
  backdrop-filter: blur(12px) !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1px dashed rgba(99,102,241,0.22) !important;
  border-radius: var(--r) !important;
  transition: all 0.2s ease;
}
div[data-testid="stFileUploader"]:hover {
  border-color: rgba(99,102,241,0.42) !important;
  background: var(--card-hover) !important;
}

/* ── DataFrame ── */
div[data-testid="stDataFrame"] {
  border-radius: var(--r) !important; overflow:hidden;
  border: 1px solid var(--border) !important; animation: fadeUp 0.5s ease-out;
}

/* ── Expander ── */
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; }
details summary { color: var(--text-2) !important; font-size:0.86rem !important; }

/* ── Alert ── */
.stAlert { border-radius: var(--r) !important; animation: fadeUp 0.35s ease-out; }

/* ── Audio / Plotly / Caption ── */
audio { width:100% !important; border-radius:var(--r) !important; }
div[data-testid="stPlotlyChart"] { animation: fadeUp 0.6s ease-out; }
.stCaption, .stCaption p { color: var(--text-muted) !important; font-size:0.74rem !important; }

/* ═══════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════ */

/* ── Panel / Glass Card ── */
.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 20px 22px;
  backdrop-filter: blur(16px);
  animation: fadeUp 0.45s ease-out;
  transition: border-color 0.22s, box-shadow 0.22s;
}
.panel:hover { border-color:rgba(99,102,241,0.24); box-shadow:0 2px 18px rgba(99,102,241,0.09); }

.panel-header {
  display:flex; align-items:center; gap:8px;
  margin-bottom:13px; padding-bottom:11px;
  border-bottom:1px solid var(--border);
}
.panel-icon  { font-size:1.1rem; }
.panel-title { font-size:0.92rem; font-weight:600; color:var(--text); letter-spacing:-0.01em; }
.panel-desc  { font-size:0.83rem; color:var(--text-2); line-height:1.6; margin:0; }

/* ── Sentiment result card ── */
.sent-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 22px;
  animation: fadeUp 0.35s ease-out;
  position: relative; overflow: hidden;
}
.sent-card::before {
  content:''; position:absolute; top:0; left:0;
  width:100%; height:3px;
  background: var(--sent-accent, var(--primary));
}

/* ── JD-match callout ── */
/* ── Insight callout ── */
.note {
  background: rgba(99,102,241,0.06);
  border: 1px solid rgba(99,102,241,0.15);
  border-left: 3px solid var(--primary);
  border-radius: 0 var(--r) var(--r) 0;
  padding: 10px 14px; margin:10px 0;
  font-size:0.82rem; color:var(--text-2); line-height:1.55;
}

/* ── Status badge ── */
.status-badge {
  display:inline-flex; align-items:center; gap:5px;
  padding:3px 10px; border-radius:999px;
  font-size:0.72rem; font-weight:600; font-family:var(--mono);
}
.status-badge.up   { background:rgba(52,211,153,0.09); border:1px solid rgba(52,211,153,0.25); color:var(--success); }
.status-badge.down { background:rgba(248,113,113,0.09); border:1px solid rgba(248,113,113,0.25); color:var(--error); }

.dot { width:6px; height:6px; border-radius:50%; display:inline-block; flex-shrink:0; }
.dot.green { background:var(--success); box-shadow:0 0 5px rgba(52,211,153,0.55); animation:pulse 2s ease-in-out infinite; }
.dot.red   { background:var(--error);   box-shadow:0 0 5px rgba(248,113,113,0.55); }

/* ── Tech pill ── */
.pill {
  display:inline-flex; align-items:center;
  padding:2px 9px; border-radius:999px;
  font-size:0.68rem; font-weight:600; font-family:var(--mono);
  background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.15); color:var(--primary);
  margin:2px; white-space:nowrap;
}
.pill.g { background:rgba(52,211,153,0.08); border-color:rgba(52,211,153,0.18); color:var(--success); }
.pill.a { background:rgba(251,191,36,0.08);  border-color:rgba(251,191,36,0.18);  color:var(--warning); }

/* Legacy feature row styles removed from the live dashboard. */
/* ── Sidebar metrics mini grid ── */
.sb-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:5px; margin:8px 0;
}
.sb-cell {
  background:rgba(9,9,15,0.72); border:1px solid var(--border);
  border-radius:8px; padding:8px 10px; text-align:center;
}
.sb-val { font-size:0.98rem; font-weight:700; letter-spacing:-0.02em; color:var(--primary); }
.sb-val.g { color:var(--success); }
.sb-val.p { color:var(--purple); }
.sb-val.a { color:var(--warning); }
.sb-lbl {
  font-size:0.58rem; color:var(--text-muted); text-transform:uppercase;
  letter-spacing:0.1em; font-family:var(--mono); margin-top:1px;
}

/* ── Horizontal divider ── */
.div { height:1px; background:var(--border); margin:18px 0; }

/* ── Footer ── */
.footer {
  text-align:center; padding:22px 0; margin-top:44px;
  border-top:1px solid var(--border);
  color:var(--text-muted); font-size:0.72rem; font-family:var(--mono); letter-spacing:0.03em;
}
.footer a { color:var(--primary); text-decoration:none; }
.footer a:hover { color:var(--success); }

/* ── Sidebar link buttons ── */
.sb-link {
  display:flex; align-items:center; gap:6px;
  padding:6px 10px; border-radius:7px;
  background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.16);
  color:var(--primary); font-size:0.76rem; font-weight:500;
  text-decoration:none; margin-bottom:5px;
  transition:background 0.18s, border-color 0.18s;
}
.sb-link:hover { background:rgba(99,102,241,0.14); border-color:rgba(99,102,241,0.36); color:#fff; }

/* ── Architecture step ── */
.arch-step {
  display:flex; align-items:flex-start; gap:12px;
  padding:11px 14px; border-radius:var(--r);
  background:var(--card); border:1px solid var(--border); margin:5px 0;
}
.arch-num {
  width:26px; height:26px; border-radius:50%;
  background:rgba(99,102,241,0.14); border:1px solid rgba(99,102,241,0.3);
  color:var(--primary); font-size:0.72rem; font-weight:700;
  display:flex; align-items:center; justify-content:center;
  flex-shrink:0; font-family:var(--mono);
}
.arch-title { font-size:0.86rem; font-weight:600; color:var(--text); }
.arch-desc  { font-size:0.76rem; color:var(--text-2); margin-top:2px; line-height:1.5; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }
header    { visibility:hidden; }

/* ── Responsive ── */
@media(max-width:768px){
  h1 { font-size:1.5rem !important; }
  .sb-grid { grid-template-columns:1fr 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: str) -> dict | list | None:
    fp = Path(path)
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def file_status(path: Path) -> dict[str, str | int | None]:
    if not path.exists():
        return {"path": str(path), "last_modified": None, "size_bytes": None}
    stat = path.stat()
    return {
        "path": str(path),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": int(stat.st_size),
    }


@st.cache_data(show_spinner=False)
def load_binary(path: str) -> bytes | None:
    fp = Path(path)
    if not fp.exists():
        return None
    return fp.read_bytes()


@st.cache_data(show_spinner=False)
def load_demo_text() -> str:
    if DEMO_TEXT_PATH.exists():
        return DEMO_TEXT_PATH.read_text(encoding="utf-8").strip()
    return DEFAULT_TAMIL_TEXT


def pkg_ok(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@st.cache_data(show_spinner=False, ttl=15)
def artifact_status() -> dict:
    return validate_artifacts(PROJECT_ROOT, verbose=False)


def confidence_band(confidence: float) -> tuple[str, str]:
    if confidence >= 0.70:
        return "High confidence", "success"
    if confidence >= 0.55:
        return "Medium confidence", "info"
    return "Low confidence — flag for human review", "warning"


def write_uploaded_audio_to_temp(uploaded_file, default_suffix: str = ".wav") -> Path:
    suffix = Path(getattr(uploaded_file, "name", "") or "").suffix.lower() or default_suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def render_stt_metadata(result: dict, mode_label: str) -> None:
    st.caption(
        f"Mode: `{mode_label}` · "
        f"Language detected: `{result.get('language_detected', 'unknown')}` · "
        f"Whisper model: `{result.get('model_name', 'base')}`"
    )


def render_audio(path: Path, audio_format: str = "audio/mp3") -> None:
    audio_bytes = load_binary(str(path))
    if audio_bytes:
        st.audio(audio_bytes, format=audio_format)
    else:
        st.warning(f"Audio file not found: `{path.name}`")


def render_png(path: Path, caption: str) -> None:
    image_bytes = load_binary(str(path))
    if not image_bytes:
        st.warning(f"Missing artifact: `{path.name}`")
        return
    encoded = base64.b64encode(image_bytes).decode("ascii")
    safe_caption = html.escape(caption)
    st.markdown(
        f"""
        <div class="panel" style="text-align:center; padding:14px;">
          <img src="data:image/png;base64,{encoded}"
               alt="{safe_caption}"
               style="width:min(100%,820px); max-width:100%; border-radius:8px;
                      border:1px solid var(--border);" />
          <p style="color:var(--text-muted); font-size:0.78rem; margin-top:9px;
                    font-family:var(--mono);">{safe_caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics_table() -> pd.DataFrame:
    comparison = load_json(str(REPORTS_DIR / "model_comparison.json"))
    rows = []
    if isinstance(comparison, dict):
        for name, metrics in comparison.get("models", {}).items():
            rows.append({
                "Model":       name,
                "Family":      metrics.get("model_family"),
                "Accuracy":    metrics.get("accuracy"),
                "Macro F1":    metrics.get("macro_f1"),
                "Weighted F1": metrics.get("weighted_f1"),
                "Champion":    "✓" if name == comparison.get("champion_model") else "",
            })
    else:
        for p in [REPORTS_DIR / "baseline_metrics.json", REPORTS_DIR / "transformer_metrics.json"]:
            m = load_json(str(p))
            if isinstance(m, dict):
                rows.append({
                    "Model":       m.get("model_name"),
                    "Family":      m.get("model_family"),
                    "Accuracy":    m.get("accuracy"),
                    "Macro F1":    m.get("macro_f1"),
                    "Weighted F1": m.get("weighted_f1"),
                    "Champion":    "✓" if p.name == "baseline_metrics.json" else "",
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=20, r=20, t=56, b=32),
    hoverlabel=dict(bgcolor="#0e0e1b", font_color="#f1f5f9", bordercolor="#6366f1"),
)


def make_confidence_gauge(confidence: float, sentiment: str) -> go.Figure:
    color = SENTIMENT_COLORS.get(sentiment, "#818cf8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number=dict(suffix="%", font=dict(size=40, color=color, family="Inter")),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#1e293b", tickwidth=1, dtick=25,
                      tickfont=dict(color="#4b5563", size=11)),
            bar=dict(color=color, thickness=0.72),
            bgcolor="rgba(14,14,27,0.5)",
            borderwidth=1, bordercolor="#1e293b",
            steps=[
                dict(range=[0,33],  color="rgba(248,113,113,0.07)"),
                dict(range=[33,66], color="rgba(251,191,36,0.06)"),
                dict(range=[66,100],color="rgba(52,211,153,0.07)"),
            ],
            threshold=dict(
                line=dict(color="rgba(255,255,255,0.5)", width=2),
                thickness=0.82, value=confidence*100,
            ),
        ),
        title=dict(text="Confidence Score", font=dict(size=12, color="#4b5563")),
    ))
    fig.update_layout(**_LAYOUT, height=250)
    return fig


def make_comparison_bar(df: pd.DataFrame) -> go.Figure:
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    colors  = ["#818cf8", "#34d399", "#c084fc"]
    layout  = {**_LAYOUT, "margin": dict(l=24, r=24, t=30, b=100)}
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric, x=df["Model"], y=df[metric],
            marker=dict(color=colors[i], line=dict(width=0), opacity=0.84),
            text=df[metric].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
            textposition="outside",
            textfont=dict(size=11, color="#64748b"),
        ))
    fig.update_layout(
        **layout, barmode="group", bargap=0.24, bargroupgap=0.07, height=420,
        xaxis=dict(gridcolor="rgba(99,102,241,0.06)", showgrid=False),
        yaxis=dict(gridcolor="rgba(99,102,241,0.06)", range=[0,1.18], dtick=0.2),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                    font=dict(size=12)),
    )
    return fig


def make_radar_chart(df: pd.DataFrame) -> go.Figure:
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    colors  = ["#818cf8", "#34d399", "#c084fc", "#f87171", "#fbbf24"]
    layout  = {**_LAYOUT, "margin": dict(l=28, r=28, t=72, b=80)}
    fig = go.Figure()
    for idx, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] if pd.notna(row[m]) else 0 for m in metrics]
        vals.append(vals[0])
        c = colors[idx % len(colors)]
        r_hex = c[1:3]; g_hex = c[3:5]; b_hex = c[5:7]
        fill_color = f"rgba({int(r_hex,16)},{int(g_hex,16)},{int(b_hex,16)},0.10)"
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]], name=row["Model"],
            fill="toself", fillcolor=fill_color,
            line=dict(color=c, width=2), marker=dict(size=5),
        ))
    fig.update_layout(
        **layout,
        title=dict(text="Model Comparison — Radar", font=dict(size=14, color="#f1f5f9"), x=0.5),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="rgba(99,102,241,0.10)", linecolor="rgba(99,102,241,0.10)"),
            radialaxis=dict(
                gridcolor="rgba(99,102,241,0.08)", linecolor="rgba(99,102,241,0.06)",
                range=[0,1], dtick=0.25,
            ),
        ),
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5),
    )
    return fig


def make_3d_confusion_chart(error_data: dict) -> go.Figure:
    labels = [str(l) for l in error_data.get("labels", [])]
    matrix = error_data.get("confusion_matrix", [])
    xs, ys, zs, hover_rows = [], [], [], []
    for ti, row in enumerate(matrix):
        if not isinstance(row, list):
            continue
        for pi, value in enumerate(row):
            count = int(value or 0)
            if count <= 0:
                continue
            xs.append(pi); ys.append(ti); zs.append(count)
            hover_rows.append([
                labels[ti] if ti < len(labels) else str(ti),
                labels[pi] if pi < len(labels) else str(pi),
            ])
    max_c = max(zs) if zs else 1
    sizes = [8 + (c / max_c) * 26 for c in zs]
    texts = [str(c) if c >= max_c * 0.08 else "" for c in zs]
    fig = go.Figure(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers+text", text=texts, textposition="top center",
        customdata=hover_rows,
        hovertemplate="True: %{customdata[0]}<br>Pred: %{customdata[1]}<br>Count: %{z}<extra></extra>",
        marker=dict(
            size=sizes, color=zs,
            colorscale=[[0.0,"#818cf8"],[0.5,"#fbbf24"],[1.0,"#f87171"]],
            opacity=0.85, showscale=True,
            colorbar=dict(title=dict(text="Count"), thickness=11),
            line=dict(color="rgba(241,245,249,0.2)", width=1),
        ),
    ))
    layout = {**_LAYOUT, "margin": dict(l=0, r=0, t=16, b=0)}
    ax = dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels,
              gridcolor="rgba(99,102,241,0.12)", zerolinecolor="rgba(99,102,241,0.18)",
              backgroundcolor="rgba(14,14,27,0.3)")
    fig.update_layout(**layout, height=500, scene=dict(
        xaxis=dict(**ax, title="Predicted"),
        yaxis=dict(**ax, title="True Label"),
        zaxis=dict(title="Count", gridcolor="rgba(99,102,241,0.1)"),
        camera=dict(eye=dict(x=1.5, y=1.6, z=1.2)),
    ))
    return fig


def make_error_donut(error_data: dict) -> go.Figure:
    total_wrong   = error_data.get("total_wrong", 0)
    total_samples = error_data.get("total_examples", error_data.get("total_test_samples", total_wrong + 100))
    correct = max(total_samples - total_wrong, 0)
    fig = go.Figure(go.Pie(
        values=[correct, total_wrong],
        labels=["Correct", "Misclassified"],
        hole=0.65,
        marker=dict(colors=["#34d399","#f87171"], line=dict(color="#0e0e1b", width=3)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#f1f5f9"),
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Classification Accuracy", font=dict(size=14, color="#f1f5f9"), x=0.5),
        height=320, showlegend=False,
        annotations=[dict(
            text=f"<b>{total_wrong}</b><br><span style='font-size:11px;color:#4b5563'>errors</span>",
            x=0.5, y=0.5, font=dict(size=20, color="#f87171", family="Inter"), showarrow=False,
        )],
    )
    return fig


def make_confidence_distribution(examples: list) -> go.Figure:
    confidences = [e.get("confidence", 0) for e in examples]
    fig = go.Figure(go.Histogram(
        x=confidences, nbinsx=14,
        marker=dict(color="rgba(129,140,248,0.55)", line=dict(color="#818cf8", width=1)),
        hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Confidence on Misclassified Examples", font=dict(size=14, color="#f1f5f9"), x=0.5),
        xaxis=dict(title="Confidence", gridcolor="rgba(99,102,241,0.07)", range=[0,1]),
        yaxis=dict(title="Count",       gridcolor="rgba(99,102,241,0.07)"),
        height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Branding ──────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:10px 0 14px;">
      <div style="font-size:2rem; margin-bottom:5px;">🧠</div>
      <div style="font-size:0.92rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em;">
        Tamil-English NLP Suite
      </div>
      <div style="font-size:0.68rem; color:#4b5563; font-family:var(--mono); margin-top:3px;">
        v3.0 &nbsp;·&nbsp; NLP Dashboard
      </div>
    </div>
    <div class="div"></div>
    """, unsafe_allow_html=True)

    # ── Pipeline status ────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.68rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:0.1em; color:#4b5563; font-family:var(--mono); margin-bottom:6px;'>"
        "Pipeline Status</div>",
        unsafe_allow_html=True,
    )

    model_path = PROJECT_ROOT / "models" / "champion_model.joblib"
    stt_ok  = pkg_ok("whisper")
    tts_ok  = pkg_ok("gtts")
    model_ok = model_path.exists()

    for label, ok in [("Sentiment Model", model_ok), ("Whisper ASR", stt_ok), ("gTTS TTS", tts_ok)]:
        dot_cls  = "green" if ok else "red"
        badge_cls = "up" if ok else "down"
        state    = "Ready" if ok else "Missing"
        st.markdown(
            f"""<div style="display:flex; align-items:center; justify-content:space-between;
                            padding:5px 9px; border-radius:7px;
                            background:rgba(9,9,15,0.65); margin:3px 0;">
                  <span style="font-size:0.78rem; color:#94a3b8;">{label}</span>
                  <span class="status-badge {'up' if ok else 'down'}">
                    <span class="dot {dot_cls}"></span>{state}
                  </span>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Model metrics ──────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.68rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:0.1em; color:#4b5563; font-family:var(--mono); margin-bottom:6px;'>"
        "Champion Model</div>",
        unsafe_allow_html=True,
    )

    comparison_sidebar = load_json(str(REPORTS_DIR / "model_comparison.json"))
    if isinstance(comparison_sidebar, dict):
        champion_name = comparison_sidebar.get("champion_model", "—")
        m_data = comparison_sidebar.get("models", {}).get(champion_name, {})
        acc  = m_data.get("accuracy", 0)
        mf1  = m_data.get("macro_f1", 0)
        wf1  = m_data.get("weighted_f1", 0)
        fam  = m_data.get("model_family", "—")

        st.markdown(
            f"""<div style="font-size:0.7rem; color:#4b5563; font-family:var(--mono);
                            background:rgba(9,9,15,0.65); border:1px solid var(--border);
                            border-radius:8px; padding:6px 10px; margin-bottom:5px;">
                  {html.escape(str(champion_name))}
                </div>
                <div class="sb-grid">
                  <div class="sb-cell"><div class="sb-val">{acc:.1%}</div><div class="sb-lbl">Accuracy</div></div>
                  <div class="sb-cell"><div class="sb-val g">{mf1:.1%}</div><div class="sb-lbl">Macro F1</div></div>
                  <div class="sb-cell"><div class="sb-val p">{wf1:.1%}</div><div class="sb-lbl">Wt. F1</div></div>
                  <div class="sb-cell"><div class="sb-val a">5</div><div class="sb-lbl">Classes</div></div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"Family: `{fam}`  ·  Macro F1 champion selection")
    else:
        st.caption("Run training pipeline to populate metrics.")

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.68rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:0.1em; color:#4b5563; font-family:var(--mono); margin-bottom:7px;'>"
        "Stack</div>",
        unsafe_allow_html=True,
    )
    pills_html = "".join([
        f'<span class="pill">{t}</span>'
        for t in ["scikit-learn","Sent-Transformers","Whisper","gTTS",
                  "FastAPI","MLflow","Docker","Streamlit"]
    ])
    st.markdown(f'<div style="line-height:2;">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Author ─────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.68rem; font-weight:700; text-transform:uppercase; "
        "letter-spacing:0.1em; color:#4b5563; font-family:var(--mono); margin-bottom:8px;'>"
        "Author</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style="font-size:0.86rem; font-weight:600; color:#f1f5f9; margin-bottom:3px;">
      Praveen Raj A
    </div>
    <div style="font-size:0.74rem; color:#4b5563; margin-bottom:10px; line-height:1.55;">
      MCA Data Science · CGPA 9.55/10<br>
      AI Engineer · Tamil-English NLP · MLOps
    </div>
    <a href="https://praveen-raj-portfolio-ai.lovable.app/" target="_blank" class="sb-link">
      🌐 &nbsp;Portfolio
    </a>
    <a href="https://github.com/praveenraj9623-sketch/tamil-english-nlp-intelligence-suite"
       target="_blank" class="sb-link">
      💻 &nbsp;GitHub Repository
    </a>
    <a href="https://www.linkedin.com/in/praveen-raj-a-b05abb2a3/" target="_blank" class="sb-link">
      💼 &nbsp;LinkedIn
    </a>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Tamil-English NLP Intelligence Suite")

st.markdown(
    """<div style="color:#94a3b8; font-size:0.96rem; margin:-2px 0 20px; line-height:1.65;">
      Code-mixed sentiment classification &nbsp;·&nbsp; Whisper ASR &nbsp;·&nbsp;
      Tamil TTS &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; MLflow &nbsp;·&nbsp; Docker
    </div>""",
    unsafe_allow_html=True,
)

# Hero KPI metrics
_comp = load_json(str(REPORTS_DIR / "model_comparison.json"))
_champ_name = _comp.get("champion_model", "—") if isinstance(_comp, dict) else "—"
_champ_m    = (_comp.get("models", {}).get(_champ_name, {}) if isinstance(_comp, dict) else {})
_acc_str  = f"{_champ_m.get('accuracy', 0):.2%}"  if _champ_m else "—"
_mf1_str  = f"{_champ_m.get('macro_f1', 0):.2%}"  if _champ_m else "—"
_wf1_str  = f"{_champ_m.get('weighted_f1', 0):.2%}" if _champ_m else "—"

hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("Champion Accuracy",  _acc_str,  help="Test-set accuracy on 80/20 stratified split")
hc2.metric("Macro F1 Score",     _mf1_str,  help="Macro F1 — champion selection criterion")
hc3.metric("Weighted F1",        _wf1_str,  help="Accounts for class imbalance")
hc4.metric("Sentiment Classes",  "5",       help="Positive · Negative · Neutral · Mixed · Other-language")

st.markdown('<div class="div" style="margin:18px 0 0;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_infer, tab_models, tab_speech, tab_errors = st.tabs([
    "⚡ Live Inference",
    "📊 Model Evaluation",
    "🎙️ Speech Pipeline",
    "🔍 Error Analysis",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_infer:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <span class="panel-icon">⚡</span>
        <span class="panel-title">Real-Time Sentiment Prediction</span>
      </div>
      <p class="panel-desc">
        Enter Tamil-English code-mixed text to classify sentiment in real time.
        The champion model returns a prediction, confidence score, and top-3 class probabilities.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if "sentiment_text_input" not in st.session_state:
        st.session_state.sentiment_text_input = random.choice(DEMO_SENTIMENT_EXAMPLES)

    shuffle_col, hint_col = st.columns([1, 4])
    if shuffle_col.button("↻ Shuffle Example", use_container_width=True):
        current = st.session_state.get("sentiment_text_input")
        choices = [e for e in DEMO_SENTIMENT_EXAMPLES if e != current]
        st.session_state.sentiment_text_input = random.choice(choices or DEMO_SENTIMENT_EXAMPLES)
        st.rerun()
    hint_col.caption(
        "Starts with a high-confidence Tanglish demo. Rotate to try more examples, "
        "or paste your own Tamil-English text."
    )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        with st.form("sentiment_form", clear_on_submit=False):
            text = st.text_area(
                "Tamil-English text",
                height=148,
                placeholder=SAMPLE_SENTENCE,
                key="sentiment_text_input",
                help="Paste any Tamil-English code-mixed sentence.",
            )
            st.caption("Press Ctrl+Enter or click the button below.")
            submitted = st.form_submit_button(
                "Analyze Sentiment →", type="primary", use_container_width=True,
            )

    with right_col:
        if submitted:
            if not text.strip():
                st.warning("⚠️ Enter a sentence to analyze.")
            else:
                with st.spinner("Running inference…"):
                    try:
                        prediction    = predict_sentiment(text)
                        sentiment     = prediction["sentiment"]
                        confidence    = prediction["confidence"]
                        model_name    = prediction["model"]
                        cleaned       = prediction["cleaned_text"]
                        top_probs     = prediction.get("top_probabilities", [])
                        band_label, band_level = confidence_band(confidence)
                        emoji = SENTIMENT_EMOJIS.get(sentiment, "🔵")
                        color = SENTIMENT_COLORS.get(sentiment, "#818cf8")

                        st.markdown(
                            f"""<div class="sent-card" style="--sent-accent:{color};">
                                  <div style="display:flex; align-items:center; gap:12px;">
                                    <span style="font-size:2.4rem;">{emoji}</span>
                                    <div>
                                      <div style="font-size:0.64rem; font-weight:700; text-transform:uppercase;
                                                  letter-spacing:0.1em; color:#4b5563; font-family:var(--mono);">
                                        Prediction
                                      </div>
                                      <div style="font-size:1.9rem; font-weight:700; color:{color};
                                                  letter-spacing:-0.03em; margin-top:2px;">
                                        {html.escape(sentiment)}
                                      </div>
                                    </div>
                                  </div>
                                </div>""",
                            unsafe_allow_html=True,
                        )

                        g_left, g_right = st.columns([1, 1])
                        with g_left:
                            fig_gauge = make_confidence_gauge(confidence, sentiment)
                            st.plotly_chart(fig_gauge, use_container_width=True,
                                            config={"displayModeBar": False})
                        with g_right:
                            st.metric("Model",           model_name)
                            st.metric("Confidence",      f"{confidence:.2%}")
                            st.metric("Confidence Band", band_label)
                            if band_level == "warning":
                                st.warning("Low confidence — flag for manual review.")

                        if top_probs:
                            probs_df = pd.DataFrame(top_probs).rename(
                                columns={"label": "Class", "probability": "Probability"}
                            )
                            with st.expander("Top-3 class probabilities", expanded=True):
                                st.dataframe(
                                    probs_df, hide_index=True, use_container_width=True,
                                    column_config={"Probability": st.column_config.ProgressColumn(
                                        "Probability", format="%.4f", min_value=0.0, max_value=1.0,
                                    )},
                                )

                        with st.expander("View cleaned text", expanded=False):
                            st.code(cleaned, language=None)

                    except FileNotFoundError as exc:
                        st.error(f"Model artifact not found: {exc}")
                    except Exception as exc:
                        st.error(f"Inference error: {exc}")
        else:
            st.markdown(
                '<div class="note" style="margin-top:16px;">Enter text on the left and click '
                '<strong>Analyze Sentiment</strong> to see the prediction here.</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_models:
    comparison = load_json(str(REPORTS_DIR / "model_comparison.json"))
    comparison = comparison if isinstance(comparison, dict) else {}
    validation = artifact_status()
    comparison_models = comparison.get("models", {}) if isinstance(comparison.get("models"), dict) else {}
    baseline_only = bool(comparison.get("baseline_only")) or len(comparison_models) < 2

    mode_text = (
        "Baseline-only mode — run the transformer training command to enable full comparison."
        if baseline_only
        else "Side-by-side evaluation: TF-IDF baseline vs. multilingual Sentence-Transformer embeddings."
    )

    st.markdown(
        f"""<div class="panel">
              <div class="panel-header">
                <span class="panel-icon">📊</span>
                <span class="panel-title">Model Evaluation Dashboard</span>
              </div>
              <p class="panel-desc">{html.escape(mode_text)}</p>
              <p style="font-size:0.76rem; color:var(--text-muted); margin:8px 0 0;">
                Champion selected by Macro F1 → Weighted F1 → Accuracy.
                Macro F1 is prioritised because class distribution is imbalanced.
              </p>
            </div>""",
        unsafe_allow_html=True,
    )

    st.write("")

    df_metrics = metrics_table()

    if df_metrics.empty:
        st.info("Run preprocessing + baseline training to populate evaluation metrics.")
    else:
        champion_row = df_metrics[df_metrics["Champion"] == "✓"]
        if not champion_row.empty:
            ch = champion_row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏆 Champion", ch["Model"])
            c2.metric("Accuracy",    f"{ch['Accuracy']:.4f}"   if pd.notna(ch["Accuracy"])    else "—")
            c3.metric("Macro F1",    f"{ch['Macro F1']:.4f}"   if pd.notna(ch["Macro F1"])    else "—")
            c4.metric("Weighted F1", f"{ch['Weighted F1']:.4f}" if pd.notna(ch["Weighted F1"]) else "—")

        v1, v2 = st.columns(2)
        v1.metric("Selection Order",      "Macro F1 → Weighted F1 → Accuracy")
        v2.metric("Artifact Consistency", "OK" if validation.get("ok") else "Needs refresh")

        if baseline_only:
            st.info("Baseline-only mode: transformer results not yet generated.")
        if validation.get("warnings"):
            st.warning("Artifact warning: " + " ".join(validation["warnings"]))

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        st.markdown("##### Performance Metrics")
        fig_bar = make_comparison_bar(df_metrics)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        if len(df_metrics) >= 2 and not baseline_only:
            st.markdown('<div class="div"></div>', unsafe_allow_html=True)
            st.markdown("##### Metric Balance — Radar")
            fig_radar = make_radar_chart(df_metrics)
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Radar chart appears when both baseline and transformer results are available.")

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        st.markdown("##### Detailed Metrics Table")
        st.dataframe(
            df_metrics, hide_index=True, use_container_width=True,
            column_config={
                "Accuracy":    st.column_config.NumberColumn(format="%.4f"),
                "Macro F1":    st.column_config.NumberColumn(format="%.4f"),
                "Weighted F1": st.column_config.NumberColumn(format="%.4f"),
                "Champion":    st.column_config.TextColumn(width="small"),
            },
        )

        error_data_for_3d = load_json(str(REPORTS_DIR / "error_analysis.json"))
        if isinstance(error_data_for_3d, dict) and error_data_for_3d.get("confusion_matrix"):
            st.markdown('<div class="div"></div>', unsafe_allow_html=True)
            st.markdown("##### 3D Confusion Landscape")
            st.markdown(
                '<div class="note">X = predicted label · Y = true label · Z = sample count. '
                'Larger, warmer markers indicate high-volume prediction paths.</div>',
                unsafe_allow_html=True,
            )
            fig_3d = make_3d_confusion_chart(error_data_for_3d)
            st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": False})

    cm_path = REPORTS_DIR / "confusion_matrix_champion.png"
    if cm_path.exists():
        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        st.markdown("##### Champion Confusion Matrix")
        render_png(cm_path, "Classes: Positive · Negative · Mixed_feelings · not-Tamil / other-language · unknown_state")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SPEECH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_speech:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <span class="panel-icon">🎙️</span>
        <span class="panel-title">Speech Processing Pipeline</span>
      </div>
      <p class="panel-desc">
        Pretrained Whisper speech-to-text (ASR) and gTTS Tamil text-to-speech (TTS) —
        integrated and ready for regional customer-support audio workflows.
        These are production ASR/TTS integrations, not trained here.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.info(
        "Whisper first run downloads the pretrained model (~145 MB). "
        "Use **Tamil / Tanglish** mode for Tamil recordings to pass a language hint; "
        "**Auto detect** when unsure. gTTS requires internet access. "
        "Supported uploads: WAV · MP3 · M4A · MP4 · AAC · OGG · FLAC"
    )

    # Status row
    s1, s2 = st.columns(2)
    for col, icon, label, ok in [
        (s1, "🔊", "Text → Speech (gTTS)", tts_ok),
        (s2, "🎤", "Speech → Text (Whisper)", stt_ok),
    ]:
        with col:
            dot = "green" if ok else "red"
            badge = "up" if ok else "down"
            st.markdown(
                f"""<div class="panel" style="text-align:center; padding:14px 10px;">
                      <div style="font-size:1.8rem; margin-bottom:5px;">{icon}</div>
                      <div style="font-size:0.76rem; color:#4b5563; font-family:var(--mono);
                                  text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                        {label}
                      </div>
                      <span class="status-badge {badge}">
                        <span class="dot {dot}"></span>{"Ready" if ok else "Missing"}
                      </span>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    asr_col, tts_col = st.columns(2, gap="large")

    # ── ASR ──────────────────────────────────────────────────────────────
    with asr_col:
        st.markdown("##### 🎤 Speech → Text (ASR)")

        speech_mode_label = st.radio(
            "Language mode",
            list(SPEECH_LANGUAGE_OPTIONS.keys()),
            index=0, horizontal=True,
            help="Forcing Tamil improves Whisper's recognition quality for Tamil/Tanglish recordings.",
        )
        speech_language = SPEECH_LANGUAGE_OPTIONS[speech_mode_label]
        st.caption("Tamil mode passes a language hint to Whisper — recommended for Tamil recordings.")

        demo_audio = load_binary(str(DEMO_AUDIO_PATH))
        if demo_audio:
            st.audio(demo_audio, format="audio/mp3")
            if st.button("Transcribe Demo", key="demo_stt", use_container_width=True):
                with st.spinner("Transcribing with Whisper…"):
                    try:
                        result = transcribe_audio(DEMO_AUDIO_PATH, language="ta")
                        st.text_area("Demo transcription", value=result["text"], height=110)
                        render_stt_metadata(result, "Tamil / Tanglish demo")
                    except Exception as exc:
                        st.error(f"Transcription error: {exc}")

        uploaded = st.file_uploader(
            "Upload audio file",
            type=["wav","mp3","m4a","mp4","aac","ogg","flac"],
            help="WAV · MP3 · M4A · MP4 · AAC · OGG · FLAC",
        )
        if uploaded is not None:
            st.caption(f"File: `{Path(uploaded.name).suffix.lower() or '(none)'}` · {uploaded.size:,} bytes")

        if uploaded is not None and st.button("Transcribe Upload", key="upload_stt", use_container_width=True):
            tmp_path = write_uploaded_audio_to_temp(uploaded)
            with st.spinner("Transcribing…"):
                try:
                    result = transcribe_audio(tmp_path, language=speech_language)
                    st.text_area("Transcription", value=result["text"], height=160)
                    render_stt_metadata(result, speech_mode_label)
                except Exception as exc:
                    st.error(f"Transcription error: {exc}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record Tamil / Tanglish audio")
            if recorded_audio is not None and st.button(
                "Transcribe Recording", key="recorded_stt", use_container_width=True,
            ):
                tmp_path = write_uploaded_audio_to_temp(recorded_audio, default_suffix=".wav")
                with st.spinner("Transcribing recording…"):
                    try:
                        result = transcribe_audio(tmp_path, language=speech_language)
                        st.text_area("Recording transcription", value=result["text"], height=160)
                        render_stt_metadata(result, speech_mode_label)
                    except Exception as exc:
                        st.error(f"Transcription error: {exc}")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
        else:
            st.info("Microphone recording requires a newer Streamlit version. Upload a WAV/MP3 file above.")

    # ── TTS ──────────────────────────────────────────────────────────────
    with tts_col:
        st.markdown("##### 🔊 Text → Speech (TTS)")
        with st.form("tts_form"):
            tts_text = st.text_area(
                "Tamil text input",
                height=160,
                value=load_demo_text(),
                help="Enter Tamil Unicode text to synthesise speech.",
            )
            tts_submit = st.form_submit_button(
                "Generate Tamil Audio →", type="primary", use_container_width=True,
            )

        if tts_submit:
            with st.spinner("Synthesising via gTTS…"):
                try:
                    audio_path = Path(text_to_speech(tts_text))
                    render_audio(audio_path, "audio/mp3")
                    st.caption(f"Output: `{audio_path.name}`")
                except Exception as exc:
                    st.error(f"TTS error: {exc}")

        st.markdown(
            '<div class="note" style="margin-top:10px;">gTTS wraps Google\'s neural Tamil TTS engine. '
            'For production use, this can be swapped for a self-hosted model (e.g. Coqui TTS) '
            'to eliminate the internet dependency and add voice customisation.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_errors:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <span class="panel-icon">🔍</span>
        <span class="panel-title">Error Analysis — Hardest Misclassifications</span>
      </div>
      <p class="panel-desc">
        The 20 most confidently wrong predictions: examples where the model was highly certain
        but incorrect. Systematic error analysis like this drives the next round of
        data cleaning, annotation, and fine-tuning.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "This Kaggle dataset contains noisy Tamil-English, Tanglish, non-Tamil, and other-language samples. "
        "Error analysis identifies class confusion and data-quality issues before production fine-tuning."
    )

    error_path        = REPORTS_DIR / "error_analysis.json"
    error_file_status = file_status(error_path)
    error_data        = load_json(str(error_path))

    if not isinstance(error_data, dict):
        st.info("Run `python -m src.error_analysis` to generate error analysis data.")
    else:
        total_wrong   = int(error_data.get("total_wrong", 0) or 0)
        total_samples = int(error_data.get("total_examples", error_data.get("total_test_samples", 0)) or 0)
        derived_acc   = (total_samples - total_wrong) / total_samples if total_samples else None
        examples      = error_data.get("hardest_misclassifications", [])
        acc_str       = f"{derived_acc:.4f}" if derived_acc is not None else "—"

        st.caption(
            f"Source: `{error_file_status['path']}` · "
            f"Modified: `{error_file_status['last_modified'] or 'missing'}` · "
            f"{total_samples:,} test samples · {total_wrong} errors · derived accuracy: `{acc_str}`"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("❌ Misclassified",   total_wrong)
        m2.metric("📝 Test Samples",    total_samples if total_samples else "—")
        m3.metric("✅ Derived Accuracy",
                  f"{derived_acc * 100:.1f}%" if derived_acc is not None else "—")

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)

        if total_wrong > 0:
            ch_left, ch_right = st.columns(2)
            with ch_left:
                st.plotly_chart(make_error_donut(error_data),
                                use_container_width=True, config={"displayModeBar": False})
            with ch_right:
                if examples:
                    st.plotly_chart(make_confidence_distribution(examples),
                                    use_container_width=True, config={"displayModeBar": False})
                    st.caption(
                        "High-confidence errors are intentional: this view focuses on the model's most "
                        "certain wrong predictions to expose class confusion and noisy labels."
                    )

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)

        if examples:
            st.markdown("##### Top Confident Misclassifications")
            st.markdown(
                '<div class="note">Filter by label pair to drill into specific class confusions — '
                'the most actionable step for data re-annotation or targeted augmentation.</div>',
                unsafe_allow_html=True,
            )
            error_df = pd.DataFrame(examples)
            f1, f2 = st.columns(2)
            true_labels = (sorted(error_df["true_label"].dropna().astype(str).unique())
                           if "true_label" in error_df else [])
            pred_labels = (sorted(error_df["predicted_label"].dropna().astype(str).unique())
                           if "predicted_label" in error_df else [])

            sel_true = f1.selectbox("Filter true label",      ["All"] + true_labels, key="err_true")
            sel_pred = f2.selectbox("Filter predicted label", ["All"] + pred_labels,  key="err_pred")

            if sel_true != "All" and "true_label" in error_df:
                error_df = error_df[error_df["true_label"].astype(str) == sel_true]
            if sel_pred != "All" and "predicted_label" in error_df:
                error_df = error_df[error_df["predicted_label"].astype(str) == sel_pred]

            vcols = [c for c in ["original_text","true_label","predicted_label","confidence"]
                     if c in error_df.columns]
            if vcols:
                st.dataframe(
                    error_df[vcols], hide_index=True, use_container_width=True,
                    column_config={
                        "original_text":   st.column_config.TextColumn("Text",        width="large"),
                        "true_label":      st.column_config.TextColumn("True Label",  width="small"),
                        "predicted_label": st.column_config.TextColumn("Predicted",   width="small"),
                        "confidence":      st.column_config.ProgressColumn(
                            "Confidence", format="%.4f", min_value=0, max_value=1,
                        ),
                    },
                )
        else:
            st.success("🎉 No misclassified examples in the current test split.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
if False:
    st.markdown("""
    <div class="panel">
      <div class="panel-header">
        <span class="panel-icon">📋</span>
        <span class="panel-title">Project Overview</span>
      </div>
      <p class="panel-desc">
        A production-style NLP portfolio project for Tamil-English code-mixed text.
        Built to demonstrate end-to-end AI engineering — from data preprocessing and model
        training to speech integration, API serving, MLflow tracking, and dashboard delivery.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ── Capability Summary ────────────────────────────────────
    st.markdown("#### Capability Overview")
    st.caption("Each capability below describes the system design and implementation scope.")

    jd_features = [
        ("🗣️", "Tamil Language Expertise",
         "Native Tamil fluency (read/write/understand) plus Tamil Unicode-preserving preprocessing, "
         "Tamil ASR testing, and Tamil TTS synthesis.",
         "Required Skill"),
        ("🔍", "Sentiment Analysis — NLP Model Development",
         "TF-IDF + Logistic Regression baseline; multilingual Sentence-Transformer embeddings; "
         "champion selection by macro F1 on a stratified 80/20 split.",
         "Core Responsibility"),
        ("🎤", "ASR — Speech Recognition",
         "OpenAI Whisper integration with Tamil language hints, demo audio, and file/microphone upload support.",
         "Key Responsibility"),
        ("🔊", "TTS — Text-to-Speech",
         "gTTS Tamil TTS integration with demo text synthesis; architecture allows drop-in replacement "
         "with a self-hosted model for production.",
         "Key Responsibility"),
        ("📊", "Performance Evaluation & Error Analysis",
         "Confusion matrix, macro/weighted F1, champion selection, top-20 confident misclassification "
         "drill-down, and class-pair filtering.",
         "Core Responsibility"),
        ("🚀", "Production API Integration",
         "FastAPI endpoints for /predict/sentiment, /speech-to-text, /text-to-speech, and /health. "
         "Deployable via uvicorn or Docker Compose.",
         "Core Responsibility"),
        ("🐳", "MLOps — Docker & MLflow",
         "Full Docker Compose setup; MLflow experiment tracking with metrics, parameters, and "
         "artifact registration per training run.",
         "Preferred Skill"),
        ("🤖", "LLM / Transformer Awareness",
         "Sentence-Transformers (multilingual embeddings), Hugging Face ecosystem, and "
         "code-mixed language challenges documented in the project.",
         "Preferred Skill"),
    ]

    for icon, title, desc, tag in jd_features:
        is_req = "Required" in tag or "Core" in tag
        tag_color = "#34d399" if is_req else "#818cf8"
        st.markdown(
            f"""<div class="feat-row">
                  <div class="feat-icon">{icon}</div>
                  <div class="feat-body">
                    <div class="feat-title">{title}</div>
                    <div class="feat-desc">{html.escape(desc)}</div>
                  </div>
                  <div style="flex-shrink:0; font-size:0.62rem; font-weight:700; font-family:var(--mono);
                               color:{tag_color}; text-transform:uppercase; letter-spacing:0.1em;
                               white-space:nowrap; align-self:flex-start; margin-top:3px;">
                    {tag}
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Architecture ──────────────────────────────────────────
    arch_col, meta_col = st.columns([3, 2], gap="large")

    with arch_col:
        st.markdown("#### System Architecture")
        arch_steps = [
            ("01", "Data Ingestion",
             "Kaggle Tamil-English sentiment dataset · CSV ingestion · Unicode validation"),
            ("02", "Preprocessing",
             "Tamil Unicode preservation · code-mixed text cleaning · Tanglish normalisation · "
             "stratified 80/20 train/test split"),
            ("03", "Baseline Model",
             "TF-IDF word n-grams (1-3) + Logistic Regression · joblib serialisation · "
             "MLflow metric logging"),
            ("04", "Transformer Model",
             "Multilingual Sentence-Transformer embeddings + Logistic Regression · "
             "champion selection by macro F1"),
            ("05", "Inference Service",
             "predict_sentiment() → champion model · confidence scoring · "
             "top-3 class probabilities · cleaned text output"),
            ("06", "Speech Integration",
             "Whisper ASR (multi-format, language-hinted) · gTTS Tamil TTS · "
             "temp-file audio pipeline"),
            ("07", "API Layer",
             "FastAPI endpoints: /predict/sentiment · /speech-to-text · /text-to-speech · /health · "
             "Uvicorn / Docker"),
            ("08", "Dashboard",
             "Streamlit multi-tab app · real-time inference · Plotly charts · "
             "error analysis · model comparison"),
        ]
        for num, title, desc in arch_steps:
            st.markdown(
                f"""<div class="arch-step">
                      <div class="arch-num">{num}</div>
                      <div>
                        <div class="arch-title">{title}</div>
                        <div class="arch-desc">{html.escape(desc)}</div>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )

    with meta_col:
        st.markdown("#### Dataset")
        st.markdown(
            '<div class="panel">'
            '<div class="panel-header"><span class="panel-icon">📁</span>'
            '<span class="panel-title">Kaggle Tamil-English Sentiment</span></div>'
            '<p class="panel-desc" style="margin-bottom:10px;">'
            'Code-mixed Tamil-English social media text with 5 sentiment labels: '
            'Positive · Negative · Mixed_feelings · not-Tamil/other-language · unknown_state.'
            '</p>'
            '<span class="pill g">5 Classes</span>'
            '<span class="pill">Code-Mixed</span>'
            '<span class="pill a">Noisy Labels</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("#### Key Artifacts")
        artifacts = [
            ("🏆", "champion_model.joblib",     "Production sentiment model"),
            ("📐", "tfidf_baseline.joblib",      "TF-IDF baseline"),
            ("📊", "model_comparison.json",      "Head-to-head metrics"),
            ("🔍", "error_analysis.json",        "Misclassification analysis"),
            ("📋", "error_analysis_top20.csv",   "Top-20 error examples"),
            ("🎵", "sample_tamil_gtts.mp3",      "TTS demo audio"),
        ]
        for icon, name, desc in artifacts:
            st.markdown(
                f'<div class="arch-step" style="padding:8px 12px;">'
                f'<div style="font-size:1rem;">{icon}</div>'
                f'<div><div class="arch-title" style="font-size:0.78rem;">{name}</div>'
                f'<div class="arch-desc" style="font-size:0.72rem;">{desc}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("#### Future Roadmap")
        roadmap = [
            "Fine-tune on larger code-mixed corpora",
            "Replace gTTS with self-hosted Coqui TTS",
            "Add model drift monitoring",
            "Extend to Telugu / Bangla support",
            "Deploy API + dashboard as separate services",
        ]
        for item in roadmap:
            st.markdown(
                f'<div style="display:flex; gap:8px; align-items:flex-start; padding:5px 0; '
                f'border-bottom:1px solid var(--border);">'
                f'<span style="color:var(--primary); font-size:0.7rem; margin-top:2px;">▸</span>'
                f'<span style="font-size:0.78rem; color:var(--text-2);">{item}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Performance summary ───────────────────────────────────
    st.markdown("#### Performance Summary")
    perf_cols = st.columns(4)
    perf_metrics = [
        ("57.66%", "Test Accuracy", "TF-IDF Baseline", "var(--primary)"),
        ("46.58%", "Macro F1",      "Champion metric",  "var(--success)"),
        ("60.54%", "Weighted F1",   "Class-weighted",   "var(--purple)"),
        ("5",      "Classes",       "Imbalanced dist.", "var(--warning)"),
    ]
    for col, (val, label, sub, color) in zip(perf_cols, perf_metrics):
        with col:
            st.markdown(
                f'<div class="panel" style="text-align:center; padding:16px 12px;">'
                f'<div style="font-size:1.5rem; font-weight:700; color:{color}; '
                f'letter-spacing:-0.03em;">{val}</div>'
                f'<div style="font-size:0.78rem; font-weight:600; color:var(--text); margin-top:3px;">'
                f'{label}</div>'
                f'<div style="font-size:0.68rem; color:var(--text-muted); font-family:var(--mono); '
                f'margin-top:2px;">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="note" style="margin-top:14px;">'
        'Accuracy of 57.66% is expected on this noisy, imbalanced code-mixed dataset — '
        'the interesting metric is macro F1 (46.58%), which treats all five classes equally '
        'and is what the champion selection is based on. '
        'Fine-tuning on more domain-specific data is the prioritised next step.'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="div" style="margin-top:44px;"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer">
      <div style="margin-bottom:5px; color:#64748b;">
        🧠 &nbsp;<strong style="color:#94a3b8;">Tamil-English NLP Intelligence Suite</strong>
        &nbsp;v3.0
      </div>
      <div>
        Sentence-Transformers &nbsp;·&nbsp; OpenAI Whisper &nbsp;·&nbsp; scikit-learn
        &nbsp;·&nbsp; MLflow &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; Docker &nbsp;·&nbsp; Streamlit
      </div>
      <div style="margin-top:4px;">
        Built by
        <a href="https://praveen-raj-portfolio-ai.lovable.app/" target="_blank">Praveen Raj A</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/praveenraj9623-sketch/tamil-english-nlp-intelligence-suite"
           target="_blank">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/praveen-raj-a-b05abb2a3/" target="_blank">LinkedIn</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
