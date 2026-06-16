from __future__ import annotations

import base64
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predict_sentiment  # noqa: E402
from src.speech_to_text import transcribe_audio  # noqa: E402
from src.text_to_speech import text_to_speech  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"
DEMO_DIR = PROJECT_ROOT / "demo"
DEMO_TEXT_PATH = DEMO_DIR / "sample_tamil_text.txt"
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


st.set_page_config(
    page_title="Tamil-English NLP Intelligence Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --surface: #111827;
      --surface-soft: #172033;
      --line: #2d3748;
      --text: #f8fafc;
      --muted: #a7b0c0;
      --green: #2dd4bf;
      --amber: #fbbf24;
      --rose: #fb7185;
    }
    .stApp {
      background:
        linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(8, 13, 22, 1)),
        #0b1020;
      color: var(--text);
    }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }
    div[data-testid="stTabs"] button {
      color: var(--muted);
      border-radius: 8px 8px 0 0;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
      color: var(--text);
      border-bottom-color: var(--green);
    }
    .stButton > button {
      background: #0f766e;
      border: 1px solid #14b8a6;
      color: white;
      border-radius: 8px;
      font-weight: 650;
    }
    .stButton > button:hover {
      background: #115e59;
      border-color: var(--green);
      color: white;
    }
    .status-card {
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(45, 212, 191, 0.10), rgba(251, 191, 36, 0.08));
      border-radius: 8px;
      padding: 16px;
      margin: 8px 0 18px;
    }
    .small-muted { color: var(--muted); font-size: 0.92rem; }
    .artifact-image {
      width: 100%;
      max-width: 980px;
      border-radius: 8px;
      border: 1px solid var(--line);
      display: block;
      margin: 18px auto 0;
    }
    .artifact-caption {
      color: var(--muted);
      font-size: 0.9rem;
      text-align: center;
      margin-top: 8px;
    }
    audio { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | list | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_binary(path: str) -> bytes | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


@st.cache_data(show_spinner=False)
def load_demo_text() -> str:
    if DEMO_TEXT_PATH.exists():
        return DEMO_TEXT_PATH.read_text(encoding="utf-8").strip()
    return DEFAULT_TAMIL_TEXT


def package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def render_png(path: Path, caption: str) -> None:
    image_bytes = load_binary(str(path))
    if not image_bytes:
        st.warning(f"Missing image artifact: {path.name}")
        return
    encoded = base64.b64encode(image_bytes).decode("ascii")
    st.markdown(
        f"""
        <figure style="margin: 0;">
          <img class="artifact-image" src="data:image/png;base64,{encoded}" alt="{caption}" />
          <figcaption class="artifact-caption">{caption}</figcaption>
        </figure>
        """,
        unsafe_allow_html=True,
    )


def render_audio(path: Path, audio_format: str = "audio/mp3") -> None:
    audio_bytes = load_binary(str(path))
    if audio_bytes:
        st.audio(audio_bytes, format=audio_format)
    else:
        st.warning(f"Missing audio artifact: {path.name}")


def metrics_table() -> pd.DataFrame:
    comparison = load_json(str(REPORTS_DIR / "model_comparison.json"))
    rows = []
    if isinstance(comparison, dict):
        for name, metrics in comparison.get("models", {}).items():
            rows.append(
                {
                    "Model": name,
                    "Family": metrics.get("model_family"),
                    "Accuracy": metrics.get("accuracy"),
                    "Macro F1": metrics.get("macro_f1"),
                    "Weighted F1": metrics.get("weighted_f1"),
                    "Champion": "Yes" if name == comparison.get("champion_model") else "",
                }
            )
    else:
        for path in [REPORTS_DIR / "baseline_metrics.json", REPORTS_DIR / "transformer_metrics.json"]:
            metrics = load_json(str(path))
            if isinstance(metrics, dict):
                rows.append(
                    {
                        "Model": metrics.get("model_name"),
                        "Family": metrics.get("model_family"),
                        "Accuracy": metrics.get("accuracy"),
                        "Macro F1": metrics.get("macro_f1"),
                        "Weighted F1": metrics.get("weighted_f1"),
                        "Champion": "Yes" if path.name == "baseline_metrics.json" else "",
                    }
                )
    return pd.DataFrame(rows)


st.title("Tamil-English NLP Intelligence Suite")
st.markdown(
    '<div class="status-card">Regional-language sentiment, speech transcription, and Tamil audio generation for code-mixed support workflows.</div>',
    unsafe_allow_html=True,
)

tab_sentiment, tab_comparison, tab_speech, tab_errors = st.tabs(
    ["Sentiment Classifier", "Model Comparison", "Speech Tools", "Error Analysis"]
)

with tab_sentiment:
    st.subheader("Sentiment Classifier")
    with st.form("sentiment_form"):
        text = st.text_area(
            "Tamil-English text",
            height=150,
            placeholder=SAMPLE_SENTENCE,
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not text.strip():
            st.warning("Enter a sentence before predicting.")
        else:
            try:
                prediction = predict_sentiment(text)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Sentiment", prediction["sentiment"])
                col_b.metric("Confidence", f"{prediction['confidence']:.2%}")
                col_c.metric("Model", prediction["model"])
                st.markdown(
                    f'<div class="small-muted">Cleaned text: {prediction["cleaned_text"]}</div>',
                    unsafe_allow_html=True,
                )
            except FileNotFoundError as exc:
                st.error(str(exc))

with tab_comparison:
    st.subheader("Model Comparison")
    df_metrics = metrics_table()
    if df_metrics.empty:
        st.info("Run preprocessing, baseline training, and transformer training to populate model metrics.")
    else:
        st.dataframe(
            df_metrics,
            hide_index=True,
            width="stretch",
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Macro F1": st.column_config.NumberColumn(format="%.4f"),
                "Weighted F1": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    cm_path = REPORTS_DIR / "confusion_matrix_champion.png"
    if cm_path.exists():
        render_png(cm_path, "Champion confusion matrix")

with tab_speech:
    st.subheader("Speech Tools")
    status_a, status_b = st.columns(2)
    status_a.metric("Text to speech", "Ready" if package_available("gtts") else "Missing")
    status_b.metric("Speech to text", "Ready" if package_available("whisper") else "Missing")

    left, right = st.columns(2)

    with left:
        demo_audio = load_binary(str(DEMO_AUDIO_PATH))
        if demo_audio:
            st.audio(demo_audio, format="audio/mp3")
            if st.button("Transcribe Demo Audio"):
                try:
                    result = transcribe_audio(DEMO_AUDIO_PATH)
                    st.text_area("Demo transcription", value=result["text"], height=130)
                    st.caption(f"Language detected: {result['language_detected']}")
                except Exception as exc:
                    st.error(str(exc))

        uploaded = st.file_uploader("Audio upload", type=["wav", "mp3", "m4a", "ogg", "flac"])
        if uploaded is not None and st.button("Transcribe Uploaded Audio"):
            suffix = Path(uploaded.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded.getbuffer())
                temp_path = temp_file.name
            try:
                result = transcribe_audio(temp_path)
                st.text_area("Transcription", value=result["text"], height=180)
                st.caption(f"Language detected: {result['language_detected']}")
            except Exception as exc:
                st.error(str(exc))
            finally:
                Path(temp_path).unlink(missing_ok=True)

    with right:
        with st.form("tts_form"):
            tts_text = st.text_area("Tamil text", height=140, value=load_demo_text())
            tts_submit = st.form_submit_button("Generate Audio")
        if tts_submit:
            try:
                audio_path = Path(text_to_speech(tts_text))
                render_audio(audio_path, "audio/mp3")
                st.caption(audio_path.name)
            except Exception as exc:
                st.error(str(exc))

with tab_errors:
    st.subheader("Error Analysis")
    error_data = load_json(str(REPORTS_DIR / "error_analysis.json"))
    if not isinstance(error_data, dict):
        st.info("Run `python -m src.error_analysis` to generate the hardest misclassifications.")
    else:
        st.metric("Total wrong predictions", error_data.get("total_wrong", 0))
        examples = error_data.get("hardest_misclassifications", [])
        if examples:
            error_df = pd.DataFrame(examples)
            visible_cols = ["original_text", "true_label", "predicted_label", "confidence"]
            st.dataframe(
                error_df[visible_cols],
                hide_index=True,
                width="stretch",
                column_config={"confidence": st.column_config.NumberColumn(format="%.4f")},
            )
        else:
            st.success("No misclassified examples found in the current test split.")
