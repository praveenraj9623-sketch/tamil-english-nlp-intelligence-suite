# Tamil-English NLP Intelligence Suite

Portfolio-grade production-style NLP suite for Tamil-English code-mixed sentiment analysis, pretrained speech-to-text, and Tamil text-to-speech demos for regional customer support workflows.

This project is designed for AI Engineer interviews and portfolio review. It uses pretrained models wherever possible and does not claim custom ASR/TTS training or transformer fine-tuning unless those steps are actually added later.

## Live Demo

Streamlit app: [Tamil-English NLP Intelligence Suite](https://tamil-english-nlp-intelligence-suite-ffndrptskn57ml3ky4jua7.streamlit.app/)

## Problem Statement

Customer support text in regional markets often mixes Tamil script, Tanglish, and English in the same message. This suite demonstrates how to clean that noisy text, train sentiment classifiers, compare model families, inspect model errors, expose FastAPI endpoints, and provide a Streamlit demo with speech tools.

The dataset is a public Kaggle sentiment dataset, not private fintech or edtech production data.

## Architecture

```text
data/raw/Tamil_sentiments.csv
        |
        v
src/preprocessing.py
  - detects or accepts text/label columns
  - preserves Tamil Unicode
  - lowercases English tokens
  - removes URLs, mentions, extra spaces
  - stratified 80/20 split
        |
        v
data/processed/train.csv + test.csv
        |
        +--> src/baseline_model.py
        |      TF-IDF word n-grams 1-2 + Logistic Regression
        |      MLflow run: tfidf_baseline
        |      writes baseline-only champion when transformer is absent
        |
        +--> src/transformer_model.py
               paraphrase-multilingual-MiniLM-L12-v2 frozen embeddings
               Logistic Regression classifier on embeddings
               MLflow run: multilingual_transformer_embeddings
               compares models when transformer artifacts exist
        |
        v
models/champion_model.joblib
        |
        +--> src/error_analysis.py
        |      confusion matrix + top confident wrong predictions
        |
        +--> api/main.py
        |      FastAPI inference and speech endpoints
        |
        +--> app.py
               Streamlit dashboard
```

## Dataset

Kaggle credit: [Sentiment Analysis in Tamil-English Text](https://www.kaggle.com/datasets/danushkumarv/sentiment-analysis-in-tamilenglish-text).

The local file is expected at:

```text
data/raw/Tamil_sentiments.csv
```

The Kaggle file used here is headerless:

```text
column 0: sentiment label
column 1: text
```

If your downloaded CSV has headers, inspect them and pass the names instead:

```powershell
Get-Content .\data\raw\Tamil_sentiments.csv -TotalCount 5
python -m src.preprocessing --input data/raw/Tamil_sentiments.csv --inspect-only
```

Common text column names: `text`, `tweet`, `comment`, `content`, `review`, `message`.
Common label column names: `sentiment`, `label`, `category`, `class`, `target`.

## Tech Stack

- Python, pandas, scikit-learn, joblib
- TF-IDF + Logistic Regression baseline
- Sentence-Transformers with `paraphrase-multilingual-MiniLM-L12-v2`
- MLflow tracking
- OpenAI Whisper `base` pretrained speech-to-text integration
- gTTS Tamil text-to-speech integration
- FastAPI, Uvicorn, Streamlit
- Plotly and Matplotlib reporting
- Docker and Docker Compose

## Local Setup

Open this folder in VS Code:

```text
C:\Users\admin\Desktop\Tamil-English NLP Intelligence Suite
```

PowerShell:

```powershell
Set-Location "C:\Users\admin\Desktop\Tamil-English NLP Intelligence Suite"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Command Prompt:

```cmd
cd /d "C:\Users\admin\Desktop\Tamil-English NLP Intelligence Suite"
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Rebuild Commands

Place the Kaggle CSV at:

```text
data/raw/Tamil_sentiments.csv
```

Run the full pipeline:

```powershell
python -m src.preprocessing --input data/raw/Tamil_sentiments.csv --label-col 0 --text-col 1
python -m src.baseline_model
python -m src.transformer_model
python -m src.error_analysis
python scripts/validate_artifacts.py
```

For a faster demo, you can stop after baseline and error analysis:

```powershell
python -m src.preprocessing --input data/raw/Tamil_sentiments.csv --label-col 0 --text-col 1
python -m src.baseline_model
python -m src.error_analysis
python scripts/validate_artifacts.py
```

In that state, the Streamlit app shows:

```text
Baseline-only mode: transformer comparison not generated yet.
```

Run the transformer command later to create a real side-by-side comparison.

## Current Metrics

Current local artifacts were regenerated from `data/raw/Tamil_sentiments.csv` with stratified 80/20 split:

| Artifact | Value |
| --- | ---: |
| Raw rows | 15,744 |
| Processed rows | 15,575 |
| Train rows | 12,460 |
| Test rows | 3,115 |
| Champion model | `tfidf_baseline` |
| Mode | `baseline_only` |
| Accuracy | 0.5752808988764045 |
| Macro F1 | 0.4646595756086166 |
| Weighted F1 | 0.6041238743665586 |
| Correct predictions | 1,792 |
| Wrong predictions | 1,323 |

Current label distribution:

| Label | Count |
| --- | ---: |
| Positive | 10,416 |
| Negative | 2,031 |
| Mixed_feelings | 1,790 |
| unknown_state | 846 |
| not-Tamil | 492 |

Champion selection is consistently defined as:

```text
Champion selected by macro F1, then weighted F1, then accuracy.
```

Macro F1 is prioritized because this dataset is imbalanced. Weighted F1 can look better when the majority class dominates, so macro F1 gives minority sentiment categories more visibility.

## Artifact Validation

Run:

```powershell
python scripts/validate_artifacts.py
```

The validator checks that:

- `error_analysis.total_examples` equals the test split row count.
- `total_wrong` equals test rows minus the confusion matrix diagonal sum.
- derived accuracy matches champion metrics.
- `models/champion_model.joblib` matches the displayed champion model.
- baseline-only mode is not mislabeled as transformer comparison.

## Run The Apps

Streamlit:

```powershell
streamlit run app.py
```

FastAPI:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

MLflow UI:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Docker Compose:

```powershell
docker compose up --build
```

## API Endpoints

- `GET /health`
- `POST /predict/sentiment`
- `POST /speech-to-text`
- `POST /text-to-speech`

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict/sentiment" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"இந்த service romba nalla irukku, support quick ah respond pannanga"}'
```

Expected response fields:

```json
{
  "sentiment": "Positive",
  "confidence": 0.0,
  "model": "baseline",
  "model_name": "tfidf_baseline",
  "cleaned_text": "...",
  "top_probabilities": [
    {"label": "Positive", "probability": 0.0}
  ]
}
```

## Speech Tools

The Streamlit Speech Tools tab has two workflows:

1. Speech-to-text: choose `Tamil / Tanglish`, `English`, or `Auto detect`, then upload WAV, MP3, M4A, MP4, AAC, OGG, or FLAC audio, record with the microphone when Streamlit supports `st.audio_input`, or use `demo/sample_tamil_gtts.mp3`. The app normalizes audio to 16 kHz mono PCM WAV before sending it to pretrained Whisper `base`.
2. Text-to-speech: type Tamil text or use `demo/sample_tamil_text.txt`. The app uses gTTS with `lang="ta"` and returns an MP3 player.

Notes:

- Whisper and gTTS are pretrained integrations, not custom-trained models.
- Tamil / Tanglish mode gives Whisper an explicit `ta` language hint. This is more reliable than auto-detection for short Tamil microphone recordings.
- First Whisper run can be slow because the pretrained model may load or download.
- gTTS requires internet access.
- Whisper needs FFmpeg. This project includes `imageio-ffmpeg` as a fallback, but a system install is also fine:

```powershell
winget install Gyan.FFmpeg
```

## Error Analysis

Generated files:

- `reports/confusion_matrix_champion.png`
- `reports/error_analysis.json`
- `reports/error_analysis_top20.csv`

The JSON includes the top 20 most confidently wrong predictions with original text, cleaned text, true label, predicted label, confidence, and top class probabilities.

The Kaggle dataset contains noisy Tamil-English, Tanglish, non-Tamil, and other-language samples. Error analysis is used to identify class confusion and data-quality issues before any production fine-tuning.

## Tests

Run:

```powershell
pytest -q
```

The tests are intentionally lightweight and avoid downloading Whisper or SentenceTransformer models.

## Demo Script For Interview

1. Sentiment Classifier: enter a Tamil-English support message and show sentiment, confidence band, top-3 class probabilities, model family, and cleaned text.
2. Model Comparison: explain baseline-only mode or, after transformer training, compare TF-IDF vs frozen multilingual embeddings. Show the 3D confusion landscape to explain true label, predicted label, and sample count.
3. Speech Tools: play demo audio, transcribe it with Whisper, then generate Tamil speech with gTTS.
4. Error Analysis: show high-confidence wrong predictions and explain how they reveal non-Tamil/noisy samples and class confusion.
5. FastAPI: open `/docs`, call `/health`, and show `/predict/sentiment`.

## Known Limitations

- Baseline accuracy is moderate because the dataset is noisy, code-mixed, and imbalanced.
- This is not production-ready without larger domain-specific labeled data and deployment monitoring.
- ASR/TTS are pretrained integrations, not custom-trained models.
- Transformer embeddings are frozen; no transformer fine-tuning is performed.
- The next modeling step is fine-tuning IndicBERT, MuRIL, XLM-R, or another suitable multilingual/Indic model on a domain-specific dataset.

## Future Work

- Add OCR for Tamil script from support screenshots, receipts, and scanned forms.
- Fine-tune IndicBERT, MuRIL, or XLM-R after collecting enough labeled Tamil-English support examples.
- Add human review queues for low-confidence or high-impact predictions.
- Add monitoring for class drift and language-mix drift.
- Add CI tests for Docker Compose and API startup.
