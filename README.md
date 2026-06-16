# Tamil-English NLP Intelligence Suite

Production-ready sentiment, speech-to-text, and Tamil text-to-speech suite for Tamil-English mixed customer support text. The core use case is regional-language sentiment understanding for fintech, edtech, and support teams where customers naturally mix Tamil, Tanglish, and English in the same message.

## Architecture

```text
Kaggle CSV in data/raw/
        |
        v
src/preprocessing.py
  - auto-detects text + sentiment columns
  - cleans URLs, mentions, whitespace
  - preserves Tamil Unicode
  - stratified 80/20 split
        |
        v
data/processed/train.csv + test.csv
        |
        +--> src/baseline_model.py
        |      TF-IDF word n-grams 1-2 + Logistic Regression
        |      MLflow run: tfidf_baseline
        |
        +--> src/transformer_model.py
               paraphrase-multilingual-MiniLM-L12-v2 embeddings
               Logistic Regression classifier
               MLflow run: multilingual_transformer_embeddings
               selects champion by macro F1, then weighted F1, then accuracy
        |
        v
models/champion_model.joblib
        |
        +--> api/main.py FastAPI endpoints
        +--> app.py Streamlit dashboard
        +--> src/error_analysis.py hardest confident mistakes
```

## Dataset

Use the Kaggle dataset: [Sentiment Analysis in Tamil-English Text](https://www.kaggle.com/datasets/danushkumarv/sentiment-analysis-in-tamilenglish-text).

The local file currently matches the headerless Kaggle format:

```text
Negative<TAB>Enna da ellam avan seyal Mari iruku
Positive<TAB>Padam vanthathum 13k dislike ...
```

For this file, column `0` is the sentiment label and column `1` is the text. If your Kaggle download has headers, check for text columns such as `text`, `tweet`, `comment`, `content`, or `review`, and label columns such as `sentiment`, `label`, `category`, `class`, or `outcome`.

## Tech Stack

- Python, pandas, scikit-learn, joblib
- TF-IDF + Logistic Regression baseline
- Hugging Face sentence-transformers with `paraphrase-multilingual-MiniLM-L12-v2`
- MLflow experiment tracking
- OpenAI Whisper `base` model for Tamil speech transcription
- gTTS Tamil text-to-speech
- FastAPI and Streamlit
- Docker and docker-compose

## Exact Run Commands

Open this folder in VS Code:

```text
C:\Users\admin\Desktop\Tamil-English NLP Intelligence Suite
```

If your VS Code terminal is PowerShell, use:

```powershell
Set-Location "C:\Users\admin\Desktop\Tamil-English NLP Intelligence Suite"
```

If your VS Code terminal is Command Prompt, use:

```cmd
cd /d "C:\Users\admin\Desktop\Tamil-English NLP Intelligence Suite"
```

Place the Kaggle CSV here. The current project already has this file at `data/raw/Tamil_sentiments.csv`; use this command only if you download a fresh copy into the project root:

```powershell
mkdir data\raw -Force
copy "Tamil_sentiments.csv" "data\raw\Tamil_sentiments.csv"
```

Inspect the raw file and confirm whether it has headers:

```powershell
Get-Content .\data\raw\Tamil_sentiments.csv -TotalCount 5
```

Create the environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Command Prompt activation, use:

```cmd
.venv\Scripts\activate.bat
```

Whisper requires `ffmpeg`. On Windows, install it if `ffmpeg -version` fails:

```powershell
winget install Gyan.FFmpeg
```

This project also includes `imageio-ffmpeg` as a Python fallback, so the app can run Whisper even when `ffmpeg` is not globally available on Windows.

Let preprocessing auto-detect the columns:

```powershell
python -m src.preprocessing --input data/raw/Tamil_sentiments.csv --inspect-only
```

For this headerless Kaggle file, run:

```powershell
python -m src.preprocessing --input data/raw/Tamil_sentiments.csv --label-col 0 --text-col 1
```

If your file has named columns, replace the names:

```powershell
python -m src.preprocessing --input data/raw/Tamil_sentiments.csv --text-col tweet --label-col sentiment
```

Train and log the TF-IDF baseline:

```powershell
python -m src.baseline_model
```

This also writes a provisional `models/champion_model.joblib` so the API can run right away. The transformer command below overwrites it after comparing both models.

Train the frozen transformer-embedding classifier, log it, compare both models, and save the champion:

```powershell
python -m src.transformer_model
```

Generate error analysis for the champion model:

```powershell
python -m src.error_analysis
```

Generate demo speech files:

```powershell
python scripts/create_demo_assets.py
```

Open MLflow:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Run the API:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Run the Streamlit app in another PowerShell window:

```powershell
Set-Location "C:\Users\admin\Desktop\Tamil-English NLP Intelligence Suite"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Run both services with Docker:

```powershell
docker compose up --build
```

FastAPI runs at `http://127.0.0.1:8000`; Streamlit runs at `http://127.0.0.1:8501`.

## API

- `GET /health`
- `POST /predict/sentiment`
- `POST /speech-to-text`
- `POST /text-to-speech`

Example sentiment request:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict/sentiment" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"இந்த service romba nalla irukku, support quick ah respond pannanga"}'
```

## Speech Tools

The Streamlit Speech Tools tab has two separate workflows:

1. Speech-to-text: upload an audio file, or use `demo/sample_tamil_gtts.mp3`, then click transcribe. The app sends the file to OpenAI Whisper `base`, which detects the language and returns text. The first run can take longer because Whisper downloads the pretrained `base` model.
2. Text-to-speech: type Tamil text, or use the default text from `demo/sample_tamil_text.txt`, then click generate audio. The app uses gTTS with `lang="ta"` and returns an MP3 player.

Demo files:

- `demo/sample_tamil_text.txt`
- `demo/sample_tamil_gtts.mp3`

## Metrics

Metrics are generated after training, not guessed in advance. Current local baseline run on `data/raw/Tamil_sentiments.csv` with a stratified 80/20 split:

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| `tfidf_baseline` | 0.5766 | 0.4658 | 0.6054 |

The transformer-embedding metrics are written after running `python -m src.transformer_model`.

- Baseline metrics: `reports/baseline_metrics.json`
- Transformer metrics: `reports/transformer_metrics.json`
- Champion comparison: `reports/model_comparison.json`
- MLflow runs: `mlruns/`

The champion is selected by `macro_f1`, then `weighted_f1`, then `accuracy`, which is appropriate for the imbalanced label distribution in this dataset.

## Error Analysis

After running `python -m src.error_analysis`, the suite writes:

- `reports/confusion_matrix_champion.png`
- `reports/error_analysis.json`
- `reports/error_analysis_top20.csv`

The JSON includes the 20 most confidently wrong predictions with original text, cleaned text, true label, predicted label, confidence, and top class probabilities.

## Future Work

- Add OCR for Tamil script from support screenshots, receipts, and scanned documents.
- Fine-tune IndicBERT or MuRIL on a labeled Tamil-English support dataset after collecting enough domain-specific examples.
- Add model monitoring for drift across new fintech or edtech customer channels.
- Add human review workflows for low-confidence or high-impact predictions.
