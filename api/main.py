from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from src.inference import load_model, predict_sentiment
from src.preprocessing import PROJECT_ROOT
from src.speech_to_text import transcribe_audio
from src.text_to_speech import text_to_speech


app = FastAPI(
    title="Tamil-English NLP Intelligence Suite",
    version="1.0.0",
    description="Sentiment, speech-to-text, and Tamil text-to-speech APIs for code-mixed support text.",
)


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)


def _cleanup_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@app.get("/health")
def health() -> dict[str, object]:
    try:
        bundle = load_model()
        return {
            "status": "ok",
            "model_ready": True,
            "champion_model": bundle.get("model_name"),
            "model_family": bundle.get("model_type"),
        }
    except FileNotFoundError as exc:
        return {"status": "degraded", "model_ready": False, "detail": str(exc)}


@app.post("/predict/sentiment")
def predict_sentiment_endpoint(payload: SentimentRequest) -> dict[str, object]:
    try:
        prediction = predict_sentiment(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "sentiment": prediction["sentiment"],
        "confidence": prediction["confidence"],
        "model": prediction["model"],
        "model_name": prediction["model_name"],
        "cleaned_text": prediction["cleaned_text"],
        "top_probabilities": prediction.get("top_probabilities", []),
    }


@app.post("/speech-to-text")
async def speech_to_text_endpoint(request: Request) -> dict[str, object]:
    try:
        form = await request.form()
    except (RuntimeError, AssertionError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Install python-multipart and remove the conflicting multipart package.",
        ) from exc

    file = form.get("file")
    if file is None or not hasattr(file, "filename"):
        raise HTTPException(status_code=400, detail="Upload an audio file in the 'file' form field.")

    language = form.get("language")
    language = str(language) if language is not None else None

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    try:
        return transcribe_audio(temp_path, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        _cleanup_file(temp_path)


@app.post("/text-to-speech")
def text_to_speech_endpoint(payload: TTSRequest) -> FileResponse:
    try:
        audio_path = text_to_speech(payload.text, output_dir=PROJECT_ROOT / "reports" / "api_audio")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=Path(audio_path).name,
        background=BackgroundTask(_cleanup_file, audio_path),
    )
