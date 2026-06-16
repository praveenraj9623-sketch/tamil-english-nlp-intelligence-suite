from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import PROJECT_ROOT


RUN_NAME = "tfidf_baseline"
EXPERIMENT_NAME = "tamil_english_sentiment"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _load_split(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing processed split files in {processed_dir}. Run: python -m src.preprocessing"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_pipeline(max_iter: int = 2000) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)[\w\u0B80-\u0BFF]+",
                    lowercase=False,
                    strip_accents=None,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: list[str],
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=35, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_model(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    reports_dir: Path,
) -> dict[str, Any]:
    y_true = test_df["label"]
    y_pred = pipeline.predict(test_df["text"].astype(str))
    labels = list(pipeline.classes_)

    metrics = {
        "model_name": RUN_NAME,
        "model_family": "baseline",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "labels": labels,
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "baseline_metrics.json"
    cm_path = reports_dir / "confusion_matrix_tfidf_baseline.png"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _confusion_matrix_plot(y_true, y_pred, labels, cm_path, "TF-IDF Baseline Confusion Matrix")
    metrics["metrics_path"] = str(metrics_path)
    metrics["confusion_matrix_path"] = str(cm_path)
    return metrics


def configure_mlflow(tracking_uri: str | None = None) -> None:
    tracking_uri = tracking_uri or f"file:{PROJECT_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_baseline(
    processed_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    mlflow_tracking_uri: str | None = None,
    max_iter: int = 2000,
) -> dict[str, Any]:
    processed_path = _path(processed_dir, PROJECT_ROOT / "data" / "processed")
    models_path = _path(models_dir, PROJECT_ROOT / "models")
    reports_path = _path(reports_dir, PROJECT_ROOT / "reports")

    train_df, test_df = _load_split(processed_path)
    pipeline = build_pipeline(max_iter=max_iter)
    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(str))

    metrics = evaluate_model(pipeline, test_df, reports_path)
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "tfidf_baseline.joblib"
    bundle = {
        "model_type": "baseline",
        "model_name": RUN_NAME,
        "pipeline": pipeline,
        "classes": list(pipeline.classes_),
        "metrics": metrics,
    }
    joblib.dump(bundle, model_path)
    metrics["model_path"] = str(model_path)

    champion_path = models_path / "champion_model.joblib"
    if not champion_path.exists():
        provisional_bundle = {
            **bundle,
            "champion": True,
            "provisional_champion": True,
            "champion_metrics": metrics,
            "selection_metric_order": ["baseline_only_until_transformer_run"],
        }
        joblib.dump(provisional_bundle, champion_path)
        metrics["provisional_champion_path"] = str(champion_path)

    configure_mlflow(mlflow_tracking_uri)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params(
            {
                "vectorizer": "TfidfVectorizer",
                "ngram_range": "1,2",
                "classifier": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": max_iter,
            }
        )
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        mlflow.log_metric("weighted_f1", metrics["weighted_f1"])
        mlflow.log_artifact(metrics["metrics_path"])
        mlflow.log_artifact(metrics["confusion_matrix_path"])
        mlflow.sklearn.log_model(pipeline, artifact_path="sklearn_pipeline")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    train_baseline(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
