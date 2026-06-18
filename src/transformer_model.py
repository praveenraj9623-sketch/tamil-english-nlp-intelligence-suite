from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.baseline_model import EXPERIMENT_NAME, configure_mlflow
from src.preprocessing import PROJECT_ROOT


RUN_NAME = "multilingual_transformer_embeddings"
DEFAULT_ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"
SELECTION_METRIC_ORDER = ["macro_f1", "weighted_f1", "accuracy"]
SELECTION_METRIC_ORDER_LABEL = "Champion selected by macro F1, then weighted F1, then accuracy."

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


def _confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: list[str],
    labels: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Greens", values_format="d", xticks_rotation=35, colorbar=False)
    ax.set_title("Transformer Embeddings Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _metric_rank(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("macro_f1", 0.0)),
        float(metrics.get("weighted_f1", 0.0)),
        float(metrics.get("accuracy", 0.0)),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def select_champion(
    transformer_metrics: dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    baseline_metrics = _read_json(reports_dir / "baseline_metrics.json")
    candidates = {RUN_NAME: transformer_metrics}
    if baseline_metrics is not None:
        candidates["tfidf_baseline"] = baseline_metrics

    champion_name, champion_metrics = max(candidates.items(), key=lambda item: _metric_rank(item[1]))
    source_model_path = models_dir / (
        "tfidf_baseline.joblib" if champion_name == "tfidf_baseline" else "transformer_embeddings.joblib"
    )
    champion_path = models_dir / "champion_model.joblib"
    champion_bundle = joblib.load(source_model_path)
    champion_bundle["champion"] = True
    champion_bundle["selection_metric_order"] = SELECTION_METRIC_ORDER
    champion_bundle["champion_metrics"] = champion_metrics
    joblib.dump(champion_bundle, champion_path)

    if champion_name == "tfidf_baseline":
        shutil.copyfile(reports_dir / "confusion_matrix_tfidf_baseline.png", reports_dir / "confusion_matrix_champion.png")
    else:
        shutil.copyfile(
            reports_dir / "confusion_matrix_multilingual_transformer_embeddings.png",
            reports_dir / "confusion_matrix_champion.png",
        )

    comparison = {
        "experiment_name": EXPERIMENT_NAME,
        "mode": "comparison",
        "baseline_only": False,
        "selection_metric_order": SELECTION_METRIC_ORDER,
        "selection_metric_order_label": SELECTION_METRIC_ORDER_LABEL,
        "champion_model": champion_name,
        "champion_family": champion_bundle.get("model_type"),
        "champion_model_path": str(champion_path),
        "models": candidates,
    }
    comparison_path = reports_dir / "model_comparison.json"
    champion_metrics_path = reports_dir / "champion_metrics.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    champion_metrics_path.write_text(json.dumps(champion_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison


def train_transformer_embeddings(
    processed_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    mlflow_tracking_uri: str | None = None,
    encoder_name: str = DEFAULT_ENCODER,
    batch_size: int = 32,
    max_iter: int = 2000,
) -> dict[str, Any]:
    processed_path = _path(processed_dir, PROJECT_ROOT / "data" / "processed")
    models_path = _path(models_dir, PROJECT_ROOT / "models")
    reports_path = _path(reports_dir, PROJECT_ROOT / "reports")
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df = _load_split(processed_path)
    encoder = SentenceTransformer(encoder_name)
    train_embeddings = encoder.encode(
        train_df["text"].astype(str).tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    test_embeddings = encoder.encode(
        test_df["text"].astype(str).tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    classifier = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
    )
    classifier.fit(train_embeddings, train_df["label"].astype(str))
    y_true = test_df["label"].astype(str)
    y_pred = classifier.predict(test_embeddings)
    labels = list(classifier.classes_)

    metrics = {
        "model_name": RUN_NAME,
        "model_family": "transformer",
        "sentence_transformer_name": encoder_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "labels": labels,
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    metrics_path = reports_path / "transformer_metrics.json"
    cm_path = reports_path / "confusion_matrix_multilingual_transformer_embeddings.png"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _confusion_matrix_plot(y_true, y_pred, labels, cm_path)
    metrics["metrics_path"] = str(metrics_path)
    metrics["confusion_matrix_path"] = str(cm_path)

    model_path = models_path / "transformer_embeddings.joblib"
    bundle = {
        "model_type": "transformer",
        "model_name": RUN_NAME,
        "sentence_transformer_name": encoder_name,
        "classifier": classifier,
        "classes": labels,
        "batch_size": batch_size,
        "metrics": metrics,
    }
    joblib.dump(bundle, model_path)
    metrics["model_path"] = str(model_path)

    configure_mlflow(mlflow_tracking_uri)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params(
            {
                "sentence_transformer": encoder_name,
                "embedding_mode": "frozen_pretrained_embeddings",
                "classifier": "LogisticRegression",
                "class_weight": "balanced",
                "batch_size": batch_size,
                "max_iter": max_iter,
            }
        )
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        mlflow.log_metric("weighted_f1", metrics["weighted_f1"])
        mlflow.log_artifact(metrics["metrics_path"])
        mlflow.log_artifact(metrics["confusion_matrix_path"])

    comparison = select_champion(metrics, models_path, reports_path)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression on frozen multilingual sentence-transformer embeddings."
    )
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    train_transformer_embeddings(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
