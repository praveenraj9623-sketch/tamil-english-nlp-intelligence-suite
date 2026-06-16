from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COMMON_TEXT_COLUMNS = {
    "text",
    "tweet",
    "comment",
    "comments",
    "content",
    "sentence",
    "review",
    "message",
    "utterance",
}
COMMON_LABEL_COLUMNS = {
    "sentiment",
    "label",
    "labels",
    "category",
    "class",
    "target",
    "outcome",
    "polarity",
}
KNOWN_LABEL_VALUES = {
    "positive",
    "negative",
    "neutral",
    "mixed_feelings",
    "mixed feelings",
    "not-tamil",
    "not tamil",
    "unknown_state",
    "unknown state",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
WHITESPACE_RE = re.compile(r"\s+")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _project_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def _default_input_csv() -> Path:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DIR}. Place the Kaggle CSV there, for example "
            f"{RAW_DIR / 'Tamil_sentiments.csv'}."
        )
    return csv_files[0]


def _read_csv_best_effort(csv_path: Path, header: int | None) -> pd.DataFrame:
    attempts: list[dict[str, Any]] = [
        {"sep": None, "engine": "python"},
        {"sep": "\t"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "|"},
    ]
    best_df: pd.DataFrame | None = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(
                csv_path,
                header=header,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
                **kwargs,
            )
        except Exception:
            continue
        if best_df is None or df.shape[1] > best_df.shape[1]:
            best_df = df
    if best_df is None or best_df.empty:
        raise ValueError(f"Could not read a non-empty CSV from {csv_path}")
    best_df = best_df.dropna(axis=1, how="all")
    return best_df


def _normalize_col_name(column: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _resolve_column(df: pd.DataFrame, selector: str | int | None) -> Any | None:
    if selector is None:
        return None
    if isinstance(selector, int):
        return df.columns[selector]
    selector_text = str(selector).strip()
    if selector_text.isdigit():
        idx = int(selector_text)
        if 0 <= idx < len(df.columns):
            return df.columns[idx]
    normalized = _normalize_col_name(selector_text)
    for column in df.columns:
        if str(column) == selector_text or _normalize_col_name(column) == normalized:
            return column
    available = ", ".join(map(str, df.columns))
    raise ValueError(f"Column '{selector}' was not found. Available columns: {available}")


def _find_named_columns(df: pd.DataFrame) -> tuple[Any | None, Any | None]:
    text_col = None
    label_col = None
    for column in df.columns:
        normalized = _normalize_col_name(column)
        if normalized in COMMON_TEXT_COLUMNS and text_col is None:
            text_col = column
        if normalized in COMMON_LABEL_COLUMNS and label_col is None:
            label_col = column
    return text_col, label_col


def _label_likeness(series: pd.Series) -> tuple[int, float, float]:
    values = series.astype(str).str.strip()
    normalized_values = values.str.lower()
    sample_size = max(len(values), 1)
    known_hits = normalized_values.isin(KNOWN_LABEL_VALUES).sum()
    unique_ratio = values.nunique(dropna=True) / sample_size
    mean_length = values.str.len().mean()
    return int(known_hits), float(unique_ratio), float(mean_length)


def _infer_columns(df: pd.DataFrame) -> tuple[Any, Any]:
    named_text_col, named_label_col = _find_named_columns(df)
    if named_text_col is not None and named_label_col is not None:
        return named_text_col, named_label_col

    column_scores: dict[Any, dict[str, float]] = {}
    for column in df.columns:
        known_hits, unique_ratio, mean_length = _label_likeness(df[column])
        column_scores[column] = {
            "known_hits": known_hits,
            "unique_ratio": unique_ratio,
            "mean_length": mean_length,
        }

    label_col = min(
        df.columns,
        key=lambda col: (
            -column_scores[col]["known_hits"],
            column_scores[col]["unique_ratio"],
            column_scores[col]["mean_length"],
        ),
    )
    text_candidates = [column for column in df.columns if column != label_col]
    if not text_candidates:
        raise ValueError("Could not infer text and label columns from a single-column CSV.")
    text_col = max(
        text_candidates,
        key=lambda col: (
            column_scores[col]["mean_length"],
            column_scores[col]["unique_ratio"],
            -column_scores[col]["known_hits"],
        ),
    )
    return text_col, label_col


def lowercase_english_tokens(text: str) -> str:
    """Lowercase ASCII English characters without altering Tamil code points."""
    return "".join(char.lower() if "A" <= char <= "Z" else char for char in text)


def clean_text(text: Any) -> str:
    """Clean code-mixed Tamil-English text while preserving Tamil Unicode."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = URL_RE.sub(" ", normalized)
    normalized = MENTION_RE.sub(" ", normalized)
    normalized = normalized.replace("#", " ")
    normalized = lowercase_english_tokens(normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def load_raw_dataset(
    csv_path: str | Path | None = None,
    text_col: str | int | None = None,
    label_col: str | int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = _project_path(csv_path, _default_input_csv()).resolve()

    if text_col is not None or label_col is not None:
        header = None if str(text_col or label_col).isdigit() else 0
        raw_df = _read_csv_best_effort(csv_path, header=header)
        detected_text_col = _resolve_column(raw_df, text_col)
        detected_label_col = _resolve_column(raw_df, label_col)
        if detected_text_col is None or detected_label_col is None:
            inferred_text_col, inferred_label_col = _infer_columns(raw_df)
            detected_text_col = detected_text_col or inferred_text_col
            detected_label_col = detected_label_col or inferred_label_col
    else:
        header_df = _read_csv_best_effort(csv_path, header=0)
        named_text_col, named_label_col = _find_named_columns(header_df)
        if named_text_col is not None and named_label_col is not None:
            raw_df = header_df
            detected_text_col, detected_label_col = named_text_col, named_label_col
        else:
            raw_df = _read_csv_best_effort(csv_path, header=None)
            detected_text_col, detected_label_col = _infer_columns(raw_df)

    processed = pd.DataFrame(
        {
            "original_text": raw_df[detected_text_col].astype(str).map(str.strip),
            "text": raw_df[detected_text_col].map(clean_text),
            "label": raw_df[detected_label_col].astype(str).map(str.strip),
        }
    )
    processed = processed[(processed["text"] != "") & (processed["label"] != "")]
    processed = processed.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    metadata = {
        "source_csv": str(csv_path),
        "raw_rows": int(len(raw_df)),
        "processed_rows": int(len(processed)),
        "text_column": str(detected_text_col),
        "label_column": str(detected_label_col),
        "label_distribution": processed["label"].value_counts().sort_index().to_dict(),
    }
    return processed, metadata


def split_and_save(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    output_path = _project_path(output_dir, PROCESSED_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    min_class_count = int(df["label"].value_counts().min())
    stratify = df["label"] if min_class_count >= 2 else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    full_path = output_path / "full_processed.csv"
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    metadata_path = output_path / "dataset_metadata.json"

    df.to_csv(full_path, index=False, encoding="utf-8")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    metadata = {
        **metadata,
        "test_size": test_size,
        "random_state": random_state,
        "stratified_split": stratify is not None,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cleaning": [
            "preserve Tamil Unicode",
            "lowercase ASCII English characters",
            "strip URLs",
            "strip @mentions",
            "collapse whitespace",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "full": full_path,
        "train": train_path,
        "test": test_path,
        "metadata": metadata_path,
    }


def inspect_dataset(csv_path: str | Path | None, text_col: str | None, label_col: str | None) -> None:
    df, metadata = load_raw_dataset(csv_path=csv_path, text_col=text_col, label_col=label_col)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Tamil-English sentiment data.")
    parser.add_argument("--input", default=None, help="CSV path. Defaults to the first CSV in data/raw/.")
    parser.add_argument("--text-col", default=None, help="Text column name or zero-based index.")
    parser.add_argument("--label-col", default=None, help="Label column name or zero-based index.")
    parser.add_argument("--output-dir", default=None, help="Processed output directory.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true", help="Print detected columns without saving.")
    args = parser.parse_args()

    if args.inspect_only:
        inspect_dataset(args.input, args.text_col, args.label_col)
        return

    df, metadata = load_raw_dataset(csv_path=args.input, text_col=args.text_col, label_col=args.label_col)
    paths = split_and_save(
        df,
        metadata,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Preprocessing complete.")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
