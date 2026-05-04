from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


VALID_LABELS = {"FAKE", "REAL"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.upper().str.strip()
    data = data[data["text"] != ""]

    invalid_labels = set(data["label"]).difference(VALID_LABELS)
    if invalid_labels:
        invalid = ", ".join(sorted(invalid_labels))
        raise ValueError(f"Invalid label(s): {invalid}. Use FAKE or REAL.")

    if len(data) < 4:
        raise ValueError("Dataset must contain at least 4 valid rows.")

    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=20_000,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000)),
        ]
    )


def train_model(data: pd.DataFrame, test_size: float = 0.25, random_state: int = 1) -> tuple[Pipeline, dict[str, Any]]:
    stratify = data["label"] if data["label"].nunique() > 1 else None
    train_x, test_x, train_y, test_y = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_pipeline()
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)
    metrics = {
        "accuracy": accuracy_score(test_y, predictions),
        "report": classification_report(test_y, predictions, zero_division=0),
        "test_rows": len(test_y),
        "train_rows": len(train_y),
    }
    return model, metrics


def save_model(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_news(model: Pipeline, text: str) -> dict[str, float | str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text cannot be empty.")

    label = str(model.predict([cleaned_text])[0])

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(probabilities[classes.index(label)])

    return {"label": label, "confidence": confidence}
