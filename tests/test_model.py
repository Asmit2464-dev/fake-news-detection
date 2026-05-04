import pandas as pd
import pytest

from src.model import load_dataset, predict_news, train_model


def test_train_model_returns_prediction():
    data = pd.DataFrame(
        {
            "text": [
                "official government report confirms policy details",
                "peer reviewed study reports clinical trial results",
                "secret cure banned by all doctors revealed online",
                "share this post to receive free money immediately",
                "city council publishes audited budget records",
                "miracle trick cures every disease overnight",
                "central bank releases monthly financial statement",
                "anonymous viral message exposes invisible satellites",
            ],
            "label": ["REAL", "REAL", "FAKE", "FAKE", "REAL", "FAKE", "REAL", "FAKE"],
        }
    )

    model, metrics = train_model(data, test_size=0.25, random_state=1)
    result = predict_news(model, "official report confirms audited government records")

    assert result["label"] in {"REAL", "FAKE"}
    assert 0 <= result["confidence"] <= 1
    assert metrics["train_rows"] == 6
    assert metrics["test_rows"] == 2


def test_load_dataset_rejects_invalid_labels():
    path = "tests/fixtures/invalid_news.csv"
    with pytest.raises(ValueError, match="Invalid label"):
        load_dataset(path)
