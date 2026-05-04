import argparse
from pathlib import Path

from src.model import load_dataset, save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fake news detection model.")
    parser.add_argument("--data", default="data/sample_news.csv", help="Path to CSV dataset.")
    parser.add_argument(
        "--model-out",
        default="models/fake_news_model.joblib",
        help="Where to save the trained model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    model, metrics = train_model(data)
    save_model(model, Path(args.model_out))

    print(f"Saved model to {args.model_out}")
    print(f"Training rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(metrics["report"])


if __name__ == "__main__":
    main()
