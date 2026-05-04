import argparse

from src.model import load_model, predict_news


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether news text is fake or real.")
    parser.add_argument("--model", default="models/fake_news_model.joblib", help="Path to trained model.")
    parser.add_argument("--text", required=True, help="News text to classify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    result = predict_news(model, args.text)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
