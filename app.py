from pathlib import Path

import streamlit as st

from src.model import load_model, predict_news


MODEL_PATH = Path("models/fake_news_model.joblib")


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("Fake News Detector")
st.write("Paste a news headline or article excerpt to estimate whether it looks real or fake.")

if not MODEL_PATH.exists():
    st.warning(
        "Model file not found. Train it first with: "
        "`python -m src.train --data data/sample_news.csv --model-out models/fake_news_model.joblib`"
    )
    st.stop()

model = load_model(MODEL_PATH)

news_text = st.text_area(
    "News text",
    height=220,
    placeholder="Paste a news headline or article body here...",
)

if st.button("Analyze", type="primary"):
    if not news_text.strip():
        st.error("Please enter some news text.")
    else:
        result = predict_news(model, news_text)
        label = result["label"]
        confidence = result["confidence"]

        if label == "FAKE":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.metric("Confidence", f"{confidence:.1%}")

        st.caption(
            "This is a machine learning estimate. Always verify important claims with trusted sources."
        )
