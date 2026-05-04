from pathlib import Path
import streamlit as st

from src.model import (
    load_model,
    predict_news,
    train_model,
    load_dataset,
    save_model,
)

MODEL_PATH = Path("models/fake_news_model.joblib")
DATA_PATH = Path("data/sample_news.csv")

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# UI
st.title("📰 Fake News Detector")
st.write("Paste a news headline or article excerpt to estimate whether it looks real or fake.")

# -------- LOAD OR TRAIN MODEL --------
def get_model():
    if MODEL_PATH.exists():
        try:
            model = load_model(MODEL_PATH)

            # 🔥 Check if model is fitted
            model.predict(["test"])

            return model

        except Exception:
            st.warning("⚠️ Model not fitted or corrupted. Retraining...")

    # Train model if not found or broken
    if not DATA_PATH.exists():
        st.error("Dataset not found. Please add data/sample_news.csv")
        st.stop()

    st.info("Training model... ⏳")

    data = load_dataset(DATA_PATH)
    model, _ = train_model(data)

    save_model(model, MODEL_PATH)

    return model


model = get_model()

# -------- INPUT --------
news_text = st.text_area(
    "News text",
    height=220,
    placeholder="Paste a news headline or article body here..."
)

# -------- PREDICTION --------
if st.button("Analyze", type="primary"):
    if not news_text.strip():
        st.error("Please enter some news text.")
    else:
        try:
            result = predict_news(model, news_text)
            label = result["label"]
            confidence = result["confidence"]

            if label == "FAKE":
                st.error(f"🚨 Prediction: {label}")
            else:
                st.success(f"✅ Prediction: {label}")

            st.metric("Confidence", f"{confidence:.1%}")

        except Exception as e:
            st.error("Error during prediction")
            st.exception(e)

# Footer
st.caption(
    "⚠️ This is a machine learning estimate. Always verify important claims with trusted sources."
)