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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------- UI HEADER ----------------
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📰 Fake News Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown("---")
st.subheader("📌 Enter News Content")

# ---------------- LOAD OR TRAIN MODEL ----------------
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

    st.info("Training model... ⏳ Please wait")

    data = load_dataset(DATA_PATH)
    model, _ = train_model(data)

    save_model(model, MODEL_PATH)

    return model


model = get_model()

# ---------------- INPUT ----------------
news_text = st.text_area(
    "",
    height=200,
    placeholder="Paste news headline or full article..."
)

# ---------------- PREDICTION ----------------
if st.button("Analyze", type="primary"):
    if not news_text.strip():
        st.error("Please enter some news text.")
    else:
        try:
            result = predict_news(model, news_text)
            label = result["label"]
            confidence = result["confidence"]

            # 🔥 SMART OUTPUT LOGIC
            if confidence < 0.6:
                st.warning("🤔 Uncertain prediction — model is not confident")

            elif label == "FAKE":
                st.error("🚨 Fake News Detected")

            else:
                st.success("✅ Real News")

            st.metric("Confidence", f"{confidence:.1%}")

        except Exception as e:
            st.error("Error during prediction")
            st.exception(e)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed using Machine Learning (TF-IDF + Logistic Regression)")