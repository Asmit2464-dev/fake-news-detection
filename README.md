# Fake News Detection Project

A beginner-friendly machine learning project that classifies news text as `FAKE` or `REAL`.

The project uses:

- TF-IDF text features
- Logistic Regression classifier
- Streamlit web app
- A small sample dataset for quick testing

For a stronger final-year or portfolio project, replace the sample dataset with a larger public dataset such as Kaggle's Fake and Real News Dataset or LIAR.

## Project Structure

```text
.
├── app.py
├── data
│   └── sample_news.csv
├── models
│   └── .gitkeep
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
└── tests
    └── test_model.py
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train The Model

```powershell
python -m src.train --data data/sample_news.csv --model-out models/fake_news_model.joblib
```

## Run The Web App

```powershell
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## Use From Command Line

```powershell
python -m src.predict --model models/fake_news_model.joblib --text "Government confirms new public health policy after expert review"
```

## Dataset Format

Your CSV should contain these columns:

```text
text,label
"News article text here","REAL"
"Misleading claim here","FAKE"
```

Labels should be `REAL` and `FAKE`.

## Important Note

Fake news detection is probabilistic. This tool should support human judgment, not replace it. Stronger results need a large, balanced, up-to-date dataset and careful evaluation.
