# ============================================================
# File: src/predict.py
# Purpose: Load saved model and predict sentiment for new text.
# ============================================================

import pickle
from pathlib import Path

from src.preprocess import full_preprocess

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.pkl"
VEC_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"
ENC_PATH = ROOT_DIR / "models" / "label_encoder.pkl"

# Emoji map for terminal output
EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
COLOR = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}


def load_artifacts():
    """Load model, vectorizer, and label encoder from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        le = pickle.load(f)
    return model, vectorizer, le


def predict_sentiment(text: str, model=None, vectorizer=None, le=None) -> dict:
    """
    Predict sentiment for a single text string.

    Returns a dict with keys:
      original_text, cleaned_text, sentiment, confidence, probabilities
    """
    # Lazy-load artifacts if not supplied
    if model is None:
        model, vectorizer, le = load_artifacts()

    cleaned  = full_preprocess(text)
    vec      = vectorizer.transform([cleaned])
    label_id = model.predict(vec)[0]
    proba    = model.predict_proba(vec)[0]

    sentiment = le.inverse_transform([label_id])[0]
    prob_dict = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(proba)}
    confidence = round(float(max(proba)) * 100, 2)

    return {
        "original_text" : text,
        "cleaned_text"  : cleaned,
        "sentiment"     : sentiment,
        "confidence"    : confidence,
        "probabilities" : prob_dict,
    }


def batch_predict(texts: list, model=None, vectorizer=None, le=None) -> list:
    """Predict sentiment for a list of texts efficiently."""
    if model is None:
        model, vectorizer, le = load_artifacts()

    cleaned_list = [full_preprocess(t) for t in texts]
    vecs         = vectorizer.transform(cleaned_list)
    label_ids    = model.predict(vecs)
    probas       = model.predict_proba(vecs)

    results = []
    for i, text in enumerate(texts):
        label     = le.inverse_transform([label_ids[i]])[0]
        prob_dict = {le.classes_[j]: round(float(p), 4)
                     for j, p in enumerate(probas[i])}
        results.append({
            "original_text": text,
            "sentiment"    : label,
            "confidence"   : round(float(max(probas[i])) * 100, 2),
            "probabilities": prob_dict,
        })
    return results


def demo():
    """Interactive demo: predict sentiment for several sample posts."""
    model, vectorizer, le = load_artifacts()

    sample_texts = [
        "I absolutely love this product! Best purchase of my life.",
        "The delivery was late and the item arrived damaged. Very disappointed.",
        "Got the package today. Will test it and share my review later.",
        "@Zomato food was cold and the delivery took 2 hours. Terrible!",
        "The new app update is clean and fast. Great improvement!",
        "Not sure about this product yet. Will use it for a week first.",
        "Horrible customer service. They kept me on hold for an hour.",
        "Amazing experience from start to finish. Highly recommended!",
    ]

    print("=" * 65)
    print("         SENTIMENT ANALYSIS — DEMO PREDICTIONS")
    print("=" * 65)

    for text in sample_texts:
        result = predict_sentiment(text, model, vectorizer, le)
        s      = result["sentiment"]
        print(f"\n📝 Text       : {text[:70]}")
        print(f"   Sentiment  : {COLOR[s]} {s.upper()}  {EMOJI[s]}  ({result['confidence']}% confidence)")
        print(f"   Proba      : {result['probabilities']}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    demo()
