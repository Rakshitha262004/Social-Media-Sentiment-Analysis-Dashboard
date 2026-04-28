# ============================================================
# File: src/train_model.py
# Purpose: Train, evaluate, and save the sentiment classifier.
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving figs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import MultinomialNB
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing    import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ─────────────────────────────────────────────────
DATA_PATH   = os.path.join("data",    "processed_posts.csv")
VEC_PATH    = os.path.join("models",  "tfidf_vectorizer.pkl")
MODEL_PATH  = os.path.join("models",  "sentiment_model.pkl")
ENC_PATH    = os.path.join("models",  "label_encoder.pkl")
CHARTS_DIR  = os.path.join("outputs", "charts")


def load_data():
    print("📂 Loading processed data …")
    df = pd.read_csv(DATA_PATH)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    X = vectorizer.transform(df["processed_text"])
    return df, vectorizer, X


def encode_labels(df: pd.DataFrame):
    le = LabelEncoder()
    y  = le.fit_transform(df["sentiment"])   # negative=0, neutral=1, positive=2
    print(f"   Classes : {list(le.classes_)}")
    return y, le


def compare_models(X_train, X_test, y_train, y_test):
    """Quick comparison of three candidate models."""
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    print("\n📊 Model comparison:")
    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {"model": model, "accuracy": acc}
        print(f"   {name:<25}  Accuracy: {acc:.4f}")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 Best model: {best_name} ({results[best_name]['accuracy']:.4f})")
    return results[best_name]["model"], best_name, results


def plot_model_comparison(results: dict):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    names  = list(results.keys())
    scores = [v["accuracy"] for v in results.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, scores, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    for bar, score in zip(bars, scores):
        ax.text(score + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=11)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Chart saved → {path}")


def plot_confusion_matrix(model, X_test, y_test, class_names):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Chart saved → {path}")


def plot_sentiment_distribution(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    colors = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3"}
    col_list = [colors.get(s, "#888") for s in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(counts.index, counts.values, color=col_list, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Sentiment Distribution (Bar)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, (idx, val) in enumerate(counts.items()):
        axes[0].text(i, val + 5, str(val), ha="center", fontsize=11)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=col_list, startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Sentiment Distribution (Pie)", fontsize=13, fontweight="bold")

    plt.suptitle("Social Media Sentiment Overview", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "sentiment_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Chart saved → {path}")


def run_training():
    os.makedirs("models",   exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    df, vectorizer, X = load_data()
    y, le             = encode_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── Compare & select best model ───────────────────────
    best_model, best_name, all_results = compare_models(
        X_train, X_test, y_train, y_test
    )

    # ── Full evaluation ───────────────────────────────────
    y_pred = best_model.predict(X_test)
    print(f"\n📋 Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
    print(f"   5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Plots ─────────────────────────────────────────────
    print("\n🎨 Saving charts …")
    plot_model_comparison(all_results)
    plot_confusion_matrix(best_model, X_test, y_test, le.classes_)
    plot_sentiment_distribution(df)

    # ── Persist model & encoder ───────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(ENC_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"\n✅ Model saved   → {MODEL_PATH}")
    print(f"✅ Encoder saved → {ENC_PATH}")
    return best_model, le, vectorizer


if __name__ == "__main__":
    run_training()