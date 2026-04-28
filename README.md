# 📊 Social Media Sentiment Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **A production-style ML pipeline that collects, cleans, classifies, and visualises social media sentiment — built entirely without API keys using synthetic data.**

---

## 🎯 Overview

This project simulates an end-to-end **Social Media Sentiment Analysis** system as used by brands like Amazon, Zomato, Netflix, and HDFC Bank to monitor customer perception in real time.

It covers the complete data-science workflow:

```
Raw Social Media Text
        ↓
Text Cleaning (URL / mention / hashtag removal)
        ↓
NLP Preprocessing (stopword removal, stemming)
        ↓
TF-IDF Feature Extraction
        ↓
Sentiment Classifier (Logistic Regression)
        ↓
Prediction (Positive / Negative / Neutral)
        ↓
Interactive Streamlit Dashboard
        ↓
Business Insights & Brand Reports
```

---

## 🔥 Problem Statement

Companies receive **millions of social media mentions** daily. Manually reading and classifying them is impossible. This project automates that process — detecting whether posts about a brand are positive, negative, or neutral — enabling:

- **Faster crisis detection** (spike in negative sentiment)
- **Campaign performance tracking** (positive growth post-launch)
- **Competitor benchmarking**
- **Customer support prioritisation**

---

## 🏭 Industry Relevance

| Company | Use Case |
|---|---|
| **Amazon / Flipkart** | Product review sentiment, delivery complaint detection |
| **Zomato / Swiggy** | Food & delivery experience monitoring |
| **Netflix** | Content reception, subscriber churn signals |
| **HDFC / Paytm** | App feedback, fraud complaint detection |
| **Political campaigns** | Voter sentiment tracking |
| **Startups** | Brand launch tracking, influencer campaign ROI |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Data | Synthetic CSV (1,000 posts, 15 brands, 6 platforms) |
| Text Cleaning | `re`, `string` (built-in) |
| Feature Extraction | `TF-IDF` (scikit-learn) |
| ML Model | `Logistic Regression` (best performer) |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Visualisation | `Matplotlib`, `Seaborn`, `Plotly` |
| Dashboard | `Streamlit` |
| Word Clouds | `WordCloud` |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│   Synthetic CSV: 1000 posts × 7 columns                  │
│   Brands: Amazon, Zomato, Netflix, HDFC, Swiggy …       │
│   Platforms: Twitter, Instagram, Reddit, App Store …     │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  TEXT CLEANING MODULE                     │
│   • Remove URLs, @mentions, #hashtags                    │
│   • Lowercase conversion                                 │
│   • Remove special characters & punctuation              │
│   • Strip extra whitespace                               │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│               NLP PREPROCESSING MODULE                    │
│   • Tokenisation (split by whitespace)                   │
│   • Stopword removal (200+ English stopwords)            │
│   • Suffix stemming (approximate lemmatisation)          │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              FEATURE EXTRACTION MODULE                    │
│   • TF-IDF Vectorizer (max 5000 features)                │
│   • Unigrams + Bigrams (ngram_range = 1,2)               │
│   • Sublinear TF scaling                                 │
│   • Sparse matrix output: (1000, 703)                    │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│            SENTIMENT CLASSIFICATION MODULE                │
│   Models compared:                                       │
│   • Logistic Regression ← SELECTED (best)               │
│   • Multinomial Naive Bayes                              │
│   • Random Forest                                        │
│   Labels: Positive | Negative | Neutral                  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  DASHBOARD MODULE                         │
│   Streamlit app with 5 pages:                            │
│   🏠 Overview  🔍 Live Predictor  📈 Analytics           │
│   🏷️ Brand Tracker  💡 Insights                         │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/
│   ├── social_media_posts.csv      ← Raw synthetic dataset
│   └── processed_posts.csv         ← Cleaned + preprocessed
│
├── src/
│   ├── __init__.py
│   ├── create_dataset.py           ← Dataset generator
│   ├── preprocess.py               ← Text cleaning + TF-IDF
│   ├── train_model.py              ← Model training + evaluation
│   └── predict.py                  ← Prediction functions
│
├── models/
│   ├── sentiment_model.pkl         ← Trained classifier
│   ├── tfidf_vectorizer.pkl        ← Fitted TF-IDF vectorizer
│   └── label_encoder.pkl           ← Label encoder
│
├── app/
│   └── dashboard.py                ← Streamlit dashboard
│
├── outputs/
│   └── charts/
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       └── sentiment_distribution.png
│
├── images/                         ← Screenshots for README
├── docs/                           ← Documentation
├── notebooks/                      ← Jupyter exploration
│
├── main.py                         ← One-command runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Social-Media-Sentiment-Analysis-Dashboard.git
cd Social-Media-Sentiment-Analysis-Dashboard
```

### Step 2 — Create virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Run the full pipeline (recommended for first run)
```bash
python main.py
```
This runs all 4 steps automatically:
1. Generates the synthetic dataset (1,000 posts)
2. Cleans and preprocesses text
3. Trains and evaluates the classifier
4. Runs demo predictions

### Option B — Run steps individually
```bash
python src/create_dataset.py    # Step 1: Create data
python src/preprocess.py        # Step 2: Clean + vectorise
python src/train_model.py       # Step 3: Train model
python src/predict.py           # Step 4: Demo predictions
```

### Step 5 — Launch the dashboard
```bash
streamlit run app/dashboard.py
```
Then open `http://localhost:8501` in your browser.

---

## 📊 Model Results

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | **100%** | Best on synthetic data |
| Naive Bayes | 100% | Fast, lightweight |
| Random Forest | 100% | Most robust on real-world data |

> **Note:** 100% accuracy is expected on synthetic data because training and test text share the same source pool. On real-world scraped data you would expect 80–92% accuracy, which is industry-standard.

**5-Fold Cross-Validation:** `1.0000 ± 0.0000`

---

## 🖥️ Dashboard Features

| Page | Features |
|---|---|
| 🏠 Overview | KPI cards, sentiment pie/bar charts, platform breakdown |
| 🔍 Live Predictor | Single text + batch prediction with confidence bars |
| 📈 Analytics | Time-series trend, engagement (likes/retweets) analysis, word clouds |
| 🏷️ Brand Tracker | Brand positivity scores, stacked bar charts, brand rankings |
| 💡 Insights | Automated business recommendations, crisis detection alerts |

---

## 📈 Sample Predictions

```
Text: "Amazing experience from start to finish. Highly recommended!"
→ POSITIVE 😊  (83.1% confidence)

Text: "@Zomato food was cold and the delivery took 2 hours. Terrible!"
→ NEGATIVE 😠  (84.5% confidence)

Text: "Got the package today. Will test it and share my review later."
→ NEUTRAL  😐  (77.6% confidence)
```

---

## 🎓 Learning Outcomes

- End-to-end ML pipeline design
- NLP text preprocessing (cleaning, tokenisation, stemming)
- TF-IDF feature engineering
- Multi-class classification with scikit-learn
- Model comparison and evaluation (precision, recall, F1, confusion matrix)
- Interactive dashboard development with Streamlit + Plotly
- Professional project structuring for GitHub portfolios

---

## 📌 No Dataset Download Required

This project uses a **fully synthetic dataset** generated by `src/create_dataset.py`. No API keys, no Twitter/X access, no Kaggle downloads required. Just clone and run.

> **Want to use real data?** Drop any CSV with `text` and `sentiment` columns into `data/` and run `python src/preprocess.py`.

---

## 👤 Author

**Rakshitha A S**  



---

## 📄 License

MIT License — free to use, modify, and share with attribution.

---

*Built as an industry-aligned portfolio project demonstrating real-world NLP and ML engineering skills.*