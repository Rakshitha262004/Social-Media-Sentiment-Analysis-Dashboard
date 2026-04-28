#!/usr/bin/env python3
# ============================================================
# File: main.py
# Purpose: One-command pipeline orchestrator.
#   Run: python main.py
# ============================================================

import os
import sys
import time

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     SOCIAL MEDIA SENTIMENT ANALYSIS DASHBOARD               ║
║     Full Pipeline Runner                                     ║
╚══════════════════════════════════════════════════════════════╝
"""

def step(num: int, title: str):
    print(f"\n{'─'*60}")
    print(f"  STEP {num}: {title}")
    print(f"{'─'*60}")


def main():
    print(BANNER)

    # ── Step 1: Create dataset ────────────────────────────
    step(1, "Generating Synthetic Social Media Dataset")
    from src.create_dataset import generate_dataset
    import pandas as pd
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(1000)
    df.to_csv(os.path.join("data", "social_media_posts.csv"), index=False)
    print(f"  ✅ Dataset: {len(df)} rows | Columns: {list(df.columns)}")
    time.sleep(0.5)

    # ── Step 2: Preprocess ────────────────────────────────
    step(2, "Cleaning & Preprocessing Text")
    from src.preprocess import run_preprocessing
    run_preprocessing()
    time.sleep(0.5)

    # ── Step 3: Train model ───────────────────────────────
    step(3, "Training Sentiment Classifier")
    from src.train_model import run_training
    model, le, vectorizer = run_training()
    time.sleep(0.5)

    # ── Step 4: Demo predictions ──────────────────────────
    step(4, "Running Sample Predictions")
    from src.predict import demo
    demo()
    time.sleep(0.5)

    # ── Done ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  ✅ ALL STEPS COMPLETE!")
    print(f"{'═'*60}")
    print("\n  📊 Next: Launch the Streamlit Dashboard with:")
    print("     streamlit run app/dashboard.py\n")
    print("  📁 Outputs saved in:")
    print("     data/           → processed datasets")
    print("     models/         → trained model + vectorizer")
    print("     outputs/charts/ → saved chart images")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()