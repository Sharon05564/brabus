"""
train_sklearn.py
----------------
Trains and evaluates classical ML baselines for document classification.

Models trained:
  1. Logistic Regression  + TF-IDF  (text pipeline)
  2. LinearSVC            + TF-IDF  (text pipeline)
  3. Logistic Regression  + numeric features from extract_features.py

The best text pipeline is saved as the sklearn artifact.
The numeric-feature pipeline demonstrates the same feature set used by the
Core ML model in the iOS app.

Outputs:
  ml/models/document_classifier_sklearn.joblib  — best pipeline
  ml/models/labels.json                         — ordered label list
  ml/models/sklearn_metrics.json                — evaluation metrics
  ml/models/sklearn_confusion_matrix.png        — confusion matrix heatmap
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Project-local import
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import FEATURE_ORDER, extract_feature_vector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "document_dataset.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SKLEARN_MODEL_PATH = MODELS_DIR / "document_classifier_sklearn.joblib"
LABELS_PATH = MODELS_DIR / "labels.json"
METRICS_PATH = MODELS_DIR / "sklearn_metrics.json"
CM_IMAGE_PATH = MODELS_DIR / "sklearn_confusion_matrix.png"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.Series, pd.Series]:
    if not DATA_PATH.exists():
        sys.exit(
            f"ERROR: Dataset not found at {DATA_PATH}.\n"
            "Run `python generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        sys.exit("ERROR: Dataset must have 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df["text"], df["label"]


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_numeric_features(texts: pd.Series) -> np.ndarray:
    """Build the numeric feature matrix from raw texts."""
    print("Extracting numeric features …")
    rows = [extract_feature_vector(t) for t in texts]
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(name: str, y_true, y_pred, labels: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    metrics = {
        "model": name,
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "per_class": {
            lbl: {
                "precision": round(report[lbl]["precision"], 4),
                "recall": round(report[lbl]["recall"], 4),
                "f1": round(report[lbl]["f1-score"], 4),
                "support": report[lbl]["support"],
            }
            for lbl in labels
            if lbl in report
        },
    }
    return metrics


def save_confusion_matrix(y_true, y_pred, labels: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Sklearn Best Model — Confusion Matrix", fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Brabus — Sklearn Baseline Training")
    print("=" * 60)

    texts, labels_series = load_data()
    labels = sorted(labels_series.unique().tolist())

    # Save label list
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Labels saved to: {LABELS_PATH}")

    # Train/test split (stratified to preserve class balance)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels_series,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_series,
    )
    print(f"\nTrain size: {len(X_train_text)} | Test size: {len(X_test_text)}\n")

    all_metrics = []
    best_f1 = -1.0
    best_pipeline = None
    best_name = ""

    # -----------------------------------------------------------------------
    # Model 1: Logistic Regression + TF-IDF
    # -----------------------------------------------------------------------
    print("Training Model 1: Logistic Regression + TF-IDF …")
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                   random_state=RANDOM_STATE)),
    ])
    lr_pipeline.fit(X_train_text, y_train)
    y_pred_lr = lr_pipeline.predict(X_test_text)
    metrics_lr = evaluate_model("LogisticRegression_TFIDF", y_test, y_pred_lr, labels)
    all_metrics.append(metrics_lr)
    print(f"  Accuracy: {metrics_lr['accuracy']:.4f}  |  F1: {metrics_lr['f1_weighted']:.4f}")

    if metrics_lr["f1_weighted"] > best_f1:
        best_f1 = metrics_lr["f1_weighted"]
        best_pipeline = lr_pipeline
        best_name = metrics_lr["model"]

    # -----------------------------------------------------------------------
    # Model 2: LinearSVC + TF-IDF
    # -----------------------------------------------------------------------
    print("Training Model 2: LinearSVC + TF-IDF …")
    svc_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ("clf", LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ])
    svc_pipeline.fit(X_train_text, y_train)
    y_pred_svc = svc_pipeline.predict(X_test_text)
    metrics_svc = evaluate_model("LinearSVC_TFIDF", y_test, y_pred_svc, labels)
    all_metrics.append(metrics_svc)
    print(f"  Accuracy: {metrics_svc['accuracy']:.4f}  |  F1: {metrics_svc['f1_weighted']:.4f}")

    if metrics_svc["f1_weighted"] > best_f1:
        best_f1 = metrics_svc["f1_weighted"]
        best_pipeline = svc_pipeline
        best_name = metrics_svc["model"]

    # -----------------------------------------------------------------------
    # Model 3: Logistic Regression + numeric features
    # NOTE: This model uses the same features as the PyTorch/Core ML model.
    #       It is included to show that numeric features are competitive and
    #       to verify the feature extractor works correctly.
    # -----------------------------------------------------------------------
    print("Training Model 3: Logistic Regression + Numeric Features …")
    X_train_num = build_numeric_features(X_train_text)
    X_test_num = build_numeric_features(X_test_text)

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                   random_state=RANDOM_STATE)),
    ])
    numeric_pipeline.fit(X_train_num, y_train)
    y_pred_num = numeric_pipeline.predict(X_test_num)
    metrics_num = evaluate_model(
        "LogisticRegression_NumericFeatures", y_test, y_pred_num, labels
    )
    all_metrics.append(metrics_num)
    print(f"  Accuracy: {metrics_num['accuracy']:.4f}  |  F1: {metrics_num['f1_weighted']:.4f}")
    print(f"  (Features used: {FEATURE_ORDER})")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(f"{'Model':<42} {'Accuracy':>9} {'F1 (wt)':>9}")
    print("-" * 62)
    for m in all_metrics:
        print(f"  {m['model']:<40} {m['accuracy']:>9.4f} {m['f1_weighted']:>9.4f}")
    print(f"\nBest model: {best_name}  (F1 = {best_f1:.4f})")

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    joblib.dump(best_pipeline, SKLEARN_MODEL_PATH)
    print(f"\nBest pipeline saved to: {SKLEARN_MODEL_PATH}")

    metrics_out = {
        "best_model": best_name,
        "feature_note": (
            "TF-IDF models are strong text baselines. "
            "The Core ML app uses the numeric-feature model because the same "
            "features can be reproduced exactly in Swift without a vocabulary."
        ),
        "models": all_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    # Confusion matrix for the best model
    if best_pipeline is not None:
        # Re-predict with the best pipeline on the correct input type
        if "Numeric" in best_name:
            y_best_pred = best_pipeline.predict(X_test_num)
        else:
            y_best_pred = best_pipeline.predict(X_test_text)
        save_confusion_matrix(y_test, y_best_pred, labels, CM_IMAGE_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
