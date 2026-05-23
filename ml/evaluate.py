"""
evaluate.py
-----------
Reloads the saved sklearn and PyTorch models and re-runs evaluation on the
held-out test set to confirm that saved artifacts work correctly.

Run this after training both models:
  python evaluate.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import FEATURE_ORDER, extract_feature_vector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "document_dataset.csv"
MODELS_DIR = Path(__file__).parent / "models"

SKLEARN_PATH = MODELS_DIR / "document_classifier_sklearn.joblib"
LABELS_PATH = MODELS_DIR / "labels.json"
PYTORCH_PATH = MODELS_DIR / "brabus_document_classifier_pytorch.pt"
METADATA_PATH = MODELS_DIR / "brabus_pytorch_metadata.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Reproduce the DocumentClassifier architecture (must match train_pytorch.py)
# ---------------------------------------------------------------------------

class DocumentClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_1: int,
        hidden_2: int,
        output_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(
            f"ERROR: {label} not found at {path}.\n"
            "Please run the training scripts first:\n"
            "  python train_sklearn.py\n"
            "  python train_pytorch.py"
        )


def _print_metrics(name: str, y_true, y_pred, labels: list[str]) -> None:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision (wt.)   : {prec:.4f}")
    print(f"  Recall    (wt.)   : {rec:.4f}")
    print(f"  F1        (wt.)   : {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Brabus — Model Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------------ data
    _check_exists(DATA_PATH, "Dataset")
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Use the same seed/split as training so the test set is identical
    texts = df["text"]
    labels_series = df["label"]

    _, X_test_text, _, y_test = train_test_split(
        texts, labels_series,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_series,
    )

    # ------------------------------------------------------------ sklearn
    _check_exists(SKLEARN_PATH, "Sklearn model")
    _check_exists(LABELS_PATH, "Labels file")

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    pipeline = joblib.load(SKLEARN_PATH)
    print(f"\nLoaded sklearn pipeline: {SKLEARN_PATH.name}")

    # Detect whether this pipeline needs text or numeric input
    pipeline_steps = [name for name, _ in pipeline.steps]
    if "tfidf" in pipeline_steps:
        y_pred_sk = pipeline.predict(X_test_text)
    else:
        X_test_num = np.array(
            [extract_feature_vector(t) for t in X_test_text], dtype=np.float32
        )
        y_pred_sk = pipeline.predict(X_test_num)

    _print_metrics("Sklearn Best Pipeline", y_test, y_pred_sk, labels)

    # ----------------------------------------------------------- pytorch
    _check_exists(PYTORCH_PATH, "PyTorch model")
    _check_exists(METADATA_PATH, "PyTorch metadata")

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    pt_labels: list[str] = metadata["labels"]
    means = np.array(metadata["normalization_means"], dtype=np.float32)
    stds = np.array(metadata["normalization_stds"], dtype=np.float32)

    model = DocumentClassifier(
        input_size=metadata["input_size"],
        hidden_1=metadata["hidden_1"],
        hidden_2=metadata["hidden_2"],
        output_size=metadata["output_size"],
        dropout=metadata["dropout"],
    )
    state_dict = torch.load(PYTORCH_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded PyTorch model: {PYTORCH_PATH.name}")

    X_test_raw = np.array(
        [extract_feature_vector(t) for t in X_test_text], dtype=np.float32
    )
    X_test_norm = (X_test_raw - means) / (stds + 1e-8)
    tensor_test = torch.tensor(X_test_norm, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor_test)
        pred_indices = logits.argmax(dim=1).numpy()

    y_pred_pt = [pt_labels[i] for i in pred_indices]
    _print_metrics("PyTorch Classifier (numeric features)", y_test, y_pred_pt, pt_labels)

    # ------------------------------------------------------- quick inference check
    print("=" * 60)
    print("Quick Inference Check")
    print("=" * 60)

    samples = [
        (
            "Applications are due March 15. Submit your resume, transcript, "
            "and recommendation letter to scholarships@gsu.edu.",
            "scholarship_notice",
        ),
        (
            "Assignment 2 is due Friday. Submit a PDF via the course portal. "
            "Rubric is attached. Late work will lose 10 points per day.",
            "assignment_sheet",
        ),
        (
            "Join us for the Spring Networking Social! RSVP by April 14. "
            "Location: Student Union Ballroom. Guest speaker from Apple.",
            "event_flyer",
        ),
        (
            "Total: $47.83. Payment: Visa ending 4821. Tax: $3.90.",
            "receipt",
        ),
    ]

    label_to_idx = {lbl: i for i, lbl in enumerate(pt_labels)}

    print(f"\n{'Text (truncated)':<50}  {'Expected':<20}  {'Predicted':<20}  Conf")
    print("-" * 100)
    for text, expected in samples:
        raw = np.array(extract_feature_vector(text), dtype=np.float32)
        normed = (raw - means) / (stds + 1e-8)
        t = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_label = pt_labels[int(np.argmax(probs))]
        conf = probs[int(np.argmax(probs))]
        status = "✓" if pred_label == expected else "✗"
        print(f"  {text[:48]:<50}  {expected:<20}  {pred_label:<20}  {conf:.0%}  {status}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
