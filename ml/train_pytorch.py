"""
train_pytorch.py
----------------
Trains a small, fully-connected PyTorch classifier on numeric document features.

Architecture
------------
  Input (12 features) → Linear(12→64) → ReLU → Dropout(0.3)
                       → Linear(64→32) → ReLU → Dropout(0.3)
                       → Linear(32→7)  → (softmax at inference)

The model is intentionally small so it can be explained in an interview,
converted to Core ML without complications, and run on-device efficiently.

Outputs
-------
  ml/models/brabus_document_classifier_pytorch.pt  — saved model weights
  ml/models/brabus_pytorch_metadata.json           — feature order, labels, norms
  ml/models/brabus_pytorch_metrics.json            — evaluation metrics
  ml/models/brabus_pytorch_confusion_matrix.png    — confusion matrix
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "brabus_document_classifier_pytorch.pt"
METADATA_PATH = MODELS_DIR / "brabus_pytorch_metadata.json"
METRICS_PATH = MODELS_DIR / "brabus_pytorch_metrics.json"
CM_IMAGE_PATH = MODELS_DIR / "brabus_pytorch_confusion_matrix.png"

# ---------------------------------------------------------------------------
# Hyperparameters — kept simple and easy to discuss in an interview
# ---------------------------------------------------------------------------
HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.3
LEARNING_RATE = 1e-3
EPOCHS = 60
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class DocumentDataset(Dataset):
    """
    Wraps a numpy feature matrix and label index array as a PyTorch Dataset.

    Parameters
    ----------
    X : np.ndarray  shape (N, num_features)  — already normalised
    y : np.ndarray  shape (N,)               — integer class indices
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class DocumentClassifier(nn.Module):
    """
    Three-layer fully-connected classifier for document type prediction.

    The architecture is deliberately shallow:
      - easy to trace and convert to Core ML
      - fast on-device inference
      - straightforward to explain

    Parameters
    ----------
    input_size  : number of numeric features (12)
    hidden_1    : first hidden layer size    (64)
    hidden_2    : second hidden layer size   (32)
    output_size : number of classes          (7)
    dropout     : dropout probability
    """

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
# Training function
# ---------------------------------------------------------------------------

def train(
    model: DocumentClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch of training. Returns average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model: DocumentClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference and return (true_labels, predicted_labels)."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(y_batch.numpy().tolist())
    return all_true, all_pred


# ---------------------------------------------------------------------------
# Predict helper
# ---------------------------------------------------------------------------

def predict(
    model: DocumentClassifier,
    text: str,
    means: np.ndarray,
    stds: np.ndarray,
    label_list: list[str],
    device: torch.device,
) -> tuple[str, float]:
    """
    Predict the document type for a single text string.

    Returns (label_name, confidence) where confidence is the softmax probability
    for the predicted class.
    """
    raw = np.array(extract_feature_vector(text), dtype=np.float32)
    normalised = (raw - means) / (stds + 1e-8)
    tensor = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    idx = int(np.argmax(probs))
    return label_list[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("PyTorch Model — Confusion Matrix", fontsize=13)

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Brabus — PyTorch Classifier Training")
    print("=" * 60)

    # ------------------------------------------------------------------ data
    if not DATA_PATH.exists():
        sys.exit(
            f"ERROR: Dataset not found at {DATA_PATH}.\n"
            "Run `python generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} rows.")

    label_list = sorted(df["label"].unique().tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
    print(f"Classes ({len(label_list)}): {label_list}")

    # --------------------------------------------------------- feature matrix
    print("\nExtracting numeric features …")
    X_raw = np.array(
        [extract_feature_vector(t) for t in df["text"]], dtype=np.float32
    )
    y_raw = np.array([label_to_idx[l] for l in df["label"]], dtype=np.int64)

    # ----------------------------------------------------------- train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    # ----------------------------------------- normalise (z-score per feature)
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)

    X_train_norm = (X_train - means) / (stds + 1e-8)
    X_test_norm = (X_test - means) / (stds + 1e-8)

    # ------------------------------------------------------- datasets / loaders
    train_dataset = DocumentDataset(X_train_norm, y_train)
    test_dataset = DocumentDataset(X_test_norm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------------------- model / optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    input_size = X_train.shape[1]
    output_size = len(label_list)

    model = DocumentClassifier(
        input_size=input_size,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        output_size=output_size,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel architecture:\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ----------------------------------------------------------------- train
    print(f"\nTraining for {EPOCHS} epochs …")
    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, train_loader, optimizer, criterion, device)
        loss_history.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss = {avg_loss:.4f}")

    # --------------------------------------------------------------- evaluate
    y_true_idx, y_pred_idx = evaluate(model, test_loader, device)
    y_true_lbls = [label_list[i] for i in y_true_idx]
    y_pred_lbls = [label_list[i] for i in y_pred_idx]

    acc = accuracy_score(y_true_idx, y_pred_idx)
    prec = precision_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
    rec = recall_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
    f1 = f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f} (weighted)")
    print(f"  Recall   : {rec:.4f} (weighted)")
    print(f"  F1       : {f1:.4f} (weighted)")
    print()
    print(classification_report(y_true_lbls, y_pred_lbls, zero_division=0))

    # ------------------------------------------------------ save model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel weights saved to: {MODEL_PATH}")

    # ------------------------------------------------------ save metadata
    metadata = {
        "feature_order": FEATURE_ORDER,
        "labels": label_list,
        "input_size": input_size,
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
        "output_size": output_size,
        "dropout": DROPOUT,
        "normalization_means": means.tolist(),
        "normalization_stds": stds.tolist(),
        "note": (
            "The Swift app must extract features in the same order as "
            "'feature_order', then normalize using these means and stds "
            "before sending to the Core ML model."
        ),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {METADATA_PATH}")

    # --------------------------------------------------------- save metrics
    report_dict = classification_report(
        y_true_lbls, y_pred_lbls, output_dict=True, zero_division=0
    )
    metrics_out = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "per_class": {
            lbl: {
                "precision": round(report_dict[lbl]["precision"], 4),
                "recall": round(report_dict[lbl]["recall"], 4),
                "f1": round(report_dict[lbl]["f1-score"], 4),
                "support": report_dict[lbl]["support"],
            }
            for lbl in label_list
            if lbl in report_dict
        },
        "training_loss_final": round(loss_history[-1], 6),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    # -------------------------------------------------- confusion matrix image
    save_confusion_matrix(y_true_idx, y_pred_idx, label_list, CM_IMAGE_PATH)

    # --------------------------------------------------- quick predict sanity check
    print("\nSanity check — predict() on a sample text:")
    sample = (
        "Applications are due March 15. Submit your resume and transcript "
        "to scholarships@gsu.edu. Award value: $5,000."
    )
    predicted_label, confidence = predict(model, sample, means, stds, label_list, device)
    print(f"  Text: \"{sample[:60]}…\"")
    print(f"  Predicted: {predicted_label}  (confidence: {confidence:.2%})")

    print("\nDone.")


if __name__ == "__main__":
    main()
