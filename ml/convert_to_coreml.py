"""
convert_to_coreml.py
--------------------
Converts the trained PyTorch DocumentClassifier into a Core ML model that
can be imported directly into an Xcode project and run on-device.

Requirements to run this script:
  - macOS (coremltools works best on macOS)
  - Python 3.10+
  - All packages from requirements.txt installed

How to run:
  cd ml
  python convert_to_coreml.py

Output:
  ios/Brabus/Models/BrabusDocumentClassifier.mlmodel

IMPORTANT — Matching the Swift app:
  The Swift app (DocumentFeatureExtractor.swift + MLModelManager.swift) must:
    1. Extract the 12 numeric features in the exact order listed in FEATURE_ORDER.
    2. Normalize each feature with the means and stds from brabus_pytorch_metadata.json.
    3. Pass the normalized float array to the Core ML model.
  The model runs fully on-device. No network request is made.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Project-local imports
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import FEATURE_ORDER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"
PYTORCH_MODEL_PATH = MODELS_DIR / "brabus_document_classifier_pytorch.pt"
METADATA_PATH = MODELS_DIR / "brabus_pytorch_metadata.json"

IOS_MODELS_DIR = Path(__file__).parent.parent / "ios" / "Brabus" / "Models"
COREML_OUTPUT_PATH = IOS_MODELS_DIR / "BrabusDocumentClassifier.mlmodel"


# ---------------------------------------------------------------------------
# Reproduce the exact model architecture from train_pytorch.py
# (must stay in sync with DocumentClassifier in train_pytorch.py)
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
# Main conversion routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Brabus — PyTorch → Core ML Conversion")
    print("=" * 60)

    # ---------------------------------------------------------------- checks
    if not PYTORCH_MODEL_PATH.exists():
        sys.exit(
            f"ERROR: PyTorch model not found at {PYTORCH_MODEL_PATH}.\n"
            "Run `python train_pytorch.py` first."
        )
    if not METADATA_PATH.exists():
        sys.exit(
            f"ERROR: Metadata not found at {METADATA_PATH}.\n"
            "Run `python train_pytorch.py` first."
        )

    IOS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    input_size: int = metadata["input_size"]
    hidden_1: int = metadata["hidden_1"]
    hidden_2: int = metadata["hidden_2"]
    output_size: int = metadata["output_size"]
    dropout: float = metadata["dropout"]
    labels: list[str] = metadata["labels"]
    means: list[float] = metadata["normalization_means"]
    stds: list[float] = metadata["normalization_stds"]

    print(f"  Input size  : {input_size}")
    print(f"  Hidden      : {hidden_1} → {hidden_2}")
    print(f"  Output size : {output_size}")
    print(f"  Labels      : {labels}")

    # --------------------------------------------------------- rebuild model
    model = DocumentClassifier(
        input_size=input_size,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        output_size=output_size,
        dropout=dropout,
    )
    state_dict = torch.load(PYTORCH_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("\nPyTorch model loaded and set to eval mode.")

    # --------------------------------------------------------- trace model
    # torch.jit.trace requires a representative input tensor.
    # Dropout is disabled during eval() so tracing is deterministic.
    example_input = torch.zeros(1, input_size, dtype=torch.float32)
    traced_model = torch.jit.trace(model, example_input)
    print("Model successfully traced with torch.jit.trace.")

    # -------------------------------------------- convert to Core ML
    try:
        import coremltools as ct
    except ImportError:
        print(
            "\nERROR: coremltools is not installed.\n"
            "Install it with: pip install coremltools\n"
            "Then re-run this script on macOS."
        )
        sys.exit(1)

    print("\nConverting to Core ML …")

    # Describe the input: a 1-D float array of length input_size
    input_features = ct.TensorType(
        name="document_features",
        shape=(1, input_size),
        dtype=float,
    )

    coreml_model = ct.convert(
        traced_model,
        inputs=[input_features],
        minimum_deployment_target=ct.target.iOS16,
    )

    # --------------------------------------------- annotate metadata
    coreml_model.short_description = (
        "Brabus document type classifier — 7 classes, 12 numeric features. "
        "Runs fully on-device. No network access required."
    )
    coreml_model.author = "Brabus ML Pipeline"
    coreml_model.license = "Educational prototype"
    coreml_model.version = "1.0"

    coreml_model.input_description["document_features"] = (
        f"Normalized numeric features in this order: {', '.join(FEATURE_ORDER)}"
    )

    # Embed the label list and normalization params as user-defined metadata
    # so the Swift app can read them at runtime if needed.
    coreml_model.user_defined_metadata["labels"] = json.dumps(labels)
    coreml_model.user_defined_metadata["feature_order"] = json.dumps(FEATURE_ORDER)
    coreml_model.user_defined_metadata["normalization_means"] = json.dumps(means)
    coreml_model.user_defined_metadata["normalization_stds"] = json.dumps(stds)

    # --------------------------------------------- save
    coreml_model.save(str(COREML_OUTPUT_PATH))
    print(f"\nCore ML model saved to:\n  {COREML_OUTPUT_PATH}")

    # --------------------------------------------- verify round-trip
    print("\nVerifying round-trip prediction …")
    loaded = ct.models.MLModel(str(COREML_OUTPUT_PATH))

    # Build a sample normalised input
    raw_sample = np.zeros(input_size, dtype=np.float32)
    raw_sample[7] = 2.0   # action_word_count = 2
    raw_sample[8] = 3.0   # education_keyword_count = 3
    normed = (raw_sample - np.array(means)) / (np.array(stds) + 1e-8)
    input_dict = {"document_features": normed.reshape(1, -1).astype(np.float32)}

    prediction = loaded.predict(input_dict)
    raw_logits = list(prediction.values())[0]
    probs = _softmax(np.array(raw_logits).flatten())
    predicted_idx = int(np.argmax(probs))
    print(f"  Sample predicted label: {labels[predicted_idx]}  "
          f"(confidence: {probs[predicted_idx]:.2%})")

    print("\nConversion complete.")
    print(
        "\nNext step in Xcode:\n"
        "  1. Open your Xcode project (ios/Brabus/).\n"
        "  2. Drag BrabusDocumentClassifier.mlmodel into the project navigator.\n"
        "  3. In the file inspector, make sure 'Target Membership' includes the Brabus app target.\n"
        "  4. Xcode will auto-generate a BrabusDocumentClassifier Swift class.\n"
        "  5. MLModelManager.swift already references this class name.\n"
        "  6. Build and run."
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


if __name__ == "__main__":
    main()
