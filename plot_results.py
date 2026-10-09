import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =========================
# 1. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_HISTORY = BASE_DIR / "baseline_history.json"
IMPROVED_HISTORY = BASE_DIR / "improved_history.json"

BASELINE_MODEL = BASE_DIR / "baseline_cnn.keras"
IMPROVED_MODEL = BASE_DIR / "improved_cnn.keras"

DATA_DIR = BASE_DIR / "data_split"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16


# =========================
# 2. LOAD TRAINING HISTORIES
# =========================

with open(BASELINE_HISTORY, "r") as f:
    baseline_history = json.load(f)

with open(IMPROVED_HISTORY, "r") as f:
    improved_history = json.load(f)


# =========================
# 3. ACCURACY GRAPH
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    baseline_history["accuracy"],
    label="Baseline - Training"
)

plt.plot(
    baseline_history["val_accuracy"],
    label="Baseline - Validation"
)

plt.plot(
    improved_history["accuracy"],
    label="Improved - Training"
)

plt.plot(
    improved_history["val_accuracy"],
    label="Improved - Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_comparison.png",
    dpi=300
)

plt.close()


# =========================
# 4. LOSS GRAPH
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    baseline_history["loss"],
    label="Baseline - Training"
)

plt.plot(
    baseline_history["val_loss"],
    label="Baseline - Validation"
)

plt.plot(
    improved_history["loss"],
    label="Improved - Training"
)

plt.plot(
    improved_history["val_loss"],
    label="Improved - Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_comparison.png",
    dpi=300
)

plt.close()


# =========================
# 5. LOAD TEST DATASET
# =========================

test_dataset = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
])


# =========================
# 6. BASELINE CONFUSION MATRIX
# =========================

baseline_test_dataset = test_dataset.map(
    lambda images, labels:
        (images / 255.0, labels)
)

baseline_model = keras.models.load_model(BASELINE_MODEL)

baseline_predictions = baseline_model.predict(
    baseline_test_dataset
)

baseline_pred = np.argmax(
    baseline_predictions,
    axis=1
)

baseline_cm = confusion_matrix(
    y_true,
    baseline_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=baseline_cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title("Baseline CNN - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "baseline_confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================
# 7. IMPROVED CONFUSION MATRIX
# =========================

improved_model = keras.models.load_model(IMPROVED_MODEL)

improved_predictions = improved_model.predict(
    test_dataset
)

improved_pred = np.argmax(
    improved_predictions,
    axis=1
)

improved_cm = confusion_matrix(
    y_true,
    improved_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=improved_cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title("Improved CNN - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "improved_confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================
# 8. PRINT RESULTS
# =========================

print("\n==============================")
print("RESULTS CREATED")
print("==============================")

print(f"\nResults folder:")
print(RESULTS_DIR)

print("\nFiles created:")

print("1. accuracy_comparison.png")
print("2. loss_comparison.png")
print("3. baseline_confusion_matrix.png")
print("4. improved_confusion_matrix.png")

print("\nClass order:")
print(class_names)

print("\nBaseline confusion matrix:")
print(baseline_cm)

print("\nImproved confusion matrix:")
print(improved_cm)

print("\nDone!")