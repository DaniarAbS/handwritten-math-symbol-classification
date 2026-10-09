import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =========================
# 1. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_split"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

BASELINE_MODEL = BASE_DIR / "baseline_cnn.keras"
IMPROVED_MODEL = BASE_DIR / "improved_cnn.keras"

# =========================
# 2. LOAD TEST DATASET
# =========================

raw_test_dataset = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = raw_test_dataset.class_names

print("\nClasses:")
print(class_names)

# =========================
# 3. PREPARE DATA
# =========================

# Baseline was trained with normalized images
baseline_test_dataset = raw_test_dataset.map(
    lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels)
)

# Improved CNN contains Rescaling inside the model,
# so it needs raw images.
improved_test_dataset = raw_test_dataset

# Get true labels
y_true = np.concatenate([
    labels.numpy()
    for images, labels in raw_test_dataset
])

# =========================
# 4. EVALUATION FUNCTION
# =========================

def evaluate_model(model_path, model_name, dataset):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    model = keras.models.load_model(model_path)

    predictions = model.predict(dataset)

    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    print("\nRows = True class")
    print("Columns = Predicted class")

    print("\nClass order:")

    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")

    return y_pred, cm


# =========================
# 5. BASELINE
# =========================

baseline_pred, baseline_cm = evaluate_model(
    BASELINE_MODEL,
    "BASELINE CNN",
    baseline_test_dataset
)

# =========================
# 6. IMPROVED
# =========================

improved_pred, improved_cm = evaluate_model(
    IMPROVED_MODEL,
    "IMPROVED CNN",
    improved_test_dataset
)

# =========================
# 7. FINAL COMPARISON
# =========================

baseline_accuracy = accuracy_score(y_true, baseline_pred)
improved_accuracy = accuracy_score(y_true, improved_pred)

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(f"Baseline CNN accuracy:  {baseline_accuracy * 100:.2f}%")
print(f"Improved CNN accuracy:  {improved_accuracy * 100:.2f}%")

difference = (improved_accuracy - baseline_accuracy) * 100

print(f"Difference:              {difference:+.2f} percentage points")