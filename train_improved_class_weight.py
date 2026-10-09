import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# =========================
# 1. PATHS AND PARAMETERS
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_split"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
SEED = 42

# =========================
# 2. LOAD DATASETS
# =========================

train_dataset = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

validation_dataset = keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClasses:")
print(class_names)

# =========================
# 3. CALCULATE CLASS COUNTS
# =========================

class_counts = np.zeros(num_classes, dtype=int)

for images, labels in train_dataset:
    for label in labels.numpy():
        class_counts[label] += 1

print("\nTraining samples per class:")

for i, class_name in enumerate(class_names):
    print(f"{class_name}: {class_counts[i]}")

# =========================
# 4. CALCULATE CLASS WEIGHTS
# =========================

total_samples = np.sum(class_counts)

class_weights = {}

for i in range(num_classes):
    class_weights[i] = total_samples / (num_classes * class_counts[i])

print("\nClass weights:")

for i, class_name in enumerate(class_names):
    print(f"{class_name}: {class_weights[i]:.4f}")

# =========================
# 5. BUILD MODEL
# =========================

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # Same successful baseline architecture
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    # Small regularization
    layers.Dropout(0.2),

    layers.Dense(num_classes, activation="softmax")
])

model.summary()

# =========================
# 6. COMPILE
# =========================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 7. CALLBACKS
# =========================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)

# =========================
# 8. TRAIN
# =========================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

import json

with open(BASE_DIR / "improved_history.json", "w") as f:
    json.dump(history.history, f)

print("Improved training history saved as improved_history.json")

# =========================
# 9. FINAL TEST
# =========================

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("FINAL IMPROVED MODEL RESULTS")
print("==============================")
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# =========================
# 10. SAVE MODEL
# =========================

model.save(BASE_DIR / "improved_cnn.keras")

print("\nModel saved as improved_cnn.keras")