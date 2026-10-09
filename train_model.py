from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_split"

TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"


# ==========================================
# 2. SETTINGS
# ==========================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

CLASS_NAMES = ["minus", "multiply", "plus"]


# ==========================================
# 3. LOAD DATA
# ==========================================

train_dataset = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_dataset = keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nClasses:")
print(CLASS_NAMES)


# ==========================================
# 4. NORMALIZATION
# ==========================================

normalization_layer = layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)


# ==========================================
# 5. BASELINE CNN
# ==========================================

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dense(3, activation="softmax")
])


# ==========================================
# 6. COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 7. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 8. TRAIN
# ==========================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

import json

with open(BASE_DIR / "baseline_history.json", "w") as f:
    json.dump(history.history, f)

print("Baseline training history saved as baseline_history.json")


# ==========================================
# 9. TEST
# ==========================================

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# ==========================================
# 10. SAVE MODEL
# ==========================================

model.save(BASE_DIR / "baseline_cnn.keras")

print("\nModel saved as baseline_cnn.keras")