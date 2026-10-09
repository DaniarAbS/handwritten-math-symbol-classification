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
# 3. LIGHT DATA AUGMENTATION
# =========================

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.03),
], name="data_augmentation")

# =========================
# 4. IMPROVED CNN
# =========================

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Light augmentation
    data_augmentation,

    # Normalize pixels
    layers.Rescaling(1.0 / 255),

    # Same convolutional structure as baseline
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    # Regularization
    layers.Dropout(0.25),

    layers.Dense(num_classes, activation="softmax")
])

model.summary()

# =========================
# 5. COMPILE
# =========================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 6. CALLBACKS
# =========================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# =========================
# 7. TRAIN
# =========================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# =========================
# 8. FINAL TEST
# =========================

# IMPORTANT:
# The model contains Rescaling inside itself,
# so the raw test images are used here.

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("FINAL IMPROVED MODEL RESULTS")
print("==============================")
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# =========================
# 9. SAVE MODEL
# =========================

model.save(BASE_DIR / "improved_cnn.keras")

print("\nModel saved as improved_cnn.keras")