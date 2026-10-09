from pathlib import Path
import random
import shutil

# Project folders
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "dataset"

# Where the split dataset will be stored
OUTPUT_DIR = BASE_DIR / "data_split"

# Classes
CLASSES = ["plus", "minus", "multiply"]

# Reproducible random split
random.seed(42)

# Split proportions
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name

    # Find images
    images = [
        file
        for file in source_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    # Shuffle
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    print(f"\nClass: {class_name}")
    print(f"Total:      {total}")
    print(f"Train:      {len(train_images)}")
    print(f"Validation: {len(validation_images)}")
    print(f"Test:       {len(test_images)}")

    # Create folders
    for split in ["train", "validation", "test"]:
        folder = OUTPUT_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

    # Copy files
    for file in train_images:
        shutil.copy2(
            file,
            OUTPUT_DIR / "train" / class_name / file.name
        )

    for file in validation_images:
        shutil.copy2(
            file,
            OUTPUT_DIR / "validation" / class_name / file.name
        )

    for file in test_images:
        shutil.copy2(
            file,
            OUTPUT_DIR / "test" / class_name / file.name
        )


print("\n==============================")
print("DATASET SPLIT FINISHED")
print("==============================")