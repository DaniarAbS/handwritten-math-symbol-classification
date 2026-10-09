
from pathlib import Path
from collections import Counter
from PIL import Image

# Папка с оригинальными изображениями
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

classes = ["plus", "minus", "multiply"]

total_images = 0

for class_name in classes:
    folder = DATASET_DIR / class_name

    print(f"\n{'=' * 45}")
    print(f"CLASS: {class_name}")
    print(f"{'=' * 45}")

    if not folder.exists():
        print("ERROR: Folder not found!")
        continue

    image_sizes = Counter()
    image_formats = Counter()
    corrupted = []
    image_count = 0

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]:
            continue

        try:
            with Image.open(file) as img:
                img.load()
                image_sizes[img.size] += 1
                image_formats[img.format] += 1
                image_count += 1

        except Exception as e:
            corrupted.append((file.name, str(e)))

    total_images += image_count

    print(f"Valid images: {image_count}")
    print(f"Formats: {dict(image_formats)}")

    print("\nImage sizes:")
    for size, count in image_sizes.most_common():
        print(f"  {size[0]} x {size[1]}: {count} images")

    if corrupted:
        print("\nCorrupted images:")
        for name, error in corrupted:
            print(f"  {name}: {error}")
    else:
        print("\nCorrupted images: 0")

print(f"\n{'=' * 45}")
print(f"TOTAL VALID IMAGES: {total_images}")
print(f"{'=' * 45}")
