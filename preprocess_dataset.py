from pathlib import Path
from PIL import Image, ImageOps

# Пути
BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "dataset_original"
OUTPUT_DIR = BASE_DIR / "dataset"

# Размер, который будет использовать CNN
IMAGE_SIZE = (128, 128)

# Наши классы
CLASSES = ["plus", "minus", "multiply"]


def resize_with_padding(image, target_size):
    """
    Resize image while keeping its proportions.
    Empty space is filled with white.
    """

    # Сохраняем пропорции и вписываем изображение в 128x128
    image.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Создаём белый квадрат 128x128
    canvas = Image.new("RGB", target_size, "white")

    # Вычисляем положение изображения по центру
    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2

    # Вставляем изображение
    canvas.paste(image, (x, y))

    return canvas


for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name
    output_folder = OUTPUT_DIR / class_name

    # Создаём папку, если её нет
    output_folder.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0

    print(f"\nProcessing: {class_name}")

    for file in sorted(source_folder.iterdir()):

        if file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        try:
            # Открываем изображение
            image = Image.open(file)

            # Переводим в RGB
            image = image.convert("RGB")

            # Resize + padding
            image = resize_with_padding(image, IMAGE_SIZE)

            # Сохраняем как JPEG
            output_file = output_folder / f"{file.stem}.jpg"
            image.save(output_file, "JPEG", quality=95)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"ERROR: {file.name} -> {e}")

    print(f"Processed: {processed}")
    print(f"Errors: {errors}")


print("\n==============================")
print("PREPROCESSING FINISHED")
print("==============================")