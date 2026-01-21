import json
from pathlib import Path

import fire
from utils import (
    class_names,
    get_image_dimensions,
    image_extensions,
    parse_yolo_annotation,
)


def convert_yolo_to_json(dataset_dir="datasets/dataset"):
    """
    Конвертирует YOLO аннотации в JSON формат.

    Args:
        dataset_dir (str): Путь к директории с датасетом.
    """
    base_path = Path(dataset_dir)

    for run_type in ["train", "valid", "test"]:
        images_directory = base_path / run_type / "images"
        labels_directory = base_path / run_type / "labels"
        output_json_path = base_path / run_type / "annotations.json"

        if not images_directory.exists():
            print(f"Нет {run_type} директории")
            continue

        # Получаем список изображений
        image_files = [
            f
            for f in images_directory.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        json_data = []
        processed_count = 0

        print(f"🔍 Найдено {len(image_files)} изображений для обработки")

        for image_path in sorted(image_files):
            try:
                width, height = get_image_dimensions(image_path)

                annotation_path = (
                    labels_directory / image_path.with_suffix(".txt").name
                )

                annotations = parse_yolo_annotation(annotation_path)

                image_data = {
                    "image_name": image_path.name,
                    "width": width,
                    "height": height,
                    "annotations": annotations,
                }

                json_data.append(image_data)
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"📊 Обработано {processed_count} изображений...")

            except Exception as e:
                print(
                    f"Ошибка при обработке изображения {image_path.name}: {e}"
                )

        # Сохраняем в JSON файл
        try:
            with output_json_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Успешно обработано {processed_count} изображений")
            print(f"💾 JSON файл сохранен: {output_json_path}")

            # Статистика
            images_with_graffiti = 0
            images_with_vandalism = 0

            for item in json_data:
                label_names = {
                    ann["label_name"] for ann in item["annotations"]
                }
                if class_names[0] in label_names:
                    images_with_graffiti += 1
                if class_names[1] in label_names:
                    images_with_vandalism += 1

            print(f"   Изображений с граффити: {images_with_graffiti}")
            print(f"   Изображений с вандализмом: {images_with_vandalism}")

        except Exception as e:
            print(f"❌ Ошибка при сохранении JSON файла: {e}")


if __name__ == "__main__":
    fire.Fire(convert_yolo_to_json)
