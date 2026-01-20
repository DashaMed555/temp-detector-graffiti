import json
import os

import fire
from utils import (
    class_names,
    get_image_dimensions,
    image_extensions,
    parse_yolo_annotation,
)


def convert_yolo_to_json(dataset_dir="dataset"):
    """
    Конвертирует YOLO аннотации в JSON формат.

    Args:
        dataset_dir (str): Путь к директории с датасетом.
    """
    for run_type in ["train", "valid", "test"]:
        images_directory = os.path.join(dataset_dir, run_type, "images")
        labels_directory = os.path.join(dataset_dir, run_type, "labels")
        output_json_path = os.path.join(
            dataset_dir, run_type, "annotations.json"
        )

        if not os.path.exists(images_directory):
            print(f"Нет {run_type} директории")
            continue

        # Получаем список изображений
        image_files = [
            f
            for f in os.listdir(images_directory)
            if f.lower().endswith(image_extensions)
        ]

        json_data = []
        processed_count = 0

        print(f"🔍 Найдено {len(image_files)} изображений для обработки")

        for image_file in sorted(image_files):
            try:
                image_path = os.path.join(images_directory, image_file)
                width, height = get_image_dimensions(image_path)

                annotation_file = os.path.splitext(image_file)[0] + ".txt"
                annotation_path = os.path.join(
                    labels_directory, annotation_file
                )

                annotations = parse_yolo_annotation(annotation_path)

                image_data = {
                    "image_name": image_file,
                    "width": width,
                    "height": height,
                    "annotations": annotations,
                }

                json_data.append(image_data)
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"📊 Обработано {processed_count} изображений...")

            except Exception as e:
                print(f"❌ Ошибка при обработке изображения {image_file}: {e}")

        # Сохраняем в JSON файл
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
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
