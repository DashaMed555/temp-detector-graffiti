import json
import random
from pathlib import Path

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
from utils import class_names


def draw_bboxes_on_image(
    image_path, annotations, output_path=None, show_image=True
):
    """
    Рисует bounding boxes на изображении
    """
    image_path = Path(image_path)
    colors = {class_names[0]: (255, 0, 0), class_names[1]: (0, 0, 255)}

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Не удалось прочитать изображение: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    for ann in annotations:
        cx = ann["cx"] * img_width
        cy = ann["cy"] * img_height
        w = ann["w"] * img_width
        h = ann["h"] * img_height

        x1 = int(np.clip(cx - w / 2, 0, img_width - 1))
        y1 = int(np.clip(cy - h / 2, 0, img_height - 1))
        x2 = int(np.clip(cx + w / 2, 0, img_width - 1))
        y2 = int(np.clip(cy + h / 2, 0, img_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            img_rgb,
            (x1, y1),
            (x2, y2),
            colors.get(ann["label_name"], (0, 255, 0)),
            2,
        )
        cv2.putText(
            img_rgb,
            ann["label_name"],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors.get(ann["label_name"], (0, 255, 0)),
            2,
        )

    if output_path:
        output_path = Path(output_path)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_bgr)
        print(f"💾 Изображение с bbox сохранено: {output_path}")

    if show_image:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(
            (
                f"Image: {image_path.name}\n"
                f"BBoxes: {len(annotations)}"
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return img_rgb


def validate_with_visualization(
    dataset_dir="datasets/dataset", run_type="train", num_samples=5, save_dir=None
):
    """
    Визуальная проверка аннотаций с отрисовкой bounding boxes.

    Args:
        dataset_dir (str): Путь к папке с датасетом.
        run_type (str): Тип выборки ('train', 'valid' или 'test').
        num_samples (int): Количество случайных изображений для проверки.
        save_dir (str, optional): Папка для сохранения результатов.
    """
    dataset_path = Path(dataset_dir)
    json_path = dataset_path / run_type / "annotations.json"
    images_dir = dataset_path / run_type / "images"

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not json_path.exists():
            print(f"❌ Файл аннотаций не найден: {json_path}")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(
            f"🔍 Визуальная проверка {num_samples} "
            f"случайных примеров из {run_type}..."
        )

        # Выбираем случайные примеры
        samples_to_check = random.sample(data, min(num_samples, len(data)))

        for i, item in enumerate(samples_to_check):
            image_name = item["image_name"]
            image_path = images_dir / image_name
            annotations = item["annotations"]

            print(f"\n📋 Пример {i+1}/{len(samples_to_check)}:")
            print(f"   Изображение: {image_name}")
            print(f"   Размер: {item['width']}x{item['height']}")
            print(f"   Аннотаций: {len(annotations)}")

            if not image_path.exists():
                print(f"❌ Изображение не найдено: {image_path}")
                continue

            output_path = None
            if save_dir:
                output_name = f"visualization_{Path(image_name).stem}.png"
                output_path = save_dir / output_name

            if annotations:
                print("   Координаты первой аннотации (нормализованные):")
                print(f"     cx: {annotations[0]['cx']:.6f}")
                print(f"     cy: {annotations[0]['cy']:.6f}")
                print(f"     w: {annotations[0]['w']:.6f}")
                print(f"     h: {annotations[0]['h']:.6f}")

            draw_bboxes_on_image(
                image_path, annotations, output_path, show_image=True
            )

            if i < len(samples_to_check) - 1:
                msg = "\n⌨️  Нажмите Enter для просмотра следующего фото..."
                input(msg)

    except Exception as e:
        print(f"❌ Ошибка при визуальной проверке: {e}")


# Основная функция
if __name__ == "__main__":
    fire.Fire(validate_with_visualization)
