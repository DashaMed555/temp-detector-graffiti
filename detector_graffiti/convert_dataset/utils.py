from pathlib import Path
from typing import Union

import fire
from PIL import Image

image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
class_names = ["graffiti", "vandalism"]


def parse_yolo_annotation(annotation_path: Union[str, Path]):
    """
    Парсит YOLO аннотацию и возвращает список bbox'ов.

    Args:
        annotation_path (str | Path): Путь к текстовому файлу аннотации.
    """
    annotations = []
    path = Path(annotation_path)

    if not path.exists():
        return annotations

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    if class_id not in (0, 1):
                        continue

                    if w <= 0 or h <= 0:
                        continue

                    annotations.append(
                        {
                            "label_name": class_names[class_id],
                            "cx": cx,
                            "cy": cy,
                            "w": w,
                            "h": h,
                        }
                    )

    except Exception as e:
        print(f"❌ Ошибка при чтении файла {path}: {e}")

    return annotations


def get_image_dimensions(image_path: Union[str, Path]):
    """
    Возвращает ширину и высоту изображения.

    Args:
        image_path (str | Path): Путь к файлу изображения.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"⚠️ Не удалось определить размер {path}: {e}")
        return 640, 480


if __name__ == "__main__":
    fire.Fire()
