import random
import shutil
from pathlib import Path

import fire
import yaml
from utils import image_extensions


def process_datasets(
    datasets_dir="data",
    output_dataset_dir="datasets/dataset",
    run_type="train",
    percentage=0.01,
    balance_classes=True,
):
    """
    Объединяет YOLO-датасеты с балансировкой по количеству bbox.

    Args:
        datasets_dir (str): Путь к исходным данным.
        output_dataset_dir (str): Куда сохранять результат.
        run_type (str): Тип выборки (train, valid, test).
        percentage (float): Какую часть данных взять (0.0 до 1.0).
        balance_classes (bool): Нужно ли балансировать классы.
    """

    datasets_path = Path(datasets_dir)
    output_path = Path(output_dataset_dir)

    if output_path.exists() and output_path.is_dir():
        print(f"🧹 Очистка существующей директории: {output_dataset_dir}")
        shutil.rmtree(output_path)

    output_images_dir = output_path / "images"
    output_labels_dir = output_path / "labels"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    target_yaml_path = output_path / "data.yaml"
    data_yaml = {
        "train": "./train/images",
        "val": "./valid/images",
        "test": "./test/images",
        "nc": 2,
        "names": ["graffiti", "vandalism"],
    }
    with open(target_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    global_counter = 0

    def analyze_annotation(path):
        """Возвращает количество bbox по классам (0 и 1)"""
        c0, c1 = 0, 0
        if not path.exists():
            return c0, c1
        try:
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    if cls == 0:
                        c0 += 1
                    elif cls == 1:
                        c1 += 1
        except Exception as e:
            print(f"⚠️ Ошибка при чтении аннотации {path}: {e}")
        return c0, c1

    # ---------- Сбор всех изображений ----------
    items = []

    print(f"📊 Сбор bbox-статистики для {run_type}...")

    if not datasets_path.exists():
        print(f"❌ Ошибка: Директория {datasets_dir} не найдена")
        return

    for dataset_folder in datasets_path.iterdir():
        if not dataset_folder.is_dir():
            continue

        images_dir = dataset_folder / run_type / "images"
        labels_dir = dataset_folder / run_type / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue

            ann_path = labels_dir / (img_path.stem + ".txt")
            c0, c1 = analyze_annotation(ann_path)

            items.append(
                {
                    "img_path": img_path,
                    "ann_path": ann_path,
                    "c0": c0,
                    "c1": c1,
                }
            )

    if not items:
        print("⚠️  Изображения не найдены.")
        return

    num_to_select = max(1, int(len(items) * percentage))
    items = random.sample(items, num_to_select)

    print(
        f"📦 Кандидатов изображений после фильтрации по проценту: {len(items)}"
    )

    # ---------- Балансировка по bbox ----------
    if balance_classes:
        random.shuffle(items)

        total_0 = sum(item["c0"] for item in items)
        total_1 = sum(item["c1"] for item in items)

        target = min(total_0, total_1)
        new_total_0 = 0
        new_total_1 = 0
        selected = []

        for item in items:
            if new_total_0 >= target and new_total_1 >= target:
                break

            add = False

            if item["c0"] > 0 and new_total_0 < target:
                add = True
            if item["c1"] > 0 and new_total_1 < target:
                add = True

            if add:
                selected.append(item)
                new_total_0 += item["c0"]
                new_total_1 += item["c1"]
    else:
        selected = items

    # Итоговая статистика
    final_0 = sum(item["c0"] for item in selected)
    final_1 = sum(item["c1"] for item in selected)
    print(f"✅ Выбранных изображений: {len(selected)}")
    print("📊 Итоговая bbox-статистика:")
    print(f"   Граффити (0): {final_0}")
    print(f"   Вандализм (1): {final_1}")
    if final_0 and final_1:
        print(f"   Соотношение: {max(final_0/final_1, final_1/final_0):.2f}:1")

    # ---------- Копирование ----------
    for item in selected:
        new_img_name = f"{global_counter:06d}{item['img_path'].suffix}"
        new_ann_name = f"{global_counter:06d}.txt"

        target_img_path = output_images_dir / new_img_name
        target_ann_path = output_labels_dir / new_ann_name

        shutil.copy2(item["img_path"], target_img_path)

        # Копируем аннотацию только если она есть
        if item["ann_path"].exists():
            shutil.copy2(item["ann_path"], target_ann_path)
        else:
            target_ann_path.touch()

        global_counter += 1

    print("🚀 Обработка завершена")


if __name__ == "__main__":
    fire.Fire(process_datasets)
