import os
import random
import shutil
import yaml

from utils import image_extensions


def process_datasets(
    datasets_dir,
    output_dataset_dir,
    run_type,
    percentage=1.0,
    balance_classes=True
):
    """
    Объединяет YOLO-датасеты с балансировкой по количеству bbox.
    Классы:
      0 — граффити
      1 — вандализм
    """

    output_images_dir = os.path.join(output_dataset_dir, "images")
    output_labels_dir = os.path.join(output_dataset_dir, "labels")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    target_yaml_path = os.path.join(output_dataset_dir, "data.yaml")
    data_yaml = {
        'train': './train/images',
        'val': './valid/images',
        'test': './test/images',
        'nc': 2,
        'names': ['graffiti', 'vandalism']
    }
    with open(target_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    global_counter = 0

    def analyze_annotation(path):
        """Возвращает количество bbox по классам"""
        c0, c1 = 0, 0
        if not os.path.exists(path):
            return c0, c1
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                    if cls == 0:
                        c0 += 1
                    elif cls == 1:
                        c1 += 1
                except:
                    continue
        return c0, c1

    # ---------- Сбор всех изображений ----------
    items = []

    print("📊 Сбор bbox-статистики...")

    for dataset_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        images_dir = os.path.join(dataset_path, run_type, "images")
        labels_dir = os.path.join(dataset_path, run_type, "labels")

        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            continue

        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(image_extensions):
                continue

            base = os.path.splitext(img_name)[0]
            ann_path = os.path.join(labels_dir, base + ".txt")
            c0, c1 = analyze_annotation(ann_path)

            items.append({
                "img_path": os.path.join(images_dir, img_name),
                "ann_path": ann_path,
                "c0": c0,
                "c1": c1
            })

    num_to_select = max(1, int(len(items) * percentage))
    items = random.sample(items, num_to_select)

    total_0 = 0
    total_1 = 0
    for item in items:
        total_0 += item['c0']
        total_1 += item['c1']

    print(f"📦 Кандидатов изображений: {len(items)}")

    # ---------- Балансировка по bbox ----------
    if balance_classes:
        random.shuffle(items)

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
    total_0 = sum(item["c0"] for item in selected)
    total_1 = sum(item["c1"] for item in selected)
    print(f"Выбранных изображений: {len(selected)}")
    print("📊 Итоговая bbox-статистика:")
    print(f"   Граффити (0): {total_0}")
    print(f"   Вандализм (1): {total_1}")
    if total_0 and total_1:
        print(f"   Соотношение: {max(total_0/total_1, total_1/total_0):.2f}:1")

    # ---------- Копирование ----------
    for item in selected:
        img_ext = os.path.splitext(item["img_path"])[1]

        new_img_name = f"{global_counter:06d}{img_ext}"
        new_ann_name = f"{global_counter:06d}.txt"

        target_img_path = os.path.join(output_images_dir, new_img_name)
        target_ann_path = os.path.join(output_labels_dir, new_ann_name)

        shutil.copy2(item["img_path"], target_img_path)

        # Копируем аннотацию только если она есть
        if os.path.exists(item["ann_path"]):
            shutil.copy2(item["ann_path"], target_ann_path)
        else:
            open(target_ann_path, "w").close()

        global_counter += 1

    print("✅ Балансировка по bbox завершена")
    print(f"📁 Изображения: {output_images_dir}")
    print(f"📁 Аннотации: {output_labels_dir}")


if __name__ == '__main__':
    run_type = 'train'

    # Настройки путей
    datasets_directory = "data"
    output_dataset_dir = "dataset"
    percentage = 0.01

    # Запуск обработки с балансировкой
    process_datasets(
        datasets_directory,
        output_dataset_dir,
        run_type,
        percentage,
        balance_classes=True
    )
