import os
import shutil
import random
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
colors = {0: (255, 0, 0), 1: (0, 0, 255)}

def process_datasets(
    datasets_dir,
    output_dataset_dir,
    run_type,
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

    total_0 = 0
    total_1 = 0

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

            total_0 += c0
            total_1 += c1

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
    print("📊 Итоговая bbox-статистика:")
    print(f"   Граффити (0): {total_0}")
    print(f"   Вандализм (1): {total_1}")
    if total_0 and total_1:
        print(f"   Соотношение: {max(total_0/total_1, total_1/total_0):.2f}:1")

    # ---------- Копирование ----------
    src_yaml_path = os.path.join(dataset_path, "data.yaml")
    target_yaml_path = os.path.join(output_dataset_dir, "data.yaml")
    if os.path.exists(src_yaml_path):
        shutil.copy2(src_yaml_path, target_yaml_path)
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

def split_dataset(
    dataset_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """
    Делит датасет на train / val / test.
    Ожидается структура:
      dataset_dir/
        images/
        labels/
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Сумма коэффициентов должна быть равна 1.0"

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Не найдена папка с изображениями: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Не найдена папка с аннотациями: {labels_dir}")

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        raise RuntimeError("В папке images нет изображений")

    random.seed(seed)
    random.shuffle(image_files)

    total = len(image_files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "valid": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:]
    }

    for split, files in splits.items():
        img_out = os.path.join(dataset_dir, split, "images")
        lbl_out = os.path.join(dataset_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_name in files:
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(img_out, img_name)
            shutil.copy2(src_img, dst_img)

            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(labels_dir, label_name)
            dst_lbl = os.path.join(lbl_out, label_name)

            # Если аннотации нет — создаём пустой файл
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()

    print("✅ Датасет успешно разделён:")
    for k, v in splits.items():
        print(f"   {k}: {len(v)} изображений")

    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)

def parse_yolo_annotation(annotation_path):
    """
    Парсит YOLO аннотацию и возвращает список bbox'ов
    """
    annotations = []
    
    if not os.path.exists(annotation_path):
        return annotations
    
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    if class_id not in (0, 1):
                        continue

                    if w <= 0 or h <= 0:
                        continue
                    
                    annotations.append({
                        "class_id": class_id,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h
                    })
    
    except Exception as e:
        print(f"❌ Ошибка при чтении файла {annotation_path}: {e}")
    
    return annotations

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except:
        return 640, 480

def convert_yolo_to_json(dataset_dir):
    """
    Конвертирует YOLO аннотации в JSON формат
    """
    for run_type in ['train', 'valid', 'test']:
        images_directory = os.path.join(dataset_dir, run_type, 'images')
        labels_directory = os.path.join(dataset_dir, run_type, 'labels')
        output_json_path = os.path.join(dataset_dir, run_type, 'annotations.json')

        if not os.path.exists(images_directory):
            print(f'Нет {run_type} директории')
            continue
        
        # Получаем список изображений
        image_files = [f for f in os.listdir(images_directory) if f.lower().endswith(image_extensions)]
        
        json_data = []
        processed_count = 0
        
        print(f"🔍 Найдено {len(image_files)} изображений для обработки")
        
        for image_file in sorted(image_files):
            try:
                image_path = os.path.join(images_directory, image_file)
                width, height = get_image_dimensions(image_path)

                annotation_file = os.path.splitext(image_file)[0] + '.txt'
                annotation_path = os.path.join(labels_directory, annotation_file)
                
                annotations = parse_yolo_annotation(annotation_path)
                
                image_data = {
                    "image_name": image_file,
                    "width": width,
                    "height": height,
                    "annotations": annotations
                }
                
                json_data.append(image_data)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"📊 Обработано {processed_count} изображений...")
                    
            except Exception as e:
                print(f"❌ Ошибка при обработке изображения {image_file}: {e}")
        
        # Сохраняем в JSON файл
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Успешно обработано {processed_count} изображений")
            print(f"💾 JSON файл сохранен: {output_json_path}")
            
            # Статистика
            images_with_graffiti = 0
            images_with_vandalism = 0

            for item in json_data:
                classes = {ann['class_id'] for ann in item['annotations']}
                if 0 in classes:
                    images_with_graffiti += 1
                if 1 in classes:
                    images_with_vandalism += 1

            print(f"   Изображений с граффити: {images_with_graffiti}")
            print(f"   Изображений с вандализмом: {images_with_vandalism}")
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении JSON файла: {e}")

def draw_bboxes_on_image(image_path, annotations, output_path=None, show_image=True):
    """
    Рисует bounding boxes на изображении
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Не удалось прочитать изображение: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    
    for ann in annotations:
        cx = ann['cx'] * img_width
        cy = ann['cy'] * img_height
        w = ann['w'] * img_width
        h = ann['h'] * img_height

        x1 = int(np.clip(cx - w / 2, 0, img_width - 1))
        y1 = int(np.clip(cy - h / 2, 0, img_height - 1))
        x2 = int(np.clip(cx + w / 2, 0, img_width - 1))
        y2 = int(np.clip(cy + h / 2, 0, img_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), colors.get(ann['class_id'], (0,255,0)), 2)
        cv2.putText(img_rgb, str(ann['class_id']), (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(ann['class_id'], (0,255,0)), 2)
    
    if output_path:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_bgr)
        print(f"💾 Изображение с bbox сохранено: {output_path}")
    
    if show_image:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f"Image: {os.path.basename(image_path)}\nBBoxes: {len(annotations)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return img_rgb

def validate_with_visualization(dataset_dir, run_type='train', num_samples=5, save_dir=None):
    """
    Визуальная проверка аннотаций с отрисовкой bounding boxes
    """
    json_path = os.path.join(dataset_dir, run_type, 'annotations.json')
    images_dir = os.path.join(dataset_dir, run_type, 'images')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"🔍 Визуальная проверка {num_samples} случайных примеров...")
        
        # Выбираем случайные примеры
        samples_to_check = random.sample(data, min(num_samples, len(data)))
        
        for i, item in enumerate(samples_to_check):
            image_name = item['image_name']
            image_path = os.path.join(images_dir, image_name)
            annotations = item['annotations']
            
            print(f"\n📋 Пример {i+1}/{len(samples_to_check)}:")
            print(f"   Изображение: {image_name}")
            print(f"   Размер: {item['width']}x{item['height']}")
            print(f"   Аннотаций: {len(annotations)}")
            
            if not os.path.exists(image_path):
                print(f"❌ Изображение не найдено: {image_path}")
                continue
            
            output_path = None
            if save_dir:
                output_name = f"visualization_{os.path.splitext(image_name)[0]}.png"
                output_path = os.path.join(save_dir, output_name)
            
            if annotations:
                print(f"   Координаты первой аннотации:")
                print(f"     cx: {annotations[0]['cx']:.6f}")
                print(f"     cy: {annotations[0]['cy']:.6f}")
                print(f"     w: {annotations[0]['w']:.6f}")
                print(f"     h: {annotations[0]['h']:.6f}")
            
            draw_bboxes_on_image(image_path, annotations, output_path, show_image=True)
            
            if i < len(samples_to_check) - 1:
                input("Нажмите Enter для просмотра следующего изображения...")
    
    except Exception as e:
        print(f"❌ Ошибка при визуальной проверке: {e}")

# Основная функция
if __name__ == "__main__":
    run_type = 'train'

    # Настройки путей
    datasets_directory = "data"
    output_dataset_dir = "dataset"

    # Запуск обработки с балансировкой
    process_datasets(
        datasets_directory,
        output_dataset_dir,
        run_type,
        balance_classes=True
    )

    split_dataset(output_dataset_dir)

    # Конвертация YOLO в JSON
    print("🔄 Конвертация YOLO → JSON...")
    convert_yolo_to_json(output_dataset_dir)
    
    # # Визуальная проверка
    print("\n🎨 Визуальная проверка результатов...") 
    validate_with_visualization(output_dataset_dir, num_samples=3)
