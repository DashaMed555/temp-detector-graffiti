import os
from PIL import Image

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
class_names = ['graffiti', 'vandalism']

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
                        "label_name": class_names[class_id],
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
