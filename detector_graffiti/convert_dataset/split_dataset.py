import os
import random
import shutil

import fire
from utils import image_extensions


def split_dataset(
    dataset_dir="dataset",
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42,
):
    """
    Делит датасет на train / val / test.
    Ожидается структура:
      dataset_dir/
        images/
        labels/
    """

    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Сумма коэффициентов должна быть равна 1.0"

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Не найдена папка с изображениями: {images_dir}"
        )
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(
            f"Не найдена папка с аннотациями: {labels_dir}"
        )

    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        raise RuntimeError("В папке images нет изображений")

    random.seed(seed)
    random.shuffle(image_files)

    total = len(image_files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    val_slice_end = n_train + n_val

    splits = {
        "train": image_files[:n_train],
        "valid": image_files[n_train:val_slice_end],
        "test": image_files[val_slice_end:],
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


if __name__ == "__main__":
    fire.Fire(split_dataset)
