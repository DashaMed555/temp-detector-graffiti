import random
import shutil
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from utils import image_extensions


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def split_dataset(config: DictConfig):
    """
    Делит датасет на train / val / test.
    Ожидается структура:
      dataset_dir/
        images/
        labels/
    """
    project_root = Path(get_original_cwd())
    dataset_path = project_root / config.data_loading.root

    train_ratio = config.data_splitting.train_ratio
    val_ratio = config.data_splitting.val_ratio
    test_ratio = config.data_splitting.test_ratio

    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Сумма коэффициентов должна быть равна 1.0"

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка с изображениями: {images_dir}"
        )
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка с аннотациями: {labels_dir}"
        )

    image_files = [
        f.name
        for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise RuntimeError("В папке images нет изображений")

    random.seed(config.data_splitting.seed)
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
        split_dir = dataset_path / split

        if split_dir.exists():
            shutil.rmtree(split_dir)

        img_out = split_dir / "images"
        lbl_out = split_dir / "labels"

        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_name in files:
            src_img = images_dir / img_name
            dst_img = img_out / img_name
            shutil.copy2(src_img, dst_img)

            label_name = Path(img_name).with_suffix(".txt").name
            src_lbl = labels_dir / label_name
            dst_lbl = lbl_out / label_name

            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # Если аннотации нет — создаём пустой файл
                dst_lbl.touch()

    print("✅ Датасет успешно разделён:")
    for k, v in splits.items():
        print(f"   {k}: {len(v)} изображений")


if __name__ == "__main__":
    split_dataset()
