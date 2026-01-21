import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import yaml
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from utils import image_extensions

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def analyze_annotation(path: Path) -> Tuple[int, int]:
    """
    Analyzes a YOLO annotation file to count occurrences of each class.

    Args:
        path (Path): Path to the .txt annotation file.

    Returns:
        Tuple[int, int]: A tuple containing (count_class_0, count_class_1).
    """
    c0, c1 = 0, 0
    if not path.exists():
        return c0, c1
    try:
        with open(path, "r", encoding="utf-8") as f:
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
        logger.error(f"Error reading annotation {path}: {e}")
    return c0, c1


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def process_datasets(config: DictConfig) -> None:
    """
    Merges multiple YOLO datasets with class balancing based on
    bounding box counts.

    The function filters images by a percentage threshold, balances
    classes if enabled, and renames files using a global counter to
    avoid naming conflicts.

    Args:
        config (DictConfig): Hydra configuration object containing:
            - data_processing.paths.input_dir: Source data directory.
            - data_processing.paths.output_path: Target directory for
              merged dataset.
            - data_processing.params.run_type: Subset type (train, valid,
              or test).
            - data_processing.params.percentage: Fraction of data to
              sample (0.0 to 1.0).
            - data_processing.params.balance_classes: Whether to perform
              class balancing.

    Returns:
        None
    """
    project_root: Path = Path(get_original_cwd())

    # Resolve absolute paths using the original working directory
    datasets_path: Path = project_root / config.data_processing.paths.input_dir
    output_path: Path = project_root / config.data_processing.paths.output_path
    run_type: str = config.data_processing.params.run_type

    # Clean up output directory if it exists
    if output_path.exists() and output_path.is_dir():
        logger.info(f"Cleaning existing directory: {output_path}")
        shutil.rmtree(output_path)

    output_images_dir: Path = output_path / "images"
    output_labels_dir: Path = output_path / "labels"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Create YOLO metadata file (data.yaml)
    target_yaml_path: Path = output_path / "data.yaml"
    data_yaml: Dict[str, Any] = {
        "train": "./train/images",
        "val": "./valid/images",
        "test": "./test/images",
        "nc": 2,
        "names": ["graffiti", "vandalism"],
    }
    with open(target_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    global_counter: int = 0
    items: List[Dict[str, Any]] = []

    logger.info(f"Gathering bbox statistics for {run_type}...")

    if not datasets_path.exists():
        logger.error(f"Directory {datasets_path} not found")
        return

    # Iterate through each dataset folder and collect image/label pairs
    for dataset_folder in datasets_path.iterdir():
        if not dataset_folder.is_dir():
            continue

        images_dir: Path = dataset_folder / run_type / "images"
        labels_dir: Path = dataset_folder / run_type / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue

            ann_path: Path = labels_dir / (img_path.stem + ".txt")
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
        logger.warning("No images found.")
        return

    # Sample a percentage of the collected data
    num_to_select: int = max(
        1, int(len(items) * config.data_processing.params.percentage)
    )
    items = random.sample(items, num_to_select)

    logger.info(f"Candidates after percentage filtering: {len(items)}")

    # Class balancing logic based on bounding box counts
    if config.data_processing.params.balance_classes:
        random.shuffle(items)

        total_0: int = sum(item["c0"] for item in items)
        total_1: int = sum(item["c1"] for item in items)

        # Aim for an equal number of bboxes based on the minority class
        target: int = min(total_0, total_1)
        new_total_0: int = 0
        new_total_1: int = 0
        selected: List[Dict[str, Any]] = []

        for item in items:
            # Stop if both classes reached the target count
            if new_total_0 >= target and new_total_1 >= target:
                break

            add_flag: bool = False
            # Check if adding this image helps balance class 0 or class 1
            if item["c0"] > 0 and new_total_0 < target:
                add_flag = True
            if item["c1"] > 0 and new_total_1 < target:
                add_flag = True

            if add_flag:
                selected.append(item)
                new_total_0 += item["c0"]
                new_total_1 += item["c1"]
    else:
        selected = items

    # Log final distribution statistics
    final_0: int = sum(item["c0"] for item in selected)
    final_1: int = sum(item["c1"] for item in selected)
    logger.info(f"Selected images: {len(selected)}")
    logger.info("Final bbox statistics:")
    logger.info(f"   Graffiti (0): {final_0}")
    logger.info(f"   Vandalism (1): {final_1}")
    if final_0 and final_1:
        ratio: float = max(final_0 / final_1, final_1 / final_0)
        logger.info(f"   Ratio: {ratio:.2f}:1")

    # Final step: Copy files with new names to the output directory
    for item in selected:
        new_img_name: str = f"{global_counter:06d}{item['img_path'].suffix}"
        new_ann_name: str = f"{global_counter:06d}.txt"

        target_img_path: Path = output_images_dir / new_img_name
        target_ann_path: Path = output_labels_dir / new_ann_name

        shutil.copy2(item["img_path"], target_img_path)

        if item["ann_path"].exists():
            shutil.copy2(item["ann_path"], target_ann_path)
        else:
            # Create an empty file for images without annotations
            target_ann_path.touch()

        global_counter += 1

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    process_datasets()
