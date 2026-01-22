import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List

from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from detector_graffiti.convert_dataset.utils import IMAGE_EXTENSIONS

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def split_dataset(config: DictConfig) -> None:
    """
    Splits the dataset into train, validation, and test sets based on
    config ratios.

    Expected directory structure:
        dataset_dir/
            images/
            labels/

    The function creates separate folders for each split (train/valid/test) and
    copies the corresponding image and label files into them.

    Args:
        config (DictConfig): Hydra configuration object containing:
            - data_loading.root (str): Relative path to the dataset root.
            - data_splitting.train_ratio (float): Proportion of training data.
            - data_splitting.val_ratio (float): Proportion of validation data.
            - data_splitting.test_ratio (float): Proportion of test data.
            - data_splitting.seed (int): Random seed for reproducibility.

    Returns:
        None

    Raises:
        FileNotFoundError: If images or labels directories are missing.
        RuntimeError: If no images are found in the source directory.
        AssertionError: If the sum of ratios is not equal to 1.0.
    """
    # Use get_original_cwd() to handle paths relative to the project root
    # since Hydra changes the current working directory.
    project_root: Path = Path(get_original_cwd())
    dataset_path: Path = project_root / config.data_loading.root

    logger.info(f"Starting dataset split in: {dataset_path}")

    # Load split ratios from the configuration
    train_ratio: float = config.data_splitting.train_ratio
    val_ratio: float = config.data_splitting.val_ratio
    test_ratio: float = config.data_splitting.test_ratio

    # Validate that ratios sum up to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        error_msg = f"Sum of ratios ({total_ratio}) must be 1.0"
        logger.error(error_msg)
        raise AssertionError(error_msg)

    # Define source directories
    images_dir: Path = dataset_path / "images"
    labels_dir: Path = dataset_path / "labels"

    # Verify source folders exist
    if not images_dir.is_dir():
        logger.error(f"Images directory not found: {images_dir}")
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        logger.error(f"Labels directory not found: {labels_dir}")
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Collect all image filenames with supported extensions
    image_files: List[str] = [
        f.name
        for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        logger.error(f"No images found in {images_dir}")
        raise RuntimeError(f"No images found in {images_dir}")

    # Set seed for reproducible shuffling
    random.seed(config.data_splitting.seed)
    random.shuffle(image_files)

    # Calculate split indices
    total: int = len(image_files)
    n_train: int = int(total * train_ratio)
    n_val: int = int(total * val_ratio)
    val_slice_end: int = n_train + n_val

    # Group files into dictionary for easier iteration
    splits: Dict[str, List[str]] = {
        "train": image_files[:n_train],
        "valid": image_files[n_train:val_slice_end],
        "test": image_files[val_slice_end:],
    }

    # Process each split: clean destination and copy files
    for split_name, files in splits.items():
        split_dir: Path = dataset_path / split_name
        logger.info(f"Processing split '{split_name}' with {len(files)} files")

        # Clear existing split directory to ensure a clean split
        if split_dir.exists():
            logger.debug(f"Removing existing directory: {split_dir}")
            shutil.rmtree(split_dir)

        # Create subdirectories for images and labels
        img_out: Path = split_dir / "images"
        lbl_out: Path = split_dir / "labels"

        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_name in files:
            # Copy image file
            src_img: Path = images_dir / img_name
            dst_img: Path = img_out / img_name
            shutil.copy2(src_img, dst_img)

            # Determine label filename (assumes .txt extension for YOLO)
            label_name: str = Path(img_name).with_suffix(".txt").name
            src_lbl: Path = labels_dir / label_name
            dst_lbl: Path = lbl_out / label_name

            # Copy label if it exists, otherwise create an empty file
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # YOLO format requires a file even for empty annotations
                logger.warning(
                    f"Label not found for {img_name}, creating empty file"
                )
                dst_lbl.touch()

    # Log final results
    logger.info("✅ Dataset splitting completed successfully")
    for split, files_list in splits.items():
        logger.info(f"   {split}: {len(files_list)} images")


if __name__ == "__main__":
    split_dataset()
