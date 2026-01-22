import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from detector_graffiti.convert_dataset.utils import (
    get_image_dimensions,
    image_extensions,
    parse_yolo_annotation,
)

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def convert_yolo_to_json(config: DictConfig) -> None:
    """
    Converts YOLO format annotations to a single JSON file for each
    dataset split.
    Uses standard logging for progress tracking and error reporting.

    This function iterates through dataset splits (train, valid, test),
    reads YOLO .txt labels, fetches image dimensions, and compiles everything
    into a structured JSON format compatible with further processing.

    Args:
        config (DictConfig): Hydra configuration object containing:
            - data_loading.root: Relative path to the dataset root.
            - params.class_names: List of class names for mapping.

    Returns:
        None
    """
    # Ensure paths are project-relative
    project_root: Path = Path(get_original_cwd())
    dataset_path: Path = project_root / config.data_loading.root
    class_names = config.params.class_names

    # Process each split of the dataset
    for run_type in ["train", "valid", "test"]:
        images_directory: Path = dataset_path / run_type / "images"
        labels_directory: Path = dataset_path / run_type / "labels"
        output_json_path: Path = dataset_path / run_type / "annotations.json"

        if not images_directory.exists():
            logger.warning(
                f"Directory for {run_type} not found at {images_directory}"
            )
            continue

        # Filter files by allowed image extensions
        image_files: List[Path] = [
            f
            for f in images_directory.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        json_data: List[Dict[str, Any]] = []
        processed_count: int = 0

        logger.info(f"Found {len(image_files)} images for '{run_type}' split")

        for image_path in sorted(image_files):
            try:
                # Retrieve image resolution
                width, height = get_image_dimensions(image_path)

                # Map image file to its corresponding .txt annotation file
                annotation_path: Path = (
                    labels_directory / image_path.with_suffix(".txt").name
                )

                # Parse YOLO format to internal representation
                annotations: List[Dict[str, Any]] = parse_yolo_annotation(
                    annotation_path, class_names
                )

                json_data.append(
                    {
                        "image_name": image_path.name,
                        "width": width,
                        "height": height,
                        "annotations": annotations,
                    }
                )
                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} images...")

            except Exception as e:
                logger.error(f"Error processing image {image_path.name}: {e}")

        # Save the compiled results into a JSON file
        try:
            with output_json_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully processed {processed_count} images")
            logger.info(f"JSON file saved: {output_json_path}")

            # Calculate and display basic dataset statistics
            stats = {name: 0 for name in class_names}
            for item in json_data:
                labels = {ann["label_name"] for ann in item["annotations"]}
                # Using class_names from config for stats mapping
                for label in labels:
                    if label in stats:
                        stats[label] += 1

            logger.info(
                f"Statistics for {run_type}: "
                f"Graffiti: {stats.get(class_names[0])}, "
                f"Vandalism: {stats.get(class_names[1])}"
            )

        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}")


if __name__ == "__main__":
    convert_yolo_to_json()
