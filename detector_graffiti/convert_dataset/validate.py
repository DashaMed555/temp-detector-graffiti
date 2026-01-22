import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def draw_bboxes_on_image(
    image_path: Union[str, Path],
    annotations: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    show_image: bool = True,
    class_names: List[str] = ["graffiti", "vandalism"],
) -> Optional[np.ndarray]:
    """
    Draws bounding boxes on an image based on provided annotations.

    Args:
        image_path: Path to the input image file.
        annotations: A list of dictionaries containing annotation data
            (cx, cy, w, h, label_name).
        output_path: Path where the resulting image will be saved.
            Defaults to None.
        show_image: Whether to display the image using matplotlib.
            Defaults to True.

    Returns:
        The image in RGB format as a numpy array if successful, None otherwise.
    """
    image_path = Path(image_path)
    # Define colors for specific classes (B, G, R format for OpenCV)
    colors = {class_names[0]: (255, 0, 0), class_names[1]: (0, 0, 255)}

    # Read image using OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error("Failed to read image: %s", image_path)
        return None

    # Convert to RGB for matplotlib and processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    for ann in annotations:
        # Convert normalized coordinates to pixel coordinates
        cx = ann["cx"] * img_width
        cy = ann["cy"] * img_height
        w = ann["w"] * img_width
        h = ann["h"] * img_height

        # Calculate box corners and clip to image boundaries
        x1 = int(np.clip(cx - w / 2, 0, img_width - 1))
        y1 = int(np.clip(cy - h / 2, 0, img_height - 1))
        x2 = int(np.clip(cx + w / 2, 0, img_width - 1))
        y2 = int(np.clip(cy + h / 2, 0, img_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        label = ann["label_name"]
        color = colors.get(label, (0, 255, 0))  # Default to green if not found

        # Draw rectangle and text label
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_rgb,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Save to disk if output_path is provided
    if output_path:
        output_path = Path(output_path)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_bgr)
        logger.info("Bbox image saved to: %s", output_path)

    # Display using matplotlib
    if show_image:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f"Image: {image_path.name}\nBBoxes: {len(annotations)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return img_rgb


def validate_with_visualization(config: DictConfig) -> None:
    """
    Performs visual validation of dataset annotations by drawing bboxes
    on random samples.

    Args:
        config: Hydra DictConfig object containing dataset and
            validation parameters.
    """
    # Get the original working directory since Hydra changes it
    project_root = Path(get_original_cwd())

    # Resolve paths from configuration
    dataset_path = project_root / config.data_loading.root
    run_type = config.data_validation.run_type
    num_samples = config.data_validation.num_samples
    class_names = config.params.class_names

    json_path = dataset_path / run_type / "annotations.json"
    images_dir = dataset_path / run_type / "images"

    # Setup output directory for visualizations if requested
    save_dir = None
    if config.data_validation.save_dir:
        save_dir = project_root / config.data_validation.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not json_path.exists():
            logger.error("Annotation file not found: %s", json_path)
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(
            "Visual validation of %d random samples from '%s'...",
            num_samples,
            run_type,
        )

        # Select random samples for visualization
        samples_to_check = random.sample(data, min(num_samples, len(data)))

        for i, item in enumerate(samples_to_check):
            image_name = item["image_name"]
            image_path = images_dir / image_name
            annotations = item["annotations"]

            logger.info(
                "Sample %d/%d: %s | Dimensions: %dx%d | BBoxes: %d",
                i + 1,
                len(samples_to_check),
                image_name,
                item["width"],
                item["height"],
                len(annotations),
            )

            if not image_path.exists():
                logger.warning("Image not found: %s", image_path)
                continue

            output_path = None
            if save_dir:
                output_name = f"visualization_{Path(image_name).stem}.png"
                output_path = save_dir / output_name

            # Log coordinates for the first annotation if available
            if annotations:
                first = annotations[0]
                logger.debug(
                    "First annotation: cx=%.4f, cy=%.4f, w=%.4f, h=%.4f",
                    first["cx"],
                    first["cy"],
                    first["w"],
                    first["h"],
                )

            # Draw and show
            draw_bboxes_on_image(
                image_path,
                annotations,
                output_path,
                show_image=True,
                class_names=class_names,
            )

            # Pause until Enter (skip for last)
            if i < len(samples_to_check) - 1:
                input("\n⌨️ Press Enter for the next image...")

    except Exception as e:
        logger.exception(f"Unexpected error during visual validation: {e}")


if __name__ == "__main__":
    validate_with_visualization()
