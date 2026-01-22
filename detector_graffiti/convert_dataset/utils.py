import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Global constant for supported image formats
IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def parse_yolo_annotation(
    annotation_path: Union[str, Path], class_names: Optional[List[str]] = None
) -> List[Dict[str, Union[str, float]]]:
    """
    Parses a YOLO format annotation file into a list of
    structured dictionaries.

    Args:
        annotation_path (Union[str, Path]): Path to the .txt
            annotation file.
        class_names (Optional[List[str]]): List of class labels
            corresponding to IDs.
            Defaults to ["graffiti", "vandalism"] if None.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of annotations
            where each item contains 'label_name', 'cx', 'cy', 'w', and 'h'.
    """
    if class_names is None:
        class_names = ["graffiti", "vandalism"]

    annotations: List[Dict[str, Union[str, float]]] = []
    path = Path(annotation_path)

    # Return empty list if the annotation file is missing
    if not path.exists():
        # Some images might intentionally have no labels
        logger.debug(f"Annotation file missing: {path}")
        return annotations

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # YOLO format expects: class_id, center_x, center_y, width, height
            if len(parts) >= 5:
                try:
                    class_id = int(parts[0])

                    if class_id < 0 or class_id >= len(class_names):
                        logger.warning(
                            f"Invalid class_id {class_id} in {path}. "
                            f"Valid range: 0-{len(class_names)-1}"
                        )
                        continue

                    cx, cy, w, h = map(float, parts[1:5])

                    if w <= 0 or h <= 0:
                        logger.warning(
                            f"Non-positive dimensions in {path}: w={w}, h={h}"
                        )
                        continue

                    annotations.append(
                        {
                            "label_name": class_names[class_id],
                            "cx": cx,
                            "cy": cy,
                            "w": w,
                            "h": h,
                        }
                    )
                except ValueError as ve:
                    logger.error(f"Value error in {path} line '{line}': {ve}")

    except Exception as e:
        logger.error(
            f"Failed to read annotation file {path}: {e}", exc_info=True
        )

    return annotations


def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Retrieves the width and height of an image file.

    Args:
        image_path (Union[str, Path]): Path to the image file.

    Returns:
        Tuple[int, int]: A tuple containing (width, height).
            Returns (640, 480) as a fallback if the file cannot be read.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception as e:
        logger.warning(
            (
                f"Could not determine dimensions for {path}: {e}. "
                "Using fallback 640x480."
            )
        )
        return 640, 480
