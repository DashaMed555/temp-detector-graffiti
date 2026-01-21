import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class JsonDataset(Dataset):
    """
    Dataset for loading images and annotations from JSON format
      for object detection.

    Args:
        json_path (str): Path to JSON file containing image annotations
        image_path (str): Directory path containing images
        label2id (Dict[str, int]): Mapping from class names to integer IDs
    """

    def __init__(self, json_path, image_path, label2id):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.image_path = image_path
        self.items = []

        for entry in data:
            w, h = entry["width"], entry["height"]
            boxes = []
            labels = []

            for ann in entry["annotations"]:
                boxes.append([ann["cx"], ann["cy"], ann["w"], ann["h"]])
                labels.append(label2id[str(ann["label_name"])])

            if len(boxes) > 0:
                boxes_cxcywh = torch.tensor(boxes, dtype=torch.float32)
                cx, cy, bw, bh = boxes_cxcywh.unbind(dim=-1)
                boxes = torch.stack([cx, cy, bw, bh], dim=-1)
                class_labels = torch.tensor(labels, dtype=torch.long)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.zeros((0,), dtype=torch.long)

            image_name = Path(self.image_path) / entry["image_name"]

            self.items.append(
                {
                    "image_path": str(image_name),
                    "size": (h, w),
                    "boxes": boxes,
                    "class_labels": class_labels,
                }
            )

    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of images in the dataset
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            Dict[str, Any]: Dictionary containing:
                - image (Image): PIL Image object in RGB format
                - image_path (str): Path to the image file
                - size (Tuple[int, int]): (height, width) of the image
                - boxes (torch.Tensor): Bounding boxes in (cx, cy, w, h)
                  format, shape (N, 4)
                - class_labels (torch.Tensor): Class IDs for each box,
                  shape (N,)
        """
        it = self.items[idx]
        image = Image.open(it["image_path"]).convert("RGB")
        return {"image": image, **it}
