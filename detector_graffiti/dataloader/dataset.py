import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class JsonDataset(Dataset):
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

            image_name = os.path.join(self.image_path, entry["image_name"])

            self.items.append(
                {
                    "image_path": image_name,
                    "size": (h, w),
                    "boxes": boxes,
                    "class_labels": class_labels,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        image = Image.open(it["image_path"]).convert("RGB")
        return {"image": image, **it}
