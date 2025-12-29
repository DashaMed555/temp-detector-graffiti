import torch
from transformers import Trainer

from .. import processor


class GroundingDINOTrainer(Trainer):
    def _build_model_inputs(self, batch, device):
        if "model_inputs" in batch:
            model_inputs = {
                k: v.to(device) for k, v in batch["model_inputs"].items()
            }
        else:
            allowed = (
                "input_ids",
                "token_type_ids",
                "attention_mask",
                "pixel_values",
                "pixel_mask",
            )
            model_inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in allowed and isinstance(v, torch.Tensor)
            }

        if "labels" in batch:
            labels_dev = []
            for item in batch["labels"]:
                cls = item["class_labels"]
                boxes = item["boxes"]
                if not isinstance(cls, torch.Tensor):
                    cls = torch.tensor(cls, dtype=torch.long)
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes, dtype=torch.float32)
                labels_dev.append(
                    {
                        "class_labels": cls.to(device),
                        "boxes": boxes.to(device),
                    }
                )
            model_inputs["labels"] = labels_dev

        return model_inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        model_inputs = self._build_model_inputs(inputs, device)
        outputs = model(**model_inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False):
        model.eval()
        device = model.device

        with torch.no_grad():
            loss, outputs = self.compute_loss(
                model, inputs, return_outputs=True
            )

        if prediction_loss_only:
            return (loss, None, None)

        if "model_inputs" in inputs:
            input_ids = inputs["model_inputs"]["input_ids"]
        else:
            input_ids = inputs["input_ids"]

        target_sizes = inputs.get("orig_sizes", None)

        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=input_ids,
            box_threshold=0.4,
            text_threshold=0.4,
            target_sizes=target_sizes,
        )

        preds = [
            torch.tensor(r["boxes"], dtype=torch.float32, device=device)
            for r in results
        ]

        W = target_sizes[0][1]
        H = target_sizes[0][0]
        labels = []
        for b in inputs["labels"]:
            boxes = b["boxes"]
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            cx, cy, bw, bh = boxes.unbind(-1)
            gt_boxes = torch.stack(
                [
                    (cx - bw / 2) * W,  # x1
                    (cy - bh / 2) * H,  # y1
                    (cx + bw / 2) * W,  # x2
                    (cy + bh / 2) * H,  # y2
                ],
                dim=-1,
            )
            labels.append(gt_boxes.to(device))
        count_boxes = torch.tensor(
            [preds[0].size()[0], labels[0].size()[0]],
            dtype=torch.float32,
            device=device,
        )
        return (loss, count_boxes, (preds, labels))
