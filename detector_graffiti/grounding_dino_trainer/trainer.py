import torch
from omegaconf import DictConfig
from transformers import Trainer


class GroundingDINOTrainer(Trainer):
    """
    Custom trainer for Grounding DINO model with specialized loss computation
    and prediction handling for object detection tasks.

    Args:
        config (DictConfig): Model configuration parameters
        processor: Text and image processor for Grounding DINO
        **kwargs: Additional arguments passed to base Trainer class
    """

    def __init__(self, config: DictConfig = None, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.config = config

    def _build_model_inputs(self, batch, device):
        """
        Prepare model inputs from batch.

        Args:
            batch (Dict[str, Any]): Batch data with inputs and labels
            device (torch.device): Target device for tensors

        Returns:
            Dict[str, Any]: Model inputs ready for forward pass
        """
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

    def compute_loss(
        self, model, inputs, return_outputs=False, *args, **kwargs
    ):
        """
        Compute loss for batch.

        Args:
            model (torch.nn.Module): Grounding DINO model
            inputs (Dict[str, Any]): Batch data
            return_outputs (bool): Return model outputs with loss

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
                Loss tensor or (loss, outputs) tuple
        """
        device = model.device
        model_inputs = self._build_model_inputs(inputs, device)
        outputs = model(**model_inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        """
        Run prediction and format outputs for metrics.

        Args:
            model (torch.nn.Module): Grounding DINO model
            inputs (Dict[str, Any]): Batch data
            prediction_loss_only (bool): Only return loss
            ignore_keys (Optional[List[str]]): Keys to ignore

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
                (loss, count_boxes, (pred_boxes, gt_boxes))
                - loss (torch.Tensor): scalar tensor
                - count_boxes (Optional[torch.Tensor]):
                tensor with [pred_count, gt_count]
                - pred_boxes (List[torch.Tensor]):
                list of predicted boxes per image
                - gt_boxes (List[torch.Tensor]):
                list of ground truth boxes per image
        """
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

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=input_ids,
            threshold=self.config.threshold,
            target_sizes=target_sizes,
        )

        preds = [
            r["boxes"].detach().clone().to(device=device, dtype=torch.float32)
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
