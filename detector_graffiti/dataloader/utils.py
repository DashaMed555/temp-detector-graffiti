from omegaconf import DictConfig


class DataCollator:
    """
    Data collator for preparing batches for Grounding DINO model training.

    Args:
        config (Optional[DictConfig]): Configuration object containing
        model parameters
        processor (Optional): Text and image processor for Grounding DINO
    """

    def __init__(self, config: DictConfig = None, processor=None):
        self.config = config
        self.processor = processor

    def __call__(self, batch):
        """
        Collate a batch of data items into model inputs.

        Args:
            batch (List[Dict[str, Any]]): List of data items, each containing:
                - image (Image): PIL Image object
                - size (Tuple[int, int]): Original image dimensions (h, w)
                - boxes (torch.Tensor): Bounding boxes in (cx, cy, w, h)
                format
                - class_labels (torch.Tensor): Class IDs for each box

        Returns:
            BatchEncoding: Dictionary containing:
                - model_inputs (Dict[str, torch.Tensor]): Processed model
                inputs
                - labels (List[Dict[str, torch.Tensor]]): Ground truth labels
                - orig_sizes (List[Tuple[int, int]]): Original image sizes
        """
        images = [b["image"] for b in batch]
        text_prompts = [self.config.fine_tuning.prompt] * len(images)

        enc = self.processor(
            images=images,
            text=text_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.model.max_length,
        )

        enc["model_inputs"] = {k: v for k, v in enc.items()}

        enc["labels"] = [
            {"class_labels": b["class_labels"], "boxes": b["boxes"]}
            for b in batch
        ]
        enc["orig_sizes"] = [b["size"] for b in batch]
        return enc
