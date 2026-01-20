from omegaconf import DictConfig


class DataCollator:
    def __init__(self, config: DictConfig = None, processor=None):
        self.config = config
        self.processor = processor

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        text_prompts = [self.config.text_prompt] * len(images)

        enc = self.processor(
            images=images,
            text=text_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )

        enc["model_inputs"] = {k: v for k, v in enc.items()}

        enc["labels"] = [
            {"class_labels": b["class_labels"], "boxes": b["boxes"]}
            for b in batch
        ]
        enc["orig_sizes"] = [b["size"] for b in batch]
        return enc
