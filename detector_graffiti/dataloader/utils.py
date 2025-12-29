from .. import processor

text_prompt = "graffiti ."


def collate_fn(batch):
    images = [b["image"] for b in batch]
    text_prompts = [text_prompt] * len(images)

    enc = processor(
        images=images,
        text=text_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )

    enc["model_inputs"] = {k: v for k, v in enc.items()}

    enc["labels"] = [
        {"class_labels": b["class_labels"], "boxes": b["boxes"]} for b in batch
    ]
    enc["orig_sizes"] = [b["size"] for b in batch]
    return enc
