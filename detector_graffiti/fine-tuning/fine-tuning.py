import json
import os

from transformers import (
    GroundingDinoForObjectDetection,
    TrainingArguments,
    set_seed,
)

from .. import processor
from ..compute_metrics import compute_metrics
from ..dataloader.dataset import JsonDataset
from ..dataloader.utils import collate_fn
from ..freeze_layers import freeze_layers
from ..grounding_dino_trainer import GroundingDINOTrainer

model_id = "IDEA-Research/grounding-dino-base"
train_json_path = "dataset/dataset/train_annotations.json"
val_json_path = "dataset/dataset/train_annotations.json"
train_image_root = "dataset/dataset/images_train"
val_image_root = "dataset/dataset/images_train"
seed = 42
output_dir = "runs/gdino-trainer1"
labels_list = []


os.makedirs(output_dir, exist_ok=True)
set_seed(seed)

with open(train_json_path, "r") as f:
    train_data = json.load(f)

    if labels_list:
        labels = labels_list
    else:
        sorted(
            {ann["label_name"] for d in train_data for ann in d["annotations"]}
        )
label2id = {c: i for i, c in enumerate(labels)}
id2label = {i: c for c, i in label2id.items()}

train_ds = JsonDataset(train_json_path, train_image_root, label2id)
val_ds = JsonDataset(val_json_path, val_image_root, label2id)

model = GroundingDinoForObjectDetection.from_pretrained(
    model_id,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
freeze_layers(model)
text_prompt = " . ".join(labels) + " ."

args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=128,
    gradient_accumulation_steps=32,
    num_train_epochs=100,
    learning_rate=1e-5,
    eval_strategy="epoch",
    eval_on_start=True,
    remove_unused_columns=False,
    weight_decay=3e-6,
    adam_beta2=0.999,
    optim="adamw_torch",
    save_strategy="best",
    load_best_model_at_end=True,
    bf16=True,
    dataloader_pin_memory=False,
    logging_dir="./logs",
    report_to="tensorboard",
    logging_strategy="epoch",
    metric_for_best_model="f1",
    greater_is_better=True,
    lr_scheduler_type="cosine",
)

trainer = GroundingDINOTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
