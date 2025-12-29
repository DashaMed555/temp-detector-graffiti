import json

from transformers import GroundingDinoForObjectDetection, TrainingArguments

from ..compute_metrics import compute_metrics
from ..dataloader.dataset import JsonDataset
from ..dataloader.utils import collate_fn
from ..grounding_dino_trainer import GroundingDINOTrainer

val_json_path = "dataset/dataset/train_annotations.json"
val_image_root = "dataset/dataset/images_train"
labels_list = []

with open(val_json_path, "r") as f:
    val_data = json.load(f)

labels = (
    labels_list
    if labels_list
    else sorted(
        {ann["label_name"] for d in val_data for ann in d["annotations"]}
    )
)
label2id = {c: i for i, c in enumerate(labels)}
id2label = {i: c for c, i in label2id.items()}

val_ds = JsonDataset(val_json_path, val_image_root, label2id)

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

model = GroundingDinoForObjectDetection.from_pretrained(
    "checkpoint-334",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
trainer_1 = GroundingDINOTrainer(
    model=model,
    args=args,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)
trainer_1.evaluate()
