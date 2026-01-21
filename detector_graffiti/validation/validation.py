import datetime
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    TrainingArguments,
)

from ..compute_metrics.compute_metrics import compute_metrics
from ..dataloader.dataset import JsonDataset
from ..dataloader.utils import DataCollator
from ..grounding_dino_trainer.trainer import GroundingDINOTrainer


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: DictConfig):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir = Path(config.validation.output_dir) / current_time
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config.data_loading.val_json_path, "r") as f:
        val_data = json.load(f)

    labels = sorted(
        {ann["label_name"] for d in val_data for ann in d["annotations"]}
    )
    label2id = {c: i for i, c in enumerate(labels)}
    id2label = {i: c for c, i in label2id.items()}

    val_ds = JsonDataset(
        config.data_loading.val_json_path,
        config.data_loading.val_image_path,
        label2id,
    )

    processor = AutoProcessor.from_pretrained(config.model.model_id)

    config_val = config.validation

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config_val.per_device_train_batch_size,
        per_device_eval_batch_size=config_val.per_device_eval_batch_size,
        eval_accumulation_steps=config_val.eval_accumulation_steps,
        gradient_accumulation_steps=config_val.gradient_accumulation_steps,
        num_train_epochs=config_val.num_train_epochs,
        learning_rate=config_val.learning_rate,
        eval_strategy=config_val.eval_strategy,
        eval_on_start=config_val.eval_on_start,
        remove_unused_columns=config_val.remove_unused_columns,
        weight_decay=config_val.weight_decay,
        adam_beta2=config_val.adam_beta2,
        optim=config_val.optim,
        report_to="none",
        save_strategy=config_val.save_strategy,
        load_best_model_at_end=config_val.load_best_model_at_end,
        dataloader_pin_memory=config_val.dataloader_pin_memory,
        logging_strategy=config_val.logging_strategy,
        metric_for_best_model=config_val.metric_for_best_model,
        greater_is_better=config_val.greater_is_better,
        lr_scheduler_type=config_val.lr_scheduler_type,
        disable_tqdm=config_val.disable_tqdm,
    )

    model = GroundingDinoForObjectDetection.from_pretrained(
        config_val.ft_model_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    collate_fn = DataCollator(processor=processor, config=config.model)

    trainer = GroundingDINOTrainer(
        model=model,
        args=args,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        config=config.model,
        processor=processor,
    )

    trainer.evaluate()


if __name__ == "__main__":
    main()
