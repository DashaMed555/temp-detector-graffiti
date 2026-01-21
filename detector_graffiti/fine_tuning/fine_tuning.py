import datetime
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from ..compute_metrics.compute_metrics import compute_metrics
from ..dataloader.dataset import JsonDataset
from ..dataloader.utils import DataCollator
from ..freeze_layers.freeze_layers import freeze_layers
from ..grounding_dino_trainer.trainer import GroundingDINOTrainer
from ..logging.logging import MLflowLogger


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: DictConfig):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir = Path(config.fine_tuning.output_dir) / current_time
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.fine_tuning.seed)

    mlflow_logger = MLflowLogger(config, str(output_dir))
    mlflow_logger.start_run(run_name=f"train_{current_time}")

    with open(config.data_loading.train_json_path, "r") as f:
        train_data = json.load(f)

    labels = sorted(
        {ann["label_name"] for d in train_data for ann in d["annotations"]}
    )
    label2id = {c: i for i, c in enumerate(labels)}
    id2label = {i: c for c, i in label2id.items()}

    train_ds = JsonDataset(
        config.data_loading.train_json_path,
        config.data_loading.train_image_path,
        label2id,
    )
    val_ds = JsonDataset(
        config.data_loading.val_json_path,
        config.data_loading.val_image_path,
        label2id,
    )

    processor = AutoProcessor.from_pretrained(config.model.model_id)

    model = GroundingDinoForObjectDetection.from_pretrained(
        config.model.model_id,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if config.freeze_layers.freeze_layers:
        freeze_layers(model, config.freeze_layers)

    config_ft = config.fine_tuning

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config_ft.per_device_train_batch_size,
        per_device_eval_batch_size=config_ft.per_device_eval_batch_size,
        eval_accumulation_steps=config_ft.eval_accumulation_steps,
        gradient_accumulation_steps=config_ft.gradient_accumulation_steps,
        num_train_epochs=config_ft.num_train_epochs,
        learning_rate=config_ft.learning_rate,
        eval_strategy=config_ft.eval_strategy,
        eval_on_start=config_ft.eval_on_start,
        remove_unused_columns=config_ft.remove_unused_columns,
        weight_decay=config_ft.weight_decay,
        adam_beta2=config_ft.adam_beta2,
        optim=config_ft.optim,
        save_strategy=config_ft.save_strategy,
        report_to="none",
        load_best_model_at_end=config_ft.load_best_model_at_end,
        dataloader_pin_memory=config_ft.dataloader_pin_memory,
        logging_strategy=config_ft.logging_strategy,
        metric_for_best_model=config_ft.metric_for_best_model,
        greater_is_better=config_ft.greater_is_better,
        lr_scheduler_type=config_ft.lr_scheduler_type,
    )

    collate_fn = DataCollator(processor=processor, config=config.model)

    class MLflowCallback(TrainerCallback):
        def __init__(self, logger):
            self.logger = logger

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.logger.log_metrics(logs)

    callbacks_list = []
    callbacks_list.append(MLflowCallback(mlflow_logger))

    trainer = GroundingDINOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        config=config.model,
        processor=processor,
        callbacks=callbacks_list,
    )

    trainer.train()

    mlflow_logger.end_run()

    model_path = output_dir / "ft_model"
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)


if __name__ == "__main__":
    main()
