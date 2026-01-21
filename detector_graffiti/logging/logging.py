import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow


class MLflowLogger:
    def __init__(self, config, output_dir):
        self.config = config
        self.git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        mlflow.set_tracking_uri(config.logging.tracking_uri)
        mlflow.set_experiment(config.logging.experiment_name)
        self.metrics = [
            "eval_loss",
            "eval_precision",
            "eval_recall",
            "eval_f1",
            "train_loss",
        ]

        self.metrics_history = {}
        self.steps = []

        self.plots_dir = Path(output_dir) / config.logging.plots_dir
        self.plots_dir.mkdir(exist_ok=True)

    def start_run(self, run_name):
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.log_param("git_commit", self.git_commit)
        mlflow.log_param(
            "batch_size", self.config.fine_tuning.per_device_train_batch_size
        )
        mlflow.log_param(
            "learning_rate", self.config.fine_tuning.learning_rate
        )
        mlflow.log_param("epochs", self.config.fine_tuning.num_train_epochs)

    def log_metrics(self, logs, step=None):
        metrics_to_log = {}
        for key in self.metrics:
            if key in logs:
                value = logs[key]
                metrics_to_log[key] = float(value)

                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(float(value))
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=step)
            if step is not None:
                self.steps.append(step)

    def save_plots(self):
        self._create_plots(
            ["train_loss", "eval_loss"],
            "loss_plot.png",
            "Training and Validation Loss",
        )

        self._create_plots(["eval_f1"], "f1_plot.png", "F1 Score")

        self._create_plots(
            ["eval_precision", "eval_recall"],
            "precision_recall_plot.png",
            "Precision and Recall",
        )

    def _create_plots(self, metric_names, filename, title):
        plt.figure(figsize=(8, 5))

        for number, metric in enumerate(metric_names):
            if metric in self.metrics_history:
                values = self.metrics_history[metric]
                x = (
                    self.steps[: len(values)]
                    if self.steps
                    else range(len(values))
                )
                plt.plot(x, values, label=metric)

        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=150)
        plt.close()

        mlflow.log_artifact(str(self.plots_dir / filename))

    def end_run(self):
        self.save_plots()
        mlflow.end_run()
