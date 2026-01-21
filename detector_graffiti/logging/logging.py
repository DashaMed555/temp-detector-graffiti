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
            "loss",
        ]

        self.metrics_history = {}

        self.plots_dir = Path(output_dir) / config.logging.plots_dir
        self.plots_dir.mkdir(exist_ok=True)

    def start_run(self, run_name):
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.log_param("git_commit", self.git_commit)
        for key, value in self.config.fine_tuning.items():
            if isinstance(value, int) or isinstance(value, float):
                mlflow.log_param(key, value)

    def log_metrics(self, logs):
        metrics_to_log = {}
        for key in self.metrics:
            if key in logs:
                value = logs[key]
                metrics_to_log[key] = float(value)

                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(float(value))

        if metrics_to_log:
            for key, value in metrics_to_log.items():
                if key in self.metrics:
                    mlflow.log_metric(key, value, step=int(logs["epoch"]))

    def save_plots(self):
        self._create_plots(
            ["loss", "eval_loss"],
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

        for metric in metric_names:
            if metric in self.metrics_history:
                values = self.metrics_history[metric]
                x = list(range(1, len(values) + 1))
                plt.plot(x, values, label=metric)

        plt.xlabel("Epoch")
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
