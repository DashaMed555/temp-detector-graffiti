import sys

import hydra
from omegaconf import DictConfig

from detector_graffiti.cli import extract_command
from detector_graffiti.convert_dataset.convert_yolo_to_json import (
    convert_yolo_to_json,
)
from detector_graffiti.convert_dataset.process_datasets import process_datasets
from detector_graffiti.convert_dataset.split_dataset import split_dataset
from detector_graffiti.convert_dataset.validate import (
    validate_with_visualization,
)
from detector_graffiti.fine_tuning.fine_tuning import main as train_main
from detector_graffiti.inference.inference import main as inference_main
from detector_graffiti.onnx_converter.convert_to_onnx import (
    main as convert_main,
)

COMMAND = extract_command(sys.argv)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig):
    if COMMAND == "inference":
        inference_main(cfg)
    elif COMMAND == "train":
        train_main(cfg)
    elif COMMAND == "convert":
        convert_main(cfg)
    elif COMMAND == "process_datasets":
        process_datasets(cfg)
    elif COMMAND == "split_dataset":
        split_dataset(cfg)
    elif COMMAND == "convert_yolo_to_json":
        convert_yolo_to_json(cfg)
    elif COMMAND == "validate":
        validate_with_visualization(cfg)
    else:
        raise ValueError(f"Unknown command: {COMMAND}")


if __name__ == "__main__":
    main()
