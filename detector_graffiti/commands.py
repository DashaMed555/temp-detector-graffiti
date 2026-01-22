import sys

import hydra
from omegaconf import DictConfig

from detector_graffiti.cli import extract_command
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
    else:
        raise ValueError(f"Unknown command: {COMMAND}")


if __name__ == "__main__":
    main()
