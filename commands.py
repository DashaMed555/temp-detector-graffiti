import sys

import click

from detector_graffiti.fine_tuning.fine_tuning import main as train_main
from detector_graffiti.inference.inference import main as inference_main
from detector_graffiti.onnx_converter.convert_to_onnx import (
    main as convert_main,
)


@click.group()
def cli():
    pass


@cli.command()
def inference():
    original_argv = sys.argv.copy()
    sys.argv = [original_argv[0]]
    inference_main()
    sys.argv = original_argv


@cli.command()
def train():
    original_argv = sys.argv.copy()
    sys.argv = [original_argv[0]]
    train_main()
    sys.argv = original_argv


@cli.command()
def convert():
    original_argv = sys.argv.copy()
    sys.argv = [original_argv[0]]
    convert_main()
    sys.argv = original_argv


if __name__ == "__main__":
    cli()
