import datetime
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def main(config: DictConfig):
    """
    Convert a fine-tuned Grounding DINO model to ONNX format
    for optimized inference.

    Args:
        config (DictConfig): Configuration object containing ONNX
        conversion parameters
    """

    conf_oc = config.onnx_converter

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir = Path(conf_oc.output_dir) / current_time
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                str(output_dir / "convert.log"), encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger.info("Start onnx converter")
    logger.info(f"Output directory: {output_dir}")

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        conf_oc.ft_model_id,
        torch_dtype=torch.float32,
        output_attentions=True,
        output_hidden_states=True,
    )
    logger.info(f"Load model from: {conf_oc.ft_model_id}")

    processor = AutoProcessor.from_pretrained(conf_oc.ft_model_id)
    logger.info(f"Load processor from: {conf_oc.ft_model_id}")

    image = Image.open(conf_oc.image_path).convert("RGB")
    logger.info(f"Load image from: {conf_oc.image_path}")
    image = image.resize((conf_oc.image_size_w, conf_oc.image_size_h))
    logger.info(
        f"Resize image to: {conf_oc.image_size_w} x {conf_oc.image_size_h}"
    )

    logger.info("Preparing model inputs")
    inputs = processor(
        images=image,
        text=conf_oc.prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=conf_oc.max_length,
    )

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = torch.zeros_like(input_ids)
    pixel_mask = torch.ones(
        pixel_values.shape[0],
        pixel_values.shape[2],
        pixel_values.shape[3],
        dtype=torch.int64,
    )

    otput_path = output_dir / "grounding_dino.onnx"

    logger.info(f"Export model to ONNX format: {otput_path}")

    torch.onnx.export(
        model,
        (pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask),
        str(otput_path),
        input_names=[
            "pixel_values",
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "pixel_mask",
        ],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "pixel_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info("ONNX export completed successfully")


if __name__ == "__main__":
    main()
