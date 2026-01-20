import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: DictConfig):
    conf_oc = config.onnx_converter

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        conf_oc.ft_model_id,
        torch_dtype=torch.float32,
        output_attentions=True,
        output_hidden_states=True,
    )
    processor = AutoProcessor.from_pretrained(conf_oc.ft_model_id)

    image = Image.open(conf_oc.image_path).convert("RGB")
    image = image.resize(conf_oc.image_size)

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

    torch.onnx.export(
        model,
        (pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask),
        "grounding_dino.onnx",
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
        opset_version=19,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    main()
