import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


model_name = "/content/grounding_dino_ft"
image_path = "/content/002350.png"
SIZE = (800, 800)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    output_attentions=True,
    output_hidden_states=True
)
processor = AutoProcessor.from_pretrained(model_name)

image = Image.open(image_path).convert("RGB")
image = image.resize(SIZE)
prompt = "abandoned object ."

inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=64
)

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
token_type_ids = torch.zeros_like(input_ids)
pixel_mask = torch.ones(pixel_values.shape[0],
                       pixel_values.shape[2],
                       pixel_values.shape[3],
                       dtype=torch.int64)

torch.onnx.export(
    model,
    (pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask),
    "grounding_dino.onnx",
    input_names=[
        "pixel_values",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "pixel_mask"
    ],
    output_names=[
        "logits",
        "pred_boxes"
    ],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "input_ids": {0: "batch_size"},
        "token_type_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "pixel_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"}
    },
    opset_version=19,
    do_constant_folding=True
)