from transformers import AutoProcessor

model_id = "./model"

processor = AutoProcessor.from_pretrained(model_id)
