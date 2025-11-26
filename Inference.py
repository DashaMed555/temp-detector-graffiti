import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.image_utils import load_image


class Inference:
    def __init__(self, model_path, target_size):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
        self.target_size = target_size

    def run(self, frame, batch_size=1):
        prompt = ['legal graffiti . illegal graffiti .'] * batch_size
        inputs = self.processor(images=frame, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=self.target_size*batch_size
        )[0]
        boxes = results["boxes"].detach().cpu().numpy()
        boxes = np.round(boxes).astype(np.int32)
        confidence = results["scores"].detach().cpu().numpy()
        class_name = results["label"].detach().cpu().numpy()
        result = [{"box": box, "class": cls, "confidence": conf} for box, cls, conf in zip(boxes, class_name, confidence)]
        return result