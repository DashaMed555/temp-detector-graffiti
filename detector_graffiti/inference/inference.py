from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection as GDModel
from transformers import AutoProcessor


class Inference:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(config.model_path)
        self.model = GDModel.from_pretrained(config.model_path)
        self.model.to(self.device)
        self.prompt = [config.prompt]

    def run(self, frames, batch_size=1):
        target_size = [frames.shape[:2]]
        prompt = self.prompt * batch_size
        inputs = self.processor(
            images=frames, text=prompt, return_tensors="pt", truncation=True
        )
        inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
            )

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.threshold,
            target_sizes=target_size * batch_size,
        )[0]
        boxes = results["boxes"].detach().cpu().numpy()
        boxes = np.round(boxes).astype(np.int32)
        confidence = results["scores"].detach().cpu().numpy()
        class_name = results["text_labels"]
        result = [
            {"box": box, "class": cls, "confidence": conf}
            for box, cls, conf in zip(boxes, class_name, confidence)
        ]
        return result


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: DictConfig):
    inf_config = config.inference

    inference = Inference(inf_config)

    input_dir = Path(inf_config.images_path)
    output_dir = Path(inf_config.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Directory {input_dir} does not exist")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_files.append(f)

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        result = inference.run(np.array(img_array))
        if result:
            for det in result:
                x1, y1, x2, y2 = det["box"]
                text = f"{det['class']}: {det['confidence']:.2f}"
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img_array,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        out_path = output_dir / f"detected_{img_path.stem}{img_path.suffix}"
        cv2.imwrite(str(out_path), img_array)


if __name__ == "__main__":
    main()
