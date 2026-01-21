import datetime
import logging
import time
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
    """
    Inference engine for Grounding DINO zero-shot object detection.

    Args:
            config (DictConfig): Inference configuration parameters
            logger (logging.Logger): Logger for tracking inference process
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Use device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(config.model.model_id)
        self.logger.info(f"Load processor from: {config.model.model_id}")

        self.model = GDModel.from_pretrained(config.model.model_id)
        self.logger.info(f"Load model from: {config.model.model_id}")
        self.model.to(self.device)

        self.prompt = [config.inference.prompt]
        self.logger.info(f"Prompt: {self.prompt[0]}")
        self.logger.info(f"Detection threshold: {config.inference.threshold}")

    def run(self, frames, batch_size=1):
        """
        Run object detection inference on input frames.

        Args:
            frames (np.ndarray): Input image array in RGB format,
            shape (H, W, 3)
            batch_size (int): Batch size for processing (currently supports 1)

        Returns:
            List[Dict[str, Any]]: List of detection dictionaries,
            each containing:
                - box (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]
                - class (str): Detected class label
                - confidence (float): Detection confidence score
        """
        target_size = [frames.shape[:2]]
        prompt = self.prompt * batch_size
        inputs = self.processor(
            images=frames, text=prompt, return_tensors="pt", truncation=True
        )
        inputs.to(self.device)

        inference_start = time.time()

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
            )

        inference_time = time.time() - inference_start
        self.logger.debug(f"Inference time: {inference_time:.3f}s")

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.inference.threshold,
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
    """
    Main inference function for detecting graffiti in images.

    Args:
        config (DictConfig): Hydra configuration object containing
        inference parameters
    """
    inf_config = config.inference

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir = Path(inf_config.output_dir) / current_time
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                str(output_dir / "inference.log"), encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger.info("Start inference")
    logger.info(f"Output directory: {output_dir}")

    inference = Inference(config, logger)
    logger.info("Init inference")

    input_dir = Path(config.data_loading.test_image_path)
    logger.info(f"Input directory: {input_dir}")

    if not input_dir.exists():
        raise ValueError(f"Directory {input_dir} does not exist")

    output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    valid_extensions = {".jpg", ".jpeg", ".png"}
    for img_path in input_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            image_files.append(img_path)

    logger.info("Starting image processing")
    processed_count = 0

    for img_path in image_files:
        name = img_path.name
        logger.info(
            f"[{processed_count}/{len(image_files)}] Processing: {name}"
        )
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        img_brg = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        result = inference.run(np.array(img_brg))
        if result:
            logger.info(f"Found {len(result)} detections")
            for det in result:
                x1, y1, x2, y2 = det["box"]
                text = f"{det['class']}: {det['confidence']:.2f}"
                cv2.rectangle(img_brg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img_brg,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
        processed_count += 1

        out_path = output_dir / f"detected_{img_path.stem}{img_path.suffix}"
        cv2.imwrite(str(out_path), img_brg)


if __name__ == "__main__":
    main()
