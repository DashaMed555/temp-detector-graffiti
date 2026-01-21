import logging


def freeze_layers(model, config):
    logger = logging.getLogger(__name__)

    logger.info("Start parameters freeze")

    base_model = model.model

    for param in base_model.parameters():
        param.requires_grad = False

    if config.encoder:
        if hasattr(base_model, "encoder"):
            for layer in base_model.encoder.layers:
                for param in layer.parameters():
                    param.requires_grad = True

    if config.reference_points_head:
        if hasattr(base_model.decoder, "reference_points_head"):
            for param in base_model.decoder.reference_points_head.parameters():
                param.requires_grad = True

    if config.bbox_embed:
        if hasattr(base_model.decoder, "bbox_embed"):
            for param in base_model.decoder.bbox_embed.parameters():
                param.requires_grad = True

    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )

    p = 100 * trainable / total

    logger.info("Freeze is complete.")
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({p:.2f}%)")
