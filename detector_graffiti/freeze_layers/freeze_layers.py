def freeze_layers(model):
    base_model = model.model

    for param in base_model.parameters():
        param.requires_grad = False

    if hasattr(base_model, "encoder"):
        for layer in base_model.encoder.layers:
            for param in layer.parameters():
                param.requires_grad = True

    if hasattr(base_model.decoder, "reference_points_head"):
        for param in base_model.decoder.reference_points_head.parameters():
            param.requires_grad = True

    if hasattr(base_model.decoder, "bbox_embed"):
        for param in base_model.decoder.bbox_embed.parameters():
            param.requires_grad = True

    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )
    print(
        (
            "Заморозка завершена. Обучаемых параметров: ",
            f"{trainable:,} / {total:,} ",
        ),
        f"({100 * trainable / total:.2f}%)",
    )
