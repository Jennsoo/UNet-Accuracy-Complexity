from model.baseline import unet2d, unet2doct


def build_baseline(baseline, n_channels, n_classes, base_lr):
    if baseline == 'unet2d':
        model = unet2d.UNet2d(n_channels, n_classes)
        return model, model.parameters()

    if baseline == 'unet2doct':
        model = unet2doct.UNet2d(n_channels, n_classes)
        return model, model.parameters()

    else:
        raise NotImplementedError
