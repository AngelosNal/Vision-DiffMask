import cv2
import numpy as np
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from torch import Tensor


def unnormalize(
    images: Tensor,
    mean: tuple[int] = (0.5, 0.5, 0.5),
    std: tuple[int] = (0.5, 0.5, 0.5),
) -> Tensor:
    """Reverts the normalization transformation applied before ViT.

    Args:
        images (Tensor): a batch of images.
        mean (tuple[int], optional): The means used for normalization. Defaults to (0.5, 0.5, 0.5).
        std (tuple[int], optional): The stds used for normalization.. Defaults to (0.5, 0.5, 0.5).

    Returns:
        Tensor: the batch of images unnormalized.
    """
    for i, (m, s) in enumerate(zip(mean, std)):
        images[:, i, :, :].mul_(s).add_(m)

    return images


def draw_mask_on_image(image: Tensor, mask: Tensor) -> Tensor:
    """This function overlays a dimming mask on the image.

    Args:
        image (Tensor): a float torch tensor in [0, 1].
        mask (Tensor): a float torch tensor in [0, 1].

    Returns:
        The default image with the cam overlay.
    """
    masked_image = image * mask

    return masked_image


def draw_heatmap_on_image(
    image: Tensor, mask: Tensor, colormap: int = cv2.COLORMAP_JET
) -> Tensor:
    """This function overlays a heatmap on the image.

    Args:
        image (Tensor): a float torch tensor in [0, 1].
        mask (Tensor): a float torch tensor in [0, 1].
        colormap (int): the OpenCV colormap to be used.

    Returns:
        The default image with the cam overlay.
    """
    # Convert image & mask to numpy
    image = image.permute(1, 2, 0).numpy()
    mask = mask.numpy()

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # Overlay heatmap on image
    masked_image = image + heatmap
    masked_image = masked_image / np.max(masked_image)

    return torch.tensor(masked_image).permute(2, 0, 1)


class DrawMaskCallback(Callback):
    def __init__(self, sample_images: Tensor):
        self.sample_images = unnormalize(sample_images)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Predict mask
        with torch.no_grad():
            pl_module.eval()
            masks = pl_module.get_mask(self.sample_images)
            pl_module.train()

        # Draw mask on sample images
        sample_images = [image for image in self.sample_images]

        sample_images_with_mask = [
            draw_mask_on_image(image, mask) for image, mask in zip(sample_images, masks)
        ]

        sample_images_with_heatmap = [
            draw_heatmap_on_image(image, mask)
            for image, mask in zip(sample_images, masks)
        ]

        # Log with tensorboard
        trainer.logger.log_image(
            key="Predicted masks for sample images",
            images=sample_images + sample_images_with_mask + sample_images_with_heatmap,
        )
