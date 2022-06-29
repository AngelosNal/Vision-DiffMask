import cv2
import numpy as np
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from torch import Tensor


@torch.no_grad()
def unnormalize(
    images: Tensor,
    mean: tuple[float] = (0.5, 0.5, 0.5),
    std: tuple[float] = (0.5, 0.5, 0.5),
) -> Tensor:
    """Reverts the normalization transformation applied before ViT.

    Args:
        images (Tensor): a batch of images
        mean (tuple[int]): the means used for normalization - defaults to (0.5, 0.5, 0.5)
        std (tuple[int]): the stds used for normalization - defaults to (0.5, 0.5, 0.5)

    Returns:
        the un-normalized batch of images
    """
    unnormalized_images = images.clone()
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_images[:, i, :, :].mul_(s).add_(m)

    return unnormalized_images


@torch.no_grad()
def smoothen(mask: Tensor, patch_size: int = 16) -> Tensor:
    """Smoothens a mask by downsampling it and re-upsampling it
     with bi-linear interpolation.

    Args:
        mask (Tensor): a 2D float torch tensor with values in [0, 1]
        patch_size (int): the patch size in pixels

    Returns:
        a smoothened mask at the pixel level
    """
    device = mask.device
    (h, w) = mask.shape
    mask = cv2.resize(
        mask.cpu().numpy(),
        (h // patch_size, w // patch_size),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(mask).to(device)


@torch.no_grad()
def draw_mask_on_image(image: Tensor, mask: Tensor) -> Tensor:
    """Overlays a dimming mask on the image.

    Args:
        image (Tensor): a float torch tensor with values in [0, 1]
        mask (Tensor): a float torch tensor with values in [0, 1]

    Returns:
        the image with parts of it dimmed according to the mask
    """
    masked_image = image * mask

    return masked_image


@torch.no_grad()
def draw_heatmap_on_image(
    image: Tensor,
    mask: Tensor,
    colormap: int = cv2.COLORMAP_JET,
) -> Tensor:
    """Overlays a heatmap on the image.

    Args:
        image (Tensor): a float torch tensor with values in [0, 1]
        mask (Tensor): a float torch tensor with values in [0, 1]
        colormap (int): the OpenCV colormap to be used

    Returns:
        the image with the heatmap overlaid
    """
    # Save the device of the image
    original_device = image.device

    # Convert image & mask to numpy
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # Overlay heatmap on image
    masked_image = image + heatmap
    masked_image = masked_image / np.max(masked_image)

    return torch.tensor(masked_image).permute(2, 0, 1).to(original_device)


def _prepare_samples(images: Tensor, masks: Tensor) -> tuple[Tensor, list[float]]:
    """Prepares the samples for the masking/heatmap visualization.

    Args:
        images (Tensor): a float torch tensor with values in [0, 1]
        masks (Tensor): a float torch tensor with values in [0, 1]

    Returns
        a tuple of image triplets (img, masked, heatmap) and their
         corresponding masking percentages
    """
    num_channels = images[0].shape[0]

    # Smoothen masks
    masks = [smoothen(m) for m in masks]

    # Un-normalize images
    if num_channels == 1:
        images = [
            torch.repeat_interleave(img, 3, 0)
            for img in unnormalize(images, mean=(0.5,), std=(0.5,))
        ]
    else:
        images = [img for img in unnormalize(images)]

    # Draw mask on sample images
    images_with_mask = [
        draw_mask_on_image(image, mask) for image, mask in zip(images, masks)
    ]

    # Draw heatmap on sample images
    images_with_heatmap = [
        draw_heatmap_on_image(image, mask) for image, mask in zip(images, masks)
    ]

    # Chunk to triplets (image, masked image, heatmap)
    samples = torch.cat(
        [
            torch.cat(images, dim=2),
            torch.cat(images_with_mask, dim=2),
            torch.cat(images_with_heatmap, dim=2),
        ],
        dim=1,
    ).chunk(len(images), dim=-1)

    # Compute masking percentages
    masked_pixels_percentages = [
        100 * (1 - torch.stack(masks)[i].mean(-1).mean(-1).item())
        for i in range(len(masks))
    ]

    return samples, masked_pixels_percentages


def log_masks(images: Tensor, masks: Tensor, key: str, logger: WandbLogger):
    """Logs a set of images with their masks to WandB.

    Args:
        images (Tensor): a float torch tensor with values in [0, 1]
        masks (Tensor): a float torch tensor with values in [0, 1]
        key (str): the key to log the images with
        logger (WandbLogger): the logger to log the images to
    """
    samples, masked_pixels_percentages = _prepare_samples(images, masks)

    # Log with wandb
    logger.log_image(
        key=key,
        images=list(samples),
        caption=[
            f"Masking: {masked_pixels_percentage:.2f}% "
            for masked_pixels_percentage in masked_pixels_percentages
        ],
    )


class DrawMaskCallback(Callback):
    def __init__(
        self,
        samples: list[tuple[Tensor, Tensor]],
        log_every_n_steps: int = 200,
        key: str = "",
    ):
        """A callback that logs VisionDiffMask masks for the sample images to WandB.

        Args:
            samples (list[tuple[Tensor, Tensor]): a list of image, label pairs
            log_every_n_steps (int): the interval in steps to log the masks to WandB
            key (str): the key to log the images with (allows for multiple batches)
        """
        self.images = torch.stack([img for img in samples[0]])
        self.labels = [label.item() for label in samples[1]]
        self.log_every_n_steps = log_every_n_steps
        self.key = key

    def _log_masks(self, trainer: Trainer, pl_module: LightningModule):
        # Predict mask
        with torch.no_grad():
            pl_module.eval()
            outputs = pl_module.get_mask(self.images)
            pl_module.train()

        # Unnest outputs
        masks = outputs["mask"]
        kl_divs = outputs["kl_div"]
        pred_classes = outputs["pred_class"].cpu()
        orig_pred_classes = outputs["orig_pred_class"].cpu()

        # Prepare masked samples for logging
        samples, masked_pixels_percentages = _prepare_samples(self.images, masks)

        # Log with wandb
        trainer.logger.log_image(
            key="DiffMask " + self.key,
            images=list(samples),
            caption=[
                f"Masking: {masked_pixels_percentage:.2f}% "
                f"\n KL-divergence: {kl_div:.4f} "
                f"\n Class: {pl_module.model.config.id2label[label]} "
                f"\n Predicted Class: {pl_module.model.config.id2label[pred_class.item()]}"
                f"\n Original Predicted Class: {pl_module.model.config.id2label[orig_pred_class.item()]}"
                for masked_pixels_percentage, kl_div, label, pred_class, orig_pred_class in zip(
                    masked_pixels_percentages, kl_divs, self.labels, pred_classes, orig_pred_classes
                )
            ],
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        # Transfer sample images to correct device
        self.images = self.images.to(pl_module.device)

        # Log sample images
        self._log_masks(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        unused: int = 0,
    ):
        # Log sample images every n steps
        if batch_idx % self.log_every_n_steps == 0:
            self._log_masks(trainer, pl_module)
