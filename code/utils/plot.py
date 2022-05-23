import cv2
import numpy as np
import torch
import torch.nn.functional as F

from math import sqrt
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_grad_cam import GradCAM
from torch import Tensor


def unnormalize(
    images: Tensor,
    mean: tuple[float] = (0.5, 0.5, 0.5),
    std: tuple[float] = (0.5, 0.5, 0.5),
) -> Tensor:
    """Reverts the normalization transformation applied before ViT.

    Args:
        images (Tensor): a batch of images.
        mean (tuple[int], optional): The means used for normalization. Defaults to (0.5, 0.5, 0.5).
        std (tuple[int], optional): The stds used for normalization.. Defaults to (0.5, 0.5, 0.5).

    Returns:
        Tensor: the batch of images unnormalized.
    """
    unnormalized_images = images.clone()
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_images[:, i, :, :].mul_(s).add_(m)

    return unnormalized_images


def smoothen(mask: Tensor, patch_size: int = 16) -> Tensor:
    """This function smoothens a mask by downsampling it and upsampling it with linear interpolation.

    Args:
        mask (Tensor): a 2D float torch tensor in [0, 1].
        patch_size (int): the patch_size in pixels.

    Returns:
        A smoothened mask.
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


def attention_rollout(images, vit, discard_ratio=0.9, head_fusion='mean'):
    # Forward pass and save attention maps
    attentions = vit(images, output_attentions=True).attentions
    
    B, _, H, W = images.shape   # Batch size, channels, height, width
    P = attentions[0].size(-1)  # Number of patches
    
    mask = torch.eye(P)
    with torch.no_grad():
        # iterate over layers
        for j, attention in enumerate(attentions):
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(B, -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            #I = torch.eye(P)
            a = (attention_heads_fused + torch.eye(P)) / 2
            a = a / a.sum(dim=-1).view(-1, P, 1)
            
            mask = a @ mask
                
    # Look at the total attention between the class token,
    # and the image patches
    mask = mask[:, 0 , 1:]
    mask = mask / torch.max(mask)

    N = int(sqrt(P))
    S = int(H / N)

    mask = mask.reshape(B, 1, N, N)
    mask = F.interpolate(mask, scale_factor=S)
    mask = mask.reshape(B, H, W)
    
    return mask


def gradcam(images, vit):
    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(
            tensor.size(0),
            height,
            width,
            tensor.size(2)
        )

        # Bring the channels to the first dimension,
        result = result.transpose(2, 3).transpose(1, 2)
        
        return result

    vit.eval()

    # Create GradCAM object
    cam = GradCAM(
        model=vit,
        target_layers=[vit.vit.encoder.layer[-1].layernorm_before],
        use_cuda=False,
        reshape_transform=reshape_transform
    )

    # Compute GradCAM masks
    grayscale_cam = cam(
        input_tensor=images,
        targets=None,
        eigen_smooth=True,
        aug_smooth=True
    )

    return torch.from_numpy(grayscale_cam)


def log_masks(sample_images: Tensor, masks: Tensor, key: str, logger: WandbLogger) -> None:
    num_channels = sample_images[0].shape[0]

    # Smoothen masks
    masks = [smoothen(m) for m in masks]

    # Draw mask on sample images
    if num_channels == 1:
        sample_images = [
            image for image in unnormalize(sample_images, mean=(0.5,), std=(0.5,))
        ]
    else:
        sample_images = [image for image in unnormalize(sample_images)]

    # Check if there are 1 or 3 channels in the image
    if num_channels == 1:
        sample_images = [
            torch.repeat_interleave(sample_image, 3, 0)
            for sample_image in sample_images
        ]

    sample_images_with_mask = [
        draw_mask_on_image(image, mask) for image, mask in zip(sample_images, masks)
    ]

    sample_images_with_heatmap = [
        draw_heatmap_on_image(image, mask) for image, mask in zip(sample_images, masks)
    ]

    # Chunk to triplets (image, masked image, heatmap)
    samples = torch.cat(
        [
            torch.cat(sample_images, dim=2),
            torch.cat(sample_images_with_mask, dim=2),
            torch.cat(sample_images_with_heatmap, dim=2),
        ],
        dim=1,
    ).chunk(len(sample_images), dim=-1)

    # Compute masking percentage
    masked_pixels_percentages = [
        100 * (1 - torch.stack(masks)[i].mean(-1).mean(-1).item())
        for i in range(len(masks))
    ]
    
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
        sample_images: list,
        log_every_n_steps: int = 200,
        key: str = "",
    ):
        self.sample_images = torch.stack([sample for sample in sample_images[0]])
        self.labels = [sample.item() for sample in sample_images[1]]
        self.log_every_n_steps = log_every_n_steps
        self.key = key
        self.num_channels = self.sample_images[0].shape[0]

    def _log_masks(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Predict mask
        with torch.no_grad():
            pl_module.eval()
            outputs = pl_module.get_mask(self.sample_images)
            pl_module.train()

        # Unnest outputs
        masks = outputs["mask"]
        kl_divs = outputs["kl_div"]
        pred_classes = outputs["pred_class"].cpu()

        # Smoothen masks
        masks = [smoothen(m) for m in masks]

        # Draw mask on sample images
        if self.num_channels == 1:
            sample_images = [
                image
                for image in unnormalize(self.sample_images, mean=(0.5,), std=(0.5,))
            ]
        else:
            sample_images = [image for image in unnormalize(self.sample_images)]

        # Check if there are 1 or 3 channels in the image
        if self.num_channels == 1:
            sample_images = [
                torch.repeat_interleave(sample_image, 3, 0)
                for sample_image in sample_images
            ]

        sample_images_with_mask = [
            draw_mask_on_image(image, mask) for image, mask in zip(sample_images, masks)
        ]

        sample_images_with_heatmap = [
            draw_heatmap_on_image(image, mask)
            for image, mask in zip(sample_images, masks)
        ]

        # Chunk to triplets (image, masked image, heatmap)
        samples = torch.cat(
            [
                torch.cat(sample_images, dim=2),
                torch.cat(sample_images_with_mask, dim=2),
                torch.cat(sample_images_with_heatmap, dim=2),
            ],
            dim=1,
        ).chunk(len(sample_images), dim=-1)

        # Compute masking percentage
        masked_pixels_percentages = [
            100 * (1 - torch.stack(masks)[i].mean(-1).mean(-1).item())
            for i in range(len(masks))
        ]

        # Log with wandb
        trainer.logger.log_image(
            key="DiffMask " + self.key,
            images=list(samples),
            caption=[
                f"Masking: {masked_pixels_percentage:.2f}% "
                f"\n KL-divergence: {kl_div:.4f} "
                f"\n Class: {pl_module.model.config.id2label[label]} "
                f"\n Predicted Class: {pl_module.model.config.id2label[pred_class.item()]}"
                for masked_pixels_percentage, kl_div, label, pred_class in zip(
                    masked_pixels_percentages, kl_divs, self.labels, pred_classes
                )
            ],
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Transfer sample images to correct device
        self.sample_images = self.sample_images.to(pl_module.device)

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
    ) -> None:
        # Log sample images every n steps
        if batch_idx % self.log_every_n_steps == 0:
            self._log_masks(trainer, pl_module)
