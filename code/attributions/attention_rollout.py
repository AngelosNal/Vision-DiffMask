import torch
import torch.nn.functional as F

from math import sqrt
from torch import Tensor
from transformers import ViTForImageClassification


@torch.no_grad()
def attention_rollout(
    images: Tensor,
    vit: ViTForImageClassification,
    discard_ratio: float = 0.9,
    head_fusion: str = "mean",
) -> Tensor:
    """Performs the Attention Rollout method on a batch of images (https://arxiv.org/pdf/2005.00928.pdf)."""
    # Forward pass and save attention maps
    attentions = vit(images, output_attentions=True).attentions

    B, _, H, W = images.shape  # Batch size, channels, height, width
    P = attentions[0].size(-1)  # Number of patches

    mask = torch.eye(P)
    # Iterate over layers
    for j, attention in enumerate(attentions):
        if head_fusion == "mean":
            attention_heads_fused = attention.mean(axis=1)
        elif head_fusion == "max":
            attention_heads_fused = attention.max(axis=1)[0]
        elif head_fusion == "min":
            attention_heads_fused = attention.min(axis=1)[0]
        else:
            raise "Attention head fusion type Not supported"

        # Drop the lowest attentions, but don't drop the class token
        flat = attention_heads_fused.view(B, -1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
        indices = indices[indices != 0]
        flat[0, indices] = 0

        # I = torch.eye(P)
        a = (attention_heads_fused + torch.eye(P)) / 2
        a = a / a.sum(dim=-1).view(-1, P, 1)

        mask = a @ mask

    # Look at the total attention between the class token and the image patches
    mask = mask[:, 0, 1:]
    mask = mask / torch.max(mask)

    N = int(sqrt(P))
    S = int(H / N)

    mask = mask.reshape(B, 1, N, N)
    mask = F.interpolate(mask, scale_factor=S)
    mask = mask.reshape(B, H, W)

    return mask
