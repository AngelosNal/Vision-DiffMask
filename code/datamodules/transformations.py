from torch import Tensor
from transformers.feature_extraction_utils import BatchFeature

import torch


class AddGaussianNoise:
    """Add Gaussian noise to an image.

    Args:
        mean (float): mean of the Gaussian noise
        std (float): standard deviation of the Gaussian noise
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class UnNest:
    """Un-nest the output after the ViTFeatureExtractor"""

    def __call__(self, x: BatchFeature) -> Tensor:
        x = torch.tensor(x["pixel_values"][0])

        if len(x) == 3:
            return x

        # ViT expects 3D tensors [C, H, W]
        return x.unsqueeze(0)
