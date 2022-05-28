from torch import Tensor
from transformers.image_utils import ImageInput

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
    """Un-nest the output of a feature extractor"""

    def __init__(self, feature_extractor: callable):
        self.feature_extractor = feature_extractor

    def __call__(self, x: ImageInput) -> Tensor:
        # Pass the input through the feature extractor
        x = self.feature_extractor(x)
        # Un-nest the pixel_values tensor
        x = torch.tensor(x["pixel_values"][0])

        # HuggingFace models expect 3D tensors [C, H, W]
        return x if len(x) == 3 else x.unsqueeze(0)
