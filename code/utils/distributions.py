"""
Hard Concrete & Rectified Strected distributions implementation.

* copied from: https://github.com/nicola-decao/diffmask/blob/master/diffmask/models/distributions.py
"""

import torch
import torch.distributions as distr
import torch.nn.functional as F

from torch import Tensor


class BinaryConcrete(distr.relaxed_bernoulli.RelaxedBernoulli):
    def __init__(self, temperature: Tensor, logits: Tensor):
        super().__init__(temperature=temperature, logits=logits)
        self.device = self.temperature.device

    def cdf(self, value: Tensor) -> Tensor:
        return torch.sigmoid(
            (torch.log(value) - torch.log(1.0 - value)) * self.temperature - self.logits
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return torch.where(
            (value > 0) & (value < 1),
            super().log_prob(value),
            torch.full_like(value, -float("inf")),
        )

    def log_expected_L0(self, value: Tensor) -> Tensor:
        return -F.softplus(
            (torch.log(value) - torch.log(1 - value)) * self.temperature - self.logits
        )


class Streched(distr.TransformedDistribution):
    def __init__(self, base_dist, l: float = -0.1, r: float = 1.1):
        super().__init__(base_dist, distr.AffineTransform(loc=l, scale=r - l))

    def log_expected_L0(self) -> Tensor:
        value = torch.tensor(0.0, device=self.base_dist.device)
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.log_expected_L0(value)
        value = self._monotonize_cdf(value)
        return value

    def expected_L0(self) -> Tensor:
        return self.log_expected_L0().exp()


class RectifiedStreched(Streched):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        x = super().rsample(sample_shape)
        return x.clamp(0, 1)
