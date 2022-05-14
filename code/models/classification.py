"""
Pytorch Lightning module for Image Classification

* modified from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
"""

import pytorch_lightning as pl
import torch.nn.functional as F

from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
from transformers import ViTForImageClassification
from typing import List, Tuple


class ImageClassificationNet(pl.LightningModule):
    """Vision Transformer wrapper for image classification.

    Args:
        model (ViTForImageClassification): a ViT for image classification
        lr (float): the learning rate used for training (no warm-up learning rate used)
    """

    def __init__(self, model: ViTForImageClassification, lr: float = 3e-4):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

        return [optimizer], [lr_scheduler]

    def _calculate_loss(
        self,
        batch: Tuple[Tensor, Tensor],
        mode: str = "train",
    ) -> Tensor:
        imgs, labels = batch

        preds = self.model(imgs).logits
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _: Tensor) -> Tensor:
        loss = self._calculate_loss(batch, mode="train")

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], _: Tensor):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch: Tuple[Tensor, Tensor], _: Tensor):
        self._calculate_loss(batch, mode="test")
