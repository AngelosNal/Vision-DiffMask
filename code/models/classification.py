"""
Pytorch Lightning module for Image Classification

* modified from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
"""

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

from transformers import ViTForImageClassification


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

    def forward(self, x):
        return self.model(x).logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )

        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch

        preds = self.model(imgs).logits
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")

        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
