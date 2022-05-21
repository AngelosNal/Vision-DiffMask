"""
Pytorch Lightning module for Image Classification

* modified from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
"""

import pytorch_lightning as pl
import torch.nn.functional as F

from argparse import ArgumentParser
from torch import Tensor
from torch.optim import AdamW, Optimizer, RAdam
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler, PreTrainedModel
from typing import List, Tuple


class ImageClassificationNet(pl.LightningModule):
    """HuggingFace model wrapper for image classification.

    Args:
        model (PreTrainedModel): a pretrained model for image classification
        num_train_steps (int): number of training steps
        optimizer (str): optimizer to use
        weight_decay (float): weight decay for optimizer
        lr (float): the learning rate used for training
    """

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Classification Model")
        parser.add_argument(
            "--optimizer",
            type=str,
            default="AdamW",
            choices=["AdamW", "RAdam"],
            help="The optimizer to use to train the model.",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-2,
            help="The optimizer's weight decay.",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=5e-5,
            help="The initial learning rate for the model.",
        )
        return parent_parser

    def __init__(
        self,
        model: PreTrainedModel,
        num_train_steps: int,
        optimizer: str = "AdamW",
        weight_decay: float = 1e-2,
        lr: float = 5e-5,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        if self.hparams.optimizer == "AdamW":
            optim_class = AdamW
        elif self.hparams.optimizer == "RAdam":
            optim_class = RAdam
        else:
            raise Exception(f"Unknown optimizer {self.hparams.optimizer}")

        optimizer = optim_class(
            self.parameters(),
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.lr,
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.hparams.num_train_steps,
        )

        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch: Tuple[Tensor, Tensor], mode: str) -> Tensor:
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
