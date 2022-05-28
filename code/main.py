from datamodules import CIFAR10QADataModule, ImageDataModule
from datamodules.utils import datamodule_factory
from functools import partial
from models import ImageInterpretationNet
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import ViTForImageClassification
from typing import Generator
from utils.plot import DrawMaskCallback

import argparse
import pytorch_lightning as pl
import torch


def get_experiment_name(args: argparse.Namespace):
    """Create a name for the experiment based on the command line arguments."""
    # Convert to dictionary
    args = vars(args)

    # Create a list with non-experiment arguments
    non_experiment_args = [
        "add_blur",
        "add_noise",
        "add_rotation",
        "base_model",
        "batch_size",
        "class_idx",
        "data_dir",
        "enable_progress_bar",
        "from_pretrained",
        "log_every_n_steps",
        "num_epochs",
        "num_workers",
        "sample_images",
        "seed",
    ]

    # Create experiment name from experiment arguments
    return "-".join(
        [
            f"{name}={value}"
            for name, value in sorted(args.items())
            if name not in non_experiment_args
        ]
    )


def sample_images_generator(
    dm: ImageDataModule, n_images: int = 8
) -> Generator[torch.Tensor, torch.Tensor, None]:
    """Get a generator that iterates through elements from the validation dataset."""
    for x, y in iter(dm.val_dataloader()):
        # Only yield as many images as requested
        yield x[:n_images], y[:n_images]


def main(args: argparse.Namespace):
    # Seed
    pl.seed_everything(args.seed)

    # Load pre-trained Transformer
    model = ViTForImageClassification.from_pretrained(args.from_pretrained)

    # Load datamodule
    dm = datamodule_factory(args)

    # Setup datamodule to sample images for the mask callback
    dm.prepare_data()
    dm.setup("fit")

    # Create Vision DiffMask for the model
    diffmask = ImageInterpretationNet(
        model_cfg=model.config,
        alpha=args.alpha,
        lr=args.lr,
        eps=args.eps,
        lr_placeholder=args.lr_placeholder,
        lr_alpha=args.lr_alpha,
        mul_activation=args.mul_activation,
        add_activation=args.add_activation,
        placeholder=not args.no_placeholder,
        weighted_layer_pred=args.weighted_layer_distribution,
    )
    diffmask.set_vision_transformer(model)

    # Create wandb logger instance
    wandb_logger = WandbLogger(
        name=get_experiment_name(args),
        project="Patch-DiffMask",
    )

    # Create checkpoint callback
    ckpt_cb = ModelCheckpoint(
        save_top_k=-1,
        dirpath=f"checkpoints/{wandb_logger.version}",
        every_n_train_steps=args.log_every_n_steps,
    )

    # Sample images & create mask callback
    sample_images = sample_images_generator(dm)
    mask_cb = partial(
        DrawMaskCallback,
        log_every_n_steps=args.log_every_n_steps,
    )
    mask_cb1 = mask_cb(next(sample_images), key="1")
    mask_cb2 = mask_cb(next(sample_images), key="2")
    mask_cb3 = mask_cb(next(sample_images), key="3")

    # Create trainer
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ckpt_cb, mask_cb1, mask_cb2, mask_cb3],
        enable_progress_bar=args.enable_progress_bar,
        logger=wandb_logger,
        max_epochs=args.num_epochs,
    )

    # Train the model
    trainer.fit(diffmask, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Trainer
    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Whether to enable the progress bar (NOT recommended when logging to file).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )

    # Logging
    parser.add_argument(
        "--sample_images",
        type=int,
        default=8,
        help="Number of images to sample for the mask callback.",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=200,
        help="Number of steps between logging media & checkpoints.",
    )

    # Base (classification) model
    parser.add_argument(
        "--base_model",
        type=str,
        default="ViT",
        choices=["ViT"],
        help="Base model architecture to train.",
    )
    parser.add_argument(
        "--from_pretrained",
        type=str,
        default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="The name of the pretrained HF model to load.",
    )

    # Interpretation model
    ImageInterpretationNet.add_model_specific_args(parser)

    # Datamodule
    ImageDataModule.add_model_specific_args(parser)
    CIFAR10QADataModule.add_model_specific_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["MNIST", "CIFAR10", "CIFAR10_QA", "toy"],
        help="The dataset to use.",
    )

    args = parser.parse_args()

    main(args)
