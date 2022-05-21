from datamodules import CIFAR10QADataModule, ImageDataModule
from datamodules.utils import datamodule_factory
from models.classification import ImageClassificationNet
from models.utils import model_factory
from pytorch_lightning.loggers import WandbLogger

import argparse
import pytorch_lightning as pl


def main(args: argparse.Namespace):
    # Seed
    pl.seed_everything(args.seed)

    # Create base model
    base = model_factory(args, own_config=True)

    # Load datamodule
    dm = datamodule_factory(args)

    model = ImageClassificationNet.load_from_checkpoint(
        args.checkpoint,
        model=base,
        num_train_steps=0,
    )

    # Create wandb logger
    wandb_logger = WandbLogger(
        name=f"{args.dataset}_eval_{args.base_model} ({args.from_pretrained})",
        project="Patch-DiffMask",
    )

    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        logger=wandb_logger,
        max_epochs=1,
        enable_progress_bar=args.enable_progress_bar,
    )

    trainer.test(model, dm)

    save_dir = f"checkpoints/{args.base_model}_{args.dataset}"
    model.model.save_pretrained(save_dir)
    dm.feature_extractor.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint to resume the training from.",
    )

    # Trainer
    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Whether to show progress bar during training. NOT recommended when logging to files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )

    # Base (classification) model
    parser.add_argument(
        "--base_model",
        type=str,
        default="ViT",
        choices=["ViT", "ConvNeXt"],
        help="Base model architecture to train.",
    )
    parser.add_argument(
        "--from_pretrained",
        type=str,
        default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="The name of the pretrained HF model to fine-tune from.",
    )

    # Datamodule
    ImageDataModule.add_model_specific_args(parser)
    CIFAR10QADataModule.add_model_specific_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10_QA",
        choices=["MNIST", "CIFAR10", "CIFAR10_QA"],
        help="The dataset to use.",
    )

    args = parser.parse_args()

    main(args)
