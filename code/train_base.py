import argparse
import pytorch_lightning as pl

from datamodules import CIFAR10DataModule, MNISTDataModule
from models.classification import ImageClassificationNet
from transformers import (
    ConvNextConfig,
    ConvNextFeatureExtractor,
    ConvNextForImageClassification,
    ViTConfig,
    ViTFeatureExtractor,
    ViTForImageClassification,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_configs(args: argparse.Namespace):
    if args.dataset == "mnist":
        model_cfg_args = {
            "image_size": 112,
            "num_channels": 1,
        }
        fe_cfg_args = {
            "image_mean": [0.5],
            "image_std": [0.5],
        }
    elif args.dataset == "cifar10":
        model_cfg_args = {
            "image_size": 224,
            "num_channels": 3,
        }
        fe_cfg_args = {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    return model_cfg_args, fe_cfg_args


def main(args: argparse.Namespace):
    # Seed
    pl.seed_everything(123)

    if args.base_model == "vit":

        # Create a new Vision Transformer
        if args.from_pretrained:
            config = ViTConfig.from_pretrained(args.from_pretrained, num_labels=10)
            fe_cfg_args = {}
        else:
            model_cfg_args, fe_cfg_args = get_configs(args)
            config = ViTConfig(num_labels=10, **model_cfg_args)

        base = ViTForImageClassification(config)
        # Create a Feature Extractor for MNIST
        feature_extractor = ViTFeatureExtractor(
            size=config.image_size,
            return_tensors="pt",
            **fe_cfg_args,
        )
    elif args.base_model == "convnext":
        # Create a new ConvNext model
        if args.from_pretrained:
            config = ConvNextConfig.from_pretrained(args.from_pretrained, num_labels=10)
            fe_cfg_args = {}
        else:
            model_cfg_args, fe_cfg_args = get_configs(args)
            config = ConvNextConfig(num_labels=10, **model_cfg_args)

        base = ConvNextForImageClassification(config)
        # Create a Feature Extractor for MNIST
        feature_extractor = ConvNextFeatureExtractor(
            size=config.image_size,
            return_tensors="pt",
            **fe_cfg_args,
        )
    else:
        raise Exception(f"Unknown base model: {args.base_model}")

    if args.checkpoint:
        model = ImageClassificationNet.load_from_checkpoint(
            args.checkpoint,
            model=base,
            lr=args.lr,
        )
    else:
        model = ImageClassificationNet(base, lr=args.lr)

    # Load datamodule
    dm_cfg = {
        "batch_size": args.batch_size,
        "feature_extractor": feature_extractor,
        "num_workers": args.num_workers,
    }
    if args.dataset == "mnist":
        dm = MNISTDataModule(**dm_cfg)
    elif args.dataset == "cifar10":
        dm = CIFAR10DataModule(**dm_cfg)
    else:
        # Add more datasets here if needed
        return

    # Create wandb logger
    wandb_logger = WandbLogger(
        name=f"{args.dataset}_training_{args.base_model}-lr={args.lr}",
        project="Patch-DiffMask",
    )

    # Create checkpoint callback
    ckpt_cb = ModelCheckpoint(dirpath=f"checkpoints/{wandb_logger.version}")
    # Create early stopping callback
    es_cb = EarlyStopping(monitor="val_acc", mode="max")

    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ckpt_cb, es_cb],
        logger=wandb_logger,
        max_epochs=args.num_epochs,
        enable_progress_bar=args.enable_progress_bar,
    )

    trainer_args = {}
    if args.checkpoint:
        trainer_args["ckpt_path"] = args.checkpoint

    trainer.fit(model, dm, **trainer_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ImageClassificationNet.add_model_specific_args(parser)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for data loading.",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="vit",
        choices=["vit", "convnext"],
        help="Base model architecture to train.",
    )

    parser.add_argument(
        "--from_pretrained",
        type=str,
        help="The name of the pretrained HF model to fine-tune from.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint to resume the training from.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to train on.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Whether to show progress bar during training. NOT recommended when logging to files.",
    )

    args = parser.parse_args()

    main(args)
