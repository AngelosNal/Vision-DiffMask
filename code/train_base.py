import argparse
import pytorch_lightning as pl

from datamodules import CIFAR10QADataModule, ImageDataModule
from datamodules.utils import datamodule_factory
from models import ImageClassificationNet
from models.utils import model_factory
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(args: argparse.Namespace):
    # Seed
    pl.seed_everything(args.seed)

    # Create base model
    base = model_factory(args)

    # Load datamodule
    dm = datamodule_factory(args)
    dm.prepare_data()
    dm.setup("fit")

    if args.checkpoint:
        # Load the model from the specified checkpoint
        model = ImageClassificationNet.load_from_checkpoint(args.checkpoint, model=base)
    else:
        # Create a new instance of the classification model
        model = ImageClassificationNet(
            model=base,
            num_train_steps=args.num_epochs * len(dm.train_dataloader()),
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            lr=args.lr,
        )

    # Create wandb logger
    wandb_logger = WandbLogger(
        name=f"{args.dataset}_training_{args.base_model} ({args.from_pretrained})",
        project="Patch-DiffMask",
    )

    # Create checkpoint callback
    ckpt_cb = ModelCheckpoint(dirpath=f"checkpoints/{wandb_logger.version}")
    # Create early stopping callback
    es_cb = EarlyStopping(monitor="val_acc", mode="max", patience=5)

    # Create trainer
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ckpt_cb, es_cb],
        logger=wandb_logger,
        max_epochs=args.num_epochs,
        enable_progress_bar=args.enable_progress_bar,
    )

    trainer_args = {}
    if args.checkpoint:
        # Resume trainer from checkpoint
        trainer_args["ckpt_path"] = args.checkpoint

    # Train the model
    trainer.fit(model, dm, **trainer_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint to resume the training from.",
    )

    # Trainer
    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Whether to show progress bar during training. NOT recommended when logging to files.",
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

    # Base (classification) model
    ImageClassificationNet.add_model_specific_args(parser)
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
        # default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="The name of the pretrained HF model to fine-tune from.",
    )

    # Datamodule
    ImageDataModule.add_model_specific_args(parser)
    CIFAR10QADataModule.add_model_specific_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="toy",
        choices=["MNIST", "CIFAR10", "CIFAR10_QA", "toy"],
        help="The dataset to use.",
    )

    args = parser.parse_args()

    main(args)
