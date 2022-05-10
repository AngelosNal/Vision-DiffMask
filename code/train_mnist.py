import argparse
import pytorch_lightning as pl

from datamodules import MNISTDataModule
from models.classification import ImageClassificationNet
from transformers import ViTConfig, ViTFeatureExtractor, ViTForImageClassification
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(args: argparse.Namespace):
    # Seed
    pl.seed_everything(123)

    # Create a new Vision Transformer
    mnist_cfg = ViTConfig(image_size=112, num_channels=1, num_labels=10)
    vit = ViTForImageClassification(mnist_cfg)
    model = ImageClassificationNet(vit)

    # Create a Feature Extractor for MNIST
    mnist_fe = ViTFeatureExtractor(
        size=mnist_cfg.image_size,
        image_mean=[0.5],
        image_std=[0.5],
        return_tensors="pt",
    )

    # Load MNIST datamodule
    dm = MNISTDataModule(batch_size=128, feature_extractor=mnist_fe)

    # Create wandb logger
    wandb_logger = WandbLogger(
        name="ViT-MNIST_training",
        project="Patch-DiffMask",
    )

    # Create checkpoint callback
    ckpt_cb = ModelCheckpoint(dirpath=f"checkpoints/{wandb_logger.version}")

    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ckpt_cb],
        logger=wandb_logger,
        max_epochs=args.num_epochs,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to train.",
    )

    args = parser.parse_args()

    main(args)
