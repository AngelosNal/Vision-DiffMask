import argparse
import pytorch_lightning as pl

from datamodules import CIFAR10DataModule
from models.interpretation import ImageInterpretationNet
from transformers import ViTFeatureExtractor, ViTForImageClassification
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args: argparse.Namespace):
    # Load pre-trained Transformer
    model = ViTForImageClassification.from_pretrained(args.vit_model)

    # Load CIFAR10 datamodule
    dm = CIFAR10DataModule(
        batch_size=8,
        feature_extractor=ViTFeatureExtractor.from_pretrained(
            args.vit_model, return_tensors="pt"
        ),
    )

    # Create Vision DiffMask for the model
    diffmask = ImageInterpretationNet(model)

    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ModelCheckpoint()],
        logger=TensorBoardLogger(
            "lightning_logs", name=args.vit_model, default_hp_metric=False
        ),
        max_epochs=args.num_epochs,
    )

    trainer.fit(diffmask, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vit_model",
        type=str,
        default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="Pre-trained Vision Transformer (ViT) model to load.",
    )

    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs to train.",
    )

    args = parser.parse_args()

    main(args)
