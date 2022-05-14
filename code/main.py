import argparse
import pytorch_lightning as pl

from datamodules import CIFAR10DataModule
from models.interpretation import ImageInterpretationNet
from transformers import ViTFeatureExtractor, ViTForImageClassification
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.plot import DrawMaskCallback


def main(args: argparse.Namespace):
    # Seed
    pl.seed_everything(123)

    # Load pre-trained Transformer
    model = ViTForImageClassification.from_pretrained(args.vit_model)

    # Load CIFAR10 datamodule
    dm = CIFAR10DataModule(
        batch_size=args.batch_size,
        feature_extractor=ViTFeatureExtractor.from_pretrained(
            args.vit_model, return_tensors="pt"
        ),
        noise=args.noise,
        rotation=args.rotation,
        blur=args.blur,
        num_workers=args.num_workers,
    )

    # Setup datamodule to sample images for the mask callback
    dm.prepare_data()
    dm.setup("fit")

    # Create Vision DiffMask for the model
    diffmask = ImageInterpretationNet(
        model_cfg=model.config,
        lr=args.lr,
        eps=args.eps,
        lr_placeholder=args.lr_placeholder,
        lr_alpha=args.lr_alpha,
        mul_activation=args.mul_activation,
        add_activation=args.add_activation,
    )
    diffmask.set_vision_transformer(model)

    # Create wandb logger instance
    wandb_logger = WandbLogger(
        name="ViT-CIFAR10",  # TODO: add experiment-related information
        project="Patch-DiffMask",
    )

    # Create checkpoint callback
    ckpt_cb = ModelCheckpoint(
        # TODO: add more args (probably monitor some metric)
        dirpath=f"checkpoints/{wandb_logger.version}"
    )

    # Sample images & create mask callback
    sample_images, _ = next(iter(dm.val_dataloader()))
    mask_cb = DrawMaskCallback(sample_images)

    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ckpt_cb, mask_cb],
        logger=wandb_logger,
        max_epochs=args.num_epochs,
    )

    trainer.fit(diffmask, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Trainer
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to train.",
    )
    
    # Classification model
    parser.add_argument(
        "--vit_model",
        type=str,
        default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="Pre-trained Vision Transformer (ViT) model to load.",
    )
    
    # Interpretation model
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for diffmask.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="KL divergence tolerance.",
    )
    parser.add_argument(
        "--lr_placeholder",
        type=float,
        default=1e-3,
        help="Learning for mask vectors.",
    )
    parser.add_argument(
        "--lr_alpha",
        type=float,
        default=0.3,
        help="Learning rate for lagrangian optimizer.",
    )
    parser.add_argument(
        "--mul_activation",
        type=float,
        default=10.0,
        help="Value to mutliply gate activations.",
    )
    parser.add_argument(
        "--add_activation",
        type=float,
        default=5.0,
        help="Value to add to gate activations.",
    )

    # Datamodule
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to use.",
    )
    parser.add_argument(
        "--dataset",
        type=float,
        default="CIFAR10",
        help="The dataset to use.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="The data directory to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of workers to use.",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Use gaussian noise augmentation.",
    )
    parser.add_argument(
        "--add_rotation",
        action="store_true",
        help="Use rotation augmentation.",
    )
    parser.add_argument(
        "--add_blur",
        action="store_true",
        help="Use blur augmentation.",
    )

    args = parser.parse_args()

    main(args)
