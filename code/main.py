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
        batch_size=8,
        feature_extractor=ViTFeatureExtractor.from_pretrained(
            args.vit_model, return_tensors="pt"
        ),
    )

    # Setup datamodule to sample images for the mask callback
    dm.prepare_data()
    dm.setup('fit')

    # Create Vision DiffMask for the model
    diffmask = ImageInterpretationNet(model.config)
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

    # Model
    parser.add_argument("--vit_model", type=str, default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
                        help="Pre-trained Vision Transformer (ViT) model to load.",)

    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train.",)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.",)
    parser.add_argument("--eps", type=float, default=0.1, help="Epsilon for the Adam optimizer.",)
    parser.add_argument("--lr_placeholder", type=float, default=1e-3, help="Epsilon for the Adam optimizer.",)
    parser.add_argument("--lr_alpha", type=float, default=0.3, help="Epsilon for the Adam optimizer.",)

    # Datamodule args
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use.",)
    parser.add_argument("--dataset", type=float, default='CIFAR10', help="The dataset to use.", )
    parser.add_argument("--data_dir", type=str, default='data/', help="The data directory to use.", )
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers to use.", )


    args = parser.parse_args()

    main(args)
