from argparse import ArgumentParser, Namespace
from attributions import attention_rollout, grad_cam
from datamodules import CIFAR10QADataModule, ImageDataModule
from datamodules.utils import datamodule_factory
from functools import partial
from models import ImageInterpretationNet
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import ViTForImageClassification
from utils.plot import DrawMaskCallback, log_masks

import pytorch_lightning as pl


def get_experiment_name(args: Namespace):
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


def setup_sample_image_logs(
    dm: ImageDataModule,
    args: Namespace,
    logger: WandbLogger,
    n_panels: int = 3,  # TODO: change?
):
    """Setup the log callbacks for sampling and plotting images."""
    images_per_panel = args.sample_images

    # Sample images
    sample_images = []
    iter_loader = iter(dm.val_dataloader())
    train_iter_loader = iter(dm.train_dataloader())
    for panel in range(n_panels):
        X, Y = next(iter_loader)
        sample_images += [(X[:images_per_panel], Y[:images_per_panel])]

    # Define mask callback
    mask_cb = partial(DrawMaskCallback, log_every_n_steps=args.log_every_n_steps)
    
    callbacks = []
    for panel in range(n_panels):
        # Initialize ViT model
        vit = ViTForImageClassification.from_pretrained(args.from_pretrained)

        # Extract samples for current panel
        samples = sample_images[panel]
        X, _ = samples

        # Log GradCAM
        gradcam_masks = grad_cam(X, vit)
        log_masks(X, gradcam_masks, f"GradCAM {panel}", logger)

        # Log Attention Rollout
        rollout_masks = attention_rollout(X, vit)
        log_masks(X, rollout_masks, f"Attention Rollout {panel}", logger)

        # Create mask callback
        callbacks += [mask_cb(samples, key=f"{panel}")]
        if panel == 0:
            X_train, y_train = next(train_iter_loader)
            samples_train = (X_train[:images_per_panel], y_train[:images_per_panel])
            callbacks += [mask_cb(samples_train, key=f"train_{panel}")]

    return callbacks


def main(args: Namespace):
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
    if args.diffmask_checkpoint:
        diffmask = ImageInterpretationNet.load_from_checkpoint(
            args.diffmask_checkpoint)

    diffmask.set_vision_transformer(model)

    # Create wandb logger instance
    wandb_logger = WandbLogger(
        name=args.experiment_name if args.experiment_name else get_experiment_name(args),
        project="Patch-DiffMask",
    )

    # Create checkpoint callback
    ckpt_cb = ModelCheckpoint(
        save_top_k=-1,
        dirpath=f"checkpoints/{wandb_logger.version}",
        every_n_train_steps=args.log_every_n_steps,
    )

    # Create mask callbacks
    mask_cbs = setup_sample_image_logs(dm, args, wandb_logger)

    # Create trainer
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ckpt_cb, *mask_cbs],
        enable_progress_bar=args.enable_progress_bar,
        logger=wandb_logger,
        max_epochs=args.num_epochs,
    )

    # Train the model
    trainer.fit(diffmask, dm)


if __name__ == "__main__":
    parser = ArgumentParser()

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

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Name of the experiment.",
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
