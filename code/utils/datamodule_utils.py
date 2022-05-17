from argparse import Namespace
import sys 
sys.path.append('..')
from datamodules import CIFAR10DataModule, MNISTDataModule
import pytorch_lightning as pl
from argparse import Namespace
from transformers import ViTFeatureExtractor, ViTConfig

def datamodule_factory(args: Namespace) -> pl.LightningDataModule:
    if args.dataset == "CIFAR10":
	    return CIFAR10DataModule(
            batch_size=args.batch_size,
                feature_extractor=ViTFeatureExtractor.from_pretrained(
                    args.vit_model, return_tensors="pt"
                ),
                noise=args.add_noise,
                rotation=args.add_rotation,
                blur=args.add_blur,
                num_workers=args.num_workers
        )
    elif args.dataset == "MNIST":
        mnist_cfg = ViTConfig(image_size=112, num_channels=1, num_labels=10)
        mnist_fe = ViTFeatureExtractor(
            size=mnist_cfg.image_size,
            image_mean=[0.5],
            image_std=[0.5],
            return_tensors="pt",
        )
        return MNISTDataModule(
            batch_size=64,
            feature_extractor=mnist_fe,
            noise=args.add_noise,
            rotation=args.add_rotation,
            blur=args.add_blur,
            num_workers=args.num_workers,
        )
    else:
	    raise Exception(f"Unknown dataset: {args.dataset}")