from .image_classification import CIFAR10DataModule, ImageDataModule, MNISTDataModule
from .visual_qa import CIFAR10QADataModule
from argparse import Namespace
from transformers import ConvNextFeatureExtractor, ViTFeatureExtractor


def get_configs(args: Namespace) -> tuple[dict, dict]:
    if args.dataset == "MNIST":
        model_cfg_args = {
            "image_size": 112,
            "num_channels": 1,
            "num_labels": 10,
        }
        fe_cfg_args = {
            "image_mean": [0.5],
            "image_std": [0.5],
        }
    elif args.dataset.startswith("CIFAR10"):
        if args.dataset not in ("CIFAR10", "CIFAR10_QA"):
            raise Exception(f"Unknown CIFAR10 variant: {args.dataset}")

        model_cfg_args = {
            "image_size": 224,
            "num_channels": 3,
            "num_labels": 5 if args.dataset == "CIFAR10_QA" else 10,
        }
        fe_cfg_args = {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    fe_cfg_args["size"] = model_cfg_args["image_size"]

    return model_cfg_args, fe_cfg_args


def datamodule_factory(args: Namespace) -> ImageDataModule:
    model_cfg_args, fe_cfg_args = get_configs(args)
    fe_cfg_args["return_tensors"] = "pt"

    if args.base_model == "ViT":
        fe_class = ViTFeatureExtractor
    elif args.base_model == "ConvNeXt":
        fe_class = ConvNextFeatureExtractor
    else:
        raise Exception(f"Unknown base model: {args.base_model}")

    if args.from_pretrained:
        feature_extractor = fe_class.from_pretrained(
            args.from_pretrained,
            **fe_cfg_args,
        )
    else:
        feature_extractor = fe_class(**fe_cfg_args)

    dm_cfg = {
        "feature_extractor": feature_extractor,
        "batch_size": args.batch_size,
        "add_noise": args.add_noise,
        "add_rotation": args.add_rotation,
        "add_blur": args.add_blur,
        "num_workers": args.num_workers,
    }

    if args.dataset.startswith("CIFAR10"):
        if args.dataset == "CIFAR10":
            dm_class = CIFAR10DataModule
        elif args.dataset == "CIFAR10_QA":
            dm_cfg["class_idx"] = args.class_idx
            dm_class = CIFAR10QADataModule
        else:
            raise Exception(f"Unknown CIFAR10 variant: {args.dataset}")
    elif args.dataset == "MNIST":
        dm_class = MNISTDataModule
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    return dm_class(**dm_cfg)
