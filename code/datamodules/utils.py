from .image_classification import CIFAR10DataModule, ImageDataModule, MNISTDataModule
from .transformations import UnNest
from .visual_qa import CIFAR10QADataModule, ToyQADataModule
from argparse import Namespace
from transformers import ConvNextFeatureExtractor, ViTFeatureExtractor


def get_configs(args: Namespace) -> tuple[dict, dict]:
    """Get the model and feature extractor configs from the command line args.

    Args:
        args (Namespace): the argparse Namespace object

    Returns:
         a tuple containing the model and feature extractor configs
    """
    if args.dataset == "MNIST":
        # We upsample the MNIST images to 112x112, with 1 channel (grayscale)
        # and 10 classes (0-9). We normalize the image to have a mean of 0.5
        # and a standard deviation of ±0.5.
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

        # We upsample the CIFAR10 images to 224x224, with 3 channels (RGB) and
        # 10 classes (0-9) for the normal dataset, or (grid_size)^2 + 1 for the
        # toy task.  We normalize the image to have a mean of 0.5 and a standard
        # deviation of ±0.5.
        model_cfg_args = {
            "image_size": 224,  # fixed to 224 because pretrained models have that size
            "num_channels": 3,
            "num_labels": (args.grid_size**2) + 1
            if args.dataset == "CIFAR10_QA"
            else 10,
        }
        fe_cfg_args = {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    elif args.dataset == "toy":
        # We use an image size so that each patch contains a single color, with
        # 3 channels (RGB) and (grid_size)^2 + 1 classes. We normalize the image
        # to have a mean of 0.5 and a standard deviation of ±0.5.
        model_cfg_args = {
            "image_size": args.grid_size * 16,
            "num_channels": 3,
            "num_labels": (args.grid_size**2) + 1,
        }
        fe_cfg_args = {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    # Set the feature extractor's size attribute to  be the same as the model's image size
    fe_cfg_args["size"] = model_cfg_args["image_size"]
    # Set the tensors' return type to PyTorch tensors
    fe_cfg_args["return_tensors"] = "pt"

    return model_cfg_args, fe_cfg_args


def datamodule_factory(args: Namespace) -> ImageDataModule:
    """A factory method for creating a datamodule based on the command line args.

    Args:
        args (Namespace): the argparse Namespace object

    Returns:
        an ImageDataModule instance
    """
    # Get the model and feature extractor configs
    model_cfg_args, fe_cfg_args = get_configs(args)

    # Set the feature extractor class based on the provided base model name
    if args.base_model == "ViT":
        fe_class = ViTFeatureExtractor
    elif args.base_model == "ConvNeXt":
        fe_class = ConvNextFeatureExtractor
    else:
        raise Exception(f"Unknown base model: {args.base_model}")

    # Create the feature extractor instance
    if args.from_pretrained:
        feature_extractor = fe_class.from_pretrained(
            args.from_pretrained, **fe_cfg_args
        )
    else:
        feature_extractor = fe_class(**fe_cfg_args)

    # Un-nest the feature extractor's output
    feature_extractor = UnNest(feature_extractor)

    # Define the datamodule's configuration
    dm_cfg = {
        "feature_extractor": feature_extractor,
        "batch_size": args.batch_size,
        "add_noise": args.add_noise,
        "add_rotation": args.add_rotation,
        "add_blur": args.add_blur,
        "num_workers": args.num_workers,
    }

    # Determine the dataset class based on the provided dataset name
    if args.dataset.startswith("CIFAR10"):
        if args.dataset == "CIFAR10":
            dm_class = CIFAR10DataModule
        elif args.dataset == "CIFAR10_QA":
            dm_cfg["class_idx"] = args.class_idx
            dm_cfg["grid_size"] = args.grid_size
            dm_class = CIFAR10QADataModule
        else:
            raise Exception(f"Unknown CIFAR10 variant: {args.dataset}")
    elif args.dataset == "MNIST":
        dm_class = MNISTDataModule
    elif args.dataset == "toy":
        dm_cfg["class_idx"] = args.class_idx
        dm_cfg["grid_size"] = args.grid_size
        dm_class = ToyQADataModule
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    return dm_class(**dm_cfg)
