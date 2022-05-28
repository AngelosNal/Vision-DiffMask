from datamodules.utils import get_configs
from transformers import (
    ConvNextConfig,
    ConvNextForImageClassification,
    PreTrainedModel,
    ViTConfig,
    ViTForImageClassification,
)

import argparse
import torch


def set_clf_head(base: PreTrainedModel, num_classes: int):
    """Set the classification head of the model in case of an output mismatch.

    Args:
        base (PreTrainedModel): the model to modify
        num_classes (int): the number of classes to use for the output layer
    """
    if base.classifier.out_features != num_classes:
        in_features = base.classifier.in_features
        base.classifier = torch.nn.Linear(in_features, num_classes)


def model_factory(
    args: argparse.Namespace,
    own_config: bool = False,
) -> PreTrainedModel:
    """A factory method for creating a HuggingFace model based on the command line args.

    Args:
        args (Namespace): the argparse Namespace object
        own_config (bool): whether to create our own model config instead of a pretrained one;
            this is recommended when the model was pre-trained on another task with a different
            amount of classes for its classifier head

    Returns:
        a PreTrainedModel instance
    """
    if args.base_model == "ViT":
        # Create a new Vision Transformer
        config_class = ViTConfig
        base_class = ViTForImageClassification
    elif args.base_model == "ConvNeXt":
        # Create a new ConvNext model
        config_class = ConvNextConfig
        base_class = ConvNextForImageClassification
    else:
        raise Exception(f"Unknown base model: {args.base_model}")

    # Get the model config
    model_cfg_args, _ = get_configs(args)
    if not own_config and args.from_pretrained:
        # Create a model from a pretrained model
        base = base_class.from_pretrained(args.from_pretrained)
        # Set the classifier head if needed
        set_clf_head(base, model_cfg_args["num_labels"])
    else:
        # Create a model based on the config
        config = config_class(**model_cfg_args)
        base = base_class(config)

    return base
