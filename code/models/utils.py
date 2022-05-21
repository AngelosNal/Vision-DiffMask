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
    if base.classifier.out_features != num_classes:
        in_features = base.classifier.in_features
        base.classifier = torch.nn.Linear(in_features, num_classes)


def model_factory(
    args: argparse.Namespace,
    own_config: bool = False,
) -> PreTrainedModel:
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

    model_cfg_args, _ = get_configs(args)
    if not own_config and args.from_pretrained:
        base = base_class.from_pretrained(args.from_pretrained)
        set_clf_head(base, model_cfg_args["num_labels"])
    else:
        config = config_class(**model_cfg_args)
        base = base_class(config)

    return base
