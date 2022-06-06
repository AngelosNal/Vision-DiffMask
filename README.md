# Vision DiffMask: Interpretability of Computer Vision models with Differentiable Patch Masking

## Overview
This repository contains *Vision DiffMask*, which is an adaptation of the
[DiffMask](https://arxiv.org/pdf/2004.14992.pdf) **[1]** algorithm for the
vision domain, and is heavily inspired by [its original PyTorch
implementation](https://github.com/nicola-decao/diffmask).

DiffMask is an attribution model that has been applied for Transformer and
RNN architectures in NLP tasks.  In particular, given a pre-trained classifier,
DiffMask predicts how much each part of the input contributes to the final prediction.

In this implementation, we extend the original idea for Computer Vision tasks.
Currently, only Vision Transformers (ViT) in the image classification setting are supported.

## Setup
We provide a conda environment for the installation of the required packages.
```bash
conda env create -f environment.yml
```

## Project Structure
The project is organized in the following way:
```
.
├── code                                Main entry point for Vision DiffMask
│   ├── attributions                    Re-implementation of other attribution methods
│   │   ├── attention_rollout.py        Attention Rollout (Abnar and Zuidema, 2019)
│   │   └── grad_cam.py                 Grad-CAM (Selvaraju et al., 2016)
│   ├── datamodules                     PL Datamodules for vision datasets
│   │   ├── base.py                     Base image datamodule
│   │   ├── image_classification.py     MNIST and CIFAR-10 datamodules
│   │   ├── transformations.py          Image transformations for torchvision
│   │   ├── utils.py                    Utilities for loading datamodules
│   │   └── visual_qa.py                Datamodule for the toy task of counting patches
│   ├── eval_base.py                    Evaluation script for a base vision model
│   ├── main.py                         Training script for the Vision DiffMask model
│   ├── models                          PL Modules for classification models & Vision DiffMask
│   │   ├── classification.py           Module for image classification
│   │   ├── gates.py                    Gating mechanisms for Vision DiffMask
│   │   ├── interpretation.pt           Main module for Vision DiffMask
│   │   └── utils.py                    Utilities for loading models
│   ├── train_base.py                   Training script for a base vision model
│   └── utils                           Various utilities and auxiliary classes
│       ├── distributions.py            Distributions used by Vision DiffMask
│       ├── getters_setters.py          Hooks for the base models used by Vision DiffMask
│       ├── metrics.py                  Metrics used for training and evaluation
│       ├── optimizer.py                Optimizers used by Vision DiffMask
│       └── plot.py                     Functions and callbacks for visualization
└── experiments                         Notebooks for experiment replication
    ├── diffmask_visualization.ipynb    Visualization of Vision Diffmask on sample images
    ├── inference_time.ipynb            Comparison of inference time with other methods
    ├── qualitative_comparison.ipynb    Qualitative comparison with other methods
    ├── quantitative_comparison.ipynb   Quantitative comparison with other methods
    └── toy_dataset.ipynb               Faithfulness verification on the toy task
```

## Training
To train a Vision DiffMask model on CIFAR-10 based on the Vision Transformer, use the following command:
```bash
python code/main.py --enable_progress_bar --num_epochs 20 --base_model ViT --dataset CIFAR10 \
                    --from_pretrained tanlq/vit-base-patch16-224-in21k-finetuned-cifar10
```
You can refer to the next section for a full list of launch options.

## Launch Arguments
<details>
<summary><b>Vision DiffMask</b></summary>

When training Vision DiffMask, the following launch options can be used:
```
Arguments:
  --enable_progress_bar
                        Whether to enable the progress bar (NOT recommended when logging to file).
  --num_epochs NUM_EPOCHS
                        Number of epochs to train.
  --seed SEED           Random seed for reproducibility.
  --sample_images SAMPLE_IMAGES
                        Number of images to sample for the mask callback.
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Number of steps between logging media & checkpoints.
  --base_model {ViT}    Base model architecture to train.
  --from_pretrained FROM_PRETRAINED
                        The name of the pretrained HF model to load.
  --dataset {MNIST,CIFAR10,CIFAR10_QA,toy}
                        The dataset to use.

Vision DiffMask:
  --alpha ALPHA         Initial value for the Lagrangian
  --lr LR               Learning rate for DiffMask.
  --eps EPS             KL divergence tolerance.
  --no_placeholder      Whether to not use placeholder
  --lr_placeholder LR_PLACEHOLDER
                        Learning for mask vectors.
  --lr_alpha LR_ALPHA   Learning rate for lagrangian optimizer.
  --mul_activation MUL_ACTIVATION
                        Value to multiply gate activations.
  --add_activation ADD_ACTIVATION
                        Value to add to gate activations.
  --weighted_layer_distribution
                        Whether to use a weighted distribution when picking a layer in DiffMask forward.

Data Modules:
  --data_dir DATA_DIR   The directory where the data is stored.
  --batch_size BATCH_SIZE
                        The batch size to use.
  --add_noise           Use gaussian noise augmentation.
  --add_rotation        Use rotation augmentation.
  --add_blur            Use blur augmentation.
  --num_workers NUM_WORKERS
                        Number of workers to use for data loading.

Visual QA:
  --class_idx CLASS_IDX
                        The class (index) to count.
  --grid_size GRID_SIZE
                        The number of images per row in the grid.
```
</details>

<details>
<summary><b>Training the base model</b></summary>

When training the base model (usually not needed as we support pretrained models from HuggingFace),
the following launch options can be used:
```
Arguments:
  --checkpoint CHECKPOINT
                        Checkpoint to resume the training from.
  --enable_progress_bar
                        Whether to show progress bar during training. NOT recommended when logging to files.
  --num_epochs NUM_EPOCHS
                        Number of epochs to train.
  --seed SEED           Random seed for reproducibility.
  --base_model {ViT,ConvNeXt}
                        Base model architecture to train.
  --from_pretrained FROM_PRETRAINED
                        The name of the pretrained HF model to fine-tune from.
  --dataset {MNIST,CIFAR10,CIFAR10_QA,toy}
                        The dataset to use.

Classification Model:
  --optimizer {AdamW,RAdam}
                        The optimizer to use to train the model.
  --weight_decay WEIGHT_DECAY
                        The optimizer's weight decay.
  --lr LR               The initial learning rate for the model.

Data Modules:
  --data_dir DATA_DIR   The directory where the data is stored.
  --batch_size BATCH_SIZE
                        The batch size to use.
  --add_noise           Use gaussian noise augmentation.
  --add_rotation        Use rotation augmentation.
  --add_blur            Use blur augmentation.
  --num_workers NUM_WORKERS
                        Number of workers to use for data loading.

Visual QA:
  --class_idx CLASS_IDX
                        The class (index) to count.
  --grid_size GRID_SIZE
                        The number of images per row in the grid.
```
</details>

<details>
<summary><b>Evaluating the base model</b></summary>

When evaluating the base model, the following launch options can be used:
```
Arguments:
  --checkpoint CHECKPOINT
                        Checkpoint to resume the training from.
  --enable_progress_bar
                        Whether to show progress bar during training. NOT recommended when logging to files.
  --seed SEED           Random seed for reproducibility.
  --base_model {ViT,ConvNeXt}
                        Base model architecture to train.
  --from_pretrained FROM_PRETRAINED
                        The name of the pretrained HF model to fine-tune from.
  --dataset {MNIST,CIFAR10,CIFAR10_QA,toy}
                        The dataset to use.

Data Modules:
  --data_dir DATA_DIR   The directory where the data is stored.
  --batch_size BATCH_SIZE
                        The batch size to use.
  --add_noise           Use gaussian noise augmentation.
  --add_rotation        Use rotation augmentation.
  --add_blur            Use blur augmentation.
  --num_workers NUM_WORKERS
                        Number of workers to use for data loading.

Visual QA:
  --class_idx CLASS_IDX
                        The class (index) to count.
  --grid_size GRID_SIZE
                        The number of images per row in the grid.
```
</details>

## Contributing
This project is licensed under the [MIT license](LICENSE).

## References
**[1]** De Cao, N., Schlichtkrull, M. S., Aziz, W., & Titov, I. (2020, November). How do Decisions Emerge across Layers in Neural Models? Interpretation with Differentiable Masking. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ (pp. 3243-3255).
