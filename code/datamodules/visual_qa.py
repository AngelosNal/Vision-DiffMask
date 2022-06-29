from .image_classification import CIFAR10DataModule
from argparse import ArgumentParser
from functools import partial
from torch import LongTensor
from torch.utils.data import random_split, Sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets import VisionDataset
from typing import Iterator, Optional

import itertools
import random
import torch


class CIFAR10QADataModule(CIFAR10DataModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Visual QA")
        parser.add_argument(
            "--class_idx",
            type=int,
            default=3,
            help="The class (index) to count.",
        )
        parser.add_argument(
            "--grid_size",
            type=int,
            default=3,
            help="The number of images per row in the grid.",
        )
        return parent_parser

    def __init__(
        self,
        class_idx: int,
        grid_size: int = 3,
        feature_extractor: callable = None,
        data_dir: str = "data/",
        batch_size: int = 32,
        add_noise: bool = False,
        add_rotation: bool = False,
        add_blur: bool = False,
        num_workers: int = 4,
    ):
        """A datamodule for a modified CIFAR10 dataset that is used for Question Answering.
        More specifically, the task is to count the number of images of a certain class in a grid.

        Args:
            class_idx (int): the class (index) to count
            grid_size (int): the number of images per row in the grid
            feature_extractor (callable): a callable feature extractor instance
            data_dir (str): the directory to store the dataset
            batch_size (int): the batch size for the train/val/test dataloaders
            add_noise (bool): whether to add noise to the images
            add_rotation (bool): whether to add rotation augmentation
            add_blur (bool): whether to add blur augmentation
            num_workers (int): the number of workers to use for data loading
        """
        super().__init__(
            feature_extractor,
            data_dir,
            (grid_size**2) * batch_size,
            add_noise,
            add_rotation,
            add_blur,
            num_workers,
        )

        # Store hyperparameters
        self.class_idx = class_idx
        self.grid_size = grid_size

        # Save the existing transformations to be applied after creating the grid
        self.post_transform = self.transform
        # Set the pre-batch transformation to be the conversion from PIL to tensor
        self.transform = transforms.PILToTensor()

        # Specify the custom collate function and samplers
        self.collate_fn = self.custom_collate_fn
        self.shuffled_sampler = partial(
            FairGridSampler,
            class_idx=class_idx,
            grid_size=grid_size,
            shuffle=True,
        )
        self.sequential_sampler = partial(
            FairGridSampler,
            class_idx=class_idx,
            grid_size=grid_size,
            shuffle=False,
        )

    def custom_collate_fn(self, batch):
        # Split the batch into groups of grid_size**2
        idx = range(len(batch))
        grids = zip(*(iter(idx),) * (self.grid_size**2))

        new_batch = []
        for grid in grids:
            # Create a grid of images from the indices in the batch
            img = torch.hstack(
                [
                    torch.dstack(
                        [batch[i][0] for i in grid[idx : idx + self.grid_size]]
                    )
                    for idx in range(
                        0, self.grid_size**2 - self.grid_size + 1, self.grid_size
                    )
                ]
            )
            # Apply the post transformations to the grid
            img = self.post_transform(img)
            # Define the target as the number of images that have the class_idx
            targets = [batch[i][1] for i in grid]
            target = targets.count(self.class_idx)
            # Append grid and target to the batch
            new_batch += [(img, target)]

        return default_collate(new_batch)


class ToyQADataModule(CIFAR10QADataModule):
    """A datamodule for the toy dataset as described in the paper."""

    def prepare_data(self):
        # No need to download anything for the toy task
        pass

    def setup(self, stage: Optional[str] = None):
        img_size = 16

        samples = []
        # Generate 6000 samples based on 6 different colors
        for r, g, b in itertools.product((0, 1), (0, 1), (0, 1)):
            if r == g == b:
                # We do not want black/white patches
                continue

            for _ in range(1000):
                patch = torch.vstack(
                    [
                        r * torch.ones(1, img_size, img_size),
                        g * torch.ones(1, img_size, img_size),
                        b * torch.ones(1, img_size, img_size),
                    ]
                )

                # Assign a unique id to each color
                target = int(f"{r}{g}{b}", 2) - 1
                # Append the patch and target to the samples
                samples += [(patch, target)]

        # Split the data to 90% train, 5% validation and 5% test
        train_size = int(len(samples) * 0.9)
        val_size = (len(samples) - train_size) // 2
        test_size = len(samples) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(
            samples,
            [
                train_size,
                val_size,
                test_size,
            ],
        )


class FairGridSampler(Sampler[int]):
    def __init__(
        self,
        dataset: VisionDataset,
        class_idx: int,
        grid_size: int,
        shuffle: bool = False,
    ):
        """A sampler that returns a grid of images from the dataset, with a uniformly random
         amount of appearances for a specific class of interest.

        Args:
            dataset (VisionDataset): the dataset to sample from
            class_idx(int): the class (index) to treat as the class of interest
            grid_size (int): the number of images per row in the grid
            shuffle (bool): whether to shuffle the dataset before sampling
        """
        super().__init__(dataset)

        # Save the hyperparameters
        self.dataset = dataset
        self.grid_size = grid_size
        self.n_images = grid_size**2

        # Get the indices of the class of interest
        self.class_indices = LongTensor(
            [i for i, x in enumerate(dataset) if x[1] == class_idx]
        )
        # Get the indices of all other classes
        self.other_indices = LongTensor(
            [i for i, x in enumerate(dataset) if x[1] != class_idx]
        )

        # Fix the seed if shuffle is False
        self.seed = None if shuffle else self._get_seed()

    @staticmethod
    def _get_seed() -> int:
        """Utility function for generating a random seed."""
        return int(torch.empty((), dtype=torch.int64).random_().item())

    def __iter__(self) -> Iterator[int]:
        # Create a torch Generator object
        seed = self.seed if self.seed is not None else self._get_seed()
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Sample the batches
        for _ in range(len(self.dataset) // self.n_images):
            # Pick the number of instances for the class of interest
            n_samples = torch.randint(self.n_images + 1, (), generator=gen).item()

            # Sample the indices from the class of interest
            idx_from_class = torch.randperm(
                len(self.class_indices),
                generator=gen,
            )[:n_samples]
            # Sample the indices from the other classes
            idx_from_other = torch.randperm(
                len(self.other_indices),
                generator=gen,
            )[: self.n_images - n_samples]

            # Concatenate the corresponding lists of patches to form a grid
            grid = (
                self.class_indices[idx_from_class].tolist()
                + self.other_indices[idx_from_other].tolist()
            )

            # Shuffle the order of the patches within the grid
            random.shuffle(grid)
            yield from grid

    def __len__(self) -> int:
        return len(self.dataset)
