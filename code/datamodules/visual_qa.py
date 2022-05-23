from .image_classification import CIFAR10DataModule
from argparse import ArgumentParser
from functools import partial
from torch import LongTensor
from torch.utils.data import default_collate, random_split, Sampler
from torchvision import transforms
from torchvision.datasets import VisionDataset
from transformers import FeatureExtractionMixin
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
        data_dir: str = "data/",
        batch_size: int = 32,
        feature_extractor: FeatureExtractionMixin = None,
        add_noise: bool = False,
        add_rotation: bool = False,
        add_blur: bool = False,
        num_workers: int = 4,
    ):
        super().__init__(
            data_dir,
            (grid_size**2) * batch_size,
            feature_extractor,
            add_noise,
            add_rotation,
            add_blur,
            num_workers,
        )

        self.class_idx = class_idx
        self.grid_size = grid_size

        self.post_transform = self.transform
        self.transform = transforms.PILToTensor()

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
        idx = range(len(batch))
        grids = zip(*(iter(idx),) * (self.grid_size**2))

        new_batch = []
        for grid in grids:
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
            img = self.post_transform(img)
            targets = [batch[i][1] for i in grid]
            target = targets.count(self.class_idx)
            new_batch += [(img, target)]

        return default_collate(new_batch)


class ToyQADataModule(CIFAR10QADataModule):
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        img_size = 16

        samples = []
        for r, g, b in itertools.product((0, 1), (0, 1), (0, 1)):
            if r == g == b:
                continue

            for _ in range(1000):
                patch = torch.vstack(
                    [
                        r * torch.ones(1, img_size, img_size),
                        g * torch.ones(1, img_size, img_size),
                        b * torch.ones(1, img_size, img_size),
                    ]
                )

                target = int(f"{r}{g}{b}", 2) - 1

                samples += [(patch, target)]

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
        super().__init__(dataset)
        self.dataset = dataset
        self.grid_size = grid_size
        self.n_images = grid_size**2

        self.class_indices = LongTensor(
            [i for i, x in enumerate(dataset) if x[1] == class_idx]
        )
        self.other_indices = LongTensor(
            [i for i, x in enumerate(dataset) if x[1] != class_idx]
        )

        self.seed = None if shuffle else self._get_seed()

    @staticmethod
    def _get_seed() -> int:
        return int(torch.empty((), dtype=torch.int64).random_().item())

    def __iter__(self) -> Iterator[int]:
        seed = self.seed if self.seed is not None else self._get_seed()
        gen = torch.Generator()
        gen.manual_seed(seed)

        for _ in range(len(self.dataset) // self.n_images):
            n_samples = torch.randint(self.n_images + 1, (), generator=gen).item()

            idx_from_class = torch.randperm(
                len(self.class_indices),
                generator=gen,
            )[:n_samples]

            idx_from_other = torch.randperm(
                len(self.other_indices),
                generator=gen,
            )[: self.n_images - n_samples]

            grid = (
                self.class_indices[idx_from_class].tolist()
                + self.other_indices[idx_from_other].tolist()
            )

            random.shuffle(grid)
            yield from grid

    def __len__(self) -> int:
        return len(self.dataset)
