from .transformations import AddGaussianNoise
from abc import abstractmethod, ABCMeta
from argparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    default_collate,
    RandomSampler,
    SequentialSampler,
)
from torchvision import transforms
from typing import Optional


class ImageDataModule(LightningDataModule, metaclass=ABCMeta):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data Modules")
        parser.add_argument(
            "--data_dir",
            type=str,
            default="data/",
            help="The directory where the data is stored.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="The batch size to use.",
        )
        parser.add_argument(
            "--add_noise",
            action="store_true",
            help="Use gaussian noise augmentation.",
        )
        parser.add_argument(
            "--add_rotation",
            action="store_true",
            help="Use rotation augmentation.",
        )
        parser.add_argument(
            "--add_blur",
            action="store_true",
            help="Use blur augmentation.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers to use for data loading.",
        )
        return parent_parser

    # Declare variables that will be initialized later
    train_data: Dataset
    val_data: Dataset
    test_data: Dataset

    def __init__(
        self,
        feature_extractor: Optional[callable] = None,
        data_dir: str = "data/",
        batch_size: int = 32,
        add_noise: bool = False,
        add_rotation: bool = False,
        add_blur: bool = False,
        num_workers: int = 4,
    ):
        """Abstract Pytorch Lightning DataModule for image datasets.

        Args:
            feature_extractor (callable): feature extractor instance
            data_dir (str): directory to store the dataset
            batch_size (int): batch size for the train/val/test dataloaders
            add_noise (bool): whether to add noise to the images
            add_rotation (bool): whether to add random rotation to the images
            add_blur (bool): whether to add blur to the images
            num_workers (int): number of workers for train/val/test dataloaders
        """
        super().__init__()

        # Store hyperparameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.num_workers = num_workers

        # Set the transforms
        # If the feature_extractor is None, then we do not split the images into features
        init_transforms = [feature_extractor] if feature_extractor else []
        self.transform = transforms.Compose(init_transforms)
        self._add_transforms(add_noise, add_rotation, add_blur)

        # Set the collate function and the samplers
        # These can be adapted in a child datamodule class to have a different behavior
        self.collate_fn = default_collate
        self.shuffled_sampler = RandomSampler
        self.sequential_sampler = SequentialSampler

    def _add_transforms(self, noise: bool, rotation: bool, blur: bool):
        """Add transforms to the module's transformations list.

        Args:
            noise (bool): whether to add noise to the images
            rotation (bool): whether to add random rotation to the images
            blur (bool): whether to add blur to the images
        """
        # TODO:
        # - Which order to add the transforms in?
        # - Applied in both train and test or just test?
        # - Check what transforms are applied by the model
        if noise:
            self.transform.transforms.append(AddGaussianNoise(0.0, 1.0))
        if rotation:
            self.transform.transforms.append(transforms.RandomRotation(20))
        if blur:
            self.transform.transforms.append(transforms.GaussianBlur(3))

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError()

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError()

    # noinspection PyTypeChecker
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.shuffled_sampler(self.train_data),
        )

    # noinspection PyTypeChecker
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.sequential_sampler(self.val_data),
        )

    # noinspection PyTypeChecker
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.sequential_sampler(self.test_data),
        )
