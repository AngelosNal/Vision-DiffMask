from .base import ImageDataModule
from torch.utils.data import random_split
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from typing import Optional
from torch.utils.data import DataLoader


class MNISTDataModule(ImageDataModule):
    """Datamodule for the MNIST dataset."""

    def prepare_data(self):
        # Download MNIST
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Set the training and validation data
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(mnist_full, [55000, 5000])

        # Set the test data
        if stage == "test" or stage is None:
            self.test_data = MNIST(self.data_dir, train=False, transform=self.transform)


class CIFAR10DataModule(ImageDataModule):
    """Datamodule for the CIFAR10 dataset."""

    def prepare_data(self):
        # Download CIFAR10
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Set the training and validation data
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(cifar10_full, [45000, 5000])

        # Set the test data
        if stage == "test" or stage is None:
            self.test_data = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

class ImageNetDataModule(ImageDataModule):
    """
    Datamodule for the ImageNet dataset.
    """
    def prepare_data(self):
        """
        No automatic download of ImageNet data. Additionally,
        no automatic extraction here, it takes quite a while to
        even check the correctness of the data.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        # Set the training and validation data
        if stage == "fit" or stage is None:
            train_data = ImageNet('/nfs/scratch/dl2john/', split='train', transform=self.transform)
            val_data = ImageNet('/nfs/scratch/dl2john/', split='val', transform=self.transform)
            self.train_data, self.val_data = train_data, val_data

        # Set the test data
        if stage == "test" or stage is None:
            self.test_data = ImageNet('/nfs/scratch/dl2john/', split='val', transform=self.transform)
    
    # noinspection PyTypeChecker
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.shuffled_sampler(self.val_data),
        )