import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from typing import Optional
import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Unnest(object):
    """
    A class to use after ViTFeatureExtractor in transforms to unnest the tensor.
    """
    def __call__(self, x):
        return x['pixel_values'][0]


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'Data/', batch_size: int = 32, feature_extractor: object = None,
                 noise: bool = False, rotation: bool = False, blur: bool = False):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        # Set the transforms
        assert feature_extractor is not None, "Feature extractor was not specified."
        # TODO: We may need to create another Unnest for other feature extractors than ViTFeatureExtractor
        self.transform = transforms.Compose([feature_extractor, Unnest()])
        self.add_transforms(noise, rotation, blur)

    def add_transforms(self, noise: bool, rotation: bool, blur: bool):
    # TODO: not sure about the order and if we should apply the same transforms both in train and test set
    # TODO: check what transforms are applied by the model
        if noise:
            self.transform.transforms.append(AddGaussianNoise(0., 1.))
        if rotation:
            self.transform.transforms.append(transforms.RandomRotation(20))
        if blur:
            self.transform.transforms.append(transforms.GaussianBlur(3))

    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class MNISTDataModule(ImageDataModule):
    def __init__(self, data_dir: str = "Data/", batch_size: int = 32, feature_extractor: object = None,
                 noise: bool = False, rotation: bool = False, blur: bool = False):
        super().__init__(data_dir, batch_size, feature_extractor, noise, rotation, blur)

        # TODO: We need to find a ViTFeatureExtractor (or create one) for MNIST

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
    def __init__(self, data_dir: str = "Data/", batch_size: int = 32, feature_extractor: object = None,
                 noise: bool = False, rotation: bool = False, blur: bool = False):
        super().__init__(data_dir, batch_size, feature_extractor, noise, rotation, blur)

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
            self.test_data = CIFAR10(self.data_dir, train=False, transform=self.transform)
