"""Module containing classes and functions for loading the dataset"""
import os

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from .norm import fn_lognorm255
from .noise import UniformNoise


class ImageEventsDataset(torch.utils.data.Dataset):
    """
    Assumes a directory struture like
    1.1.1/
    ├── some_filename_1
    ├── some_filename_2
    ├── ...
    1.1.2/
    ├── some_filename_1
    ├── some_filename_2
    ├── ...
    ...
    with the same filenames in each directory where one filename corresponds to one event
    and the top-level subdirectories corresponding to the labels.

    Will generate one instance as a set of 40 images of a single event.
    """

    def __init__(self, path, transform):
        super().__init__()
        self.path = path
        self.transform = transform
        self.subdirs = sorted(os.listdir(path))
        self.filenames = sorted(
            os.listdir(os.path.join(path, self.subdirs[0]))
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, event_idx):
        filename = self.filenames[event_idx]
        images = []
        for subdir in self.subdirs:
            image = torchvision.datasets.folder.default_loader(
                os.path.join(self.path, subdir, filename)
            )
            image = self.transform(image)
            images.append(image)
        return torch.stack(images), torch.arange(40)


def load_dataset(data_path: str, num_workers: int, shuffle: bool):
    """Load dataset from path

    Args:
        data_path (str): path to dataset
        num_workers (int): number of workers in DataLoader
        shuffle (bool): shuffle dataset

    Returns:
        torch.utils.data.DataLoader: generates batches of 40 images
    """
    train_dataset = ImageEventsDataset(
        data_path,
        transform=transforms.Compose(
            [
                transforms.Pad((0, 3, 0, 3)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Lambda(fn_lognorm255),
                UniformNoise(scale=4e-3),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    return DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x[0])
