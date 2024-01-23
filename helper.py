import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose

from mmd import MMD

ROOT = "data"


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Reshape(torch.nn.Module):
    def __init__(self, outer_shape: tuple):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), *self.outer_shape)


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.Generator().manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


TRANSFORM = Compose([ToTensor(), Flatten()])


def load_data(
    fashion: bool = True,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Loads pytorch datasets for MNIST / FashionMNIST.

    Args:
        fashion: bool
            Load FashionMNIST.

    Returns:
        Training and test sets.

    Raises:
        None.
    """
    if not os.path.exists(ROOT):
        os.makedirs(ROOT)

    if fashion:
        train = datasets.FashionMNIST(
            root=ROOT,
            train=True,
            download=True,
            transform=TRANSFORM,
        )
        test = datasets.FashionMNIST(
            root=ROOT,
            train=False,
            download=True,
            transform=TRANSFORM,
        )
        return train, test
    else:
        train = datasets.MNIST(
            root=ROOT,
            train=True,
            download=True,
            transform=TRANSFORM,
        )
        test = datasets.MNIST(
            root=ROOT,
            train=False,
            download=True,
            transform=TRANSFORM,
        )
        return train, test