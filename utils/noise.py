"""Module hosting classes which add noise to a torch tensor"""
import numpy as np
import torch
from torchvision import transforms

class UniformNoise:
    """
    Creates a callable object which adds uniform noise to a tensor.
    """

    __slots__ = (
        "scale",
        "inplace",
    )

    def __init__(self, scale: float = 4e-3, inplace: bool = True):
        """
        Set scale of uniform noise.

        Args:
            scale (float, optional): scale of noise level. Defaults to 4e-3.
            inplace (bool, optional): use inplace addition to tensor. Defaults to True.
        """
        self.scale = scale
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        if self.inplace:
            return tensor.add_(self.scale * torch.rand_like(tensor))
        return tensor.add(self.scale * torch.rand_like(tensor))

    def __repr__(self):
        return f"UniformNoise(scale={self.scale}, inplace={self.inplace})"


class GaussianNoise:
    """
    Creates a callable objects which adds gaussian noise to a tensor.
    """

    __slots__ = (
        "std",
        "mean",
        "inplace",
    )

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, inplace: bool = True
    ):
        """
        Set gaussian mean, std and inplace.

        Args:
            mean (float, optional): mean of gaussian. Defaults to 0..
            std (float, optional): standard deviation of gaussian. Defaults to 1..
            inplace (bool, optional): use inplace addition to tensor. Defaults to True.
        """
        self.std = std
        self.mean = mean
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        if self.inplace:
            return tensor.add_(
                torch.randn_like(tensor) * self.std + self.mean
            )
        return tensor.add(torch.randn_like(tensor) * self.std + self.mean)

    def __repr__(self):
        return f"GaussianNoise(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class CenterCropLongEdge:
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class RandomCropLongEdge:
    """Crops the given PIL Image on the long edge with a random start point.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = 0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0])
        j = 0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1])
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__
