"""Module with utilitiy functions for norming and denorming tensors."""

import math

import torch


def fn_lognorm255(tensor: torch.Tensor):
    """
    Takes a tensor with values [0,1], mapping it to the same range.
    Small values will be larger and closer to mid range.
    This is more derivative friendly.
    Args:
        tensor (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """
    return torch.log(255 * tensor + 1) / math.log(256)


def fn_lognorm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Map a tensor with uint8 [0 .. 255] values to [0 .. 1]

    Args:
        tensor (torch.Tensor): input tensor
    Returns:
        torch.Tensor: normalized tensor
    """
    return torch.log(tensor + 1) / math.log(256)


def denorm(tensor: torch.Tensor):
    """Procedure for denorming the tensor

    Args:
        tensor (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """
    out = tensor.mul_(0.5).add_(0.5)
    out = torch.pow(256, out).add_(-1).clamp_(0, 255)  # .permute(1,2,0)
    out = out[:, :, 3:-3, :]
    return out
