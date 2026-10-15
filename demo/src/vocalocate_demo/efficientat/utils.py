import math
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch import Tensor


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(x: Tensor, dim: int, mode: str = "pool", pool_fn:  Callable[[Tensor, int], Tensor] = torch.mean,
                 combine_dim: int = None):
    """
    Collapses dimension of multi-dimensional tensor by pooling or combining dimensions
    :param x: input Tensor
    :param dim: dimension to collapse
    :param mode: 'pool' or 'combine'
    :param pool_fn: function to be applied in case of pooling
    :param combine_dim: dimension to join 'dim' to
    :return: collapsed tensor
    """
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)


class CollapseDim(nn.Module):
    def __init__(self, dim: int, mode: str = "pool", pool_fn:  Callable[[Tensor, int], Tensor] = torch.mean,
                 combine_dim: int = None):
        super(CollapseDim, self).__init__()
        self.dim = dim
        self.mode = mode
        self.pool_fn = pool_fn
        self.combine_dim = combine_dim

    def forward(self, x):
        return collapse_dim(x, dim=self.dim, mode=self.mode, pool_fn=self.pool_fn, combine_dim=self.combine_dim)


class ConvNormActivation(nn.Sequential):
    """Conv2d + norm + activation, matching torchvision.ops.misc.ConvNormActivation's layout and
    parameter names, so pretrained EfficientAT state dicts load without torchvision installed."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, dilation=1, inplace=True, bias=None):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                            groups=groups, bias=bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace) if inplace is not None else activation_layer())
        super().__init__(*layers)
        self.out_channels = out_channels


def NAME_TO_WIDTH(name):
    """Width multiplier from an EfficientAT model name, e.g. mn10_as -> 1.0 (from helpers/utils.py)."""
    widths = {"mn01": 0.1, "mn02": 0.2, "mn04": 0.4, "mn05": 0.5, "mn06": 0.6, "mn08": 0.8, "mn10": 1.0,
              "mn12": 1.2, "mn14": 1.4, "mn16": 1.6, "mn20": 2.0, "mn30": 3.0, "mn40": 4.0}
    return widths.get(name.split("_")[0], 1.0)
