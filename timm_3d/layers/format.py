from enum import Enum
from typing import Union

import torch


class Format(str, Enum):
    NCHWD = 'NCHWD'
    NHWDC = 'NHWDC'
    NCL = 'NCL'
    NLC = 'NLC'


FormatT = Union[str, Format]


def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWDC:
        dim = (1, 2, 3)
    else:
        dim = (2, 3, 4)
    return dim


def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWDC:
        dim = 4
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchwd_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWDC:
        x = x.permute(0, 2, 3, 4, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwdc_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NCHWD:
        x = x.permute(0, 4, 1, 2, 3)
    elif fmt == Format.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(1, 2).transpose(1, 2)
    return x
