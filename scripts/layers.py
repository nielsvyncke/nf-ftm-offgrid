"""Generic channels-first building blocks used by the segmentation backbone.

This module is self-contained and has no external model dependencies. It
provides a small ``partialize`` helper for lazily-configured modules together
with channels-first ``Linear``, ``LayerNorm``, ``MLP`` and ``DoubleConv``
layers that operate on inputs of shape ``(B, C, *spatial)``.
"""

from typing import Any, Optional, Sequence
from functools import partial
import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair


def as_tuple(obj: Any) -> tuple[Any, ...]:
    """Convert an object to a tuple.

    Sequences (other than strings) are converted directly; any other object is
    wrapped in a single-element tuple.
    """
    if not isinstance(obj, Sequence) or isinstance(obj, str):
        return (obj,)
    return tuple(obj)


def partialize(obj):
    """Wrap ``obj`` into a partial callable with pre-bound arguments.

    ``obj`` is either a callable (returned as-is) or a tuple whose first element
    is a callable and whose remaining elements are positional arguments (given
    as tuples) and/or keyword arguments (given as dicts).
    """
    if callable(obj):
        return obj

    if isinstance(obj, Sequence) and callable(obj[0]):
        callable_obj = obj[0]
        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        for item in obj[1:]:
            if isinstance(item, dict):
                kwargs.update(item)
            elif isinstance(item, Sequence) and not isinstance(item, str):
                args.extend(item)
            else:
                args.append(item)

        return partial(callable_obj, *args, **kwargs)

    raise TypeError(f"Expected a callable or valid tuple, got {type(obj).__name__}")


class Linear(nn.Module):
    """Linear layer for channels-first inputs.

    Applies a 1x1 convolution over the flattened spatial dimensions, i.e. a
    per-position linear transformation over the channel dimension.

    Shape:
        - Input: ``(N, C_in, *)``
        - Output: ``(N, C_out, *)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = self.flatten(x)  # flatten spatial dimensions
        x = self.linear(x)
        x = x.view(original_shape[0], -1, *original_shape[2:])
        return x


class LayerNorm(nn.Module):
    """Layer normalization over the channel dimension for channels-first inputs.

    Normalizes inputs of shape ``(B, C, *spatial)`` across the channel
    dimension ``C``.
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S1, S2, ..., Sp)
        out = torch.einsum("b c ... -> b ... c", x)
        out = self.norm(out)
        out = torch.einsum("b ... c -> b c ...", out)
        return out


class MLP(nn.Module):
    """Channels-first feed-forward network with one hidden layer.

    Shape:
        - Input: ``(N, C_in, *)``
        - Output: ``(N, C_out, *)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        ratio: float = 4.0,
        dropout: float | tuple[float, float] = 0.0,
        **kwargs,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or int(ratio * in_channels)
        dropout = _pair(dropout)  # ensure a (hidden, output) pair of rates

        self.block = nn.Sequential(
            Linear(in_channels, hidden_channels, **kwargs),
            nn.GELU(),
            nn.Dropout(dropout[0]),
            Linear(hidden_channels, out_channels, **kwargs),
            nn.Dropout(dropout[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """(Conv -- Drop -- Norm -- Act) applied twice.

    Default convolutional block for the generic U-Net backbone.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv = partialize(conv)
        drop = partialize(drop)
        norm = partialize(norm)
        act = partialize(act)

        self.block1 = nn.Sequential(
            conv(in_channels, mid_channels, stride=stride),
            drop(),
            norm(mid_channels),
            act(),
        )

        self.block2 = nn.Sequential(
            conv(mid_channels, out_channels, stride=1),
            drop(),
            norm(out_channels),
            act(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        return out
