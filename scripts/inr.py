"""
INR modules implemented from scratch (no external library dependency).

Models
------
FourierFeatureMLP   Fixed random Fourier encoding + ReLU MLP (Tancik et al., 2020)
SIREN               Sinusoidal activations with SIREN init (Sitzmann et al., 2020)
FINER               Variable-periodic activation (Liu et al., 2023)
WIRE                Real Gabor / wavelet activations (Saragadam et al., 2023)

All four share the same public API::

    net = Model(in_features, out_features, hidden_features, hidden_layers, ...)
    y   = net(x)   # x: (*, in_features) -> y: (*, out_features)

The final output activation is ``tanh`` for all models so that zeroing the
last nn.Linear in ``build_inr`` produces exactly-zero output at step 0.

Parameter-count notes (in=2, hidden=H, layers=L, out=O)
---------------------------------------------------------
SIREN/FINER:  3H + L*(H²+H) + H*O + O           (FINER adds H*(1+L) alpha params)
FF-MLP:       (2F+1)*H + L*(H²+H) + H*O + O     (extra (2F-2)*H vs SIREN)
WIRE:         6H + L*(2H²+2H) + H*O + O          (~2× hidden-layer params vs SIREN)

Because the output layer H*O dominates (O = 2*n*d²  with d=C/n≫H), the
absolute differences are small (<5 %) in all practical configurations.
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor


# ─── Fourier-Feature MLP ──────────────────────────────────────────────────────

class FourierFeatureMLP(nn.Module):
    """Fourier-Feature MLP (Tancik et al., NeurIPS 2020).

    Encodes input coordinates with a fixed random Fourier basis::

        γ(x) = [cos(2π B x), sin(2π B x)]   B ∈ R^{F × in}

    then passes γ(x) through a standard ReLU MLP.  The frequency matrix B
    is sampled once at construction and stored as a non-trainable buffer.

    Parameters
    ----------
    num_frequencies : int
        Number of random frequency vectors F.  Encoding dim = 2F.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        num_frequencies: int = 16,
    ):
        super().__init__()
        # Frequency coordinates sampled uniformly in [-1, 1].
        B = torch.empty(num_frequencies, in_features).uniform_(-1.0, 1.0)
        self.register_buffer("B", B)

        dims = [2 * num_frequencies] + [hidden_features] * (hidden_layers + 1)
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-1], out_features), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        proj = (2.0 * math.pi * x) @ self.B.T  # (*, F)
        return self.net(torch.cat([proj.cos(), proj.sin()], dim=-1))


# ─── SIREN ────────────────────────────────────────────────────────────────────

import math

import torch
from torch import Tensor, nn

from typing import Callable

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        layer_class: type[nn.Module] = nn.Linear,
        output_activation: Callable[[Tensor], Tensor] = torch.tanh,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([layer_class(in_features, hidden_features, **kwargs)])
        for _ in range(hidden_layers):
            self.layers.append(layer_class(hidden_features, hidden_features, **kwargs))

        self.layers.append(nn.Linear(hidden_features, out_features))

        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return self.output_activation(x)

class ReLUMLP(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 40,
        bias: bool = True,
        init_weights: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.bias = bias

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(
                -math.sqrt(6 / self.in_features) / self.omega,
                math.sqrt(6 / self.in_features) / self.omega,
            )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(x))


class SIREN(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        omega: float = 30,
        bias: bool = True,
        init_weights: bool = True,
        output_activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=SineLayer,
            output_activation=output_activation,
            omega=omega,
            bias=bias,
            init_weights=init_weights,
        )
class RealGaborLayer(nn.Module):
    """
    Real Gabor layer.

    Args:
        in_features: Input features
        out_features: Output features
        omega: Frequency of Gabor sinusoid term
        scale: Scaling of Gabor Gaussian term
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        scale: float = 30.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.scale = scale
        self.bias = bias

        self.linear = nn.Linear(in_features, 2 * out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        proj = self.linear(x)
        omega, scale = proj.chunk(2, dim=-1)
        omega = self.omega * omega
        scale = self.scale * scale
        return torch.cos(omega) * torch.exp(-(scale**2))


class WIRE(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        omega: float = 20,
        scale: float = 10,
        bias: bool = True,
        output_activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        super().__init__(
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            layer_class=RealGaborLayer,
            omega=omega,
            scale=scale,
            bias=bias,
            output_activation=output_activation,
        )

# ─── FINER ────────────────────────────────────────────────────────────────────

class _FinerLayer(nn.Module):
    """FINER layer with variable-periodic activation.

    φ_i(u_i) = sin(ω · u_i · (1 + |α_i| · |u_i|))

    where u_i = (Wx + b)_i is the i-th pre-activation and α_i is a
    learnable per-neuron frequency scalar initialised from U(-1, 1).
    At α=0 the activation reduces to standard SIREN.

    Reference: Liu et al., "FINER: Flexible Spectral-bias Tuning in Implicit
    NEural Representations by Variable-periodic Activation Functions", 2023.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        self.alpha = nn.Parameter(torch.empty(out_features).uniform_(-1.0, 1.0))
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega
                self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        u = self.linear(x)
        return torch.sin(self.omega * u * (1.0 + self.alpha.abs() * u.abs()))


class FINER(nn.Module):
    """FINER: Flexible INR with variable-periodic activations.

    Extends SIREN with a learnable per-neuron α_i that allows each neuron
    to cover a broader and more adaptive frequency range.  The α parameters
    add H*(1+L) extra scalars over SIREN (< 1 % overhead for typical H).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        omega: float = 30.0,
        output_activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [_FinerLayer(in_features, hidden_features, omega, is_first=True)]
        )
        for _ in range(hidden_layers):
            self.layers.append(_FinerLayer(hidden_features, hidden_features, omega))
        self.layers.append(nn.Linear(hidden_features, out_features))
        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.output_activation(self.layers[-1](x))