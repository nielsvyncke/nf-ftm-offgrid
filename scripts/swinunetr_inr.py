import torch
from torch import nn
import math

from .layers import partialize, Linear, LayerNorm, MLP
from .unet import UNet

from .inr import FourierFeatureMLP, SIREN, FINER, WIRE, ReLUMLP as INRMLP


def build_inr(name, in_features, out_features, hidden_features=64,
              hidden_layers=2, omega=30.0, scale=10.0,
              ffmlp_num_frequencies=16):
    """Build a small INR that maps a coordinate to kernel weights."""
    name = name.lower()
    common = dict(in_features=in_features, out_features=out_features,
                  hidden_features=hidden_features, hidden_layers=hidden_layers)
    
    identity = nn.Identity()
    if name == "mlp":
        net = INRMLP(**common, output_activation=identity)
    elif name == "siren":
        net = SIREN(**common, omega=omega, output_activation=identity)
    elif name == "wire":
        net = WIRE(**common, omega=omega, scale=scale, output_activation=identity)
    elif name == "finer":
        net = FINER(**common, omega=omega, output_activation=identity)
    elif name == "ffmlp":
        net = FourierFeatureMLP(**common,
                                num_frequencies=ffmlp_num_frequencies)
    else:
        raise ValueError(
            f"Unknown INR '{name}'. Choose from: siren, wire, finer, ffmlp."
        )

    last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
    return net


class NFFTM(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
        inr_hidden_dim=None,
        inr_hidden_layers=2,
        inr_model="siren",
        inr_omega=30.0,
        inr_scale=10.0,
        inr_wire_omega=None,
        inr_wire_scale=None,
        inr_ffmlp_num_frequencies=16,
        **_,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)

        is_wire = inr_model == "wire"
        omega = inr_wire_omega if is_wire and inr_wire_omega is not None else inr_omega
        scale = inr_wire_scale if is_wire and inr_wire_scale is not None else inr_scale

        hidden = inr_hidden_dim or out_channels * 5 // 2

        self.kernel = build_inr(
            inr_model,
            in_features=4,
            out_features=out_channels,  # depthwise: channel mixing handled by MLP
            hidden_features=hidden,
            hidden_layers=inr_hidden_layers,
            omega=omega,
            scale=scale,
            ffmlp_num_frequencies=inr_ffmlp_num_frequencies,
        )

    def _coords(self, h, w, device):
        key = (h, w, device.type, device.index)
        if not hasattr(self, "_coords_cache"):
            self._coords_cache = {}
        if key not in self._coords_cache:
            self._coords_cache[key] = self._compute_coords(h, w, device)
        return self._coords_cache[key]

    @staticmethod
    def _compute_coords(h, w, device, dtype=torch.float32):
        ky_idx = torch.arange(h, device=device)
        kx_idx = torch.arange(w // 2 + 1, device=device)

        ky, kx = torch.meshgrid(ky_idx, kx_idx, indexing="ij")
        ky_conj = (-ky) % h
        kx_conj = (-kx) % w

        theta_y = 2.0 * torch.pi * torch.arange(
            h, device=device, dtype=dtype
        ) / h

        theta_x = 2.0 * torch.pi * torch.arange(
            w, device=device, dtype=dtype
        ) / w

        emb_y = torch.stack((torch.cos(theta_y), torch.sin(theta_y)), dim=-1)
        emb_x = torch.stack((torch.cos(theta_x), torch.sin(theta_x)), dim=-1)

        coords = torch.cat(
            (emb_y[ky], emb_x[kx]),
            dim=-1,
        ).flatten(0, 1)

        conjugate_coords = torch.cat(
            (emb_y[ky_conj], emb_x[kx_conj]),
            dim=-1,
        ).flatten(0, 1)

        return coords, conjugate_coords

    def _spectral_kernel(self, h, w, device):
        coords, conjugate_coords = self._coords(h, w, device)
        # Evaluate both sets in one batched call.
        values = self.kernel(torch.cat((coords, conjugate_coords), dim=0))
        values, conjugate_values = values.chunk(2, dim=0)
        # Even/odd decomposition enforces Hermitian symmetry → real spatial kernel.
        re = 0.5 * (values + conjugate_values)   # even part → real
        im = 0.5 * (values - conjugate_values)   # odd part  → imaginary

        # kernel: (H*(W//2+1), C) — starts near zero, 1+ gives identity filter at init
        return torch.complex(re, im).reshape(-1, self.out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x.float(), dim=(2, 3), norm="ortho")
        kernel = self._spectral_kernel(h, w, x.device)  # (H*(W//2+1), C)
        x_ft = x_ft.reshape(b, c, -1) * kernel.T.unsqueeze(0)  # (B, C, H*(W//2+1))
        x_ft = x_ft.reshape(b, c, h, w // 2 + 1)
        out = torch.fft.irfft2(x_ft, s=(h, w), dim=(2, 3), norm="ortho").to(x.dtype)

        return self.dropout(out)

class TransformerBlock(nn.Module):

    def __init__(self, channels, norm=LayerNorm, dropout=0.0, mlp_ratio=4,
                 spatial_size=None, **kwargs):
        super().__init__()

        self.norm1 = partialize(norm)(channels)
        self.dcm = NFFTM(channels, channels, dropout=dropout, **kwargs)

        self.norm2 = partialize(norm)(channels)
        self.mlp = MLP(channels, ratio=mlp_ratio, dropout=dropout)


    def forward(self, x):
        out = x
        out = out + self.dcm(self.norm1(out))
        out = out + self.mlp(self.norm2(out))

        return out

class TransformerStage(nn.Module):
    """One backbone stage: a sequence of NF-FTM MetaFormer blocks."""


    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size=None,
        depth=1,
        adapter=(Linear, {"bias": False}),
        **kwargs,
    ):
        super().__init__()
        if in_channels != out_channels:
            self.adapter = partialize(adapter)(in_channels, out_channels)


        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                TransformerBlock(
                    out_channels,
                    spatial_size=spatial_size,
                    **kwargs,
                )
            )


    def forward(self, x):
        # x: (B, C, *)
        out = self.adapter(x) if hasattr(self, "adapter") else x
        for blk in self.blocks:
            out = blk(out)


        return out

class Stem(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size=(4, 4), norm=LayerNorm):
        spatial_dims = len(patch_size)
        _conv = getattr(nn, f"Conv{spatial_dims}d")
        _norm = partialize(norm)
        super().__init__(
            _conv(in_channels, out_channels, patch_size, stride=patch_size),
            _norm(out_channels),
        )

class SwinUNETRINR(UNet):


    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_dims=3,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        upsample=None,
        head=None,
        num_deep_supr=False,
        **kwargs,
    ):
        num_stages = len(encoder_depth) + len(decoder_depth)
        block = num_stages * [TransformerStage]
        if stem is None:
            stem = (
                getattr(nn, f"Conv{spatial_dims}d"),
                {"kernel_size": 3, "padding": 1, "bias": False},
            )
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=spatial_dims,
            encoder_depth=encoder_depth,
            encoder_width=encoder_width,
            strides=strides,
            decoder_depth=decoder_depth,
            stem=stem,
            downsample=downsample,
            block=block,
            upsample=upsample,
            head=head,
            num_deep_supr=num_deep_supr,
            **kwargs,
        )
