# ABOUTME: Phase 2 sequence networks: LSTM, temporal CNN and Transformer encoder.
# ABOUTME: Each maps (batch, window, sensors) to RUL / 125 per window, shape (batch,); train.py rescales.
import math
from functools import lru_cache
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class LSTMNet(nn.Module):
    """Stacked LSTM; the hidden state at the last cycle goes through dropout to a linear head."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        # PyTorch applies LSTM dropout only between layers, so a single layer gets none there.
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden
        self.head = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden state after the last cycle, shape (batch, out_dim)."""
        out, _ = self.lstm(x)
        return out[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dropout sits here rather than in encode, so all three encode() outputs are undropped.
        return self.head(self.drop(self.encode(x))).squeeze(-1)


def causal_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """conv applied to x (batch, channels, time) with left padding only, so output t sees cycles <= t.

    Same result as conv(F.pad(x, ((k - 1) * d, 0))) for a conv with groups=1 and no padding
    of its own, computed as k shifted slices and one
    matrix product, because PyTorch's CPU dilated convolution loops over samples one at a
    time and made the TCN about five times slower than the LSTM.
    """
    k, d = conv.kernel_size[0], conv.dilation[0]
    x = F.pad(x, ((k - 1) * d, 0))
    steps = x.shape[-1] - (k - 1) * d
    taps = torch.stack([x[..., j * d: j * d + steps] for j in range(k)], dim=2)
    out = torch.einsum("oik,bikt->bot", conv.weight, taps)
    return out if conv.bias is None else out + conv.bias[:, None]


class CausalBlock(nn.Module):
    """Two dilated causal convolutions with ReLU and dropout, added to a residual path."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(causal_conv(x, self.conv1)))
        h = self.drop(F.relu(causal_conv(h, self.conv2)))
        return F.relu(h + self.skip(x))


class TCNNet(nn.Module):
    """Causal residual blocks with dilations 1, 2, 4, ...; the last cycle feeds a linear head."""

    def __init__(self, n_features: int, channels: int = 32, layers: int = 3, kernel: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(*(
            CausalBlock(n_features if i == 0 else channels, channels, kernel, 2**i, dropout)
            for i in range(layers)))
        self.out_dim = channels
        self.head = nn.Linear(channels, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Last cycle's channels before the head, shape (batch, out_dim)."""
        return self.blocks(x.transpose(1, 2))[:, :, -1]  # Conv1d wants (batch, channels, time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


def receptive_field(kernel: int, layers: int) -> int:
    """Cycles the last TCN output can see: two convolutions per block, dilation doubling."""
    return 1 + 2 * (kernel - 1) * (2**layers - 1)


@lru_cache(maxsize=32)
def positional_encoding(length: int, d_model: int) -> torch.Tensor:
    """Sinusoidal position codes of shape (length, d_model), as in Vaswani et al. (2017).

    Cached per (length, d_model), so treat the returned tensor as read only.
    """
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)[:, : d_model // 2]
    return pe


class TransformerNet(nn.Module):
    """Linear embedding plus sinusoidal positions, a Transformer encoder, mean pooled to one value."""

    def __init__(self, n_features: int, d_model: int = 32, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.out_dim = d_model
        self.head = nn.Linear(d_model, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output averaged over cycles, shape (batch, out_dim)."""
        # Codes are cached per window length, so any length works without a fixed maximum.
        h = self.embed(x) + positional_encoding(x.shape[1], self.out_dim).to(x)
        return self.encoder(self.drop(h)).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


NETS = {"lstm": LSTMNet, "tcn": TCNNet, "transformer": TransformerNet}


def build_net(name: Literal["lstm", "tcn", "transformer"], n_features: int, **params) -> nn.Module:
    """Build a sequence network that returns one value per window, shape (batch,).

    The output is RUL divided by 125, not cycles: fit_net and predict_net multiply it by
    y_scale=125. Calling net(x) directly gives values near 1, so multiply by 125 yourself.
    Every net also has encode(x), its window summary of shape (batch, net.out_dim) before
    the head and without dropout, for models that add their own head.
    """
    if name not in NETS:
        raise ValueError(f"unknown net {name!r}; choose lstm, tcn, or transformer")
    return NETS[name](n_features, **params)


def count_params(net: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
