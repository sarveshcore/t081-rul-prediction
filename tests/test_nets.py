# ABOUTME: Tests for the Phase 2 sequence networks.
# ABOUTME: Output shape for any window length, TCN causality and receptive field, parameter counts.
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rul.nets import build_net, causal_conv, count_params, positional_encoding, receptive_field
from rul.train import set_seed

N_FEATURES = 14


@pytest.mark.parametrize("name", ["lstm", "tcn", "transformer"])
@pytest.mark.parametrize("window", [1, 7, 30, 64])
def test_one_value_per_window_for_any_length(name, window):
    net = build_net(name, N_FEATURES).eval()
    out = net(torch.randn(5, window, N_FEATURES))
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name,params,expected", [
    ("lstm", {"hidden": 32, "layers": 1, "dropout": 0.3}, 6_177),
    ("tcn", {"channels": 64, "layers": 4, "dropout": 0.1}, 90_241),
    ("transformer", {"d_model": 64, "layers": 1, "dropout": 0.1}, 51_009),
])
def test_parameter_counts_of_the_selected_configs(name, params, expected):
    """The counts quoted in the report tables and results/dl_best_params.json."""
    n = count_params(build_net(name, N_FEATURES, **params))
    print(f"{name}: {n} parameters")
    assert n == expected


@pytest.mark.parametrize("name", ["lstm", "tcn", "transformer"])
def test_encode_gives_the_summary_the_head_reads(name):
    set_seed(0)
    net = build_net(name, N_FEATURES).eval()
    x = torch.randn(4, 30, N_FEATURES)
    z = net.encode(x)
    assert z.shape == (4, net.out_dim)
    assert torch.allclose(net.head(z).squeeze(-1), net(x))


def test_unknown_name_is_rejected():
    with pytest.raises(ValueError, match="unknown net"):
        build_net("gru", N_FEATURES)


def test_tcn_last_output_sees_exactly_its_receptive_field():
    """A cycle just inside the receptive field moves the output; one just outside does not."""
    set_seed(0)
    kernel, layers, window = 3, 3, 40
    rf = receptive_field(kernel, layers)
    assert rf == 29
    net = build_net("tcn", N_FEATURES, kernel=kernel, layers=layers, dropout=0.0).eval()
    x = torch.randn(1, window, N_FEATURES)
    base = net(x)
    inside, outside = x.clone(), x.clone()
    inside[:, window - rf] += 5.0
    outside[:, window - rf - 1] += 5.0
    assert not torch.allclose(net(inside), base)
    assert torch.allclose(net(outside), base)


@pytest.mark.parametrize("kernel,dilation,bias", [(3, 1, True), (3, 2, True), (3, 8, True),
                                                  (2, 3, True), (5, 2, False)])
def test_causal_conv_matches_padded_conv1d(kernel, dilation, bias):
    set_seed(0)
    conv = nn.Conv1d(4, 6, kernel, dilation=dilation, bias=bias)
    x = torch.randn(2, 4, 11)
    expected = conv(F.pad(x, ((kernel - 1) * dilation, 0)))
    assert torch.allclose(causal_conv(x, conv), expected, atol=1e-6)


def test_tcn_is_causal():
    """Changing the last cycle leaves every earlier block output unchanged."""
    set_seed(0)
    net = build_net("tcn", N_FEATURES, dropout=0.0).eval()
    x = torch.randn(2, 20, N_FEATURES)
    changed = x.clone()
    changed[:, -1] += 5.0
    early = net.blocks(x.transpose(1, 2))[:, :, :-1]
    early_changed = net.blocks(changed.transpose(1, 2))[:, :, :-1]
    assert torch.allclose(early, early_changed)


def test_transformer_follows_input_dtype():
    net = build_net("transformer", N_FEATURES).double().eval()
    assert net(torch.randn(2, 30, N_FEATURES, dtype=torch.float64)).dtype == torch.float64


def test_positional_encoding_handles_odd_width():
    pe = positional_encoding(10, 7)
    assert pe.shape == (10, 7)
    assert torch.allclose(pe[0, 1::2], torch.ones(3))  # cos(0) = 1 at position 0


def test_transformer_uses_cycle_order():
    """Reversing the window changes the output, which mean pooling alone would not."""
    set_seed(0)
    net = build_net("transformer", N_FEATURES, dropout=0.0).eval()
    x = torch.randn(3, 30, N_FEATURES)
    assert not torch.allclose(net(x), net(x.flip(1)))
