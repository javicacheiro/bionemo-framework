# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for the vortex-style FP8 padding wrapper in ``te_compat``.

These cover the regression that broke LoRA fine-tuning of the vortex-style-fp8 projection
layers: Megatron-Bridge LoRA enables ``return_layernorm_output`` on the wrapped
``TELayerNormColumnParallelLinear`` so the adapter can read the post-layernorm activations,
which makes the wrapped forward return ``((out, ln_out), bias)`` (or ``(out, bias, ln_out)``)
instead of ``(out, bias)``. ``fp8_padded_forward`` must pass that structure through unchanged
(and only unpad sequence-first activation tensors), rather than hard-coding a ``(out, bias)``
unpack.

``te.fp8_autocast`` is patched to a no-op context manager so the logic runs on CPU without
Transformer Engine / CUDA.
"""

import contextlib

import torch

from bionemo.evo2.models.megatron.hyena import te_compat
from bionemo.evo2.models.megatron.hyena.te_compat import _unpad_seq, fp8_padded_forward


@contextlib.contextmanager
def _noop_autocast(*args, **kwargs):
    yield


class _FakeParent:
    """Stand-in for the parent TE layer: echoes the (padded) sequence length in its outputs."""

    def __init__(self, return_pattern, out_features=64):
        self.return_pattern = return_pattern
        self.out_features = out_features

    def forward(self, x):
        seq, batch, _ = x.shape
        out = torch.ones(seq, batch, self.out_features)
        bias = torch.zeros(self.out_features)  # 1-D, must never be sliced along seq
        ln_out = torch.full((seq, batch, x.shape[-1]), 2.0)
        if self.return_pattern == "out_bias":
            return out, bias
        if self.return_pattern == "nested":  # ((out, ln_out), bias)
            return (out, ln_out), bias
        if self.return_pattern == "triple":  # (out, bias, ln_out)
            return out, bias, ln_out
        raise ValueError(self.return_pattern)


def test_unpad_seq_only_slices_multidim_seq_tensors():
    L = 13
    act = torch.ones(16, 2, 8)  # padded sequence-first activation
    assert _unpad_seq(act, L).shape[0] == L
    bias = torch.zeros(64)  # 1-D, longer than L, must pass through untouched
    assert _unpad_seq(bias, L).shape[0] == 64
    short = torch.ones(L, 2, 8)  # already at length L -> unchanged
    assert _unpad_seq(short, L).shape[0] == L
    assert _unpad_seq(None, L) is None


def test_fp8_padded_forward_no_padding_preserves_structure(monkeypatch):
    """Seq length already a multiple of 8 -> parent return passed through verbatim."""
    monkeypatch.setattr(te_compat.te, "fp8_autocast", _noop_autocast)
    x = torch.ones(16, 2, 8)  # 16 % 8 == 0, no padding

    out, bias = fp8_padded_forward(_FakeParent("out_bias"), None, x)
    assert out.shape == (16, 2, 64) and bias.shape == (64,)

    # The LoRA case that used to crash: nested ((out, ln_out), bias).
    (out, ln_out), bias = fp8_padded_forward(_FakeParent("nested"), None, x)
    assert out.shape == (16, 2, 64)
    assert ln_out.shape == (16, 2, 8)
    assert bias.shape == (64,)

    out, bias, ln_out = fp8_padded_forward(_FakeParent("triple"), None, x)
    assert out.shape == (16, 2, 64) and ln_out.shape == (16, 2, 8) and bias.shape == (64,)


def test_fp8_padded_forward_unpads_each_structure(monkeypatch):
    """Seq length not a multiple of 8 -> activations unpadded to L, bias untouched."""
    monkeypatch.setattr(te_compat.te, "fp8_autocast", _noop_autocast)
    L = 13
    x = torch.ones(L, 2, 8)  # padded to 16 internally

    out, bias = fp8_padded_forward(_FakeParent("out_bias"), None, x)
    assert out.shape[0] == L and bias.shape[0] == 64  # bias (out_features) not sliced

    (out, ln_out), bias = fp8_padded_forward(_FakeParent("nested"), None, x)
    assert out.shape[0] == L and ln_out.shape[0] == L and bias.shape[0] == 64

    out, bias, ln_out = fp8_padded_forward(_FakeParent("triple"), None, x)
    assert out.shape[0] == L and ln_out.shape[0] == L and bias.shape[0] == 64
