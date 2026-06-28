# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import pickle

import numpy as np
import pytest
import torch

from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import load_savanna_state_dict


def test_load_savanna_state_dict_strips_prefixes(tmp_path):
    """A plain tensor-only checkpoint loads and has its module/ prefixes stripped."""
    ckpt = {
        "module": {
            "module.sequential.0.weight": torch.zeros(2, 3),
            "sequential.1.bias": torch.ones(4),
        }
    }
    path = tmp_path / "savanna_clean.pt"
    torch.save(ckpt, path)

    sd = load_savanna_state_dict(path)

    assert set(sd.keys()) == {"sequential.0.weight", "sequential.1.bias"}
    assert sd["sequential.0.weight"].shape == (2, 3)
    assert torch.equal(sd["sequential.1.bias"], torch.ones(4))


def test_load_savanna_state_dict_with_numpy_metadata(tmp_path):
    """Regression: Savanna checkpoints pickle non-tensor numpy objects.

    PyTorch >=2.6 defaults ``torch.load(weights_only=True)``, which raises
    ``UnpicklingError`` on the numpy globals these checkpoints contain.
    ``load_savanna_state_dict`` must fall back to a full unpickle so the
    Savanna->MBridge conversion CLI works out of the box.
    """
    ckpt = {
        "module": {
            "module.sequential.0.weight": torch.zeros(2, 3),
            "sequential.1.bias": torch.ones(4),
        },
        # Non-tensor numpy objects, as found in real Savanna training checkpoints.
        "metadata": {"step": np.int64(42), "loss_curve": np.arange(5)},
    }
    path = tmp_path / "savanna_numpy.pt"
    torch.save(ckpt, path)

    # Sanity check that this checkpoint genuinely trips the strict loader, so the
    # test actually exercises the fallback path rather than passing vacuously.
    with pytest.raises(pickle.UnpicklingError):
        torch.load(str(path), map_location="cpu", weights_only=True)

    sd = load_savanna_state_dict(path)

    assert set(sd.keys()) == {"sequential.0.weight", "sequential.1.bias"}
    assert sd["sequential.0.weight"].shape == (2, 3)
    assert torch.equal(sd["sequential.1.bias"], torch.ones(4))
