# Recipe source changes for the Evo2 viral-LoRA work

This documents the changes made to the **`evo2_megatron` recipe source** (`src/…`, `tests/…`) that
were required to run the viral-LoRA experiments logged in `LOG_evo2_*.md`. They are general recipe
improvements (enabling Savanna-checkpoint loading, LoRA logging, and FP8 LoRA training) — not
specific to the viral corpus — so they live in the recipe tree, while the data, run logs, and
`COMMANDS.sh` live here in `Evo2_virus/`.

All changes are on branch `evo2-viral-lora-wandb`. None of the edited files is listed in
`ci/scripts/check_copied_files.py`, so no copied-file regeneration was needed. Every change shipped
with focused CPU-only tests.

| Commit | Area | Summary |
|---|---|---|
| `7c12e89` | `utils/checkpoint/savanna_to_mbridge.py` | Load Savanna checkpoints under PyTorch ≥2.6 `weights_only` default (multi-part shards) |
| `c801003` | `run/train.py` | Make LoRA fine-tuning compatible with gradient-norm logging (wandb/tensorboard) |
| `864a377` | `run/train.py`, `models/megatron/hyena/te_compat.py` | Enable **vortex-style FP8** for LoRA fine-tuning in `train_evo2` |

---

## 1. Load Savanna checkpoints under PyTorch 2.6 `weights_only` (`7c12e89`)

**File:** `src/bionemo/evo2/utils/checkpoint/savanna_to_mbridge.py` (+ test
`tests/…/utils/checkpoint/test_savanna_to_mbridge.py`).

**Why:** The ARC **Savanna** HF checkpoints (`arcinstitute/savanna_evo2_20b`, `…_40b`) are published
as multi-part pickles (`.pt.part0..N`) containing non-tensor objects (numpy metadata). PyTorch 2.6
flipped `torch.load` to `weights_only=True` by default, which rejects those objects and aborts the
conversion.

**Change:** In the checkpoint loader, fall back to a trusted-source load (`weights_only=False`) when
the default `weights_only=True` load fails on the non-tensor pickle content, and join the multi-part
shards in order. This is what makes `evo2_convert_savanna_to_mbridge` work for the 20B/40B at all
(Step 1 of `LOG_evo2_20b.md` / `LOG_evo2_40b.md`).

## 2. LoRA-compatible gradient-norm logging (`c801003`)

**File:** `src/bionemo/evo2/run/train.py` (+6 lines; + test in `tests/…/run/test_train.py`).

**Why:** With `--lora-finetune`, the base model is frozen. The training loop's
`report_l2_norm_grad()` iterates over **all** parameters reading `.main_grad`, which the frozen LoRA
base parameters never receive → `AttributeError: 'Parameter' object has no attribute 'main_grad'` at
the first log step. `log_l2_norm_grad_to_tensorboard=True` is hard-coded in the recipe with no CLI
flag, and the metrics block runs whenever *any* logger (tensorboard **or** wandb) is active — so
turning on wandb re-triggered the crash.

**Change:** In `train()`, when `args.lora_finetune` is set, disable the offending logger:

```python
if args.lora_finetune:
    cfg.logger.log_l2_norm_grad_to_tensorboard = False
```

This lets wandb (and tensorboard) run normally for LoRA; the older
`--disable-tensorboard-logger` workaround is no longer needed.

## 3. Vortex-style FP8 for LoRA fine-tuning (`864a377`)

**Files:** `src/bionemo/evo2/run/train.py` (+15), `src/bionemo/evo2/models/megatron/hyena/te_compat.py`
(+41/−6); + tests `tests/…/run/test_train.py::test_vortex_style_fp8_sets_model_flag` and
`tests/…/models/megatron/hyena/test_te_compat.py`.

**Why:** The FP8/Hopper-sensitive Savanna checkpoints (README support matrix: Hopper FP8 ✅, Hopper
BF16 ❌) must run in their native **vortex-style FP8** regime — FP8 only on the dense-projection
matmuls, bf16 elsewhere. That path was wired into `predict.py`/`infer.py` but **not** `train_evo2`,
so these checkpoints could not be fine-tuned in the precision they were trained with. Enabling it
also exposed a LoRA incompatibility in the FP8 projection forward.

**Change A — CLI flag (`train.py`):** add `--vortex-style-fp8` (mirroring `predict.py`/`infer.py`)
and, after the config is built, set `cfg.model.vortex_style_fp8 = True`:

```python
parser.add_argument("--vortex-style-fp8", action="store_true", default=False, help=...)
...
if args.vortex_style_fp8:
    cfg.model.vortex_style_fp8 = True
```

**Change B — LoRA-safe `fp8_padded_forward` (`te_compat.py`):** the wrapper hard-coded
`x, bias = cls.forward(x)`, assuming a 2-tuple return. Megatron-Bridge LoRA wraps `dense_projection`
and enables `return_layernorm_output` so its adapter can read the post-layernorm activations, making
the wrapped forward return `((out, ln_out), bias)` — the old code then did `x = (out, ln_out)` →
`AttributeError: 'tuple' object has no attribute 'shape'`. The fix passes the parent's return
**structure** through unchanged and unpads only sequence-first activation tensors (never the bias).
It is behavior-preserving for the inference / non-LoRA path and enables LoRA on the FP8 projection
layers.

**Result:** `train_evo2 --mixed-precision-recipe bf16_mixed --vortex-style-fp8 --lora-finetune`
works for the Savanna 20B/40B (verified end to end on 8× H200, 0 NaN). This is the enablement behind
every 20B/40B run in the logs.

---

## Not a source change: parallelism / memory configuration

The 40B and the 32k/128k context runs required specific **runtime** flags (no code change) —
`--tensor-model-parallel-size`, `--context-parallel-size`, and
`--activation-checkpoint-recompute-num-layers` — because Megatron pre-allocates a full fp32
`main_grad` buffer for all parameters (including the frozen LoRA base), which is sharded by tensor
parallelism, not data parallelism. Those configs and the reasoning are recorded in
`COMMANDS.sh` and `LOG_evo2_40b.md` / `LOG_evo2_context_ablation.md`, not here, since they are
invocation choices rather than recipe-code edits. (Context parallelism itself was already
implemented and gradient-verified for the Hyena mixers upstream; we only exercised it.)
