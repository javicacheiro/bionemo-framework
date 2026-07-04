# Evo2_virus run log — 20B model (`arcinstitute/savanna_evo2_20b`, vortex-style FP8)

Repeating the verified viral LoRA pipeline (see `../running_evo2_in_aws.md` "Viral LoRA …"
sections and `LOG_evo2_7b_8k.md`) with the **20B** base checkpoint. Unlike the 1B/7B runs,
the 20B is an ARC **Savanna** checkpoint from HuggingFace **and** is FP8/Hopper-sensitive
(README support matrix: Hopper FP8 ✅, Hopper BF16 ❌ — same profile as the 40B). We are on
8× H200 (Hopper), so the run is done in the checkpoint's **native FP8 regime**:
**vortex-style FP8** (FP8 only on the dense-projection matmuls, bf16 elsewhere — the FP8
configuration the model was trained with in Savanna). Megatron-style full-FP8 recipes are a
*different* config the README warns against for these sensitive checkpoints.

Host: 8× NVIDIA H200 (143 GB). Container: `evo2:20260628` (`sleepy_germain`, started `--rm`
with the `/data` bind-mount). The venv is on `PATH`; `BIONEMO_DATA_SOURCE=ngc`.

> **Follow-up (`LOG_evo2_longer_3000steps.md`):** the 1000-step plateau below was a LR-schedule
> artifact. A fresh 3000-step run (full cosine decay) improved this 20B to val PPL **2.754** and
> base→LoRA **−17.5%**. See that log for the 20B-vs-40B longer-training comparison, and
> `LOG_evo2_context_ablation.md` for the 16k/32k/128k context sweep (context length gives no gain).

What changes vs the 7B run:
- Base checkpoint: `arcinstitute/savanna_evo2_20b` (HF Savanna) → convert with
  `evo2_convert_savanna_to_mbridge --model-size evo2_20b` (not the NeMo2 converter).
- **Code change required (Step 0).** `train_evo2` did not expose `--vortex-style-fp8`
  (only `predict`/`infer` did), and the FP8 projection-layer forward was not LoRA-safe.
  Both were fixed (see Step 0). Training and evaluation both run with
  `--mixed-precision-recipe bf16_mixed --vortex-style-fp8`.
- Reuse the already-preprocessed viral data (`/data/viral/preprocessed/...`),
  `/data/viral/viral_dataset.yaml`, `/data/viral/valid_cap8192.fasta`,
  `/data/viral/manifest.tsv.gz`, `/data/viral/aggregate_ppl.py` — same tokenizer
  (`nucleotide_fast_tokenizer_512`), so **no re-preprocessing**.

## Status checklist

| Step | Status |
|---|---|
| 0. Add `--vortex-style-fp8` to `train_evo2` + LoRA-safe FP8 forward (+ tests) | ✅ done |
| 1. Convert Savanna 20B → MBridge (`/data/evo2_20b_mbridge`, bf16) | ✅ done |
| 2. FP8+LoRA smoke run (validate the untested combination) | ✅ done |
| 3. LoRA train at seq-16384, vortex-FP8 (`/data/viral/lora_run_20b_16k`) | ✅ done |
| 4. Base vs LoRA scoring + per-`genome` PPL (vortex-FP8) | ✅ done |

---

## Step 0 — Recipe code change (FP8 LoRA training enablement)

The 20B needs to run in vortex-style FP8, but that path was inference-only and not
LoRA-compatible. Two changes in the recipe (host repo = source of truth; also applied to
the container's editable copy under `/workspace/bionemo/src`):

1. **`src/bionemo/evo2/run/train.py`** — added the `--vortex-style-fp8` CLI flag and, after
   the config is built (`cfg = pretrain_config(...)`), set `cfg.model.vortex_style_fp8 = True`
   when the flag is passed. This mirrors `predict.py` / `infer.py`. Neither `train.py` nor
   the file below is in `ci/scripts/check_copied_files.py`, so no copy regeneration.

2. **`src/bionemo/evo2/models/megatron/hyena/te_compat.py`** — fixed `fp8_padded_forward`.
   It hard-coded `x, bias = cls.forward(x)`, assuming a 2-tuple return. Megatron-Bridge LoRA
   wraps `dense_projection` and enables `return_layernorm_output` so its adapter can read the
   post-layernorm activations, which makes the wrapped forward return `((out, ln_out), bias)`
   (see `megatron.bridge.peft.adapter_wrapper.base_linear_forward`). The old code then did
   `x = (out, ln_out)` → `AttributeError: 'tuple' object has no attribute 'shape'`. The fix
   passes the parent's return **structure** through unchanged and unpads only sequence-first
   activation tensors (never bias). Behavior-preserving for inference / non-LoRA; enables
   LoRA on the FP8 projection layers.

Tests added (CPU-only, both green):
- `tests/bionemo/evo2/run/test_train.py::test_vortex_style_fp8_sets_model_flag`
- `tests/bionemo/evo2/models/megatron/hyena/test_te_compat.py` (3 cases: `_unpad_seq`
  bias-vs-activation safety; structure pass-through with and without padding).

## Step 1 — Convert Savanna 20B → MBridge

`arcinstitute/savanna_evo2_20b` is published as multi-part shards (`.pt.part0..2`). The
PyTorch-2.6 `weights_only` fallback is already in the code. Convert in `bf16_mixed` (vortex
is a runtime choice, not baked into the checkpoint; the savanna converter doesn't accept the
flag). Preflight: `/data` 22 TB free, host ~1.9 TB RAM.

```bash
evo2_convert_savanna_to_mbridge \
  --savanna-ckpt-path arcinstitute/savanna_evo2_20b \
  --mbridge-ckpt-dir /data/evo2_20b_mbridge \
  --model-size evo2_20b \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --seq-length 1048576 \
  --mixed-precision-recipe bf16_mixed
```

**Results (exit 0):** `Converting with pattern=SDH*SDHSDH*SDHSDH*SDHSDH` (24-layer evo2_20b),
`Converted 247 keys`, benign `Unmapped savanna keys (58)` (the usual
`*.outer_mlp_layernorm` / `*.post_attention_layernorm` / `*.rotary_emb.inv_freq` /
`*.mixer.filter.t`). Output `/data/evo2_20b_mbridge` (**37 GB**): `iter_0000001/` (distcp
shards + `.metadata`), `latest_checkpointed_iteration.txt`, `latest_train_state.pt`.

## Step 2 — FP8+LoRA smoke run

The FP8 + LoRA combination had no test coverage, so a 10-step run on the real viral data
validated it before the full run (same config as Step 3, `--max-steps 10`). The **first
attempt surfaced the `fp8_padded_forward` bug above**; after the Step-0 fix it passed:

- LoRA grafted onto `decoder.layers.*.mixer.dense_projection` (the FP8 layer) +
  `linear_fc1/linear_fc2`; PEFT statistics printed.
- 10 iters, finite `lm loss` ~1.13–1.21, **0 NaN / 0 skipped**, validation **PPL 3.19**,
  adapter checkpoint saved. ~9 s/step, ~404 TFLOP/s/GPU, **no OOM in pure data-parallel**
  (TP=1) at `--activation-checkpoint-recompute-num-layers 2`.

## Step 3 — LoRA training (seq 16384, vortex-FP8)

Reuses the already-preprocessed viral data (172 MB train / 17 MB valid). 8× H200 pure
data-parallel (no TP needed), `bf16_mixed --vortex-style-fp8`. Saved to
`/data/viral/train_20b_lora.sh`, run via `nohup` (`/data/viral/train_20b_lora.log`):

```bash
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_20b_mbridge --model-size evo2_20b \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 2 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 \
  --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_20b_16k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-20b-seq16384-vfp8 \
  --lora-finetune --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
  --lora-target-modules "dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"
```

**Results (TRAIN_EXIT=0):**
- **Memory fits in pure DP** at seq-16384, no TP/PP/CP. ~404 TFLOP/s/GPU, **~9.0 s/step**,
  100% util, **0 NaN / 0 skipped** for all 1000 steps. 1000 steps in **~2.5 h**
  (`progress.txt`: 0.26 B tokens, 2.91e19 FLOPs, cumulative 392.2 MODEL_TFLOP/s/GPU).
  The FP8 dense-projection makes the 20B run *faster per FLOP* than the 7B bf16 run
  (~404 vs ~325 TFLOP/s/GPU), though ~2× slower per step (20B is ~3× the params).
- Training-loss curve (wandb `viral-lora-20b-seq16384-vfp8`, GBS 16, lr 3e-4):

  | iter | lm loss |
  |---:|---:|
  | 10 | 1.194 |
  | 100 | 1.144 |
  | 250 | 1.109 |
  | 500 | 1.075 |
  | 750 | 1.052 |
  | 1000 | 1.053 |

- **Validation (species-holdout `validation` split, `--eval-iters 20`):**

  | iter | val lm loss | val PPL | 7B PPL (same iter) |
  |---:|---:|---:|---:|
  | 250 | 1.132 | **3.103** | 3.235 |
  | 500 | 1.107 | **3.026** | 3.149 |
  | 750 | 1.090 | **2.974** | 3.092 |
  | 1000 | 1.088 | **2.969** | 3.087 |

  **20B beats the 7B at every eval point** and improves monotonically. Best checkpoint =
  **iter 1000** (PPL 2.969). End-of-training full eval (more iters):
  `validation set PPL 2.939`, `test set PPL 2.946` (test ≈ validation by design).
- Checkpoints (adapter-only, **611 MB** each) at
  `/data/viral/lora_run_20b_16k/evo2/checkpoints/iter_0000{250,500,750,1000}`. Each
  `run_config.yaml` records the `peft:` section + `pretrained_checkpoint:
  /data/evo2_20b_mbridge`. wandb run:
  `https://wandb.ai/alcachi-cesga/evo2-viral-lora/runs/tlyvhguh`.

## Step 4 — Base vs LoRA scoring + stratified per-`genome` PPL

Report items 2 (stratified validation PPL) and 3 (base vs LoRA) from `Evo2_virus/README.md`,
via `predict_evo2` on the species-holdout valid set. As in the 7B run, every record is capped
to **8192 bp** (`/data/viral/valid_cap8192.fasta`, 1349 records) and the **same cap +
precision (vortex-FP8) is applied to both** base and LoRA, so the delta is exact and the
absolute PPL is "PPL over the first ≤8192 bp of each record." Both scored with
`--mixed-precision-recipe bf16_mixed --vortex-style-fp8` (the accurate Hopper regime,
consistent with training), in parallel on GPU 0 (base) and GPU 1 (LoRA):

```bash
# base 20B (vortex-fp8) — GPU 0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/evo2_20b_mbridge/iter_0000001 \
  --output-dir /data/viral/pred_20b_base_vfp8 \
  --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8

# LoRA 20B (vortex-fp8; PEFT auto-detected → reloads base from pretrained_checkpoint + grafts adapters) — GPU 1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_20b_16k/evo2/checkpoints/iter_0001000 \
  --output-dir /data/viral/pred_20b_lora_vfp8 \
  --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8
```

```bash
python3 /data/viral/aggregate_ppl.py \
  /data/viral/pred_20b_base_vfp8 /data/viral/pred_20b_lora_vfp8 /data/viral/manifest.tsv.gz
```

**Results (both exit 0; 1349/1349 records scored).** LoRA log:
`Loading base model weights from: /data/evo2_20b_mbridge/iter_0000001` →
`Loading adapter weights from: .../iter_0001000` (PEFT auto-detected via `run_config.yaml`).

**Base vs LoRA — mean per-record perplexity (`exp(-mean_logprob)`), valid set capped 8192 bp, vortex-FP8:**

| genome class | n | base PPL | LoRA PPL | Δ | rel % |
|---|---:|---:|---:|---:|---:|
| **OVERALL** | 1349 | 3.5784 | **3.0879** | −0.4905 | **−13.71%** |
| ssRNA(−) | 353 | 3.6305 | 3.0234 | −0.6072 | −16.72% |
| ssRNA(+) | 337 | 3.5632 | 3.2152 | −0.3480 | −9.77% |
| ssDNA | 289 | 3.5879 | 2.7373 | −0.8506 | −23.71% |
| dsRNA | 179 | 3.7361 | 3.6845 | −0.0516 | −1.38% |
| dsDNA | 127 | 3.3202 | 2.8100 | −0.5102 | −15.37% |
| ssRNA(other) | 64 | 3.3999 | 3.2403 | −0.1597 | −4.70% |

- **LoRA improves 1313/1349 records (97.3%)**; every genome class improves. Largest gains on
  ssDNA (−24%), ssRNA(−)/dsDNA (−15–17%); smallest on dsRNA (−1%). Same class ordering as the
  7B, with larger magnitudes.
- This capped per-record PPL (LoRA 3.088) differs in methodology from the in-training windowed
  validation (PPL 2.969 over full 16384 windows) — both show a clear, sizeable LoRA
  improvement. (Same methodology caveat the 1B/7B runbooks note.)

---

## Summary — 20B vs the earlier 7B / 1B viral LoRA runs

| | **20B** (this run) | 7B (`LOG_evo2_7b_8k.md`) | 1B (`running_evo2_in_aws.md`) |
|---|---|---|---|
| Base ckpt | `arcinstitute/savanna_evo2_20b` (HF Savanna) → `/data/evo2_20b_mbridge` (37 GB) | `evo2/7b-8k:1.0` (NeMo2) → `/data/evo2_7b_mbridge` (13 GB) | `evo2/1b-8k-bf16:1.0` → `/data/evo2_1b_mbridge` |
| Precision | **bf16 + vortex-style FP8** (FP8-sensitive ckpt; native Hopper regime) | bf16 | bf16 |
| Train throughput | ~9.0 s/step, ~404 TFLOP/s/GPU | ~4.34 s/step, ~325 TFLOP/s/GPU | ~1.27 s/step, ~255 TFLOP/s/GPU |
| Parallelism / mem | pure DP, no TP, recompute-2 | pure DP, no TP | pure DP, no TP |
| Adapter ckpt size | 611 MB | 405 MB | 149 MB |
| In-train val PPL @1000 | **2.969** | 3.087 | 3.172 |
| Best val PPL | **2.969** (iter 1000) | 3.087 (iter 1000) | 3.167 (iter 750) |
| Base→LoRA (capped predict) | 3.578 → **3.088 (−13.7%)**, 97.3% improved | 3.612 → 3.218 (−10.9%), 96.8% improved | 3.533 → 3.509 (−0.7%) |

**The 20B vortex-FP8 LoRA gives the lowest validation PPL and the largest base→LoRA viral-fit
improvement of the three** (−13.7%), while still fitting comfortably on 8× H200 in pure
data-parallel at seq-16384. Best checkpoint:
`/data/viral/lora_run_20b_16k/evo2/checkpoints/iter_0001000`. wandb run
`viral-lora-20b-seq16384-vfp8` (project `evo2-viral-lora`).

**Recipe note:** this run also required (and validated) enabling **FP8 LoRA training** in the
recipe — `--vortex-style-fp8` in `train_evo2` plus a LoRA-compatible `fp8_padded_forward`
(see Step 0). These are general improvements, not specific to the viral corpus.
