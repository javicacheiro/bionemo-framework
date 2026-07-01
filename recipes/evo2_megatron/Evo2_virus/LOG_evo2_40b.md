# Evo2_virus run log — 40B model (`arcinstitute/savanna_evo2_40b`, vortex-style FP8)

Repeating the verified viral LoRA pipeline (see `LOG_evo2_20b.md`, `LOG_evo2_7b_8k.md`, and
`../running_evo2_in_aws.md` "Viral LoRA …") with the **40B** base checkpoint. Like the 20B,
the 40B is an ARC **Savanna** checkpoint from HuggingFace and is FP8/Hopper-sensitive (README
support matrix: Hopper FP8 ✅, Hopper BF16 ❌). We are on 8× H200 (Hopper), so it runs in the
checkpoint's **native FP8 regime**: **vortex-style FP8** (FP8 only on the dense-projection
matmuls, bf16 elsewhere — the FP8 configuration the model was trained with in Savanna).

Host: 8× NVIDIA H200 (143 GB). Container: `evo2:20260628` (`sleepy_germain`, started `--rm`
with the `/data` bind-mount). The venv is on `PATH`; `BIONEMO_DATA_SOURCE=ngc`.

> **Follow-up (`LOG_evo2_longer_3000steps.md`):** a fresh 3000-step run (full cosine decay) improved
> this 40B to val PPL **2.767** and base→LoRA **−17.2%** — but it still does not beat the 20B
> (2.754 / −17.5%) even at 3000 steps.

What changes vs the 20B run:
- Base checkpoint: `arcinstitute/savanna_evo2_40b` (HF Savanna) → convert with
  `evo2_convert_savanna_to_mbridge --model-size evo2_40b`. 50 layers (vs 24 for the 20B),
  same hidden 8192 / ffn 22528 / 1M-context config.
- **No recipe code change needed.** The 20B run already enabled FP8 LoRA training
  (`--vortex-style-fp8` in `train_evo2` + the LoRA-safe `fp8_padded_forward`, committed on
  this branch); that work covers the 40B unchanged.
- **Parallelism had to change (the only real new difficulty).** The 20B fit in *pure
  data-parallel*; the 40B does **not**. It needs **TP=4 + activation-recompute-num-layers 1**
  (see Step 2). Everything else (data, hyperparameters, eval) is identical to the 20B run.
- Reuse the already-preprocessed viral data (`/data/viral/preprocessed/...`),
  `/data/viral/viral_dataset.yaml`, `/data/viral/valid_cap8192.fasta`,
  `/data/viral/manifest.tsv.gz`, `/data/viral/aggregate_ppl.py` — same tokenizer
  (`nucleotide_fast_tokenizer_512`), so **no re-preprocessing**.

## Status checklist

| Step | Status |
|---|---|
| 0. Prerequisites (assets, vortex flag, disk/GPU) — no code change | ✅ done |
| 1. Convert Savanna 40B → MBridge (`/data/evo2_40b_mbridge`, 77 GB) | ✅ done |
| 2. FP8+LoRA smoke run + find the memory-fitting config | ✅ done |
| 3. LoRA train at seq-16384, vortex-FP8 (`/data/viral/lora_run_40b_16k`) | ✅ done |
| 4. Base vs LoRA scoring + per-`genome` PPL (vortex-FP8) | ✅ done |

---

## Step 1 — Convert Savanna 40B → MBridge

`arcinstitute/savanna_evo2_40b` is published as multi-part shards (`.pt.part0..4`). Convert in
`bf16_mixed` (vortex is a runtime choice, not baked into the checkpoint). Preflight: `/data`
21 TB free, host ~1.9 TB RAM.

```bash
evo2_convert_savanna_to_mbridge \
  --savanna-ckpt-path arcinstitute/savanna_evo2_40b \
  --mbridge-ckpt-dir /data/evo2_40b_mbridge \
  --model-size evo2_40b \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --seq-length 1048576 \
  --mixed-precision-recipe bf16_mixed
```

**Results (exit 0):** `Converting with pattern=SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*SDH*SDHSDH*SDHSDH*`
(50-layer evo2_40b), `Converted 506 keys`, benign `Unmapped savanna keys (122)` (the usual
`*.outer_mlp_layernorm` / `*.post_attention_layernorm` / `*.rotary_emb.inv_freq` /
`*.mixer.filter.t`). Output `/data/evo2_40b_mbridge` (**77 GB**): `iter_0000001/` (distcp shards
+ `.metadata`), `latest_checkpointed_iteration.txt`, `latest_train_state.pt`. ~2× the 20B's 37 GB.

## Step 2 — Smoke run **and** finding the memory-fitting config

The 40B is ~2× the 20B, and the 20B's *pure data-parallel* config OOMs. Finding what fits took
four smoke attempts (each 10 steps on the real viral data); the failures are recorded because the
fix is non-obvious and matters for anyone repeating this:

| Attempt | Config | Result | Per-GPU peak |
|---|---|---|---|
| 1 | pure DP (TP=1), recompute-num-layers **50** | OOM in backward | ~132 GiB |
| 2 | TP=2, recompute-num-layers 50 | OOM in backward | ~129 GiB |
| 3 | TP=4, recompute-num-layers 50 | OOM in backward | ~133 GiB |
| 4 | pure DP, recompute-num-layers **1** | OOM | ~134 GiB |
| **5** | **TP=4, recompute-num-layers 1** | **✅ fits** | **~66 GiB** |

Two independent things had to be right at once, which is why single-lever attempts all failed:

1. **Megatron pre-allocates a full fp32 `main_grad` buffer for *all* 40B params** — including the
   frozen LoRA base (these are the same `.main_grad` buffers the l2-norm-grad logger tripped over
   in the 1B/7B/20B runs). In pure DP each rank holds the whole thing: ~80 GB bf16 weights +
   ~160 GB fp32 grad buffer ≫ 140 GB → impossible regardless of activation recompute (attempt 4).
   This buffer **is** tensor-parallel-sharded, so TP is required (TP=4 → ~20 GB weights + ~40 GB
   grad buffer per GPU).
2. **`--activation-checkpoint-recompute-num-layers` must be *small*, not large.** With
   `recompute_method: uniform`, the value is the *chunk size*: `50` recomputes the entire 50-layer
   stack as one chunk, so the backward re-materializes all 50 layers' activations at once — the
   worst case (attempts 1–3, ~130 GiB even with TP). `1` recomputes one layer at a time → peak
   activation is one layer's worth. (The 20B run used `2`.) Higher is **not** more aggressive.

Winning smoke (10 steps, `--max-steps 10`, TP=4, recompute-1,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): finite `lm loss` ~1.13–1.21, **0 NaN /
0 skipped**, validation **PPL 3.204**, adapter checkpoint saved (validation + checkpoint-save
phases also fit). ~23.4 s/step, ~324 TFLOP/s/GPU, peak **~66 GiB/GPU** (comfortable headroom).

## Step 3 — LoRA training (seq 16384, vortex-FP8)

Reuses the already-preprocessed viral data (172 MB train / 17 MB valid). 8× H200, **TP=4
(DP=2), sequence-parallel auto-enabled**, `bf16_mixed --vortex-style-fp8`,
`--activation-checkpoint-recompute-num-layers 1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Saved to `/data/viral/train_40b_lora.sh`,
run via `nohup` (`/data/viral/train_40b_lora.log`):

```bash
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --tensor-model-parallel-size 4 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_40b_mbridge --model-size evo2_40b \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 \
  --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_40b_16k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-40b-seq16384-vfp8 \
  --lora-finetune --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
  --lora-target-modules "dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"
```

**Results (TRAIN_EXIT=0):**
- **~23.4 s/step, ~324 TFLOP/s/GPU, 100% util, 0 NaN / 0 skipped for all 1000 steps.**
  1000 steps in **~6 h 42 min** (`progress.txt`: 0.26 B tokens, 6.07e19 FLOPs, cumulative
  314.8 MODEL_TFLOP/s/GPU). Per-FLOP throughput is a touch below the 20B (~324 vs ~404
  TFLOP/s/GPU) — the price of TP=4 cross-GPU communication + per-layer recompute — and ~2.6×
  slower per step (50 vs 24 layers, plus TP comm).
- Training-loss curve (wandb `viral-lora-40b-seq16384-vfp8`, GBS 16, lr 3e-4):

  | iter | lm loss |
  |---:|---:|
  | 10 | 1.194 |
  | 250 | 1.109 |
  | 500 | 1.076 |
  | 750 | 1.054 |
  | 1000 | 1.055 |

- **Validation (species-holdout `validation` split, `--eval-iters 20`):**

  | iter | val lm loss | val PPL | 20B PPL (same iter) |
  |---:|---:|---:|---:|
  | 250 | 1.132 | **3.102** | 3.103 |
  | 500 | 1.108 | **3.029** | 3.026 |
  | 750 | 1.092 | **2.979** | 2.974 |
  | 1000 | 1.090 | **2.976** | 2.969 |

  **The 40B in-training validation is essentially tied with the 20B** (within ~0.007 PPL at
  every eval point) — i.e. **no improvement from 20B → 40B on this windowed validation**, a
  clear diminishing-returns signal for this viral corpus. Best checkpoint = **iter 1000**
  (PPL 2.976). End-of-training full eval: `validation set PPL 2.946`, `test set PPL 2.952`
  (test ≈ validation by design).
- Checkpoints (adapter + sharded distributed-optimizer state, ~1.3 GB each across the 8 distcp
  shards) at `/data/viral/lora_run_40b_16k/evo2/checkpoints/iter_0000{250,500,750,1000}`. Each
  `run_config.yaml` records the `peft:` section + `pretrained_checkpoint: /data/evo2_40b_mbridge`.
  wandb run: `https://wandb.ai/alcachi-cesga/evo2-viral-lora/runs/ytv0owwe`.

## Step 4 — Base vs LoRA scoring + stratified per-`genome` PPL

Report items 2 (stratified validation PPL) and 3 (base vs LoRA) from `Evo2_virus/README.md`, via
`predict_evo2` on the species-holdout valid set. As in the 7B/20B runs, every record is capped to
**8192 bp** (`/data/viral/valid_cap8192.fasta`, 1349 records) and the **same cap + precision
(vortex-FP8)** is applied to both base and LoRA, so the delta is exact and the absolute PPL is
"PPL over the first ≤8192 bp of each record." Unlike training, `predict` allocates **no grad
buffer**, so a single 40B (~80 GB weights) fits on one 140 GB H200 at **TP=1** — base on GPU 0
and LoRA on GPU 1 run in parallel (~115 GB/GPU):

```bash
# base 40B (vortex-fp8) — GPU 0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/evo2_40b_mbridge/iter_0000001 \
  --output-dir /data/viral/pred_40b_base_vfp8 \
  --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8

# LoRA 40B (vortex-fp8; PEFT auto-detected → reloads base + grafts adapters) — GPU 1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_40b_16k/evo2/checkpoints/iter_0001000 \
  --output-dir /data/viral/pred_40b_lora_vfp8 \
  --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8

python3 /data/viral/aggregate_ppl.py \
  /data/viral/pred_40b_base_vfp8 /data/viral/pred_40b_lora_vfp8 /data/viral/manifest.tsv.gz
```

**Results (both exit 0; 1349/1349 records scored).** LoRA log:
`Loading base model weights from: /data/evo2_40b_mbridge/iter_0000001` → `Loading adapter weights
from: .../iter_0001000` (PEFT auto-detected via `run_config.yaml`; LoRA grafted onto all 50 layers'
`mixer.dense_projection` + `linear_qkv`/`linear_fc1`/`linear_fc2`).

**Base vs LoRA — mean per-record perplexity (`exp(-mean_logprob)`), valid set capped 8192 bp, vortex-FP8:**

| genome class | n | base PPL | LoRA PPL | Δ | rel % |
|---|---:|---:|---:|---:|---:|
| **OVERALL** | 1349 | 3.5784 | **3.0950** | −0.4834 | **−13.51%** |
| ssRNA(−) | 353 | 3.6308 | 3.0328 | −0.5981 | −16.47% |
| ssRNA(+) | 337 | 3.5631 | 3.2190 | −0.3440 | −9.66% |
| ssDNA | 289 | 3.5877 | 2.7476 | −0.8401 | −23.42% |
| dsRNA | 179 | 3.7359 | 3.6835 | −0.0524 | −1.40% |
| dsDNA | 127 | 3.3202 | 2.8130 | −0.5072 | −15.28% |
| ssRNA(other) | 64 | 3.4000 | 3.2683 | −0.1317 | −3.87% |

- **LoRA improves 1312/1349 records (97.3%)**; every genome class improves. Largest gains on ssDNA
  (−23%), ssRNA(−)/dsDNA (−15–16%); smallest on dsRNA (−1%). **Same class ordering and nearly
  identical magnitudes as the 20B** (20B was −13.71% overall, 97.3% improved).
- This capped per-record PPL (LoRA 3.095) differs in methodology from the in-training windowed
  validation (PPL 2.976 over full 16384 windows) — both show a clear, sizeable LoRA improvement.
  (Same methodology caveat the 1B/7B/20B runbooks note.)

---

## Summary — 40B vs the earlier 20B / 7B / 1B viral LoRA runs

| | **40B** (this run) | 20B (`LOG_evo2_20b.md`) | 7B (`LOG_evo2_7b_8k.md`) | 1B (`running_evo2_in_aws.md`) |
|---|---|---|---|---|
| Base ckpt | `arcinstitute/savanna_evo2_40b` (HF Savanna) → `/data/evo2_40b_mbridge` (77 GB) | `arcinstitute/savanna_evo2_20b` → `/data/evo2_20b_mbridge` (37 GB) | `evo2/7b-8k:1.0` (NeMo2) → 13 GB | `evo2/1b-8k-bf16:1.0` |
| Layers | 50 | 24 | — | — |
| Precision | **bf16 + vortex-style FP8** | bf16 + vortex-style FP8 | bf16 | bf16 |
| Parallelism / mem | **TP=4 (DP=2), recompute-1** (pure DP OOMs) | pure DP, recompute-2 | pure DP | pure DP |
| Train throughput | ~23.4 s/step, ~324 TFLOP/s/GPU | ~9.0 s/step, ~404 TFLOP/s/GPU | ~4.34 s/step, ~325 TFLOP/s/GPU | ~1.27 s/step, ~255 TFLOP/s/GPU |
| Wall-clock (1000 steps) | ~6 h 42 min | ~2.5 h | — | — |
| Adapter ckpt size | ~1.3 GB (incl. dist-opt shards) | 611 MB | 405 MB | 149 MB |
| In-train val PPL @1000 | 2.976 | **2.969** | 3.087 | 3.172 |
| Best val PPL | 2.976 (iter 1000) | **2.969** (iter 1000) | 3.087 (iter 1000) | 3.167 (iter 750) |
| Base→LoRA (capped predict) | 3.578 → 3.095 (**−13.51%**), 97.3% improved | 3.578 → **3.088 (−13.71%)**, 97.3% improved | 3.612 → 3.218 (−10.9%), 96.8% | 3.533 → 3.509 (−0.7%) |

**The 40B does not beat the 20B.** Its in-training validation PPL (2.976) and capped base→LoRA
improvement (−13.51%) are within noise of — and marginally behind — the 20B (2.969 / −13.71%),
despite ~2.6× the per-step cost and ~2.7× the wall-clock. This is a clear **diminishing-returns**
signal: for this eukaryotic-host viral corpus, the 20B vortex-FP8 LoRA remains the best
price/performance point, and scaling the base to 40B buys nothing here. Best 40B checkpoint:
`/data/viral/lora_run_40b_16k/evo2/checkpoints/iter_0001000`. wandb run
`viral-lora-40b-seq16384-vfp8` (project `evo2-viral-lora`,
`https://wandb.ai/alcachi-cesga/evo2-viral-lora/runs/ytv0owwe`).

**Recipe note:** unlike the 20B run, the 40B required **no recipe code change** — the FP8-LoRA
enablement from the 20B work already covers it. The only new operational requirement is the
parallelism/recompute config: **TP=4 + `--activation-checkpoint-recompute-num-layers 1`** (the
40B does not fit in pure data-parallel; see Step 2).

**Recipe note:** unlike the 20B run, the 40B required **no recipe code change** — the FP8-LoRA
enablement from the 20B work already covers it. The only new operational requirement is the
parallelism/recompute config: **TP=4 + `--activation-checkpoint-recompute-num-layers 1`** (the
40B does not fit in pure data-parallel; see Step 2).
