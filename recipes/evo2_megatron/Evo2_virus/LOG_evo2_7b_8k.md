# Evo2_virus run log — 7B model (`evo2/7b-8k:1.0`)

Repeating the verified 1B viral LoRA pipeline (see `../running_evo2_in_aws.md`,
"Viral LoRA …" sections) with the **7B** base checkpoint instead of 1B.

Host: 8× NVIDIA H200. Container: `evo2:20260628` (`sleepy_germain`, started `--rm`
with the `/data` bind-mount). The venv is on `PATH`; `BIONEMO_DATA_SOURCE=ngc`.

What changes vs the 1B run:
- Base checkpoint: `evo2/7b-8k:1.0` (NeMo2) → convert with `--model-size evo2_7b_base`.
- Reuse the already-preprocessed viral data (`/data/viral/preprocessed/...`) and
  `/data/viral/viral_dataset.yaml` — same tokenizer (`nucleotide_fast_tokenizer_512`),
  so **no re-preprocessing**.
- Training parallelism/batch sizing will differ (7B is ~7× larger than 1B).

## Status checklist

| Step | Status |
|---|---|
| 1. Download `evo2/7b-8k:1.0` (NeMo2) | ✅ done |
| 2. Convert NeMo2 → MBridge (`/data/evo2_7b_mbridge`) | ✅ done |
| 3. LoRA train at seq-16384 (`/data/viral/lora_run_7b_16k`) | ✅ done |
| 4. Base vs LoRA scoring + per-`genome` PPL | ✅ done |

---

## Step 1 — Download `evo2/7b-8k:1.0`

Run inside the container (`BIONEMO_DATA_SOURCE=ngc` is already set). The command
prints the local checkpoint path on stdout, which becomes `$NEMO_CKPT` for Step 2:

```bash
download_bionemo_data evo2/7b-8k:1.0
```

(As actually run, redirected so the path persists on the host bind-mount:)

```bash
docker exec sleepy_germain bash -lc \
  'download_bionemo_data evo2/7b-8k:1.0 > /data/dl_7b_8k.log 2>&1'
```

**Results:**
- Resource confirmed available: `evo2/7b-8k:1.0  ngc,pbss` (via
  `download_bionemo_data --list-resources | grep evo2/7b`).
- Source archive: `nvidia/clara/evo2-7b-8k-nemo2:1.0` → cached at
  `~/.cache/bionemo/<hash>-nemo2_evo2_7b_8k.tar.gz` (**10.3 GB**), then extracted to
  the sibling `...-nemo2_evo2_7b_8k.tar.gz.untar/` directory.
- Disk: `/data` has ~22 TB free.
- **Final checkpoint path** (exit 0):
  `/root/.cache/bionemo/<hash>-nemo2_evo2_7b_8k.tar.gz.untar`
  (full hash `78fc05536e1a9bd2febacea079a4beedf93ddcba1c69ac24690a5f7b649a0655`).
- Extracted NeMo2 layout (13 GB): `context/{model.yaml,io.json}` +
  `weights/{__0_0.distcp,__0_1.distcp,common.pt,.metadata,metadata.json}` — a valid
  NeMo2 distributed checkpoint.

## Step 2 — Convert NeMo2 → MBridge

Mirrors the 1B conversion (`running_evo2_in_aws.md`) but with `--model-size
evo2_7b_base` and output dir `/data/evo2_7b_mbridge`. The `evo2/7b-8k` checkpoint is
8k-context, so `--seq-length 8192`; tokenizer `nucleotide_fast_tokenizer_512`;
precision `bf16_mixed` (matches how the 1B was converted and the LoRA recipe below).

```bash
NEMO_CKPT=$(download_bionemo_data evo2/7b-8k:1.0)   # path from Step 1

evo2_convert_nemo2_to_mbridge \
  --mixed-precision-recipe bf16_mixed \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --model-size evo2_7b_base \
  --seq-length 8192 \
  --nemo2-ckpt-dir "$NEMO_CKPT" \
  --mbridge-ckpt-dir /data/evo2_7b_mbridge
```

**Results (exit 0):**
- Log: `Loading 325 tensors into memory (Approx 12.96 GB)...` → `Keys munged. saving to
  /data/evo2_7b_mbridge/iter_0000001...` → `Conversion complete.`
- Output `/data/evo2_7b_mbridge` (**13 GB**): `iter_0000001/` (distcp shards +
  `.metadata`), `latest_checkpointed_iteration.txt`, `latest_train_state.pt` — a valid
  MBridge checkpoint, same layout as the 1B `/data/evo2_1b_mbridge`.

## Step 3 — LoRA training (seq 16384)

Reuses the **already-preprocessed** viral data from the 1B run (same
`nucleotide_fast_tokenizer_512`, no re-preprocessing):
- `/data/viral/preprocessed/viral_train_..._train.{bin,idx}` (172 MB train)
- `/data/viral/preprocessed/viral_valid_..._val.{bin,idx}` (17 MB valid)
- `/data/viral/viral_dataset.yaml` (train / validation / test = valid prefix)

Same 8× H200 pure-data-parallel config as the verified 1B seq-16384 run, only
`--model-size evo2_7b_base` and `--finetune-ckpt-dir /data/evo2_7b_mbridge` change.
Saved to `/data/viral/train_7b_lora.sh` and run via `nohup`
(log: `/data/viral/train_7b_lora.log`):

```bash
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_7b_mbridge \
  --model-size evo2_7b_base \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 \
  --mixed-precision-recipe bf16_mixed \
  --seq-length 16384 \
  --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 \
  --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_7b_16k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-7b-seq16384 \
  --lora-finetune --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
  --lora-target-modules "dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"
```

**Results:** _(in progress)_
- Base checkpoint loaded OK (`successfully loaded checkpoint from /data/evo2_7b_mbridge
  ... at iteration 0`); `Starting training loop at iteration 0`.
- **Memory fits comfortably** at seq-16384 pure-DP: ~20 GB/GPU after load, **~42–44
  GB/GPU during training** (of 143 GB), GPUs at 100% util. No OOM — no need for tensor
  parallel or extra recompute. (Benign `AccumulateGrad ... stream does not match`
  warnings appear with LoRA; harmless.)
- **Throughput: ~4.4–4.6 s/step, ~308–321 TFLOP/s/GPU** (7B is ~3.5× slower per step
  than the 1B's ~1.27 s/step, as expected). 1000 steps ≈ ~76 min.
- Training-loss curve (wandb `viral-lora-7b-seq16384`, GBS 16, lr 3e-4):

  | iter | lm loss |
  |---:|---:|
  | 10 | 1.323 |
  | 20 | 1.268 |
  | 100 | 1.209 |
  | 200 | 1.181 |
  | 300 | 1.152 |
  | 400 | 1.132 |
  | 500 | 1.123 |
  | 600 | 1.110 |
  | 700 | 1.109 |
  | 800 | 1.073 |
  | 1000 | 1.106 |

- **Validation (species-holdout `validation` split, `--eval-iters 20`):**

  | iter | val lm loss | val PPL | 1B PPL (same iter) |
  |---:|---:|---:|---:|
  | 250 | 1.174 | **3.235** | 3.263 |
  | 500 | 1.147 | **3.149** | 3.207 |
  | 750 | 1.129 | **3.092** | 3.167 |
  | 1000 | 1.127 | **3.087** | 3.172 |

  **7B beats the 1B at every eval point** and improves monotonically. Base 1B was ~3.45.
  Best checkpoint = **iter 1000** (PPL 3.087). End-of-training full eval (more iters):
  `validation set PPL 3.071`, `test set PPL 3.066` (test ≈ validation by design — the
  `test` entry points at the valid prefix).

- **TRAIN_EXIT=0.** 1000 steps in ~75 min (~4.34 s/step). `progress.txt`: 0.26 B tokens
  seen, cumulative 313.5 MODEL_TFLOP/s/GPU.
- Checkpoints (adapter-only, **405 MB** each — larger than 1B's 149 MB) saved at each
  eval: `/data/viral/lora_run_7b_16k/evo2/checkpoints/iter_0000{250,500,750,1000}`. Each
  `run_config.yaml` records the `peft:` section + `pretrained_checkpoint:
  /data/evo2_7b_mbridge`.

## Step 4 — Base vs LoRA scoring + stratified per-`genome` PPL

Report items 2 (stratified validation PPL) and 3 (base vs LoRA) from
`Evo2_virus/README.md`, via `predict_evo2` on the species-holdout valid set.

**Length cap.** `predict_evo2` has no truncation flag and scores each sequence at its
full length; the valid set ranges 246 bp → 1.16 Mb (median 3.9 kb, p90 12 kb), so whole-
genome scoring of the long dsDNA records is impractical. We **cap every record to 8192 bp**
(the 7B base's native context) — covers 1077/1349 records whole, truncates the rest — and
apply the **same cap to both** base and LoRA, so the base-vs-LoRA delta is exact and the
per-class absolute PPL is "PPL over the first ≤8192 bp of each record":

```bash
# build /data/viral/valid_cap8192.fasta — all 1349 records, each truncated to 8192 bp
# (host: /opt/dlami/nvme/evo2/evo2_data/viral/valid_cap8192.fasta)
```

Score twice, in parallel on separate GPUs (base on GPU 0, LoRA on GPU 1):

```bash
# base 7B
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/evo2_7b_mbridge/iter_0000001 \
  --output-dir /data/viral/pred_7b_base \
  --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean

# LoRA 7B (PEFT auto-detected → reloads base from pretrained_checkpoint + grafts adapters)
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_7b_16k/evo2/checkpoints/iter_0001000 \
  --output-dir /data/viral/pred_7b_lora \
  --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean
```

Aggregate with `/data/viral/aggregate_ppl.py` (joins each `record` → manifest `genome`,
buckets to canonical classes, PPL = `exp(-mean_logprob)`):

```bash
python3 /data/viral/aggregate_ppl.py \
  /data/viral/pred_7b_base /data/viral/pred_7b_lora /data/viral/manifest.tsv.gz
```

**Results (both exit 0; 1349/1349 records scored):** `predictions__rank_0__dp_rank_0.pt`
(dict: `log_probs_seqs` (1349,), `seq_idx` (1349,)) + `seq_idx_map.json` (record→idx) in
each `pred_7b_{base,lora}/`.

**Base vs LoRA — mean per-record perplexity (`exp(-mean_logprob)`), valid set capped 8192 bp:**

| genome class | n | base PPL | LoRA PPL | Δ | rel % |
|---|---:|---:|---:|---:|---:|
| **OVERALL** | 1349 | 3.6122 | **3.2184** | −0.3938 | **−10.90%** |
| ssRNA(−) | 353 | 3.6399 | 3.1986 | −0.4413 | −12.12% |
| ssRNA(+) | 337 | 3.6010 | 3.3131 | −0.2878 | −7.99% |
| ssDNA | 289 | 3.6576 | 2.9230 | −0.7346 | −20.08% |
| dsRNA | 179 | 3.7425 | 3.7029 | −0.0396 | −1.06% |
| dsDNA | 127 | 3.3626 | 2.9210 | −0.4417 | −13.13% |
| ssRNA(other) | 64 | 3.4447 | 3.3981 | −0.0465 | −1.35% |

- **LoRA improves 1306/1349 records (96.8%)**; every genome class improves. Largest gains
  on ssDNA (−20%) and dsDNA/ssRNA(−) (−12–13%); smallest on dsRNA (−1%).
- This capped per-record PPL (LoRA 3.218) differs in methodology from the in-training
  windowed validation (PPL 3.087 over full 16384 windows) — both show a clear, sizeable
  LoRA improvement. (Same methodology caveat the 1B runbook notes.)

---

## Summary — 7B vs the earlier 1B viral LoRA run

| | **7B** (this run) | 1B (`running_evo2_in_aws.md`) |
|---|---|---|
| Base ckpt | `evo2/7b-8k:1.0` → `/data/evo2_7b_mbridge` (13 GB) | `evo2/1b-8k-bf16:1.0` → `/data/evo2_1b_mbridge` |
| Train throughput | ~4.34 s/step, ~325 TFLOP/s/GPU | ~1.27 s/step, ~255 TFLOP/s/GPU |
| Mem during train | ~42–44 GB/GPU (pure DP, no TP needed) | ~16 GB/GPU |
| Adapter ckpt size | 405 MB | 149 MB |
| In-train val PPL @1000 | **3.087** | 3.172 |
| Best val PPL | **3.087** (iter 1000) | 3.167 (iter 750) |
| Base→LoRA (capped predict) | 3.612 → **3.218 (−10.9%)** | 3.533 → 3.509 (−0.7%) |

**The 7B LoRA shows a much larger viral-fit improvement than the 1B** (−10.9% vs −0.7% on
the capped per-record comparison), and a lower absolute validation PPL, while still fitting
comfortably on 8× H200 in pure data-parallel at seq-16384. Best checkpoint:
`/data/viral/lora_run_7b_16k/evo2/checkpoints/iter_0001000`. wandb run
`viral-lora-7b-seq16384` (project `evo2-viral-lora`).
