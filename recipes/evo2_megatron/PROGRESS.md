# Evo2 examples — progress tracker

Status of running every example in [`README.md`](README.md) on the AWS H200 host.
See [`AGENTS.md`](AGENTS.md) for environment/conventions and
[`running_evo2_in_aws.md`](running_evo2_in_aws.md) for the commands + results.

**Legend:** ✅ done & verified · 🟡 in progress / partial · ⬜ not started ·
⏭️ skipped (with reason) · ❌ failed (see notes)

_Last updated: 2026-06-28_

## Setup / prerequisites

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 0a | Build venv (`.ci_build.sh` / `.ci_test_env.sh`) | Installation | ✅ | Baked into `evo2:20260628` image; venv on `PATH` at `/workspace/.venv` |
| 0b | Launch container with GPUs | running_evo2_in_aws.md §1 | ✅ | Container running (`evo2:20260628`), `/data` bind-mount |
| 0c | Convert 1B NeMo2 → MBridge (`evo2_convert_nemo2_to_mbridge`) | running_evo2_in_aws.md §2 | ✅ | `/data/evo2_1b_mbridge/iter_0000001`, `bf16_mixed` |

## Quick start

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 1 | Training with mock data (Hyena, `train_evo2`, `striped_hyena_1b_nv_parallel`) | Quick start | ✅ | Confirmed run; documented in running_evo2_in_aws.md (`--result-dir tmpfp8`) |
| 2 | Autoregressive generation (`infer_evo2`) | Quick start | ✅ | `/data/generated.jsonl`, greedy (`--top-k 1`), valid ACGT continuation |
| 3 | Batch sequence scoring (`predict_evo2`) | Quick start | ✅ | 3-seq FASTA → `/data/predictions/*.pt`; mean log-probs `[-0.31, -0.60, -0.31]` |
| 4 | Data preprocessing (`preprocess_evo2`) | Quick start | ✅ | 12-seq FASTA → `/data/preproc_out` train/val/test `.bin/.idx` (24 samples); CPU-only |
| 5 | Transcript extraction (`splice_evo2`) | Quick start | ✅ | chr1 + 2-exon GTF → `/data/transcripts.fa`; spliced seq matches expected. GTF needs gbkey/transcript_biotype + all values quoted |

## Checkpoint maintenance

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 6 | Remove optimizer state (`evo2_remove_optimizer`) | Removing optimizer state | ✅ | tmpfp8 iter_0000012: 16 GB → 2.3 GB → `/data/evo2_1b_weights_only`; CPU-only |

## Fine-tuning

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 7 | Fine-tune from NeMo2 ckpt (`train_evo2 --finetune-ckpt-dir`) | Fine-tuning → NeMo2 | ✅ | 8-step mock fine-tune from `/data/evo2_1b_mbridge` → `/data/ft_nemo2`; `finetune: true` in run_config |
| 8 | Convert Savanna → MBridge (`evo2_convert_savanna_to_mbridge`) | Fine-tuning → Savanna | 🟡 | **Fails as written** (PT2.6 `weights_only=True` vs numpy globals). Verified the only blocker — works with `weights_only=False`; produced `/data/mbridge_1b_savanna`. Needs recipe fix (see runbook) |
| 9 | LoRA fine-tuning (`train_evo2 --lora-finetune`) | LoRA Fine-tuning | ✅ | 8-step adapter-only ckpt (149 MB) → `/data/lora_run`. Needs `--decay-steps/--warmup-steps` + `--disable-tensorboard-logger` (see runbook gotchas) |
| 10 | Inference on a LoRA checkpoint (`infer_evo2` / `predict_evo2`) | LoRA → Running inference | ✅ | Both auto-reload base from `pretrained_checkpoint`; log-probs match base 1B |

## Export

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 11 | Export MBridge → Vortex (`evo2_export_mbridge_to_vortex`) | Exporting to Vortex | ✅ | `/data/evo2_1b_vortex.pt` (~2.2 GB, 270 tensors) + `config.json`; CPU-only |
| 12 | Savanna → MBridge → Vortex round-trip | Exporting to Vortex | 🟡 | Step 2 (Vortex export) verified → `/data/evo2_1b_savanna_vortex.pt` (~2.2 GB). Step 1 inherits the example-8 `weights_only` caveat |

## Notebooks (`examples/`)

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 13 | `zeroshot_brca1.ipynb` — zero-shot BRCA1 VEP (1B) | Examples | ✅ | End-to-end `NBEXIT=0`; **AUROC 0.74**. Needs venv kernel + output-strip (see runbook) |
| 14 | `fine-tuning-tutorial.ipynb` — fine-tune 1B on human chromosomes | Examples | ✅ | `FAST_CI_MODE=1` end-to-end `NBEXIT=0` on 8× H200 (chr20/21/22 → preprocess → convert → train, iter_0000010) |
| 15 | `lora-fine-tuning-tutorial.ipynb` — LoRA splice-site classification | Examples | ✅ | `FAST_CI_MODE=1` `NBEXIT=0` (baseline 0.33% params / LoRA 1.42%). **Run single-GPU** — 8-GPU FAST_CI hits ZeroDivisionError (see runbook) |

## Build

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 16 | Docker build (`docker build`) | Docker build | ✅ | Image `evo2:20260628` already built and in use |

## Status summary

All 16 examples have been exercised: **14 fully verified (✅)** and **2 partial
(🟡)**. The only outstanding issue is a recipe bug:

- **Example 8 / 12 (Savanna→MBridge):** `evo2_convert_savanna_to_mbridge` fails
  out of the box on PyTorch 2.6 because `load_savanna_state_dict` uses
  `torch.load(weights_only=True)` against a checkpoint that pickles numpy
  objects. Verified this is the only blocker (works with `weights_only=False`).
  **Action:** fix `load_savanna_state_dict` to allowlist the numpy globals (or
  load ARC's trusted checkpoint with `weights_only=False`); then re-verify
  examples 8 and 12 as ✅.

Other notes captured during the run (see `running_evo2_in_aws.md` for details):

- LoRA training (`train_evo2 --lora-finetune`) needs
  `--disable-tensorboard-logger` (frozen-param `main_grad` crash) and explicit
  `--decay-steps`/`--warmup-steps` for short runs.
- Notebooks must run with a **venv ipykernel** (system `python3` kernel lacks
  `bionemo`/`seaborn`) and need their shipped outputs stripped before nbconvert.
- The LoRA classifier notebook's `FAST_CI_MODE` smoke test must run on a **single
  GPU**.
