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
| 4 | Data preprocessing (`preprocess_evo2`) | Quick start | ⬜ | Needs a `preprocess_config.yaml` + input FASTA |
| 5 | Transcript extraction (`splice_evo2`) | Quick start | ⬜ | Needs genome FASTA + GTF |

## Checkpoint maintenance

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 6 | Remove optimizer state (`evo2_remove_optimizer`) | Removing optimizer state | ✅ | tmpfp8 iter_0000012: 16 GB → 2.3 GB → `/data/evo2_1b_weights_only`; CPU-only |

## Fine-tuning

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 7 | Fine-tune from NeMo2 ckpt (`train_evo2 --finetune-ckpt-dir`) | Fine-tuning → NeMo2 | ✅ | 8-step mock fine-tune from `/data/evo2_1b_mbridge` → `/data/ft_nemo2`; `finetune: true` in run_config |
| 8 | Convert Savanna → MBridge (`evo2_convert_savanna_to_mbridge`) | Fine-tuning → Savanna | ⬜ | Pulls `arcinstitute/savanna_evo2_1b_base` from HF |
| 9 | LoRA fine-tuning (`train_evo2 --lora-finetune`) | LoRA Fine-tuning | ✅ | 8-step adapter-only ckpt (149 MB) → `/data/lora_run`. Needs `--decay-steps/--warmup-steps` + `--disable-tensorboard-logger` (see runbook gotchas) |
| 10 | Inference on a LoRA checkpoint (`infer_evo2` / `predict_evo2`) | LoRA → Running inference | ✅ | Both auto-reload base from `pretrained_checkpoint`; log-probs match base 1B |

## Export

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 11 | Export MBridge → Vortex (`evo2_export_mbridge_to_vortex`) | Exporting to Vortex | ✅ | `/data/evo2_1b_vortex.pt` (~2.2 GB, 270 tensors) + `config.json`; CPU-only |
| 12 | Savanna → MBridge → Vortex round-trip | Exporting to Vortex | ⬜ | Depends on example 8 |

## Notebooks (`examples/`)

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 13 | `zeroshot_brca1.ipynb` — zero-shot BRCA1 VEP (1B) | Examples | ⬜ | Reuse converted 1B checkpoint |
| 14 | `fine-tuning-tutorial.ipynb` — fine-tune 1B on human chromosomes | Examples | ⬜ | Includes `preprocess_evo2` workflow |
| 15 | `lora-fine-tuning-tutorial.ipynb` — LoRA splice-site classification | Examples | ⬜ | Head-only baseline comparison |

## Build

| # | Example | README ref | Status | Notes |
|---|---------|-----------|--------|-------|
| 16 | Docker build (`docker build`) | Docker build | ✅ | Image `evo2:20260628` already built and in use |

## Suggested next steps

1. **Example 3** (`predict_evo2`) and **example 11** (Vortex export) — both reuse
   the existing 1B checkpoint, no new downloads.
2. **Example 6** (`evo2_remove_optimizer`) — run against the training checkpoint
   produced by example 1 (`tmpfp8`).
3. **Example 7** (fine-tune from NeMo2) and **example 13** (`zeroshot_brca1.ipynb`)
   — both reuse the converted 1B checkpoint.
