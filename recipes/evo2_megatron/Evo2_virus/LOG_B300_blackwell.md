# B300 (Blackwell) — 40B NVIDIA-checkpoint fine-tuning

Separate track from the Hopper capacity study. Node: 8× **NVIDIA B300 SXM6, ~268 GB each**
(Blackwell), driven via tmux windows `b300-1/2/3` → container `evo2run` (`-v .../evo2_data:/data`,
wandb offline). **Hard deadline: 11:30 UTC 2026-07-13** (node returned then). Started ~22:38 UTC
2026-07-12 → ~13 h.

## Purpose
1. **Verify fine-tuning works on Blackwell** (the recipe/pipeline runs on B300 at all).
2. **Match the 40B result** we got on Hopper with the **Arc** checkpoint under vortex-FP8
   (16k@3000 LoRA: val 2.767 / test 2.774; base→LoRA capped −17.20%). On Blackwell we CANNOT use the
   Arc Hopper checkpoints (Hopper-FP8/vortex-specific) — must use the **NVIDIA** checkpoint
   `evo2/40b-1m-fp8-bf16:1.0` (no 20B NVIDIA ckpt exists; 40B is the option). Question: equivalent
   results?
3. **Full fine-tune** (not just LoRA) if time — B300's 268 GB enables it where 143 GB H200 couldn't.

## Environment (confirmed)
- 8× B300 SXM6, 275040 MiB. Container `evo2run` (image evo2:20260628) started fresh (wasn't running).
- `/data/viral` present (copied from Hopper): preprocessed train/valid datasets, `viral_dataset.yaml`,
  `valid_cap8192.fasta`, `manifest.tsv.gz`, `aggregate_ppl.py`, base preds NOT present (score base fresh).
- Tools OK (train_evo2, predict_evo2, download_bionemo_data, evo2_convert_nemo2_to_mbridge). SRC=ngc.

## Plan (deadline-aware)
1. **Download** `evo2/40b-1m-fp8-bf16:1.0` (NeMo2) — RUNNING (→ `/root/.cache/bionemo/...nemo2...tar.gz`).
2. **Convert** NeMo2 → MBridge (`evo2_convert_nemo2_to_mbridge`, `--model-size evo2_40b[_base]?` — check
   converter options; `--mixed-precision-recipe bf16_mixed`) → `/data/evo2_40b_nv_mbridge`.
3. **Smoke** (~20–50 steps) LoRA to (a) confirm no NaN on Blackwell and (b) measure step-time →
   decide step budget. Precision: start **bf16_mixed** (Blackwell-native; Arc needed vortex-FP8 only
   because of a *Hopper*-bf16 instability — test whether Blackwell bf16 is stable). If NaN → try
   Blackwell FP8 recipe.
4. **Full LoRA run** matching the Arc 40B config (dim16/α32, 16k, GBS16) for as many steps as the
   budget allows (target 3000 to compare to Arc@3000; fall back to 1000 = Arc@1000 −13.51%). Score
   capped per-genome vs base.
5. **Full fine-tune** (if time): no `--lora-finetune`; full 40B on B300 (268 GB, likely needs TP;
   smoke first).

## Confirmed facts (2026-07-13 ~00:45 UTC)
- **Download DONE + validated** (`.checked` marker). NeMo2 ckpt untarred at:
  `/root/.cache/bionemo/544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680-evo2_40b_bf16_finetune_wandb_Ji2IRcrz_step_119.tar.gz.untar`
  (standard NeMo2 layout: `context/` + `weights/`). This is the **NVIDIA** 40B, name `evo2_40b_bf16_finetune...step_119`.
- **Do NOT use `/data/evo2_40b_mbridge` (77 GB)** — that is the **Arc** 40B MBridge copied from Hopper
  (Hopper-FP8/vortex-specific). Convert the NVIDIA NeMo2 → a NEW dir `/data/evo2_40b_nv_mbridge`.
- **Converter args (verified via --help):** `evo2_convert_nemo2_to_mbridge --nemo2-ckpt-dir <untar>
  --tokenizer-path <tok> --mbridge-ckpt-dir <out> [--mixed-precision-recipe ...]`. **No `--model-size`**
  (size inferred from the checkpoint `context`). Blackwell-native precision options exist if bf16
  training is unstable: `bf16_with_nvfp4_mixed` (NVFP4/FP4), `bf16_with_mxfp8_mixed` (MXFP8),
  `bf16_with_fp8_current_scaling_mixed`.
- **B300 GPUs BUSY:** another user task holds all 8 GPUs (267 GB each, PIDs 614880–887). It **overran**
  its ~00:00 UTC estimate. HOLDING all B300 GPU work. Waiter armed on **b300-1** →
  emits `B300_GPUS_FREE_MARKER` when compute-apps clear (polls every 60 s).

## Conversion command (CORRECTED — running as of ~06:35 UTC)
`--model-size` AND `--seq-length` are **required** (my first --help grep missed them). Match the Arc
40B Savanna conversion convention: `--model-size evo2_40b --seq-length 1048576` (the `40b-1m` NVIDIA
ckpt is the extended-context model, analogous to Arc Savanna 40B → `evo2_40b`, NOT the 8k `_base`).
```bash
docker exec evo2run bash -lc 'cd /workspace/bionemo; nohup evo2_convert_nemo2_to_mbridge \
  --nemo2-ckpt-dir /root/.cache/bionemo/544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680-evo2_40b_bf16_finetune_wandb_Ji2IRcrz_step_119.tar.gz.untar \
  --mbridge-ckpt-dir /data/evo2_40b_nv_mbridge \
  --model-size evo2_40b --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --seq-length 1048576 --mixed-precision-recipe bf16_mixed \
  > /data/viral/convert_40b_nv.log 2>&1 &'
```
LESSON: don't inline long backgrounded commands into a BUSY tmux pane — keystrokes concatenate with
the pending line and silently corrupt the command (I chased a phantom "argparse error" that was really
the stale log from the first no-args attempt). Verify with a foreground `timeout 45 ...` run first;
if it prints "Reading metadata..." the args are good, then relaunch backgrounded into a FREE pane.
Then smoke LoRA (~30 steps) with `--finetune-ckpt-dir /data/evo2_40b_nv_mbridge --model-size evo2_40b`,
precision **bf16_mixed first** (test Blackwell bf16 stability; Arc needed vortex-FP8 only for Hopper).
If NaN → retry with `--mixed-precision-recipe bf16_with_fp8_current_scaling_mixed` (Blackwell FP8).

## Timeline
- 22:40 UTC: NVIDIA 40B download started (b300-2). Waiter armed (b300-3 blocking).
- 2026-07-13 ~00:45 UTC: download confirmed DONE+validated. Converter args verified. B300 still busy
  with another task (overran). GPU-free waiter armed on b300-1. Conversion command staged above.
- 2026-07-13 ~06:15 UTC: the "other task" identified as a persistent **vLLM inference server** (8 TP
  workers, PGID 613983, up 9.5 h) — never going to exit on its own. User authorized reclaiming the node.
- 2026-07-13 ~06:20 UTC: `kill -9 -613983` (process-group kill) → all 8 GPUs freed (0 MiB).
- 2026-07-13 ~06:26 UTC: conversion COMPLETE (77 GB, iter_0000001, .distcp shards).
- 2026-07-13 ~06:30 UTC: **SMOKE (30-step LoRA, TP1/pure-DP8, bf16_mixed NO vortex-FP8)** → key findings:
  - **Blackwell bf16 is STABLE**: 0 NaN / 0 skipped through warmup+steady-state. Arc 40B needed
    vortex-FP8 *only* because of a Hopper-bf16 instability — **not needed on Blackwell.**
  - **TP1 works**: frozen 40B (~80 GB) + activations fits one B300 card. Steady-state **~243 GB/GPU**
    used (recompute-2) → **full fine-tune (all-param optimizer state) will NOT fit this slot** → LoRA.
  - **Fast**: step5 13.76 s (compile) → **step10 ~11.0 s/step, 690 TFLOP/s/GPU**. vs Hopper Arc 40B
    **23.4 s/step, ~320 TFLOP/s, TP4+vortex-FP8**. B300 ≈ **2× faster, 2× TFLOP/s, simpler parallelism.**
- 2026-07-13 ~06:35 UTC: **MAIN RUN launched** — 40B LoRA, TP1/DP8, bf16_mixed, dim16/α32 (Arc-40B
  config), seq16k, GBS16, **1000 steps** (~3 h @11 s → done ~09:40), eval 250. `lora_40b_nv_b300_1k`.
  Comparison target: Arc 40B@1000 = −13.51% (val 2.976). Score vs FRESH NVIDIA-40B base (must generate;
  Hopper base preds are 20B/Arc, not comparable). Base+LoRA scoring runs after the 8-GPU training frees.
- 2026-07-13 ~09:43 UTC: **MAIN RUN COMPLETE** — 1000/1000, **0 NaN**, loss 1.22→1.06, 688 TFLOP/s,
  ~11 s/step, ~3.1 h wall. Checkpoint iter_0001000 saved.
- 2026-07-13 ~10:05 UTC: **SCORED** (base+LoRA predicts in parallel GPU0/GPU1, bf16, no vortex).

## RESULT — Blackwell 40B fine-tuning VERIFIED (primary + secondary goals met)
| 40B @1000 | base PPL | LoRA PPL | OVERALL | coverage | dsRNA | precision | parallel | s/step |
|---|---:|---:|---:|---:|---:|---|---|---:|
| Arc (Hopper) | 3.5784 | — | **−13.51%** | 97.3% | — | vortex-FP8 | TP4 | 23.4 |
| **NVIDIA (B300)** | 3.6472 | 3.0921 | **−15.22%** | **98.3%** | −2.65% | **bf16 (no FP8)** | **TP1/DP8** | **~11** |

Per-genome (B300): ssDNA −25.54%, ssRNA(-) −18.13%, dsDNA −17.01%, ssRNA(+) −11.05%, ssRNA(other)
−6.53%, dsRNA −2.65%. Coverage 1325/1348 = **98.3%** (highest of any run).

**Conclusions:**
1. **Blackwell CAN fine-tune Evo2 40B** end-to-end — 0 NaN, clean loss decrease, valid predictions.
2. **bf16 is stable on Blackwell — vortex-FP8 NOT needed** (it was a Hopper-bf16 workaround). Simpler recipe.
3. **Results equivalent-to-better than Arc-40B/Hopper** at matched steps: −15.22% vs −13.51%, coverage
   98.3% vs 97.3%. (Caveat: different base ckpts — NV base 3.647 vs Arc 3.578 — and precision; magnitude
   comparison, not bit-exact.)
4. **~2× faster with simpler parallelism**: TP1/pure-DP8 (frozen 40B fits one 268 GB card) vs Hopper's
   TP4; ~11 s/step & 688 TFLOP/s vs 23.4 s/step & ~320.
5. **Full fine-tune**: LoRA already uses ~243 GB/GPU at TP1 → full-param optimizer state won't fit at
   TP1; needs TP sharding. Stretch-goal smoke: see below.

## STRETCH GOAL — full fine-tune (all params) FEASIBLE on B300 ✓
2026-07-13 ~10:05 UTC: `smoke_fullft_40b_b300` — full 40B FT (NO LoRA), **TP4/DP2**, bf16_mixed,
recompute-2, seq16k, GBS16, lr 5e-6, 20 steps.
- **Runs cleanly: 0 NaN / 0 skipped**, loss moving (1.225 → 1.166), grad norm 0.3–0.6.
- **~14.5 s/step steady** (18.8 s step-2 compile), **587 TFLOP/s/GPU**.
- **Memory only ~145 GB/GPU** (mem-max-reserved 144.77) → **~123 GB headroom** under 268 GB. Full-FT
  could run at TP2 (less comm) or larger MBS/batch. Distributed optimizer shards fp32 master+m+v
  across DP; TP4 shards weights/grads → comfortably fits.
- **Conclusion: B300 can do a 40B FULL fine-tune** (not just LoRA) — the 268 GB/GPU is the enabler
  (H200's 143 GB cannot). A production full-FT would just need many steps + a proper LR schedule; this
  smoke proves the pipeline + memory + stability. Did NOT launch a long full-FT (wouldn't finish+score
  before the 11:30 UTC node deadline; feasibility is the deliverable).

## FINAL STATUS: all B300 goals met (verify FT ✓, match Arc-40B ✓ exceeded, full-FT feasible ✓ bonus).
