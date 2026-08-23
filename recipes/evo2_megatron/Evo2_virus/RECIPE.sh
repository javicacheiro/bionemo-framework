#!/usr/bin/env bash
# =============================================================================
# Evo2 viral-LoRA — RECOMMENDED PRODUCTION RECIPE
# =============================================================================
# Distilled from the cross-scale study (7B/20B/40B). See RESULTS_SUMMARY.md and
# EXPLORATION_LOG.md for the full derivation.
#
#   BEST RESULT:  20B, dim256, alpha 1024, dropout 0.3, all-5 targets, 16k seq,
#                 6000 steps  ->  -25.44% mean PPL / 93.7% coverage
#                 (1.85x the dim16 baseline of -13.71%).
#   PRICE/PERF:   same config, dropout 0.2, 3000 steps  ->  -24.33% (half the cost).
#
# =============================================================================
# IF YOU ADD DATA, SCALE THE STEPS TO KEEP EPOCHS CONSTANT.
# =============================================================================
# The 6000 steps above are tuned for train.fasta (171.80 Mtok) and give 9.155 passes over it:
#     epochs = max_steps x global_batch x seq_len / corpus_tokens
#            = 6000 x 16 x 16384 / 171.80e6 = 9.155
# Enlarging the corpus WITHOUT raising --max-steps silently trains for fewer passes, and that costs
# out-of-corpus transfer. To keep a bigger corpus comparable:
#     max_steps = 9.155 x corpus_tokens / (16 x 16384)
# e.g. train+valid (188.46 Mtok) needs 6582 steps, not 6000. Set --decay-steps to match --max-steps
# so the cosine keeps its shape (leaving it at 6000 parks the tail at the min-lr floor).
#
# ...BUT ONLY FOR SMALL STRETCHES. A stretched cosine holds the LR ABOVE the champion's trajectory at
# every early iteration, and this recipe sits on an alpha x LR stability edge (alpha 1024 @ lr 3e-4).
# The +10% stretch above (6000 -> 6582) ran clean at n=3. A +50% stretch (6000 -> 9000) killed 2 of 4
# seeds against 0 of 5 for the champion -- and the SAME seeds trained fine at 9000 steps when
# --decay-steps was left at 6000. If you need to stretch much beyond ~10%, either keep --decay-steps
# at 6000 (extra steps then run at the min-lr floor) or lower the LR/alpha first. See Phase 16-EPOCH-B.
#
# AND DO NOT EXPECT MORE EPOCHS TO HELP. 9000 steps (13.73 epochs) was tested against this recipe's
# 9.155 at n=3: 31 of 63 pairwise downstream comparisons favoured it, i.e. a coin flip, with held-out
# PPL flat. The epoch response has PLATEAUED by ~9. The scaling rule above exists to RESTORE passes
# when the corpus grows -- it is not a reason to add passes. (Phase 16-EPOCH-B, 2026-08-17.)
#
# HISTORY, because this file previously said the opposite. Phase 16 (2026-08-04/06) merged valid into
# train at a FIXED 6000 steps, measured worse out-of-corpus transfer, and concluded "do not merge".
# That comparison was CONFOUNDED: at fixed steps the larger corpus got 8.35 epochs vs 9.155. Redoing
# it at matched epochs (Phase 16-EPOCH, 2026-08-10, n=3) recovered ~59% of the deficit on average and
# 100% on the cleanest benchmark (HA Perth: 0.0653 -> 0.1140 vs champion 0.1113). A residual ~40%
# remains on two benchmarks but is within seed noise and is not claimed.
#
# STILL TRUE, and the real reason to keep a held-out split:
#   * Merging valid into train forfeits EVERY held-out metric by construction -- the -25.44% headline
#     above becomes unquotable, because valid_cap8192 is then trained-on. Keep a held-out split unless
#     you have an external benchmark to evaluate on.
#   * 1 of 3 full-corpus seeds diverged (grad norm -> 0, val PPL 502) vs 0 of 5 for this recipe. Small
#     numbers, but watch iters 100-300 and see the divergence signature in EXPLORATION_LOG.md.
# See EXPLORATION_LOG.md "Phase 16-EPOCH" for the correction and the full numbers.
# =============================================================================
#
# Run INSIDE the evo2 container (image evo2:20260628); the venv (train_evo2,
# predict_evo2, ...) is already on PATH. From the host:
#   docker exec -it $(docker ps -q -f ancestor=evo2:20260628) bash
#   cd /workspace/bionemo && bash /data/viral/RECIPE.sh
# =============================================================================
set -euo pipefail

TOK=tokenizers/nucleotide_fast_tokenizer_512
DATA=/data/viral/viral_dataset.yaml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=/data/viral/PROD_20b_best          # result dir (keeps every checkpoint)
LORA="--lora-finetune --lora-dim 256 --lora-alpha 1024 --lora-dropout 0.3 \
  --lora-target-modules dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"

# -----------------------------------------------------------------------------
# 1. TRAIN  — 20B best config (8x H200, pure DP, vortex-FP8, recompute-1)
#    ~9 s/step -> ~15 h for 6000 steps. Drop to --max-steps/--decay-steps 3000
#    (dropout 0.2) for the price/perf variant (~7.5 h, -24.33%).
# -----------------------------------------------------------------------------
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config "$DATA" \
  --finetune-ckpt-dir /data/evo2_20b_mbridge --model-size evo2_20b \
  --hf-tokenizer-model-path "$TOK" \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 6000 --warmup-steps 10 --decay-steps 6000 \
  --eval-interval 500 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 20 \
  --most-recent-k -1 \
  --result-dir "$RUN" \
  --wandb-project evo2-viral-lora --wandb-run-name prod-20b-best \
  $LORA

# -----------------------------------------------------------------------------
# 2. SCORE  — base vs LoRA, capped-8192 per-genome PPL (base on GPU0, LoRA GPU1).
#    Use the SAME precision (vortex-FP8) for base and LoRA so the delta is exact.
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta --ckpt-dir /data/evo2_20b_mbridge/iter_0000001 \
  --output-dir /data/viral/PROD_pred_base --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta --ckpt-dir "$RUN"/evo2/checkpoints/iter_0006000 \
  --output-dir /data/viral/PROD_pred_lora --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
wait
python3 /data/viral/aggregate_ppl.py \
  /data/viral/PROD_pred_base /data/viral/PROD_pred_lora /data/viral/manifest.tsv.gz
# Expect OVERALL ~ -25% (6000 steps) / -24.3% (3000 steps).

# =============================================================================
# PER-SCALE CONFIGS  — scale alpha with model width (do NOT copy alpha across sizes)
# =============================================================================
#   model | ckpt / model-size            | precision            | dim  alpha  dropout | result
#   ------+------------------------------+----------------------+---------------------+--------------------
#   7B    | evo2_7b_mbridge / evo2_7b_base| bf16 (NO vortex-FP8) | 256   512   0.2     | -23.3%@3k / -24.4%@6k
#   20B   | evo2_20b_mbridge / evo2_20b   | bf16 + vortex-FP8    | 256  1024   0.2/0.3 | -24.3%@3k / -25.4%@6k  <-- sweet spot
#   40B   | evo2_40b_mbridge / evo2_40b   | bf16 + vortex-FP8    | 256   768   0.2     | -23.95%@3k (alpha1024 DIVERGES)
#
# KEY RULES (from the study):
#  * alpha has an "alpha x LR" ceiling: too-high alpha diverges at the standard LR (3e-4); a lower LR
#    restores stability but never beats the standard-alpha peak (no free lunch). Stay at/below the
#    per-scale alpha above.
#  * dropout optimum rises with model size: 7B wants 0.2 (0.3 over-regularizes); 20B wants 0.3 for long
#    (6k) runs. Raise dropout when you raise steps.
#  * The MLP adapters (linear_fc1/fc2) carry ~all the benefit; attn/mixer projections add ~0.4pp.
#  * dim256 is the right capacity at every scale; length helps to ~6k then hits the data-limited ceiling.
#  * 20B is the price/perf sweet spot -- the 40B does NOT surpass it and is far more fragile.
#  * 7B uses bf16 without vortex-FP8 and loads as --model-size evo2_7b_base (11008 MLP dim; evo2_7b errors).
#
# MEASURED PARALLELISM / MEMORY  (2026-08-15 audit, 8xH200 143 GB, seq 16384, mbs 1, recompute-1,
# LoRA dim256; 60-step runs. Logs: /fsx/evo2/evo2_data/logs/memaudit/)
#
#   config      result   params(B)  theor W+opt(MB)  peak reserved(GB)
#   7B  TP=1    ok        5.04       36 019           42.7
#   20B TP=1    ok       15.31      109 476          133.3
#   20B TP=2    ok       15.31       65 685           67.0
#   20B TP=4    ok       15.31       43 790           34.2
#   40B TP=1    OOM        --           --              --
#   40B TP=2    ok       31.88      136 825          139.1     <-- WORKS; use this, not TP=4
#   40B TP=4    ok       31.88       91 217           70.9
#
#  * **The 40B runs at TP=2**, at 139.1 GB of 143 GB. Earlier work used TP=4 and never tried TP=2,
#    because a (wrong) theory said an FP32 grad buffer was allocated for the FROZEN base params and
#    would not fit. It is not: megatron.core 0.17.0rc0 skips frozen params before buffer allocation
#    (`distributed_data_parallel.py`: `if not param.requires_grad: continue`). TP=2 halves the
#    tensor-parallel communication vs TP=4. Headroom is only ~4 GB, so keep
#    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and do not raise seq-length or mbs without
#    re-measuring; drop to TP=4 if you do.
#  * Peak memory scales as ~1/TP (20B: 133.3 -> 67.0 -> 34.2), tracking the theoretical
#    weight+optimizer figure. That is weights and optimizer state being sharded -- the 40B OOMs at
#    TP=1 simply because ~32 B params do not fit unsharded alongside activations.
# =============================================================================
#
# OPTIONAL — dsRNA reverse-complement (RC) augmentation  [Phase 14; NOT part of the default recipe]
# -----------------------------------------------------------------------------------------------
# The dsRNA class is under-represented (2.45% of tokens, 1574 records) and stays the worst class
# (~-10%). Adding the RC of the dsRNA records (a biologically valid, DISTINCT strand -> doubles unique
# dsRNA windows), optionally upweighted, improves IN-DISTRIBUTION dsRNA/aggregate held-out PPL:
#     dsRNA:   plain -9.69% -> RC-natural -10.02% -> RC + dsRNA-upweight-to-10% -13.13%
#     overall: plain -25.44% -> RC-natural -25.51% -> RC + upweight-to-6% -25.74%
#   (Reweighting WITHOUT RC backfires, +5.28% -- it memorises the tiny set; RC supplies real unique data.
#    Whole-corpus RC-doubling underfits at fixed steps and is NOT worth it -- only augment the thin class.)
# HOWEVER: downstream validation (SARS-CoV-2 RBD DMS, ssRNA(+)) shows this PPL gain does NOT transfer --
# the PLAIN champion beats both RC adapters on variant-effect prediction (Spearman 0.368 vs 0.325/0.295),
# and RC matches plain on ssRNA(+) held-out PPL yet scores lower downstream (regression ORTHOGONAL to PPL;
# effect is NOT RC-specific -- see Phase 15: over-optimizing held-out PPL hurts downstream generally). => Use dsRNA-RC ONLY when in-distribution
# dsRNA likelihood is itself the deliverable; keep the plain config above as the general recipe.
# Pipeline: rc_augment.py (IUPAC-aware) -> preprocess_evo2 -> blend rest+dsRNA(both strands) in the dataset yaml.
# =============================================================================
