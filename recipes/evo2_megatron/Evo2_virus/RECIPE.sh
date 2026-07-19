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
# =============================================================================
