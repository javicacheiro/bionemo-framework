#!/bin/bash
# =============================================================================
# COMMANDS.sh — reproducible command log for the Evo2 viral-LoRA runs
# =============================================================================
# Every command that produced the results in the LOG_evo2_*.md files, in order.
# This is a RECORD to repeat runs, not a single auto-run script: the training
# runs take hours-to-days each and share all 8 GPUs, so run the blocks you want
# one at a time (long jobs were launched with `nohup <script> &`).
#
# Host:      AWS 8x NVIDIA H200 (143 GB).
# Container: evo2:20260628 (started --rm with /opt/dlami/nvme/evo2/evo2_data -> /data).
# Enter it:  docker exec -it $(docker ps -q -f ancestor=evo2:20260628) bash
# Inside:    venv already on PATH (train_evo2, predict_evo2, preprocess_evo2,
#            evo2_convert_*); BIONEMO_DATA_SOURCE=ngc; run `wandb login` once.
#            cd /workspace/bionemo   (recipe root; tokenizers/ live here)
# Persist everything under /data (host bind-mount).
#
# Config cheat-sheet (why the flags differ per model — all vortex-FP8 for the
# FP8/Hopper-sensitive Savanna 20B/40B):
#   1B  (24L) : pure DP,  recompute 1,           bf16
#   7B        : pure DP,  recompute 1,           bf16
#   20B (24L) : pure DP,  recompute 2, vortex-FP8              @16k
#   20B @32k  : TP=2,     recompute 1, vortex-FP8, expandable_segments
#   20B @128k : TP=4,CP=1 recompute 1, vortex-FP8, expandable_segments
#   40B (50L) : TP=4,     recompute 1, vortex-FP8, expandable_segments  (pure DP OOMs)
# Memory driver: Megatron pre-allocates an fp32 main_grad buffer for ALL params
# incl. the frozen LoRA base (~6 B/param fixed footprint); it is sharded by TP
# (not DP), so big models / long context need TP. recompute-num-layers uses
# method=uniform where N is the CHUNK SIZE -> use N=1 (smallest) for least memory.
# =============================================================================

set -uo pipefail
TOK=tokenizers/nucleotide_fast_tokenizer_512   # HF tokenizer shipped in the image
REPO=/opt/dlami/nvme/evo2/bionemo-recipes/recipes/evo2_megatron/Evo2_virus  # this repo (data/ here)

# =============================================================================
# 0. DATA PREP  (one-time; produces the reusable /data/viral assets)
# =============================================================================
mkdir -p /data/viral/preprocessed

# 0a. JSONL -> FASTA (verbatim: text->sequence, record->header). Run from $REPO.
for split in train valid; do
  python3 -c '
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f, open(sys.argv[2], "w") as g:
    for line in f:
        r = json.loads(line)
        g.write(">" + r["record"] + "\n" + r["text"] + "\n")
' "$REPO/data/${split}.jsonl.gz" /data/viral/${split}.fasta
done

# 0b. Preprocess to Megatron indexed datasets. viral_preprocess.yaml forces each
#     FASTA entirely into one split (train->train_split 1.0, valid->valid_split 1.0)
#     and uses DISTINCT output_prefix per entry (else the 2nd run is skipped).
cat > /data/viral/viral_preprocess.yaml << 'YAML'
- datapaths: ["/data/viral/train.fasta"]
  output_dir: /data/viral/preprocessed
  output_prefix: viral_train
  hf_tokenizer_model_path: /workspace/bionemo/tokenizers/nucleotide_fast_tokenizer_512
  train_split: 1.0
  valid_split: 0.0
  test_split: 0.0
  append_eod: true
  transcribe: null
  random_reverse_complement: 0.0
  embed_reverse_complement: false
  force_uppercase: false
  workers: 8
- datapaths: ["/data/viral/valid.fasta"]
  output_dir: /data/viral/preprocessed
  output_prefix: viral_valid
  hf_tokenizer_model_path: /workspace/bionemo/tokenizers/nucleotide_fast_tokenizer_512
  train_split: 0.0
  valid_split: 1.0
  test_split: 0.0
  append_eod: true
  transcribe: null
  random_reverse_complement: 0.0
  embed_reverse_complement: false
  force_uppercase: false
  workers: 8
YAML
preprocess_evo2 --config /data/viral/viral_preprocess.yaml
# -> /data/viral/preprocessed/viral_train_nucleotide_fast_tokenizer_512_train.{bin,idx} (~172 MB)
#    /data/viral/preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val.{bin,idx}   (~17 MB)

# 0c. Blended dataset config used by every train_evo2 run below. A `test` entry
#     is REQUIRED (provider builds train/validation/test); point it at valid.
cat > /data/viral/viral_dataset.yaml << 'YAML'
- dataset_prefix: /data/viral/preprocessed/viral_train_nucleotide_fast_tokenizer_512_train
  dataset_weight: 1.0
  dataset_split: train
- dataset_prefix: /data/viral/preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val
  dataset_weight: 1.0
  dataset_split: validation
- dataset_prefix: /data/viral/preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val
  dataset_weight: 1.0
  dataset_split: test
YAML

# 0d. Scoring inputs: cap each valid record to 8192 bp, and stage the manifest.
python3 -c '
seqs={}; name=None; buf=[]
import sys
for line in open("/data/viral/valid.fasta"):
    line=line.rstrip("\n")
    if line.startswith(">"):
        if name is not None: seqs[name]="".join(buf)
        name=line[1:]; buf=[]
    else: buf.append(line)
if name is not None: seqs[name]="".join(buf)
with open("/data/viral/valid_cap8192.fasta","w") as g:
    for k,v in seqs.items(): g.write(">"+k+"\n"+v[:8192]+"\n")
'
cp "$REPO/metadata/eukaryotic_host_core_manifest.tsv.gz" /data/viral/manifest.tsv.gz
# aggregate_ppl.py (base-vs-LoRA per-genome perplexity join) lives at /data/viral/aggregate_ppl.py

# =============================================================================
# 1. BASE CHECKPOINT CONVERSIONS  (NeMo2/Savanna -> MBridge, one-time each)
# =============================================================================
# 1B (NeMo2 -> MBridge, bf16). See running_evo2_in_aws.md for the exact download.
#   download_bionemo_data evo2/1b-8k-bf16:1.0 --source ngc
#   evo2_convert_nemo2_to_mbridge --nemo2-ckpt-path <dl> --mbridge-ckpt-dir /data/evo2_1b_mbridge \
#     --model-size evo2_1b_base --tokenizer-path $TOK --mixed-precision-recipe bf16_mixed

# 7B (NeMo2 -> MBridge, bf16) -> /data/evo2_7b_mbridge (13 GB)
#   download_bionemo_data evo2/7b-8k:1.0 --source ngc
#   evo2_convert_nemo2_to_mbridge --nemo2-ckpt-path <dl> --mbridge-ckpt-dir /data/evo2_7b_mbridge \
#     --model-size evo2_7b_base --tokenizer-path $TOK --mixed-precision-recipe bf16_mixed

# 20B (Savanna HF, multi-part shards -> MBridge, 37 GB)
evo2_convert_savanna_to_mbridge \
  --savanna-ckpt-path arcinstitute/savanna_evo2_20b \
  --mbridge-ckpt-dir /data/evo2_20b_mbridge \
  --model-size evo2_20b \
  --tokenizer-path $TOK \
  --seq-length 1048576 \
  --mixed-precision-recipe bf16_mixed

# 40B (Savanna HF, 5 shards .pt.part0..4 -> MBridge, 77 GB)
evo2_convert_savanna_to_mbridge \
  --savanna-ckpt-path arcinstitute/savanna_evo2_40b \
  --mbridge-ckpt-dir /data/evo2_40b_mbridge \
  --model-size evo2_40b \
  --tokenizer-path $TOK \
  --seq-length 1048576 \
  --mixed-precision-recipe bf16_mixed

# =============================================================================
# 2. TRAINING RUNS  (each is a long job; launch with `nohup bash <block> &`)
#    Shared LoRA flags: --lora-finetune --lora-dim 16 --lora-alpha 32
#      --lora-dropout 0.1 --lora-target-modules
#      "dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"
#    Shared batch: --micro-batch-size 1 --global-batch-size 16
# =============================================================================
LORA='--lora-finetune --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 --lora-target-modules dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2'

# --- 1B  @16k, 1000 steps  (pure DP, bf16) -> val PPL ~3.17 -----------------
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_1b_mbridge --model-size evo2_1b_base \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_16k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-1b-seq16384 $LORA

# --- 7B  @16k, 1000 steps  (pure DP, bf16) -> val PPL 3.087 -----------------
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_7b_mbridge --model-size evo2_7b_base \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_7b_16k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-7b-seq16384 $LORA

# --- 20B @16k, 1000 steps  (pure DP, recompute 2, vortex-FP8) -> val PPL 2.969
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_20b_mbridge --model-size evo2_20b \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 2 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_20b_16k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-20b-seq16384-vfp8 $LORA

# --- 40B @16k, 1000 steps  (TP=4, recompute 1, vortex-FP8) -> val PPL 2.976 --
# NOTE: pure DP and TP=2 OOM for the 40B; TP=4 + recompute-1 is required.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --tensor-model-parallel-size 4 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_40b_mbridge --model-size evo2_40b \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_40b_16k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-40b-seq16384-vfp8 $LORA

# --- 20B @16k, 3000 steps  (longer training; full cosine decay 3000) -> 2.754
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_20b_mbridge --model-size evo2_20b \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 2 \
  --max-steps 3000 --warmup-steps 10 --decay-steps 3000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_20b_16k_3k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-20b-seq16384-vfp8-3k $LORA

# --- 40B @16k, 3000 steps  (longer training) -> val PPL 2.767 ---------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --tensor-model-parallel-size 4 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_40b_mbridge --model-size evo2_40b \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 16384 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 3000 --warmup-steps 10 --decay-steps 3000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_40b_16k_3k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-40b-seq16384-vfp8-3k $LORA

# --- 20B @32k, 3000 steps  (context ablation; TP=2, recompute 1) -> 2.755 ----
# Peak ~64 GB/GPU, ~20.6 s/step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --tensor-model-parallel-size 2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_20b_mbridge --model-size evo2_20b \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 32768 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 3000 --warmup-steps 10 --decay-steps 3000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_20b_32k_3k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-20b-seq32768-vfp8-3k $LORA

# --- 20B @128k, 1000 steps (context ablation; TP=4, CP=1, recompute 1) -> 2.806
# Peak ~66.7 GB/GPU, ~110 s/step (~583 TFLOP/s/GPU), ~31 h. CP not needed at 128k.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --tensor-model-parallel-size 4 --context-parallel-size 1 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_20b_mbridge --model-size evo2_20b \
  --hf-tokenizer-model-path $TOK --mixed-precision-recipe bf16_mixed --vortex-style-fp8 \
  --seq-length 131072 --micro-batch-size 1 --global-batch-size 16 \
  --activation-checkpoint-recompute-num-layers 1 \
  --max-steps 1000 --warmup-steps 10 --decay-steps 1000 \
  --eval-interval 250 --eval-iters 20 --lr 3e-4 --min-lr 3e-5 --log-interval 10 \
  --result-dir /data/viral/lora_run_20b_128k_1k \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-20b-seq131072-vfp8-1k $LORA

# =============================================================================
# 3. SCORING  (base vs LoRA, capped 8192 bp valid set; per-genome PPL)
#    predict has NO training grad buffer, so a single model fits one GPU at TP=1
#    -> run base on GPU 0 and LoRA on GPU 1 in parallel. Use the SAME precision
#    (vortex-FP8 for 20B/40B) for base and LoRA so the delta is exact.
#    aggregate_ppl.py: exp(-mean_logprob) per record, joined to manifest `genome`.
# =============================================================================
# Template (substitute <BASE_CKPT>, <LORA_CKPT>, <OUTDIR_BASE>, <OUTDIR_LORA>):
#   CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
#     --fasta /data/viral/valid_cap8192.fasta --ckpt-dir <BASE_CKPT> \
#     --output-dir <OUTDIR_BASE> --micro-batch-size 1 --write-interval epoch \
#     --output-log-prob-seqs --log-prob-collapse-option mean \
#     --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
#   CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
#     --fasta /data/viral/valid_cap8192.fasta --ckpt-dir <LORA_CKPT> \
#     --output-dir <OUTDIR_LORA> ... (same flags) &  ; wait
#   python3 /data/viral/aggregate_ppl.py <OUTDIR_BASE> <OUTDIR_LORA> /data/viral/manifest.tsv.gz
#
# Base checkpoints (PEFT adapters auto-detected from the LoRA run_config.yaml):
#   20B base: /data/evo2_20b_mbridge/iter_0000001
#   40B base: /data/evo2_40b_mbridge/iter_0000001
#   (1B/7B base predicts omit --vortex-style-fp8; use bf16_mixed only.)

# 20B  base -> LoRA@1000
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta --ckpt-dir /data/evo2_20b_mbridge/iter_0000001 \
  --output-dir /data/viral/pred_20b_base_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_20b_16k/evo2/checkpoints/iter_0001000 \
  --output-dir /data/viral/pred_20b_lora_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
wait
python3 /data/viral/aggregate_ppl.py /data/viral/pred_20b_base_vfp8 /data/viral/pred_20b_lora_vfp8 /data/viral/manifest.tsv.gz

# 40B  base -> LoRA@1000  (reuse base preds for later @3000 comparisons)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta --ckpt-dir /data/evo2_40b_mbridge/iter_0000001 \
  --output-dir /data/viral/pred_40b_base_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_40b_16k/evo2/checkpoints/iter_0001000 \
  --output-dir /data/viral/pred_40b_lora_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
wait
python3 /data/viral/aggregate_ppl.py /data/viral/pred_40b_base_vfp8 /data/viral/pred_40b_lora_vfp8 /data/viral/manifest.tsv.gz

# 3000-step LoRA@3000 (20B on GPU0, 40B on GPU1), scored vs the reused base preds
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_20b_16k_3k/evo2/checkpoints/iter_0003000 \
  --output-dir /data/viral/pred_20b_lora_3k_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_40b_16k_3k/evo2/checkpoints/iter_0003000 \
  --output-dir /data/viral/pred_40b_lora_3k_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8 &
wait
python3 /data/viral/aggregate_ppl.py /data/viral/pred_20b_base_vfp8 /data/viral/pred_20b_lora_3k_vfp8 /data/viral/manifest.tsv.gz
python3 /data/viral/aggregate_ppl.py /data/viral/pred_40b_base_vfp8 /data/viral/pred_40b_lora_3k_vfp8 /data/viral/manifest.tsv.gz

# 32k LoRA@3000 and 128k LoRA@1000 (20B), scored vs the reused 20B base preds
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_20b_32k_3k/evo2/checkpoints/iter_0003000 \
  --output-dir /data/viral/pred_20b_32k_lora_3k_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8
python3 /data/viral/aggregate_ppl.py /data/viral/pred_20b_base_vfp8 /data/viral/pred_20b_32k_lora_3k_vfp8 /data/viral/manifest.tsv.gz

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --no-python predict_evo2 \
  --fasta /data/viral/valid_cap8192.fasta \
  --ckpt-dir /data/viral/lora_run_20b_128k_1k/evo2/checkpoints/iter_0001000 \
  --output-dir /data/viral/pred_20b_128k_lora_1k_vfp8 --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean \
  --mixed-precision-recipe bf16_mixed --vortex-style-fp8
python3 /data/viral/aggregate_ppl.py /data/viral/pred_20b_base_vfp8 /data/viral/pred_20b_128k_lora_1k_vfp8 /data/viral/manifest.tsv.gz

# =============================================================================
# 4. OPTIONAL — per-genome stratified validation loss curve source
#    In-training validation PPL is read from wandb (project evo2-viral-lora).
#    Runs / results are written up in:  LOG_evo2_7b_8k.md, LOG_evo2_20b.md,
#    LOG_evo2_40b.md, LOG_evo2_longer_3000steps.md, LOG_evo2_context_ablation.md
# =============================================================================
