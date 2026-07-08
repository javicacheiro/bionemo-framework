#!/bin/bash
# Uncapped length-stratified scoring, two passes (CP=1 short / CP=8 long) to fit
# memory on the long records. base vs 16k@3000 LoRA vs 128k@3000 LoRA.
set -uo pipefail
cd /workspace/bionemo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SHORT=/data/viral/valid_le32k.fasta      # <=32768 bp
LONG=/data/viral/valid_gt32k.fasta       # >32768 bp, N-padded to mult of 1024
BASE=/data/evo2_20b_mbridge/iter_0000001
L16=/data/viral/lora_run_20b_16k_3k/evo2/checkpoints/iter_0003000
L128=/data/viral/lora_run_20b_128k_3k/evo2/checkpoints/iter_0003000

pshort () {  # gpu ckpt outdir
  CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node 1 --no-python predict_evo2 \
    --fasta "$SHORT" --ckpt-dir "$2" --output-dir "$3" \
    --micro-batch-size 1 --write-interval epoch \
    --output-log-prob-seqs --log-prob-collapse-option mean \
    --mixed-precision-recipe bf16_mixed --vortex-style-fp8
}
plong () {   # ckpt outdir   (CP=8, all GPUs)
  torchrun --nproc_per_node 8 --no-python predict_evo2 --context-parallel-size 8 \
    --fasta "$LONG" --ckpt-dir "$1" --output-dir "$2" \
    --micro-batch-size 1 --write-interval epoch \
    --output-log-prob-seqs --log-prob-collapse-option mean \
    --mixed-precision-recipe bf16_mixed --vortex-style-fp8
}

echo "[$(date)] PASS 1 (short <=32k, CP=1, 3 models parallel)..."
pshort 0 "$BASE" /data/viral/pred_base_le32k  > /data/viral/sc_base_short.log 2>&1 &
pshort 1 "$L16"  /data/viral/pred_16k_le32k   > /data/viral/sc_16k_short.log  2>&1 &
pshort 2 "$L128" /data/viral/pred_128k_le32k  > /data/viral/sc_128k_short.log 2>&1 &
wait
echo "[$(date)] PASS 2 (long >32k, CP=8, sequential)..."
plong "$BASE" /data/viral/pred_base_gt32k  > /data/viral/sc_base_long.log 2>&1
plong "$L16"  /data/viral/pred_16k_gt32k   > /data/viral/sc_16k_long.log  2>&1
plong "$L128" /data/viral/pred_128k_gt32k  > /data/viral/sc_128k_long.log 2>&1

echo "[$(date)] checking outputs..."
ok=1
for o in pred_base_le32k pred_16k_le32k pred_128k_le32k pred_base_gt32k pred_16k_gt32k pred_128k_gt32k; do
  ls /data/viral/$o/predictions__rank_*.pt >/dev/null 2>&1 && echo "  OK $o" || { echo "  FAILED $o"; ok=0; }
done
[ $ok -eq 1 ] || { echo "[$(date)] some passes failed, aborting aggregate"; exit 1; }

echo "[$(date)] aggregating (short+long merged per model) ->"
python3 /data/viral/aggregate_ppl_bylen.py \
  "/data/viral/pred_base_le32k,/data/viral/pred_base_gt32k" \
  "/data/viral/pred_16k_le32k,/data/viral/pred_16k_gt32k" \
  "/data/viral/pred_128k_le32k,/data/viral/pred_128k_gt32k" \
  /data/viral/valid_record_lengths.tsv /data/viral/manifest.tsv.gz \
  | tee /data/viral/RESULTS_uncapped_bylen.txt
echo "[$(date)] DONE -> /data/viral/RESULTS_uncapped_bylen.txt"
