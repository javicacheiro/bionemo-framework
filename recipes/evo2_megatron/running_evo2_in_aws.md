# Running Evo2 inference in AWS

Quick smoke test of `infer_evo2` (`--prompt`) using the prebuilt Docker image on an
AWS GPU host (8× NVIDIA H200).

## Prerequisites

- Docker image built: `evo2:20260628`
- NVIDIA GPUs visible on the host (`nvidia-smi`)
- NGC access for downloading the checkpoint (`BIONEMO_DATA_SOURCE=ngc`)

### 1. Launch the container with GPUs

The venv is already on `PATH` at `/workspace/.venv`, so the CLI tools
(`infer_evo2`, `download_bionemo_data`, etc.) work directly — no activation needed.
A host bind-mount keeps the downloaded/converted checkpoint across runs.

```bash
cd /opt/dlami/nvme/evo2
mkdir -p evo2_data
docker run --rm -it --gpus all --shm-size=16g \
  -v ./evo2_data:/data \
  -e BIONEMO_DATA_SOURCE=ngc \
  evo2:20260628 bash
```

### 2. Get a 1B MBridge checkpoint (download NeMo2 → convert)

`infer_evo2` needs an MBridge checkpoint; none ships in the image. The cheapest
one to produce is the 1B model (this mirrors the test fixture in
`tests/bionemo/evo2/conftest.py`).

```bash
NEMO_CKPT=$(download_bionemo_data evo2/1b-8k-bf16:1.0)

evo2_convert_nemo2_to_mbridge \
  --mixed-precision-recipe bf16_mixed \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --model-size evo2_1b_base \
  --seq-length 8192 \
  --nemo2-ckpt-dir "$NEMO_CKPT" \
  --mbridge-ckpt-dir /data/evo2_1b_mbridge
```

This produces `/data/evo2_1b_mbridge/iter_0000001`.

## Running Training with mock data (Hyena)
```bash
torchrun --nproc-per-node 2 --no-python \
  train_evo2 \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_256 \
  --model-size striped_hyena_1b_nv_parallel --max-steps 12 --eval-interval 10 \
  --eval-iters 3 --mock-data \
  --micro-batch-size 16 --global-batch-size 32 --seq-length 1024 \
  --tensor-model-parallel 1 \
  --use-precision-aware-optimizer --dataset-seed 33 \
  --seed 41 --spike-no-more-embedding-init \
  --no-weight-decay-embeddings --cross-entropy-loss-fusion \
  --align-param-gather --overlap-param-gather --grad-reduce-in-fp32 \
  --decay-steps 100 --warmup-steps 10 \
  --mixed-precision-recipe bf16_with_fp8_current_scaling_mixed \
  --no-fp32-residual-connection --activation-checkpoint-recompute-num-layers 1 \
  --attention-dropout 0.001 --hidden-dropout 0.001 \
  --eod-pad-in-loss-mask --enable-preemption \
  --log-interval 5 --debug-ddp-parity-freq 10 \
  --result-dir tmpfp8 --no-renormalize-loss \
  --use-subquadratic-ops
```


### Run Autoregressive generation (infer_evo2)
Generate DNA sequences from a prompt using an MBridge checkpoint:

```bash
torchrun --nproc_per_node 1 --no-python \
  infer_evo2 \
  --ckpt-dir /data/evo2_1b_mbridge/iter_0000001 \
  --prompt "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" \
  --max-new-tokens 10 \
  --temperature 1.0 \
  --top-k 1 \
  --output-file /data/generated.jsonl
```

Greedy decoding (`--top-k 1`) for a deterministic result, and a prompt length
divisible by 8 (matters if FP8 kicks in), matching `test_infer.py`.

Success = it exits cleanly and writes valid DNA (ACGT) continuation tokens into
`/data/generated.jsonl`. Bump `--max-new-tokens` to 200 once the pipeline is confirmed.

### Batch sequence scoring (predict_evo2)

Compute per-sequence log-likelihoods for sequences in a FASTA file, reusing the
same 1B checkpoint. We made a tiny 3-sequence FASTA at `/data/test_seqs.fasta`:

```
>seq1
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>seq2
TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAA
>seq3
ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC
```

```bash
torchrun --nproc_per_node 1 --no-python \
  predict_evo2 \
  --fasta /data/test_seqs.fasta \
  --ckpt-dir /data/evo2_1b_mbridge/iter_0000001 \
  --output-dir /data/predictions \
  --micro-batch-size 1 \
  --write-interval epoch \
  --output-log-prob-seqs \
  --log-prob-collapse-option mean
```

Success = it exits cleanly and writes `/data/predictions/predictions__rank_0__dp_rank_0.pt`
(plus `seq_idx_map.json`). The `.pt` is a dict with keys `log_probs_seqs` and
`seq_idx`. Verified result (mean log-prob per sequence):

```
seq_idx:        tensor([0, 1, 2])
log_probs_seqs: tensor([-0.3077, -0.5972, -0.3130])
```

Three finite, negative mean log-probs — one per input sequence — confirm the
scoring path works. Skipped `--use-subquadratic-ops` for this small smoke test
(it only pays off on larger datasets due to the one-time kernel compile).

### Export to Vortex format (evo2_export_mbridge_to_vortex)

Convert the 1B MBridge checkpoint into ARC's single-file Vortex `.pt` format.
This is a CPU-only conversion (no `torchrun`/GPU needed):

```bash
evo2_export_mbridge_to_vortex \
  --mbridge-ckpt-dir /data/evo2_1b_mbridge/iter_0000001 \
  --output-path /data/evo2_1b_vortex.pt \
  --model-size evo2_1b_base
```

Success = it exits cleanly and writes `/data/evo2_1b_vortex.pt` plus a sibling
`/data/config.json`. Verified result:

- Logs: `Loaded 254 keys` → `Converted to 270 vortex keys` →
  `Saved vortex checkpoint`.
- `evo2_1b_vortex.pt` is ~2.2 GB and loads with `torch.load(..., weights_only=True)`
  into a 270-tensor state dict with Vortex-style keys, e.g.
  `embedding_layer.weight` (shape `(512, 1920)`, `bfloat16`), `unembed.weight`,
  `blocks.0.pre_norm.scale`.

### Strip optimizer state from a checkpoint (evo2_remove_optimizer)

The mock-data training run above (`--result-dir tmpfp8`) writes full training
checkpoints under `/workspace/bionemo/tmpfp8/evo2/checkpoints/` (these live
inside the container, not in `/data`). Strip the optimizer state to produce a
small weights-only checkpoint in `/data` so it persists:

```bash
evo2_remove_optimizer \
  --src-ckpt-dir /workspace/bionemo/tmpfp8/evo2/checkpoints \
  --dst-ckpt-dir /data/evo2_1b_weights_only
```

The tool auto-selects the latest `iter_*` (here `iter_0000012`). Success = it
exits cleanly and writes `/data/evo2_1b_weights_only/iter_0000012` plus
`latest_checkpointed_iteration.txt` / `latest_train_state.pt`. Verified result:

- Logs: `Loading 258 model-weight keys (skipping 112 optimizer/other keys)`.
- Size dropped from **16 GB → 2.3 GB** for the iter dir (~7×; larger than the
  README's "roughly triples" because this run used
  `--use-precision-aware-optimizer`, which stores extra master-weight state).

### Fine-tune from the converted NeMo2 checkpoint (train_evo2 --finetune-ckpt-dir)

Fine-tune the converted 1B checkpoint on mock data. Point `--finetune-ckpt-dir`
at the converted MBridge dir (the one containing `iter_0000001`). We used
`bf16_mixed` to match how the checkpoint was converted, a small step count for a
smoke test, and `--result-dir /data/...` so the output persists:

```bash
torchrun --nproc-per-node 2 --no-python \
  train_evo2 \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 \
  --model-size evo2_1b_base --max-steps 8 --eval-interval 10 \
  --eval-iters 2 --mock-data \
  --micro-batch-size 8 --global-batch-size 16 --seq-length 1024 \
  --tensor-model-parallel 1 \
  --use-precision-aware-optimizer --dataset-seed 33 \
  --seed 41 \
  --cross-entropy-loss-fusion \
  --align-param-gather --overlap-param-gather --grad-reduce-in-fp32 \
  --decay-steps 100 --warmup-steps 10 \
  --mixed-precision-recipe bf16_mixed \
  --no-fp32-residual-connection --activation-checkpoint-recompute-num-layers 1 \
  --attention-dropout 0.001 --hidden-dropout 0.001 \
  --eod-pad-in-loss-mask --enable-preemption \
  --log-interval 2 \
  --result-dir /data/ft_nemo2 --no-renormalize-loss \
  --finetune-ckpt-dir /data/evo2_1b_mbridge
```

Success = it exits cleanly, saves a checkpoint, and runs validation. Verified:

- Trained 8 iterations (~0.31 s/step, ~178 TFLOP/s/GPU) and wrote
  `/data/ft_nemo2/evo2/checkpoints/iter_0000008`.
- The saved `run_config.yaml` records `finetune: true` and
  `pretrained_checkpoint: /data/evo2_1b_mbridge`, confirming weights were loaded
  from the converted checkpoint rather than randomly initialised.
- Validation/test `lm loss` ≈ 11.1 — meaningless on **mock** random data; this
  example checks the fine-tuning *pipeline*, not convergence.

## Notes

- `--temperature 1.0` is required (MCore rejects 0); `--top-k 1` gives greedy decoding.
- The model size is auto-detected from the checkpoint path (`infer.py`), so keeping
  `evo2_1b_mbridge` in the path is the right naming.
- Skip `--use-subquadratic-ops` for a first smoke test — it adds a one-time CUDA kernel
  compile and only helps the prefill phase. Add it later when processing many prompts.
- If `download_bionemo_data` fails with an auth/NGC-URL error, run
  `download_bionemo_data --list-resources` to confirm access; the fallback data source
  is `BIONEMO_DATA_SOURCE=pbss`.

