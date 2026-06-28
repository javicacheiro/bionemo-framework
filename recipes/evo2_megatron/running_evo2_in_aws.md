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

### LoRA fine-tuning (train_evo2 --lora-finetune)

LoRA fine-tune the converted 1B checkpoint on mock data. Two adjustments were
needed versus the README snippet — see the gotchas below:

```bash
torchrun --nproc-per-node 2 --no-python \
  train_evo2 \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 \
  --model-size evo2_1b_base --max-steps 8 --eval-interval 10 \
  --eval-iters 2 --mock-data \
  --micro-batch-size 4 --global-batch-size 8 --seq-length 1024 \
  --mixed-precision-recipe bf16_mixed \
  --decay-steps 100 --warmup-steps 10 \
  --log-interval 2 --disable-tensorboard-logger \
  --result-dir /data/lora_run \
  --finetune-ckpt-dir /data/evo2_1b_mbridge \
  --lora-finetune --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
  --lora-target-modules "dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"
```

Success = it exits 0, grafts LoRA adapters, trains, and saves an adapter-only
checkpoint. Verified result:

- Logs `Adding lora to: decoder.layers.N....` for the targeted modules and
  `[Evo2LoRA+Recompute] Patched HyenaStack.forward ...`.
- Trained 8 iterations and saved `/data/lora_run/evo2/checkpoints/iter_0000008`,
  which is only **149 MB** (adapters only — the base weights are not duplicated).
- `run_config.yaml` contains a `peft:` section
  (`_target_: bionemo.evo2.models.evo2_lora.Evo2LoRA`) plus
  `pretrained_checkpoint: /data/evo2_1b_mbridge` — this is what `infer_evo2` /
  `predict_evo2` use to reload the base model (see example 10).

**Two gotchas (both needed to make the README snippet run as a short smoke test):**

1. **`--decay-steps` / `--warmup-steps` are required for short runs.** Without
   them the job dies early with
   `ValueError: lr_decay_steps must be > 0, got -39936`. The README LoRA snippet
   omits them but uses `--max-steps 500`; for a small step count you must pass
   them explicitly (we used `--decay-steps 100 --warmup-steps 10`).
2. **`--disable-tensorboard-logger` is required for LoRA.** With tensorboard
   logging enabled, at the first tensorboard log interval the run crashes with
   `AttributeError: 'Parameter' object has no attribute 'main_grad'` inside
   megatron-bridge's `report_l2_norm_grad`. The Evo2 recipe hard-codes
   `log_l2_norm_grad_to_tensorboard=True` (`recipes/evo2.py`), and that code path
   touches `.main_grad` on the **frozen** LoRA base parameters, which never get a
   grad buffer. There is no dedicated CLI flag to disable only the l2-norm
   logging, so disabling the tensorboard logger entirely is the workaround.
   (Standard non-LoRA training is unaffected because all params are trainable.)

### Inference on a LoRA checkpoint (infer_evo2 / predict_evo2)

A LoRA checkpoint stores only adapter tensors; `infer_evo2` / `predict_evo2`
detect the `peft` section in `run_config.yaml`, reload the dense base model from
the recorded `pretrained_checkpoint`, graft the adapters, then load the adapter
tensors. Point `--ckpt-dir` at the LoRA `iter_*` directory.

```bash
# Autoregressive generation on the LoRA checkpoint
torchrun --nproc_per_node 1 --no-python \
  infer_evo2 \
  --ckpt-dir /data/lora_run/evo2/checkpoints/iter_0000008 \
  --prompt "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" \
  --max-new-tokens 10 --temperature 1.0 --top-k 1 \
  --output-file /data/lora_generated.jsonl

# Batch scoring on the LoRA checkpoint
torchrun --nproc_per_node 1 --no-python \
  predict_evo2 \
  --fasta /data/test_seqs.fasta \
  --ckpt-dir /data/lora_run/evo2/checkpoints/iter_0000008 \
  --output-dir /data/lora_predictions \
  --micro-batch-size 1 --write-interval epoch \
  --output-log-prob-seqs --log-prob-collapse-option mean
```

Success = both exit cleanly and reload the base model automatically. Verified:

- `infer_evo2` logs `PEFT checkpoint detected. Loading base weights from:
  /data/evo2_1b_mbridge/iter_0000001` → `Applying PEFT adapter structure` and
  writes a valid greedy continuation to `/data/lora_generated.jsonl`.
- `predict_evo2` logs `Loading adapter weights from: .../iter_0000008` and writes
  `/data/lora_predictions/...pt`. Its mean log-probs
  (`[-0.3063, -0.5971, -0.3118]`) are essentially identical to the base 1B model's
  (`[-0.3077, -0.5972, -0.3130]`), as expected after only 8 mock-data steps — a
  good sanity check that the base+adapter load path is correct.
- The base checkpoint at `pretrained_checkpoint` must still exist on disk; it does
  (`/data/evo2_1b_mbridge`).

### Viral LoRA continue-pretraining end-to-end (Evo2_virus dataset)

Full JSONL → FASTA → `preprocess_evo2` → `train_evo2 --lora-finetune` → `predict_evo2`
base-vs-LoRA pipeline on the real viral corpus in `Evo2_virus/`. The recipe-specific
instructions live in [`Evo2_virus/README.md`](Evo2_virus/README.md); this is the verified run.

**1. JSONL → FASTA** (the dataset's `text`→sequence, `record`→header; data is already
U→T/uppercased). Written into `/data/viral/` so the container sees it (`Evo2_virus/data/` is in
the host repo, not the `/data` bind mount):

```bash
for split in train valid; do
  zcat Evo2_virus/data/${split}.jsonl.gz | python3 -c '
import json, sys
with open(sys.argv[1], "w") as g:
    for line in sys.stdin:
        r = json.loads(line); g.write(">" + r["record"] + "\n" + r["text"] + "\n")
' /data/viral/${split}.fasta
done
```

Verified: 12,944 train / 1,349 valid records (matches the dataset's split table).

**2. `preprocess_evo2`** (CPU-only, ~55 s for 172 MB of train FASTA with `workers: 8`). Config is
a YAML **list**, one entry per split, each forced entirely into one split to preserve the
species-holdout split (`train_split: 1.0` for train; `valid_split: 1.0` for valid), tokenizer
`nucleotide_fast_tokenizer_512`, `transcribe: null` + `embed_reverse_complement: false`
(the data is already processed). Produced
`viral_train_..._train.{bin,idx}` (172 MB) and `viral_valid_..._val.{bin,idx}` (17 MB) in
`/data/viral/preprocessed/`.

> **Gotcha — the two entries must use different `output_prefix`.** `preprocess_evo2` writes a
> `{train,val,test}` triple per entry and **skips a run if any output with that prefix already
> exists** (`overwrite: false` default). Sharing one prefix → the second (valid) entry is silently
> skipped and `_val.bin` is left at 0 bytes. Used `viral_train` / `viral_valid`.

**3. Blended dataset YAML** (`/data/viral/viral_dataset.yaml`):

```yaml
- {dataset_prefix: /data/viral/preprocessed/viral_train_nucleotide_fast_tokenizer_512_train, dataset_weight: 1.0, dataset_split: train}
- {dataset_prefix: /data/viral/preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val,   dataset_weight: 1.0, dataset_split: validation}
- {dataset_prefix: /data/viral/preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val,   dataset_weight: 1.0, dataset_split: test}
```

> **Gotcha — a `test` split entry is required.** The Evo2 dataset provider always calls
> `get_blend_from_list(paths["test"])`; with no test entry, `train_evo2` aborts during data setup
> with `ValueError: not enough values to unpack (expected 2, got 0)`. This corpus has no test set,
> so point `test` at the valid prefix (validation and test losses then come out identical).

**4. LoRA training smoke** (real viral data, 8 steps, `--seq-length 1024` for speed; same two
LoRA gotchas as the mock-data example above — `--disable-tensorboard-logger`,
`--warmup-steps`/`--decay-steps`):

```bash
torchrun --nproc-per-node 2 --no-python train_evo2 \
  --dataset-config /data/viral/viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_1b_mbridge --model-size evo2_1b_base \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 \
  --mixed-precision-recipe bf16_mixed --seq-length 1024 \
  --micro-batch-size 4 --global-batch-size 8 \
  --max-steps 8 --warmup-steps 10 --decay-steps 100 --eval-interval 10 --eval-iters 2 \
  --lr 3e-4 --min-lr 3e-5 --log-interval 2 --disable-tensorboard-logger \
  --result-dir /data/viral/lora_run \
  --lora-finetune --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
  --lora-target-modules "dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"
```

Verified: trained 8 iters (~0.22 s/step, ~100 TFLOP/s/GPU), `lm loss` fell 1.40 → ~1.24 on **real**
viral data (vs the meaningless ~11 on mock random data), saved a **149 MB** adapter-only checkpoint
at `/data/viral/lora_run/evo2/checkpoints/iter_0000008`. Reported **validation `lm loss` 1.241,
PPL 3.458** (4-letter alphabet → max PPL 4, so this is a real, sensible number).

**5. Base vs LoRA scoring** with `predict_evo2` on a length-capped 5-record valid subset
(`--micro-batch-size 1 --write-interval epoch --output-log-prob-seqs --log-prob-collapse-option
mean`), once with `--ckpt-dir /data/evo2_1b_mbridge/iter_0000001` and once with
`--ckpt-dir /data/viral/lora_run/evo2/checkpoints/iter_0000008`. The LoRA run logs
`Loading base model weights from: /data/evo2_1b_mbridge/iter_0000001` → `Loading adapter weights
from: .../iter_0000008` (PEFT auto-detected via `run_config.yaml`). Per-sequence perplexity
`exp(-mean_logprob)`: mean **base 3.533 → LoRA 3.528**, lower on all 5 sequences — the expected
direction (and expectedly tiny after only 8 smoke steps). Pipeline confirmed end-to-end; a real
run uses `--seq-length 16384` and a real `--max-steps`.

### Data preprocessing (preprocess_evo2)

Convert FASTA → Megatron indexed binary. The config is a YAML **list** (note the
leading `-`); full schema is in `src/bionemo/evo2/data/README.md`. We made a tiny
12-sequence random FASTA at `/data/preproc_input.fasta` and this config at
`/data/preprocess_config.yaml`:

```yaml
- datapaths: ["/data/preproc_input.fasta"]
  output_dir: "/data/preproc_out"
  output_prefix: smoke
  train_split: 0.6
  valid_split: 0.2
  test_split: 0.2
  overwrite: true
  embed_reverse_complement: true
  transcribe: "back_transcribe"
  force_uppercase: true
  indexed_dataset_dtype: "uint8"
  tokenizer_type: "Byte-Level"
  fast_hf_tokenizer: true
  append_eod: true
  workers: 1
  chunksize: 25
  drop_empty_sequences: true
  nnn_filter: true
  seed: 42
```

```bash
preprocess_evo2 -c /data/preprocess_config.yaml
```

Success = it exits cleanly and writes `.bin`/`.idx` pairs for each split. Verified:

- Output files in `/data/preproc_out/`:
  `smoke_nucleotide_fast_tokenizer_256_{train,val,test}.{bin,idx}`.
- They load via `megatron.core.datasets.indexed_dataset.IndexedDataset`: 14 train
  / 4 val / 6 test samples (24 total = 12 sequences × 2 from
  `embed_reverse_complement`), byte-level tokens (65=A, 67=C, 71=G, 84=T).
- This is CPU-only — no `torchrun`/GPU needed.

> Note: the output filename embeds `nucleotide_fast_tokenizer_256` even though the
> config requests `Byte-Level`; with `fast_hf_tokenizer: true` the byte-level DNA
> tokenizer is realised as the bundled 256-vocab fast tokenizer.

### Transcript extraction (splice_evo2)

Extract spliced transcripts (concatenated exons) from a genome FASTA + GTF. We
built a 120 bp single-contig genome `/data/genome.fa` (`>chr1`) and a minimal GTF
`/data/genes.gtf` with one gene / one transcript / two exons (1–30 and 61–90,
`+` strand):

```
chr1	test	gene	1	90	.	+	.	gene_id "gene1";
chr1	test	transcript	1	90	.	+	.	gene_id "gene1"; transcript_id "t1"; gbkey "mRNA"; transcript_biotype "mRNA";
chr1	test	exon	1	30	.	+	.	gene_id "gene1"; transcript_id "t1"; exon_number "1";
chr1	test	exon	61	90	.	+	.	gene_id "gene1"; transcript_id "t1"; exon_number "2";
```

```bash
splice_evo2 \
  --fasta-path /data/genome.fa \
  --gtf-path /data/genes.gtf \
  --output-path /data/transcripts.fa \
  --only-longest-transcript
```

Success = it exits cleanly and writes the spliced transcript. Verified: the
output `/data/transcripts.fa` is

```
>chr1|gene1|t1
TTTCCTCATGCAATTCAAAACCATGTCCGTGAGGATACCAAATTCCTCCTTATTCAGGAC
```

i.e. exactly `genome[0:30] + genome[60:90]` (60 bp = exon1 + exon2 concatenated,
introns removed) — matching the value we computed independently. CPU-only.

**GTF format gotchas (the parser is strict / NCBI-RefSeq-shaped):**

1. **`transcript` features must carry `gbkey` and `transcript_biotype`** in
   addition to `gene_id`/`transcript_id` — otherwise `KeyError: 'gbkey'`.
2. **Every attribute value must be double-quoted**, including numeric ones like
   `exon_number "1"`. An unquoted value (`exon_number 1;`) raises
   `IndexError: list index out of range` in `parse_gtf_attributes`.

### Convert Savanna → MBridge (evo2_convert_savanna_to_mbridge)

Convert ARC's HuggingFace Savanna checkpoint to MBridge:

```bash
evo2_convert_savanna_to_mbridge \
  --savanna-ckpt-path arcinstitute/savanna_evo2_1b_base \
  --mbridge-ckpt-dir /data/mbridge_1b_savanna \
  --model-size evo2_1b_base \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_256
```

The HF download succeeds (unauthenticated is fine; ~3.5 GB into
`~/.cache/huggingface`). Verified end-to-end: `Converted 254 keys` →
`MBridge checkpoint saved to /data/mbridge_1b_savanna` (contains `iter_0000001`).
(There is a benign `Unmapped savanna keys (54): ...` warning for
`*.outer_mlp_layernorm` / `*.post_attention_layernorm` / `*.rotary_emb.inv_freq`
keys that the TE mapping does not consume.)

> **History — fixed a PyTorch 2.6 incompatibility.** As originally shipped this
> command failed on this image with
> `_pickle.UnpicklingError: ... Unsupported global: GLOBAL numpy.core.multiarray._reconstruct`.
> `load_savanna_state_dict` loaded the `.pt` with `torch.load(weights_only=True)`,
> but Savanna checkpoints pickle numpy training metadata, which PyTorch ≥2.6
> refuses under the new `weights_only=True` default. We fixed
> `load_savanna_state_dict` to fall back to `weights_only=False` (ARC's published
> checkpoint is a trusted source) when the strict load raises `UnpicklingError`,
> with a regression test in
> `tests/bionemo/evo2/utils/checkpoint/test_savanna_to_mbridge.py`. The command
> above now works out of the box; the log shows a one-line warning when the
> fallback triggers.

### Savanna → MBridge → Vortex round-trip

Chains the two converters. Step 1 is the Savanna→MBridge conversion above; step 2
is the Vortex export:

```bash
# Step 2: MBridge -> Vortex (using the checkpoint produced above)
evo2_export_mbridge_to_vortex \
  --mbridge-ckpt-dir /data/mbridge_1b_savanna/iter_0000001 \
  --output-path /data/evo2_1b_savanna_vortex.pt \
  --model-size evo2_1b_base
```

Verified: `Loaded 254 keys` → `Converted to 270 vortex keys` →
`/data/evo2_1b_savanna_vortex.pt` (~2.2 GB) + `config.json`. The full
Savanna→MBridge→Vortex chain is functional.

## Example notebooks (Jupyter)

### How to run the notebooks headless (important kernel setup)

The `examples/*.ipynb` notebooks need two adjustments to run headless in this
container:

1. **Use the venv kernel, not the default `python3` kernel.** `jupyter` on
   `PATH` is the *system* install (`/usr/local/bin`), whose `python3` kernel does
   **not** have `bionemo`, `seaborn`, etc. The Evo2 stack lives in the venv
   (`/workspace/.venv`), which already ships `ipykernel` + `nbconvert`. Register a
   venv kernel once and target it:

   ```bash
   /workspace/.venv/bin/python -m ipykernel install --user --name evo2venv
   ```

   Symptom if you skip this: `ModuleNotFoundError: No module named 'seaborn'`
   (or `bionemo`) even though `python -c "import seaborn"` works in the shell.

2. **Strip the shipped cell outputs first.** The notebooks ship with at least one
   stream output missing the required `name` field, which makes `nbconvert`
   abort with `NotebookValidationError: 'name' is a required property` before it
   even runs. Clear outputs via raw JSON, then execute:

   ```python
   import json
   nb = json.load(open("nb.ipynb"))
   for c in nb["cells"]:
       if c["cell_type"] == "code":
           c["outputs"] = []; c["execution_count"] = None
   json.dump(nb, open("nb.ipynb", "w"))
   ```

We ran each notebook from a copy under `/data` (so downloads/outputs persist and
the repo tree stays clean):

```bash
cd /data && /workspace/.venv/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=evo2venv \
  --ExecutePreprocessor.timeout=5400 \
  <notebook>.ipynb
```

### zeroshot_brca1.ipynb — zero-shot BRCA1 variant effect prediction (1B)

Self-contained: it `wget`s the BRCA1 supplementary table + chr17 genome from
ARC's GitHub, downloads & converts the 1B checkpoint to
`evo2_1b_base_mbridge/` (≈2.1 GB) if absent, builds reference/variant FASTAs,
runs `predict_evo2 --use-subquadratic-ops`, and computes an AUROC.

Verified: completed end-to-end (`NBEXIT=0`) on 1× H200 in ~3–4 min after the
kernel fixes above. Artifacts under `/data/brca1*/`, predictions in
`reference_predictions/` and `variant_predictions/`. Headline result:

```
Zero-shot prediction AUROC: 0.74
```

This is the expected ballpark for the **1B** model (the 7B/40B checkpoints reach
~0.87–0.88 in the README's AUC table).

### fine-tuning-tutorial.ipynb — fine-tune the 1B on human chromosomes

Full data→train pipeline: `wget`s hg38 chr20/chr21/chr22, concatenates them,
runs `preprocess_evo2` (→ `/data/preprocessed_data`), downloads & converts the
`evo2/1b-8k:1.0` checkpoint to `evo2_1b_fp8_mbridge/` (≈2.1 GB), then fine-tunes
with `train_evo2 --finetune-ckpt-dir`. Set `FAST_CI_MODE=1` to use a 4-layer
subset and `MAX_STEPS=10` so it finishes quickly:

```bash
cd /data && FAST_CI_MODE=1 /workspace/.venv/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=evo2venv \
  --ExecutePreprocessor.timeout=7200 \
  fine-tuning-tutorial.ipynb
```

Verified: completed end-to-end (`NBEXIT=0`) on 8× H200. Evidence:

- Preprocessing wrote train/val/test `.bin`/`.idx` to `/data/preprocessed_data`
  (the val `.bin` alone is ~123 MB — these are real human chromosomes).
- The notebook auto-derives `num_gpus=8` and trains with
  `--context-parallel-size 8` at `--seq-length 8192`.
- Saved checkpoints at iteration 5 and 10
  (`/data/pretraining_demo/evo2/checkpoints/iter_0000010`); `progress.txt`
  records `# GPUs: 8 ... Iteration: 10 ... Saved checkpoint`.

Same headless-notebook setup as above (venv kernel + output strip). With
`FAST_CI_MODE=1` the whole notebook runs in roughly the download +
preprocess time (preprocessing the three chromosomes dominates wall-clock).

### lora-fine-tuning-tutorial.ipynb — LoRA splice-site classification

Trains a splice-site classifier two ways via the bundled `evo2_classifier.py`:
a head-only baseline and a LoRA+head run, then compares test accuracy and
trainable-parameter counts. It pulls the `InstaDeepAI/nucleotide_transformer_
downstream_tasks_revised` dataset from HF (`datasets` lib), writes
`splice_{train,val,test}.jsonl`, and converts `evo2/1b-8k-bf16:1.0` to
`evo2_1b_bf16_mbridge/`.

This notebook needs **`evo2_classifier.py` alongside it** (its torchrun command
runs `evo2_classifier.py`), so copy both into the working dir:

```bash
cp /workspace/bionemo/examples/lora-fine-tuning-tutorial.ipynb /data/
cp /workspace/bionemo/examples/evo2_classifier.py /data/
cd /data && CUDA_VISIBLE_DEVICES=0 FAST_CI_MODE=1 /workspace/.venv/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=evo2venv \
  --ExecutePreprocessor.timeout=7200 \
  lora-fine-tuning-tutorial.ipynb
```

Verified: completed end-to-end (`NBEXIT=0`). `FAST_CI_MODE=1` runs 40 iters each
on a 600/180/180 split. Results (these are **smoke-test** numbers — 40 iters on a
3-class task is near chance, not a real benchmark):

| Run               | Trainable params      | % of 1.1B | Test accuracy |
| ----------------- | --------------------- | --------- | ------------- |
| Head-only baseline| 3,697,923             | 0.33%     | 0.4833        |
| LoRA + head       | 15,985,923 (12.3M LoRA)| 1.42%    | 0.3389        |

The point demonstrated is the **trainable-parameter breakdown** (head-only vs
head+LoRA on a frozen 1.1B backbone), not converged accuracy.

> **Gotcha — run the FAST_CI smoke test on a single GPU.** The notebook
> auto-detects `NUM_GPUS = torch.cuda.device_count()` (8 here). On 8 GPUs the
> LoRA stage dies with `ZeroDivisionError: integer division or modulo by zero`
> in the data sampler — the tiny `FAST_CI_MODE` split (600 train) can't be
> sharded across 8 ranks. Forcing `CUDA_VISIBLE_DEVICES=0` (single GPU) runs both
> stages cleanly. The full-scale (non-FAST_CI) config with its larger dataset is
> the one intended for multi-GPU.

## Notes

- `--temperature 1.0` is required (MCore rejects 0); `--top-k 1` gives greedy decoding.
- The model size is auto-detected from the checkpoint path (`infer.py`), so keeping
  `evo2_1b_mbridge` in the path is the right naming.
- Skip `--use-subquadratic-ops` for a first smoke test — it adds a one-time CUDA kernel
  compile and only helps the prefill phase. Add it later when processing many prompts.
- If `download_bionemo_data` fails with an auth/NGC-URL error, run
  `download_bionemo_data --list-resources` to confirm access; the fallback data source
  is `BIONEMO_DATA_SOURCE=pbss`.

