# Evo2 Viral LoRA Training Data

This repository contains the minimal data handoff for Evo2 LoRA continue-pretraining on eukaryotic-host / broad-host viral genomes.

Use only the files in this repository for the first training run.

## Files

| Path | Use |
|---|---|
| `data/train.jsonl.gz` | Training JSONL. One viral genome or viral segment per line. |
| `data/valid.jsonl.gz` | Validation JSONL for checkpoint / hyperparameter selection. |
| `metadata/eukaryotic_host_core_manifest.tsv.gz` | Per-record metadata for audit and stratified evaluation. Not used directly by the training dataloader. |
| `savanna/preprocess_commands.sh` | Converts the JSONL files into Savanna/Evo2 indexed mmap datasets. See [Appendix: original Savanna framework](#appendix-original-savanna-framework). |
| `savanna/data_config_viral_lora.json` | Dataset path snippet after Savanna preprocessing. See [Appendix](#appendix-original-savanna-framework). |

## Dataset format

The JSONL files are gzip-compressed. Each line has this format:

```json
{"record": "DQ665917.1|ictv:VMR1000033", "text": "CCCCAAGCGCCCCCCCGGCGCCATCTCCG..."}
```

Only `text` is model input. `record` is a traceable sequence ID.

The nucleotide text has already been processed:

- RNA `U` was converted to `T`;
- sequences are uppercase;
- database/reference orientation is preserved;
- negative-sense RNA was not forcibly converted to coding sense;
- reverse-complement augmentation was not added;
- exact and near-duplicate reduction were applied;
- bacteria-host phage-like and archaea-host viruses were excluded.

For segmented viruses, each segment is one document. Segments from the same `genome_group_id` are kept in the same split.

## Current split

| Split | Records | Bases | Species | Human-priority records |
|---|---:|---:|---:|---:|
| train | 12,944 | 171,790,984 | 7,621 | 715 |
| valid | 1,349 | 17,266,753 | 848 | 36 |

There is no separate test set. The validation set is a species-holdout split relative to train.

After choosing the training recipe, the validation set can be folded back into train for a final adapter run. If this is done, do not report that same set as an unbiased validation/test set.

## Preprocess and train with the BioNeMo evo2_megatron recipe

This is the path for the host this dataset was handed to, which runs the BioNeMo
**`evo2_megatron`** recipe (`train_evo2`, `preprocess_evo2`, `predict_evo2`, MBridge
checkpoints). The original Arc Institute **Savanna** instructions are preserved in the
[Appendix](#appendix-original-savanna-framework) but will not run as-is here.

Two differences from the Savanna path drive the steps below:

- `preprocess_evo2` ingests **FASTA**, not JSONL — so we convert first (Step 1).
- This recipe uses the HuggingFace `nucleotide_fast_tokenizer_512` (the same tokenizer as
  the recipe's `fine-tuning-tutorial.ipynb` and `lora-fine-tuning-tutorial.ipynb`), not the
  Savanna `CharLevelTokenizer`.

Run the commands below from the root of this repository.

### Step 1 — Convert JSONL to FASTA

The text is already U→T-converted, uppercased, and orientation-preserved (see
[Dataset format](#dataset-format)), so the conversion is verbatim — `text` becomes the FASTA
sequence and `record` becomes the header:

```bash
for split in train valid; do
  python3 -c '
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f, open(sys.argv[2], "w") as g:
    for line in f:
        r = json.loads(line)
        g.write(">" + r["record"] + "\n" + r["text"] + "\n")
' data/${split}.jsonl.gz data/${split}.fasta
done
```

(The recipe's `bionemo_fasta_to_jsonl` only goes the other direction and emits different field
names, so the small snippet above is the conversion to use.)

### Step 2 — Preprocess to Megatron indexed datasets (`preprocess_evo2`)

Write a `viral_preprocess.yaml` with one entry per split. To preserve the pre-made
species-holdout split, each FASTA is forced entirely into one split (train file →
`train_split: 1.0`; valid file → `valid_split: 1.0`). The processing flags are set so
preprocessing adds nothing the data does not already have (no reverse-complement augmentation,
no transcription, no taxonomy tokens):

```yaml
- datapaths: ["data/train.fasta"]
  output_dir: preprocessed
  output_prefix: viral_train
  hf_tokenizer_model_path: tokenizers/nucleotide_fast_tokenizer_512
  train_split: 1.0
  valid_split: 0.0
  test_split: 0.0
  append_eod: true
  transcribe: null
  random_reverse_complement: 0.0
  embed_reverse_complement: false
  force_uppercase: false
  workers: 8
- datapaths: ["data/valid.fasta"]
  output_dir: preprocessed
  output_prefix: viral_valid
  hf_tokenizer_model_path: tokenizers/nucleotide_fast_tokenizer_512
  train_split: 0.0
  valid_split: 1.0
  test_split: 0.0
  append_eod: true
  transcribe: null
  random_reverse_complement: 0.0
  embed_reverse_complement: false
  force_uppercase: false
  workers: 8
```

> **The two entries must use different `output_prefix` values** (`viral_train` vs `viral_valid`).
> `preprocess_evo2` always emits a `{train,val,test}` triple per entry and **skips a run entirely
> if any output file with that prefix already exists** (`overwrite` defaults to `false`). If both
> entries share a prefix, the second (valid) run is silently skipped and its `_val.bin` is left
> empty (0 bytes). Verified on this host.

Then run:

```bash
preprocess_evo2 --config viral_preprocess.yaml
```

`preprocess_evo2` auto-names the outputs `{output_prefix}_{tokenizer_name}_{split}.{bin,idx}`.
With the config above, the two files that actually carry data are:

```text
preprocessed/viral_train_nucleotide_fast_tokenizer_512_train.{bin,idx}   # ~172 MB .bin (train)
preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val.{bin,idx}     # ~17 MB .bin (valid)
```

(Each run also writes empty `.bin` companions for the splits it did not fill — e.g.
`viral_train_..._val.bin` at 0 bytes — which is normal; just use the two non-empty stems above.)
Use the path **without** the `.bin`/`.idx` extension as the dataset prefix in Step 3.

### Step 3 — Dataset config

`evo2_megatron` does not use the Savanna `data_config_viral_lora.json`. Instead, point
`train_evo2` at a blended dataset YAML (`viral_dataset.yaml`) that references the prefixes from
Step 2:

```yaml
- dataset_prefix: preprocessed/viral_train_nucleotide_fast_tokenizer_512_train
  dataset_weight: 1.0
  dataset_split: train
- dataset_prefix: preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val
  dataset_weight: 1.0
  dataset_split: validation
- dataset_prefix: preprocessed/viral_valid_nucleotide_fast_tokenizer_512_val
  dataset_weight: 1.0
  dataset_split: test
```

Notes:

- The split label is `validation` (not `valid`), and the prefix is the indexed-dataset stem, not
  the original `.jsonl.gz`.
- **A `test` split entry is required.** The Evo2 dataset provider always builds a blend for all
  three of train/validation/test; with no `test` entry, `train_evo2` aborts during data setup with
  `ValueError: not enough values to unpack (expected 2, got 0)`. Since this corpus has no separate
  test set, point `test` at the same valid prefix (as above). It does not affect training and the
  reported validation and test losses will simply be identical. Verified on this host.

### Context length

The context length is a training configuration choice; do not pre-split the JSONL into fixed
windows. In this recipe it is the `train_evo2 --seq-length` flag (default `8192`).

For the first LoRA model-selection run, use:

```text
context length = 16,384 tokens   (--seq-length 16384)
```

If memory or throughput is limiting, `8,192` tokens is acceptable. If resources are comfortable,
`32,768` tokens is a useful ablation. Do not use 1M context as the default first run; it is much
more expensive and most records in this viral corpus are far shorter than 1M.

Note on the base checkpoint: the checkpoint converted to MBridge on this host is **Evo2 1B / 8k**
(`/data/evo2_1b_mbridge`). It still runs at 16k/32k context, but a true long-context base would
need a separate NeMo2→MBridge conversion (`evo2_convert_nemo2_to_mbridge`).

Please report the exact context length used together with the training loss curve and validation
metrics.

### LoRA training

Start from the Evo2 long-context checkpoint as the base model and use this corpus for LoRA
continue-pretraining. The command below mirrors the recipe's LoRA example; batch sizes, step
counts, and eval intervals are placeholders to tune for your hardware:

```bash
torchrun --nproc-per-node 8 --no-python train_evo2 \
  --dataset-config viral_dataset.yaml \
  --finetune-ckpt-dir /data/evo2_1b_mbridge \
  --model-size evo2_1b_base \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_512 \
  --mixed-precision-recipe bf16_mixed \
  --seq-length 16384 \
  --micro-batch-size <N> --global-batch-size <N> \
  --max-steps <N> --warmup-steps <N> --decay-steps <N> \
  --eval-interval <N> --eval-iters <N> \
  --lr 3e-4 --min-lr 3e-5 \
  --result-dir /data/viral_lora_run \
  --wandb-project evo2-viral-lora --wandb-run-name viral-lora-1b-seq16384 \
  --lora-finetune --lora-dim 16 --lora-alpha 32 --lora-dropout 0.1 \
  --lora-target-modules "dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2"
```

Notes (all verified in `../running_evo2_in_aws.md`):

- **`--warmup-steps` / `--decay-steps` are required.** The LR scheduler needs them set explicitly;
  otherwise short runs die with `ValueError: lr_decay_steps must be > 0`. Set `--decay-steps` near
  your `--max-steps` and a small `--warmup-steps`.
- **wandb** logs the loss curve (run `wandb login` in the container once). `--wandb-project` turns
  it on; `--wandb-run-name` is optional. See "Logging training to Weights & Biases" in the runbook.
- **LoRA + any logger used to crash** with `AttributeError: 'Parameter' object has no attribute
  'main_grad'` (l2-norm-grad logging reading `.main_grad` on the frozen base params). This is now
  handled in `run/train.py`: with `--lora-finetune` it disables that one logger, so wandb (and
  tensorboard) work normally. No `--disable-tensorboard-logger` workaround is needed anymore.
- For long context, fits on 8× H200 with `--micro-batch-size 1 --global-batch-size 16
  --activation-checkpoint-recompute-num-layers 1` at ~1.27 s/step (~255 TFLOP/s/GPU).

The LoRA adapter checkpoint is written to `/data/viral_lora_run/evo2/checkpoints/iter_<N>` (adapter
tensors only — the base weights are not duplicated) with a `run_config.yaml` that records a `peft:`
section and `pretrained_checkpoint: /data/evo2_1b_mbridge`.

## Requested training report metrics

Please report only the following core metrics for the first training handoff. The mechanics below
are for the `evo2_megatron` recipe.

1. **Training loss curve.**
   - Read from the `--wandb-project` run. (Tensorboard is disabled for LoRA — see the gotcha
     above — so wandb is the loss-curve source.)
   - Include steps or tokens seen on the x-axis, and learning rate if available.
   - Perplexity, if reported, is `exp(loss)`.

2. **Viral species-holdout validation loss / perplexity.**
   - The overall value is computed during training on the `validation` split (`--eval-interval`,
     `--eval-iters`).
   - For the breakdown stratified by the manifest column `genome` (`ssRNA(+)`, `ssRNA(-)`,
     `dsRNA`, `dsDNA`, `ssDNA`), score each record directly with `predict_evo2`:

     ```bash
     torchrun --nproc_per_node 1 --no-python predict_evo2 \
       --fasta data/valid.fasta \
       --ckpt-dir <checkpoint> \
       --output-dir <out_dir> \
       --micro-batch-size 1 --write-interval epoch \
       --output-log-prob-seqs \
       --log-prob-collapse-option mean
     ```

     Per-sequence perplexity is `exp(-mean_logprob)`, read from `predictions__rank_*.pt` together
     with `seq_idx_map.json`. Join each `record` to its `genome` value in
     `metadata/eukaryotic_host_core_manifest.tsv.gz` and aggregate per group.
     `examples/zeroshot_brca1.ipynb` is the template for the predict → load → aggregate pattern.

3. **Base Evo2 versus viral LoRA comparison on the same validation set.**
   - Run the `predict_evo2` scoring above twice on the same `data/valid.fasta`:
     - **base:** `--ckpt-dir /data/evo2_1b_mbridge/iter_0000001`
     - **LoRA:** `--ckpt-dir /data/viral_lora_run/evo2/checkpoints/iter_<N>` — the PEFT adapter is
       auto-detected via its `run_config.yaml`, which reloads the base model from the recorded
       `pretrained_checkpoint` and grafts the adapters.
   - Report the absolute and relative change in validation loss / perplexity.

## Appendix: original Savanna framework

The commands in this appendix target Arc Institute's original Evo2/Savanna repository and will
**not** run as-is on the BioNeMo `evo2_megatron` recipe. Use the
[evo2_megatron path above](#preprocess-and-train-with-the-bionemo-evo2_megatron-recipe) on this
host; this section is retained for teams using the Savanna framework.

### Preprocess for Evo2/Savanna

From the root of this repository, run:

```bash
bash savanna/preprocess_commands.sh
```

The script runs:

```bash
python tools/preprocess_data.py \
  --input data/train.jsonl.gz \
  --output-prefix savanna/viral_euk_host_core_train \
  --tokenizer-type CharLevelTokenizer \
  --jsonl-keys text \
  --append-eod \
  --dataset-impl mmap
```

and the equivalent command for `data/valid.jsonl.gz`.

If `tools/preprocess_data.py` lives in a separate Evo2/Savanna repository, either:

1. copy this repository's `data/` and `savanna/` directories into that training repository, then run the script there; or
2. edit `savanna/preprocess_commands.sh` so `python tools/preprocess_data.py` points to the correct script path.

### Savanna training config

After preprocessing, use these dataset prefixes in the Evo2/Savanna training config:

```json
{
  "train-data-paths": [
    "savanna/viral_euk_host_core_train_text_CharLevelTokenizer_document"
  ],
  "valid-data-paths": [
    "savanna/viral_euk_host_core_valid_text_CharLevelTokenizer_document"
  ],
  "test-data-paths": [],
  "tokenizer-type": "CharLevelTokenizer"
}
```

The same content is provided in:

```text
savanna/data_config_viral_lora.json
```

Use the preprocessed dataset prefix, not the original `.jsonl.gz`, as the training data path.
