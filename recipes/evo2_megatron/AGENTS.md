# AGENTS.md — Evo2 recipe evaluation on AWS

Working notes for any agent helping run and document the examples in
[`README.md`](README.md) on this AWS GPU host. For repo-wide development rules
(copied files, model/recipe boundaries, TE test harness, linting) see the
top-level [`bionemo-recipes/AGENTS.md`](../../AGENTS.md).

## What we are doing

Running each runnable example from `README.md`, end to end, on real hardware and
recording the outcome. Two living documents track this:

- [`running_evo2_in_aws.md`](running_evo2_in_aws.md) — the human-facing runbook:
  prerequisites, the exact commands we ran, and notes/gotchas per example.
- [`PROGRESS.md`](PROGRESS.md) — the checklist of every example and its status,
  so work can be resumed across sessions.

When you finish (or attempt) an example: update **both** — flip its row in
`PROGRESS.md` and, if anything was non-obvious, add the working command and a
note to `running_evo2_in_aws.md`.

## Environment

- **Host:** AWS GPU instance, 8× NVIDIA H200. Confirm with `nvidia-smi`.
- **Container:** `evo2:20260628` is already running (see `docker ps`; recent
  name `sleepy_germain`). It was started with:

  ```bash
  cd /opt/dlami/nvme/evo2
  docker run --rm -it --gpus all --shm-size=16g \
    -v ./evo2_data:/data \
    -e BIONEMO_DATA_SOURCE=ngc \
    evo2:20260628 bash
  ```

- **Exec into the running container** instead of launching a new one (the
  bind-mount and downloaded checkpoint persist):

  ```bash
  docker exec -it $(docker ps -q -f ancestor=evo2:20260628) bash
  ```

  Note: the container was started `--rm`, so do not `docker stop` it unless you
  intend to lose the session — the host bind-mount `/opt/dlami/nvme/evo2/evo2_data`
  (→ `/data` inside) survives regardless.

## Key facts about the container

- The venv is already on `PATH` at `/workspace/.venv` — CLI tools (`train_evo2`,
  `infer_evo2`, `predict_evo2`, `download_bionemo_data`, the converters, …) work
  directly. No `source .ci_test_env.sh` needed.
- `BIONEMO_DATA_SOURCE=ngc` is set for `download_bionemo_data`. Fallback is
  `BIONEMO_DATA_SOURCE=pbss`. If a download fails, run
  `download_bionemo_data --list-resources` to confirm access.
- Tokenizers ship in the image under `tokenizers/` (e.g.
  `nucleotide_fast_tokenizer_256`, `nucleotide_fast_tokenizer_512`).
- Persist anything you want to keep across container restarts under `/data`.

## Assets already produced (don't redo)

- `/data/evo2_1b_mbridge/iter_0000001` — 1B MBridge checkpoint, converted from
  `evo2/1b-8k-bf16:1.0` (NeMo2 → MBridge, `bf16_mixed`). Reuse this for any
  example needing a 1B `--ckpt-dir`.
- `/data/generated.jsonl` — output of the `infer_evo2` smoke test.

## Conventions for running examples

- Prefer the **smallest** model that exercises the code path: `striped_hyena_test`
  / `striped_hyena_1b_nv_parallel` for training, the 1B checkpoint for
  inference/prediction. Avoid 7B/40B unless explicitly asked — they are large
  downloads and may need multi-GPU.
- For a deterministic smoke test of generation use greedy decoding
  (`--temperature 1.0 --top-k 1`) and a prompt length divisible by 8 (matters if
  FP8 kicks in), matching `tests/.../test_infer.py`.
- Skip `--use-subquadratic-ops` on first runs (one-time CUDA kernel compile); add
  it once a pipeline is confirmed working.
- Keep run outputs (`--result-dir`, `--output-dir`, `--mbridge-ckpt-dir`, …)
  under `/data` so they persist and are inspectable from the host.
- "Success" = process exits 0 **and** the expected artifact is present and
  sensible (valid ACGT continuation, a checkpoint dir, predictions file, etc.).
  Record both the command and the evidence.

## Pointers

- Examples list / status: [`PROGRESS.md`](PROGRESS.md)
- Runbook with results: [`running_evo2_in_aws.md`](running_evo2_in_aws.md)
- Source of all CLI tools: `pyproject.toml` `[project.scripts]`
- Notebooks: `examples/` (`zeroshot_brca1.ipynb`,
  `fine-tuning-tutorial.ipynb`, `lora-fine-tuning-tutorial.ipynb`)
