# Evo2_virus run log — longer training (3000 steps) for 20B and 40B

Follow-up to `LOG_evo2_20b.md` and `LOG_evo2_40b.md`. Those runs trained the viral LoRA for **1000
steps at seq-16384** and appeared to plateau (20B val PPL 750→1000: 2.974→2.969; 40B: 2.979→2.976).
This log tests whether **more training** helps, and answers a "more steps vs longer context (1M)"
question.

## Why more steps, not 1M context

The viral corpus is **short**, which rules out long-context training as a useful direction:

| split | n | median | p90 | p95 | >16k | >32k | >131k | ≥1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 12,944 | **3,968** | 12,926 | 28,941 | 6.7% | 4.5% | 2.3% | **0.1% (17)** |
| valid | 1,349 | 3,862 | 12,133 | 26,209 | 5.6% | 4.0% | 2.1% | 0.1% (1) |

seq-16384 already fully contains ~93% of records. Training at 1M context would be ~64× the
cost for a window that is ~99.9% padding on this data, is likely infeasible on 8×H200 for the
20B/40B (can't shard the ~80 GB weights while also context-parallel-splitting a 1M sequence), and
the README explicitly warns against it. So the experiment is **longer training at seq-16384**.

## Why a fresh run, not a resume

The 1000-step runs used `--decay-steps 1000`, so the cosine LR had **already decayed to the floor
(`min-lr` 3e-5) by step 1000**. A plain resume would run the extra steps pinned at 3e-5 (Megatron
restores the scheduler state; no override flag is exposed) → a near-flat curve. Instead we ran
**fresh 3000-step runs with a full cosine schedule** (`--decay-steps 3000`, lr 3e-4→3e-5), same peak
LR as before, just a 3× longer decay. Originals were preserved (new result dirs + wandb names).

## Configs

Both reuse the verified per-model configs (precision, parallelism, recompute, LoRA dims/targets,
`viral_dataset.yaml`, `nucleotide_fast_tokenizer_512`), changing only `--max-steps 3000
--decay-steps 3000`, the result dir, and the wandb run name. Scripts:
`/data/viral/train_{20b,40b}_lora_3k.sh`.

- **20B:** pure data-parallel, `--activation-checkpoint-recompute-num-layers 2`, `--vortex-style-fp8`.
  ~9.0 s/step, ~393 TFLOP/s/GPU. **~7 h 43 min** for 3000 steps. wandb
  `viral-lora-20b-seq16384-vfp8-3k` (`5yl8arbe`). Dir `/data/viral/lora_run_20b_16k_3k`.
- **40B:** `--tensor-model-parallel-size 4`, `--activation-checkpoint-recompute-num-layers 1`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--vortex-style-fp8`. ~23.4 s/step,
  ~316 TFLOP/s/GPU. **~20 h** for 3000 steps. wandb `viral-lora-40b-seq16384-vfp8-3k` (`z2y3z951`).
  Dir `/data/viral/lora_run_40b_16k_3k`.

Both: 3000 steps, **0 NaN / 0 skipped**, TRAIN_EXIT=0. 0.79 B tokens each. Adapter checkpoints at
`iter_0000{2000,2250,2500,2750,3000}` (+ earlier) in each run dir.

## Result — longer training helps; the "plateau" was a schedule artifact

Both models keep improving well past step 1000 once the LR isn't pinned at the floor. Validation
`lm loss` PPL on the species-holdout `validation` split (`--eval-iters 20`, so noisy):

| iter | 20B val PPL | 40B val PPL |
|---:|---:|---:|
| 250 | 3.085 | 3.086 |
| 500 | 3.006 | 3.010 |
| 750 | 2.941 | 2.946 |
| 1000 | 2.924 | 2.931 |
| 1250 | 2.895 | 2.902 |
| 1500 | 2.813 | 2.824 |
| 1750 | 2.858 | 2.870 |
| 2000 | 2.802 | 2.814 |
| 2250 | 2.827 | 2.838 |
| 2500 | 2.792 | 2.806 |
| 2750 | **2.768** | 2.784 |
| 3000 | 2.786 | 2.801 |
| **end-of-train (full val)** | **2.754** | **2.767** |
| end-of-train (test) | 2.759 | 2.774 |

- **vs the 1000-step runs:** 20B end-val **2.939 → 2.754** and 40B **2.946 → 2.767** — about
  **−0.19 PPL (~−6.4%)** for each. The apparent 1000-step plateau was caused by the LR hitting its
  floor, not true convergence; a proper 3× schedule keeps the loss dropping (train loss also fell
  ~1.09 → 0.94).
- **20B still edges out the 40B at every matched step** (end-val 2.754 vs 2.767), consistent with
  the 1000-step finding — the extra 26 layers of the 40B still buy nothing on this viral corpus.

## Base vs LoRA@3000 — capped per-record PPL (valid capped 8192 bp, vortex-FP8)

Base predictions reused from the earlier runs (`/data/viral/pred_{20b,40b}_base_vfp8`, base is
unchanged); only the `iter_0003000` adapters re-scored (`/data/viral/pred_{20b,40b}_lora_3k_vfp8`),
then `aggregate_ppl.py`.

Both exit 0, 1349/1349 records scored.

**20B — base vs LoRA@3000:** overall base 3.5784 → **2.9528 (−17.48%)**, 1305/1349 (96.7%) improved.
**40B — base vs LoRA@3000:** overall base 3.5784 → **2.9630 (−17.20%)**, 1301/1349 (96.4%) improved.

| genome class | n | base PPL | 20B LoRA@3000 | rel % | 40B LoRA@3000 | rel % |
|---|---:|---:|---:|---:|---:|---:|
| **OVERALL** | 1349 | 3.5784 | **2.9528** | **−17.48%** | **2.9630** | **−17.20%** |
| ssDNA | 289 | 3.588 | 2.5751 | −28.23% | 2.5856 | −27.93% |
| ssRNA(−) | 353 | 3.631 | 2.8610 | −21.20% | 2.8717 | −20.91% |
| dsDNA | 127 | 3.320 | 2.6998 | −18.69% | 2.7043 | −18.55% |
| ssRNA(+) | 337 | 3.563 | 3.0855 | −13.41% | 3.0964 | −13.10% |
| ssRNA(other) | 64 | 3.400 | 3.0344 | −10.75% | 3.0645 | −9.87% |
| dsRNA | 179 | 3.736 | 3.6441 | −2.46% | 3.6484 | −2.34% |

**vs @1000:** the base→LoRA improvement widened from −13.71% → **−17.48%** (20B) and −13.51% →
**−17.20%** (40B). Same class ordering as before (ssDNA biggest, dsRNA smallest); the improved-record
fraction dipped slightly (97.3% → ~96.5%) as a handful of dsRNA outliers moved the wrong way, but
overall PPL dropped substantially.

---

## Verdict

1. **Longer training helps, clearly.** The 1000-step "plateau" was a learning-rate-schedule artifact
   (LR had decayed to floor). With a proper 3000-step cosine schedule, both models improved ~6% on
   in-training validation PPL (20B 2.939→**2.754**, 40B 2.946→**2.767**) and the capped base→LoRA
   improvement grew from ~−13.6% to **~−17.3%**. If pushing further, a longer schedule is the lever
   that works on this corpus — not more context.
2. **20B ≥ 40B, still.** At 3000 steps the 20B remains marginally better than the 40B on every
   metric (in-train val 2.754 vs 2.767; capped predict 2.953 vs 2.963), at ~1/2.6 the per-step cost
   and ~1/2.6 the wall-clock. **The 20B vortex-FP8 LoRA remains the recommended model** for this
   eukaryotic-host viral corpus.
3. Best checkpoints: `/data/viral/lora_run_20b_16k_3k/evo2/checkpoints/iter_0003000` (and the 40B
   equivalent). wandb: `viral-lora-20b-seq16384-vfp8-3k` (`5yl8arbe`),
   `viral-lora-40b-seq16384-vfp8-3k` (`z2y3z951`), project `evo2-viral-lora`.
