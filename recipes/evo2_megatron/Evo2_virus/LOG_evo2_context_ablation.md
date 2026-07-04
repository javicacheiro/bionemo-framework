# Evo2_virus run log — context-length ablation on the 20B (16k / 32k / 128k)

Follow-up to `LOG_evo2_longer_3000steps.md`. That log showed longer *training* helps (16k, 3000
steps → val PPL 2.754). This log asks the orthogonal question: does a longer *context window* help
on this viral corpus? We test the 20B (a native 1M-context base) at **32k** and **128k** vs the
**16k** baseline, all vortex-FP8, on 8× H200.

## Motivation and the data-length prior

The viral sequences are short (train median 3,968 bp; only 6.7% exceed 16,384 and 0.1% reach 1M —
see `LOG_evo2_longer_3000steps.md` for the full distribution). So 16k already fully contains ~93% of
records. The one plausible upside of longer context is the **large-genome tail** (segmented / large
dsDNA viruses up to ~2 Mb), which holds a disproportionate share of total bases (16k covers only
~43% of nucleotides; 128k ~72%). This ablation tests whether that tail — especially the **dsDNA**
class — benefits from longer within-genome attention.

## Runs

All 20B, `--mixed-precision-recipe bf16_mixed --vortex-style-fp8`, GBS 16, LoRA dim 16, same
`viral_dataset.yaml`, full cosine schedule (lr 3e-4→3e-5, `--decay-steps = --max-steps`). Memory
config was smoke-tested per context; scripts `/data/viral/train_20b_{32k_3k,128k_1k}.sh`.

| context | parallelism | recompute | steps | peak GB/GPU | step time | TFLOP/s/GPU | wall-clock |
|---|---|---|---:|---:|---:|---:|---:|
| 16k | pure DP | 2 | 3000 | <143 | ~9.0 s | ~393 | ~7.7 h |
| 32k | TP=2 | 1 | 3000 | 63.8 | ~20.6 s | ~414 | ~17 h |
| 128k | TP=4 | 1 | 1000 | 66.7 | ~110.7 s | ~583 | ~31 h |

- **128k memory:** TP≥2 is required to shard the ~120 GB fixed weights+grad footprint (activations
  are not the bottleneck). TP=4/CP=1/recompute-1 fit at **66.7 GB/GPU** — context-parallelism (which
  is implemented and grad-verified for the Hyena mixers in this recipe) was **not needed** at 128k.
- All runs: **0 NaN / 0 skipped**, TRAIN_EXIT=0.
- Note the step-count asymmetry: 128k ran **1000** steps (each ~2–3 min; 3000 would be ~4 days),
  vs 3000 for 16k/32k. But 128k still saw **more tokens** (2.10 B) than 16k@3000 (0.79 B) and
  32k@3000 (1.57 B), so "not enough tokens" does not explain its lack of advantage.

## Results

**In-training validation PPL (species-holdout `validation` split):**

| context | steps | tokens | val PPL @ iter1000 | val PPL (end-of-train) |
|---|---:|---:|---:|---:|
| 16k | 3000 | 0.79 B | 2.924 | **2.754** |
| 32k | 3000 | 1.57 B | 2.883 | **2.755** |
| 128k | 1000 | 2.10 B | 2.783 | 2.806 |

**Base → LoRA capped per-record PPL (valid capped 8192 bp, vortex-FP8, same base for all):**

| genome class | n | base | 16k@3000 | 32k@3000 | 128k@1000 |
|---|---:|---:|---:|---:|---:|
| **OVERALL** | 1349 | 3.578 | **−17.48%** | **−17.46%** | −16.45% |
| ssDNA | 289 | 3.588 | −28.23% | −27.90% | −26.93% |
| ssRNA(−) | 353 | 3.631 | −21.20% | −21.31% | −20.04% |
| dsDNA | 127 | 3.320 | −18.69% | −18.39% | −16.84% |
| ssRNA(+) | 337 | 3.563 | −13.41% | −13.36% | −12.53% |
| ssRNA(other) | 64 | 3.400 | −10.75% | −11.31% | −9.50% |
| dsRNA | 179 | 3.736 | −2.46% | −2.66% | −2.37% |

## Verdict — context length does not help this corpus

1. **32k ≈ 16k, exactly.** End-of-train val PPL 2.755 vs 2.754; capped base→LoRA −17.46% vs −17.48%;
   per-genome within noise. Doubling context bought nothing at 2.3× the cost.
2. **128k shows no benefit either — including for the large-genome dsDNA class** (−16.84% vs −18.69%
   at 16k). The one hypothesis for longer context (big dsDNA viruses gain from long-range attention)
   is **not** supported — dsDNA is, if anything, marginally worse. 128k's slightly lower overall
   number is partly its shorter 1000-step schedule, but the per-genome pattern and the token-count
   argument both say context is not the lever.
3. **16k is the right context** for this eukaryotic-host viral corpus. The best model remains the
   16k (or equivalently 32k) run at 3000 steps, val PPL 2.754. **Training length is the lever that
   works; context length is not.**
4. **Feasibility banked:** 128k LoRA on the 20B trains cleanly and efficiently on 8× H200 (TP=4,
   66.7 GB/GPU, ~583 TFLOP/s/GPU), and context-parallelism is available/verified for pushing beyond
   128k — useful for other datasets, just unnecessary here.

Checkpoints: `/data/viral/lora_run_20b_32k_3k/evo2/checkpoints/iter_0003000`,
`/data/viral/lora_run_20b_128k_1k/evo2/checkpoints/iter_0001000`. wandb (project `evo2-viral-lora`):
`viral-lora-20b-seq32768-vfp8-3k` (`pw1y5ab6`), `viral-lora-20b-seq131072-vfp8-1k` (`ls97x5b1`).
