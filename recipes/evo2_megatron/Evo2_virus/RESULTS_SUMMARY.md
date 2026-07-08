# Evo2 viral-LoRA — consolidated results (for next-steps analysis)

Scope: all viral continue-pretraining LoRA runs on the BioNeMo `evo2_megatron` recipe, 8× H200
(container `evo2:20260628`), corpus `Evo2_virus/` (eukaryotic-host viral genomes). Two metrics
throughout:
- **in-training val PPL** — species-holdout `validation` split, `--eval-iters 20` (noisy), lower=better.
- **base→LoRA capped PPL** — per-record perplexity on the valid set capped to 8192 bp, base vs LoRA
  scored with identical precision; reported as % change (more negative = bigger gain).

Shared LoRA config unless noted: dim 16, alpha 32, dropout 0.1, targets
`dense_projection,linear_qkv,linear_proj,linear_fc1,linear_fc2`; GBS 16, MBS 1; cosine LR 3e-4→3e-5,
warmup 10; eval/save every 250 steps.

---

## 0. The corpus (the prior that drives everything)

Sequences are **short**: seq-16384 already fully contains ~93% of records.

| split | n | median | p90 | p95 | >16k | >32k | >131k | ≥1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 12,944 | 3,968 | 12,926 | 28,941 | 6.7% | 4.5% | 2.3% | 0.1% (17) |
| valid | 1,349 | 3,862 | 12,133 | 26,209 | 5.6% | 4.0% | 2.1% | 0.1% (1) |

Base-nucleotide coverage (the large-genome tail holds most bases): 16k covers ~43% of nucleotides,
128k ~72%.

---

## 1. Model-size sweep — seq-16384, 1000 steps

| model | val PPL (best@1000) | base→LoRA overall | % improved | s/step | TFLOP/s/GPU | wall-clock | parallelism |
|---|---:|---:|---:|---:|---:|---:|---|
| 1B  | ~3.17 (3.167@750) | ~−0.7% † | — | ~1.27 | ~255 | — | pure DP, bf16 |
| 7B  | 3.087 | −10.90% ‡ | 96.8% | ~4.34 | ~313–325 | ~76 min | pure DP, bf16 |
| 20B | **2.969** | **−13.71%** | 97.3% | ~9.0 | ~393–404 | ~2.5 h | pure DP + vortex-FP8, recompute-2 |
| 40B | 2.976 | −13.51% | 97.3% | ~23.4 | ~316–324 | ~6.7 h | TP=4 + vortex-FP8, recompute-1 |

† 1B from the runbook/memory (approx); scored vs bf16 base. ‡ **7B/1B base scored vs bf16 base
(overall base 3.6122); 20B/40B vs vortex-FP8 base (overall base 3.5784)** — the base differs, so 7B%
is not perfectly comparable to 20B/40B%. val-PPL comparison is clean.

**Finding:** 20B is the sweet spot. **40B does not beat 20B** (20B marginally better on both metrics
at ~1/2.6 the cost). Monotonic 1B < 7B < 20B ≈ 40B.

---

## 2. Training-length ablation — seq-16384, 1000 → 3000 steps

| model | val PPL @1000 | val PPL @3000 (end) | overall @1000 | overall @3000 |
|---|---:|---:|---:|---:|
| 20B | 2.969 | **2.754** | −13.71% | **−17.48%** |
| 40B | 2.976 | 2.767 | −13.51% | −17.20% |

**Finding:** longer training is **the lever**. The apparent 1000-step "plateau" was an LR-schedule
artifact (the `--decay-steps 1000` cosine had annealed to the 3e-5 floor). With a full 3000-step
cosine both models keep improving well past 1000. **20B still edges 40B at 3000** (2.754 vs 2.767).

20B @3000 val-PPL trajectory (noisy): 250:3.085, 500:3.006, 750:2.941, 1000:2.924, 1250:2.895,
1500:2.813, 1750:2.858, 2000:2.802, 2250:2.827, 2500:2.792, 2750:**2.768**, 3000:2.786, full-val:2.754.
→ curve is near-floor but still gently descending by 2750; **not obviously converged**.

---

## 3. Context-length ablation — 20B, 16k / 32k / 128k

| context | steps | tokens | parallelism / peak | s/step | val PPL (end) | base→LoRA overall |
|---|---:|---:|---|---:|---:|---:|
| 16k  | 3000 | 0.79 B | pure DP, recompute-2 | ~9.0  | **2.754** | **−17.48%** |
| 32k  | 3000 | 1.57 B | TP=2, 63.8 GB | ~20.6 | 2.755 | −17.46% |
| 128k | 1000 | 2.10 B | TP=4/CP=1, 66.7 GB | ~110.7 | 2.806 | −16.45% |
| **128k** | **3000** | **~6.3 B** | TP=4/CP=1, ~66 GB, ~583 TFLOP/s | ~111 | **2.750** (test 2.740) | see §3b |

128k@3000 completed cleanly (0 nan/0 skipped, LR annealed to exactly 3e-5). Resumed from the
1000-step checkpoint via `--workers 0` (see §5) — saved ~31 h of compute. Noisy in-training evals
1250–3000: 2.805/2.738/2.756/2.723/2.762/2.701/2.755/2.763 (mean ~2.74).

**Finding:** context length is **not a lever** for this corpus. Step-matched at 3000, all three land
within ±0.005: **16k 2.754, 32k 2.755, 128k 2.750** — despite 128k reading ~8× the tokens and
costing ~8–12× the wall-clock. (The 0.004 "edge" for 128k is inside eval noise + the extra-tokens
confound.) Consistent with the short-sequence prior. **Feasibility banked:** 128k LoRA on 20B trains
cleanly on 8×H200 (TP=4, 66.7 GB, ~583 TFLOP/s); CP available/grad-verified for >128k.

> NOTE the above uses the standard metric, which (like the capped per-genome tables in §4) cannot
> see a long-range benefit. §3b is the metric that can.

---

## 3b. Uncapped, length-stratified re-scoring (the context test the capped metric can't do)

The §3/§4 metrics cap every record at 8192 bp, so they are **structurally blind** to any long-range
benefit (a longer-context model can only show its edge on tokens past position 8192). To actually
test it, all valid records were re-scored **at full length** (capped only at 131072 bp) with base vs
16k@3000-LoRA vs 128k@3000-LoRA, then bucketed by **original record length**. Long records (>32k)
scored with context-parallel sharding (CP=8); short records at CP=1. (N-padding on long records is
common-mode across all three models → cancels in the 128k-vs-16k column.)

| length bucket | n | base | 16k-LoRA | 128k-LoRA | 16k Δ | 128k Δ | **128k vs 16k** |
|---|---:|---:|---:|---:|---:|---:|---:|
| A ≤8192 (control) | 1077 | 3.593 | **2.947** | 2.979 | −17.98% | −17.09% | **+1.09%** |
| B 8192–16k | 197 | 3.607 | **2.966** | 2.979 | −17.79% | −17.42% | +0.45% |
| C 16k–32k | 21 | 3.463 | **2.898** | 2.933 | −16.32% | −15.32% | +1.20% |
| D 32k–128k | 25 | 3.305 | 2.705 | **2.693** | −18.15% | −18.53% | **−0.46%** |
| E >128k (cap 131072) | 28 | 3.143 | 2.785 | **2.763** | −11.41% | −12.08% | **−0.76%** |
| **OVERALL** | 1348 | 3.579 | **2.941** | 2.969 | −17.81% | −17.04% | **+0.94%** |

Last column: negative = 128k-trained adapter beats 16k-trained adapter. Genome-class within long
records (>8192, n=271): dsDNA (n=64) **−0.03%** (tie), ssRNA(−) (105) +0.24%, ssRNA(+) (87) +0.95%,
ssRNA(other) (14) −1.82% (noisy). Full table: `/data/viral/RESULTS_uncapped_bylen.txt`.

**Findings:**
1. **A genuine long-context crossover exists — and only this metric reveals it.** On records **>32 kb
   (D, E)** the 128k adapter finally beats the 16k adapter (−0.46%, −0.76%). The capped metric put
   everything in bucket A and saw none of this.
2. **But it's practically negligible.** The effect is <1%, on only **~4% of records (53/1348)**, and
   it's *paid for* by the 128k adapter being ~1% **worse** on the 96% of records ≤32 kb (it
   over-specializes to long context). **Net: 16k wins overall (−17.81% vs −17.04%; 2.941 vs 2.969).**
3. **The motivating hypothesis is refuted even here.** "Large dsDNA genomes need long context" — on
   long dsDNA records the two adapters **tie (−0.03%)**. The small D/E win is driven by a few noisy
   ssRNA(other) records, not the big-genome class.

**Bottom line:** long-context training does *something* on the long-genome tail, but it's a hair, on a
sliver of the data, for the wrong class, and it degrades the common case. **16k remains the right
choice.** The value of this experiment: it upgrades the verdict from "context doesn't help (but our
metric was blind)" to "context helps ~0.5–0.8% on >32 kb records, still not worth 8–12× cost."

---

## 4. Per-genome breakdown (base→LoRA capped 8192 bp, % improvement)

Base column = vortex-FP8 base (3.578-overall runs). Rows sorted by 20B@3000 gain.

| class | n | base | 20B@1k | 40B@1k | 20B@3k(16k) | 40B@3k | 32k@3k | 128k@1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OVERALL** | 1349 | 3.578 | −13.71% | −13.51% | **−17.48%** | −17.20% | −17.46% | −16.45% |
| ssDNA | 289 | 3.588 | −23.71% | −23.42% | **−28.23%** | −27.93% | −27.90% | −26.93% |
| ssRNA(−) | 353 | 3.631 | −16.72% | −16.47% | −21.20% | −20.91% | −21.31% | −20.04% |
| dsDNA | 127 | 3.320 | −15.37% | −15.28% | −18.69% | −18.55% | −18.39% | −16.84% |
| ssRNA(+) | 337 | 3.563 | −9.77% | −9.66% | −13.41% | −13.10% | −13.36% | −12.53% |
| ssRNA(other) | 64 | 3.400 | −4.70% | −3.87% | −10.75% | −9.87% | −11.31% | −9.50% |
| dsRNA | 179 | 3.736 | −1.38% | −1.40% | −2.46% | −2.34% | −2.66% | −2.37% |

(7B@1k, vs bf16 base 3.612: OVERALL −10.90, ssDNA −20.08, ssRNA(−) −12.12, dsDNA −13.13, ssRNA(+)
−7.99, ssRNA(other) −1.35, dsRNA −1.06.)

**Finding:** consistent ordering everywhere — **ssDNA gains most (~−28%), dsRNA barely moves (~−2.5%)**.
dsRNA is the clear outlier: high base PPL (3.736, hardest) *and* smallest LoRA gain. ssRNA(other) (n=64)
is noisy.

---

## 5. Engineering / feasibility (measured)

| run | parallelism | recompute | peak GB/GPU | s/step | TFLOP/s/GPU |
|---|---|---:|---:|---:|---:|
| 20B @16k | pure DP | 2 | fits (<143) | ~9.0 | ~400 |
| 40B @16k | TP=4 | 1 | ~66 | ~23.4 | ~320 |
| 20B @32k | TP=2 | 1 | 63.8 | ~20.6 | ~414 |
| 20B @128k | TP=4, CP=1 | 1 | 66.7 | ~110.7 | ~583 |

- **FP8-LoRA enablement** was required for 20B/40B (Hopper BF16 unstable on Savanna 20B/40B): added
  `--vortex-style-fp8` to `train_evo2` + made the FP8 projection LoRA-safe.
- **Memory driver:** Megatron pre-allocates an fp32 main_grad buffer for ALL params incl. the frozen
  LoRA base (~6 B/param), sharded by TP not DP → big models / long context need TP. `recompute-num-layers`
  is a *chunk size* (uniform method) → use 1 for least memory.
- **Resume/extend is supported** (validated 2026-07-05): re-run same command, same parallelism, at a
  `--result-dir` that already has `iter_*`; base loads via PEFT pre-wrap hook, then adapter+optimizer+
  RNG+iteration+data-position restore. `override_opt_param_scheduler=True` means the LR curve is
  **reshaped from the new CLI** (`--decay-steps`) — resume with a larger horizon trains at a useful LR
  (not pinned at floor). **Two resume gotchas:** (a) keep `--eval-interval 250` so the valid dataloader
  builds enough samples (else `no samples left to consume`); (b) **`--workers 0` is required** — the
  default async dataloader deadlocks one DP replica on the first post-resume batch (NCCL ALLREDUCE
  timeout; a bigger timeout does not help).

---

## 6. Caveats / data-quality flags (important for the analysis)

1. **Possible train/eval leakage:** `KF740664.1|ictv:VMR1024671` was flagged as contaminated against
   Evo2's *own base pretraining* data and slated for rescoring. The per-genome tables above **predate**
   that rescoring — some capped numbers may shift. **Recommend a full leakage audit (valid vs Evo2
   pretraining) before drawing final per-genome conclusions.**
2. **Eval noise:** in-training val PPL uses only `--eval-iters 20` → ±~0.02–0.05 jitter (e.g. 128k@3000
   1500→1750 went 2.738→2.756, within noise). End-of-train full-val numbers are the reliable ones.
3. **Base-precision mismatch:** 1B/7B scored vs a bf16 base (3.612); 20B/40B/context vs a vortex-FP8
   base (3.578). Cross-size % comparisons are approximate; val-PPL comparisons are clean.
4. **Base preds reused** across @1000 and @3000 comparisons (same base, exact deltas).
5. The §4 capped ablation reports per-class **% only**; §3b now provides absolute uncapped PPL by
   length bucket (the metric that can see context effects).
6. **§3b long-record scoring** N-pads records >32k to a CP=8-friendly multiple (≤1023 extra tokens on
   ≥32769-long seqs, <3%); the bias is common-mode across base/16k/128k so it cancels in the
   128k-vs-16k column, but the absolute per-bucket PPLs for D/E carry a small upward bias.

---

## 7. Candidate next steps (to scope in the in-depth analysis)

**A. Close the current threads**
- ✅ DONE: 128k@3000 finished (full-val 2.750 ≈ 16k/32k) — §3; context not a lever on the standard metric.
- ✅ DONE: uncapped length-stratified re-scoring — §3b; long-context edge is real but negligible
  (<1% on ~4% of records) and 16k wins overall. Question settled.
- TODO: rescore after a leakage audit (KF740664.1); regenerate per-genome tables on a clean valid set.
- Optional follow-up to §3b: **position-resolved** scoring (mean logprob over tokens *beyond* 8192
  only, not whole-record) would sharpen the D/E signal; and unpadded CP scoring to remove the small
  N-pad bias. Low priority given the effect size.

**B. Push the lever that works (training length)**
- 20B @16k **beyond 3000** — the 3000 curve was still gently descending (2.768@2750). Does 4000–6000
  keep helping, or is it converged? Cheap now that resume+extend works (`--workers 0`).

**C. Adapter capacity (mostly untested)**
- Only LoRA dim 16 tried. Sweep dim/alpha (e.g. 32/64) — does more capacity help the weak classes
  (dsRNA, ssRNA(+))? Compare vs full fine-tuning of the 20B as an upper bound.

**D. The dsRNA outlier**
- dsRNA has the highest base PPL and smallest gain (~−2.5%) across every run. Investigate: class data
  quantity (n=179), sequence characteristics, tokenization, or genuine domain difficulty. Consider
  class-reweighting the blend (upweight dsRNA / ssRNA(other)).

**E. Does PPL translate downstream?**
- All gains are PPL-only. Add a functional eval (variant-effect / zero-shot, cf. the `zeroshot_brca1`
  notebook adapted to viral tasks) to check the −17% PPL actually buys downstream performance.

**F. Efficiency**
- 20B is the price/perf winner; 40B not worth it. If deploying, quantify 20B@3000 inference cost and
  whether a smaller adapter or distillation retains the gain.

---

### Artifacts (all under `/data/viral/`, host bind-mount)
| run | ckpt dir | wandb (project evo2-viral-lora) |
|---|---|---|
| 1B @16k@1k | `lora_run_16k/.../iter_0001000` | `viral-lora-1b-seq16384` |
| 7B @16k@1k | `lora_run_7b_16k/.../iter_0001000` | `viral-lora-7b-seq16384` |
| 20B @16k@1k | `lora_run_20b_16k/.../iter_0001000` | `viral-lora-20b-seq16384-vfp8` (`tlyvhguh`) |
| 40B @16k@1k | `lora_run_40b_16k/.../iter_0001000` | `viral-lora-40b-seq16384-vfp8` (`ytv0owwe`) |
| 20B @16k@3k | `lora_run_20b_16k_3k/.../iter_0003000` | `...-3k` (`5yl8arbe`) |
| 40B @16k@3k | `lora_run_40b_16k_3k/.../iter_0003000` | `...-3k` (`z2y3z951`) |
| 20B @32k@3k | `lora_run_20b_32k_3k/.../iter_0003000` | `viral-lora-20b-seq32768-vfp8-3k` (`pw1y5ab6`) |
| 20B @128k@1k | `lora_run_20b_128k_1k/.../iter_0001000` | `viral-lora-20b-seq131072-vfp8-1k` (`ls97x5b1`) |
| 20B @128k@3k | `lora_run_20b_128k_3k/` (running) | `viral-lora-20b-seq131072-vfp8-3k` |

Source logs: `Evo2_virus/LOG_evo2_{7b_8k,20b,40b,longer_3000steps,context_ablation}.md`; commands in
`Evo2_virus/COMMANDS.sh`.
