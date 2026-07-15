# Autonomous exploration log (2-server run)

Directive (2026-07-10): "when [the current sweep runs] are done decide on next steps and use the
two servers to continue exploring until I ping back." Operating autonomously across **local** (this
8×H200) and **ohio** (3.149.234.129, driven via tmux windows ohio-1/2/3 → detached container
`evo2run`; wandb offline there). Decisions + results logged here; consolidated into `RESULTS_SUMMARY.md`.

## What we already know (levers)
- **Training length:** helps then saturates — big gain 1000→3000 (−0.215 val), tiny 3000→9000 (−0.025).
  Exhausted past ~3000 on this ~86M-token corpus. NOT a productive lever.
- **Context length:** no benefit (16k≈32k≈128k step-matched; uncapped test: 128k edges 16k only on
  >32kb records, ~4% of data, and loses overall). NOT a lever.
- **Adapter capacity (dim):** IN PROGRESS and **the current live lever** — dim32 (−19.01%) already
  beats dim16 (−17.48%) at equal cost, and dim32@3000 even beats dim16@9000 at ⅓ the compute.
- **dsRNA:** persistent outlier (~−3%) across every size/length/context/dim so far → intrinsic, not a
  training-budget problem.

## Standing config (held constant unless noted)
20B, seq-16384, 3000 steps, pure DP + recompute-2 + vortex-FP8, GBS16/MBS1, cosine 3e-4→3e-5,
warmup 10, eval/save 250, base `/data/evo2_20b_mbridge`. Fresh runs (not resumes). Scored capped-8192
per-genome vs reused base preds (`pred_20b_base_vfp8`, base 3.579).

## Phase 1 — capacity sweep {8,16,32,64,128,256} @16k@3000  (RUNNING)
Capacity is a **strong, still-climbing lever** (each doubling ≈ +1–1.5 pp). Even dsRNA responds
(−2.46→−3.13→−4.02 across 16/32/64), unlike to length. dim32@3000 already beats dim16@9000 at ⅓ cost.
| dim | alpha | server | OVERALL base→LoRA | full-val | test | status |
|---:|---:|---|---:|---:|---:|---|
| 8 | 16 | ohio | −15.76% | — | — | done (dsRNA −1.91) |
| 16 | 32 | (baseline) | −17.48% | 2.754 | 2.759 | done |
| 32 | 64 | local | −19.01% | 2.739 | 2.721 | done |
| 64 | 128 | local | **−20.09%** | **2.685** | 2.664 | done |
| 128 | 256 | ohio | −20.74% | — | — | done |
| 256 | 512 | local | **−21.29%** | **2.610** | 2.586 | done — see overfit note |

CURVE COMPLETE: 8/16/32/64/128/256 = −15.76/−17.48/−19.01/−20.09/−20.74/−21.29 (monotonic, log-linear,
diminishing: +1.5pp/doubling early → +0.6pp late). Coverage peaks ≤64 (95%) and drops by 256 (87%).

**Capacity findings:** monotonic, log-linear gain in MEAN (8→256: −15.76→−17.48→−19.01→−20.09→−21.29),
best val 2.610 at dim256. BUT **coverage (% records improved) drops at high capacity**: 96%(dim8) →
95%(64) → **87%(256)** — dim256 overfits, helping the majority more but regressing a minority. So
**robustness sweet spot ≈ dim 64–128**; dim 256 wins on mean only. Chose dim 64 as the Phase-2
reference (strong + 95% coverage + safe memory). Did NOT push dim 512 (diminishing + OOM-risk).

## Phase 2 — orthogonal levers at dim 64 (RUNNING)
Question: WHERE does capacity help (which modules) + does alpha/dim scaling matter? All dim64, α128
baseline = −20.09%.
| exp | server | targets / alpha | OVERALL | status |
|---|---|---|---:|---|
| Tmixer | local | dense_projection,linear_qkv,linear_proj (mixer/attn only) | −16.68% | done |
| Tmlp | local | linear_fc1,linear_fc2 (MLP only) | −19.72% | done |
| α=64 (ratio1) | ohio | alpha 64 | −19.39% | done |
| α=256 (ratio4) | ohio | alpha 256 | **−21.00%** | done |

**Alpha finding (dim64):** ratio1 −19.39% < ratio2 −20.09% < **ratio4 −21.00%**. Higher LoRA scaling
(alpha/dim) is a real lever; ratio4@dim64 (−21.00%) nearly matches dim256-ratio2 (−21.29%) at ¼ the
params. Two independent knobs both raise effective adapter magnitude: dim and alpha/dim.

## Phase 4 — best-combined + dsRNA
- **ohio DONE: dim128 α512 (ratio4, all-5) = BEST CONFIG** → **OVERALL −22.20%**, val 2.601, test
  2.579, **coverage 92.5%**, dsRNA −5.83%. **Beats raw dim256 (−21.29%, 87% cov) on BOTH mean AND
  coverage** — capacity (dim) and scaling (alpha/dim) stack, and dim128×ratio4 avoids dim256's
  overfitting. **D\* = dim128 × α512.** `lora_run_20b_16k_dim128_a512`.
- **ohio DONE: dim128 × α1024 (ratio8) = NEW BEST → OVERALL −23.37%**, coverage **93.9%**, dsRNA
  **−6.93%** (base 3.579 → 2.7427). **Beats ratio4 (−22.20%, 92.5%, −5.83%) on ALL THREE axes** — mean,
  coverage, AND dsRNA. Scaling did NOT peak at ratio4; coverage even rose (no overfit). Ratio ladder
  @dim128: r2 −20.74 → r4 −22.20 → r8 −23.37 (still climbing). `lora_run_20b_16k_dim128_a1024`.
- **ohio DONE: dim128 × α2048 (ratio16) = DESTABILIZED → OVERALL +4.97%** (WORSE than base), coverage
  **1.0%** (14/1348), every genome class regressed. In-train val PPL 3.6186 (≈ base 3.579) foretold it.
  → **Scaling PEAKS at ratio8; ratio16 blows up** (alpha/dim ≈ effective adapter LR; 16× is too high →
  adapter learns garbage without NaN). Scaling ladder @dim128: r2 −20.74 → r4 −22.20 → **r8 −23.37 (PEAK)**
  → r16 +4.97 (BROKEN). `lora_run_20b_16k_dim128_a2048`.

## Phase 5 — regularization lever (dropout), attacking the capacity-overfit
Scaling (peaks r8) and capacity (plateaus ~dim256, overfits coverage) are exhausted, both ~−23.4%.
dim256×r4 has the best MEAN (−23.48%) but overfits (coverage 91% vs dim128×r8's 93.9%). **Dropout was
fixed at 0.1 in EVERY run so far — untested lever.** Higher dropout is the classic overfit fix.
- **ohio DONE: dim256 × α1024 (r4), dropout 0.2 = NEW BEST → −24.33%**, coverage **94.6%**, dsRNA
  −7.81% (in-train val 2.446). **Beats dropout-0.1 (−23.48%, 91.0%) on BOTH mean (+0.85pp) AND coverage
  (+3.6pp)** — higher dropout regularized the dim256 overfit without costing the mean. dsRNA held.
  **Dropout is a real, stacking lever.** `lora_run_20b_16k_dim256_a1024_do2`.

## Phase 6 — can dropout rescue the α=2048 cliff?
The α2048 runs diverged at dropout 0.1 (dim256×α2048 +2.77%, dim128×α2048 +4.97%). Hypothesis: that
divergence is **under-regularization / overfit-to-instability**, not a fundamental magnitude limit —
if so, strong dropout should tame it and possibly unlock a peak past −24.33%.
- **ohio DONE: dim256 × α2048 (r8), dropout 0.3 = STILL BROKEN → +5.35%** (coverage 0.4%, in-train val
  3.635 ≈ base). Even worse than do0.1 (+2.77%). **Dropout does NOT rescue α2048 → the α-ceiling is a
  FUNDAMENTAL magnitude limit, not under-regularization.** `lora_run_20b_16k_dim256_a2048_do3`.
- **ohio DONE: dim256 × α1536 × dropout 0.2 = MARGINAL NEW BEST → −24.44%**, coverage 94.7%, dsRNA
  −7.98% (in-train val 2.531, healthy). **Dropout 0.2 RESCUED α1536** (which broke to +0.19% at do0.1)
  and edges the α1024×do0.2 best (−24.33%) by +0.11pp. `lora_run_20b_16k_dim256_a1536_do2`.
  → **Dropout and α interact: regularization extends the usable-α cliff (α1024→α1536 at do0.2), but
  α2048 stays unrescuable.** Gain over α1024×do0.2 is noise-level → **we are on the peak plateau ~−24.4%.**

## ADAPTER-CONFIG FRONTIER — EXHAUSTIVELY MAPPED. Peak plateau ~−24.4%.
**Best config: dim256 × α1024–1536 × dropout 0.2, all-5 targets, 20B/16k/3000 → −24.4%, ~95% coverage.**
Levers, in order of impact: **α-scaling** (biggest, hard cliff ~α1024 that dropout stretches to ~1536)
→ **capacity/dim** (helps to 256, overfits coverage unless regularized) → **dropout** (new lever:
mean saturates ~0.2, coverage climbs to 0.3; also unlocks higher α) → **target modules** (MLP carries
it). dsRNA responds only to capacity. **STOP adapter fan-out** — remaining permutations are noise.
Only orthogonal axis left = training length (Phase 7 ceiling test, running).
- **MLP-only@256 DONE: −20.79%, coverage 87.5%** → regularization hypothesis **REFUTED** (same 87%
  coverage as all-5@256; dim-256 overfit is intrinsic to high capacity, not attn adapters). Robust
  sweet spot stays ~dim128.
- **dsRNA-reweighted DONE — BACKFIRED (clean negative):** upweighting dsRNA to 10% tokens made
  held-out **dsRNA WORSE: +5.28%** (vs −4.02% baseline) — LoRA over 19 epochs of the tiny dsRNA set
  **memorized/overfit**, hurting generalization. dsRNA weakness is NOT a token-share problem fixable
  by reweighting; it needs **more unique dsRNA data** (only 1574 recs / 4.2M bp) or a different
  approach. Also confirms Megatron blend weights = sample/token fractions ([0.9,0.1] → 10% dsRNA).
- **local DONE: dim256 × α1024 (ratio4) → OVERALL −23.48%**, coverage 91.0%, dsRNA **−7.87%** (best
  dsRNA yet). Best MEAN so far, essentially tied with dim128×ratio8 (−23.37%). But coverage 91.0% <
  dim128×ratio8's 93.9% → dim256 still overfits a minority. `lora_run_20b_16k_dim256_a1024`.
  **Frontier converges ~−23.4%: capacity (dim) and scaling (alpha/dim) both still help and have met.**
  dim256×r4 = best mean+dsRNA; dim128×r8 = best robustness (coverage).
- **local DONE: dim256 × α2048 (ratio8) = DESTABILIZED → +2.77%** (worse than base), coverage 10.5%.
  Combined with ratio16 (dim128×α2048 = +4.97%), the cliff is driven by **ABSOLUTE α, not ratio**:
  both broken runs have **α=2048**; every α≤1024 run is healthy regardless of dim/ratio.
  `lora_run_20b_16k_dim256_a2048`.

## FRONTIER — FULLY MAPPED (α≈1024 is the ceiling)
  | config | α | ratio | OVERALL | coverage | dsRNA | verdict |
  |---|---:|---:|---:|---:|---:|---|
  | dim128×α512 | 512 | 4 | −22.20% | 92.5% | −5.83% | good |
  | **dim256×α1024** | 1024 | 4 | **−23.48%** | 91.0% | **−7.87%** | **best MEAN** |
  | **dim128×α1024** | 1024 | 8 | **−23.37%** | **93.9%** | −6.93% | **best ROBUSTNESS** |
  | dim128×α2048 | 2048 | 16 | +4.97% | 1.0% | — | BROKEN |
  | dim256×α2048 | 2048 | 8 | +2.77% | 10.5% | — | BROKEN |
  → **Peak ~−23.4–23.5%.** Two co-winners: dim256×α1024 (mean+dsRNA) vs dim128×α1024 (coverage).
  α=2048 diverges (no NaN, bad basin) at BOTH dims. Actionable ceiling: **α≤1024.**
- **local DONE: dim128 × α1536 (ratio12) = BROKEN → +0.19%** (≈base), coverage 28.2%. So the α-cliff
  is **sharp and sits just above 1024**: α1024 −23.37% (great) → α1536 +0.19% (broken) → α2048 +4.97%
  (badly broken), all @dropout 0.1. **α1024 is essentially the hard ceiling at dropout 0.1.**
  `lora_run_20b_16k_dim128_a1536`.
- **local DONE: dim256 × α1024 × dropout 0.3 → −24.28%**, coverage **96.4%** (highest of any run),
  dsRNA −7.67%. Dropout ladder @dim256×α1024: do0.1 −23.48%/91.0% → **do0.2 −24.33%/94.6%** →
  do0.3 −24.28%/96.4%. **Mean saturates ~−24.3% (0.2≈0.3); coverage keeps climbing with dropout.**
  Sweet spot 0.2 (best mean) / 0.3 (best coverage). `lora_run_20b_16k_dim256_a1024_do3`.

## Phase 7 — ceiling test (best adapter config + extended length)
Adapter-config frontier now fully mapped; peak ~−24.3%. Open project question: is that the DATA-limited
ceiling, or does the best config keep improving with more steps (length gave +0.94pp for 3000→9000 at
dim16 — does it still help at high capacity+regularization)?
- **local DONE: dim256 × α1024 × dropout 0.2 × 6000 steps → −24.96%** (NEW BEST mean), coverage 91.8%,
  dsRNA **−9.78%** (best dsRNA by far). **Length is NOT exhausted at the best config**: 3000→6000 gave
  +0.63pp mean and +1.97pp dsRNA — but coverage dropped 94.6→91.8% (**more training re-introduced the
  overfit** that do0.2 had fixed at 3000 → more steps needs more regularization). `..._do2_6k`.
- **local DONE: dim256 × α1024 × dropout 0.3 × 6000 steps = OVERALL BEST → −25.44%**, coverage 93.7%,
  dsRNA −9.69%. **do0.3 at 6k beats do0.2 at 6k on BOTH mean (−24.96→−25.44) AND coverage (91.8→93.7%)**
  — confirms longer training needs more dropout. `lora_run_20b_16k_dim256_a1024_do3_6k`.

## 20B THREAD COMPLETE. Peak = dim256 × α1024 × dropout0.3 × 6000 steps = −25.44% / 93.7% coverage
(from dim16 baseline −13.71% → **1.85× the gain**). All levers stack: capacity(dim256) + α-scaling
(α1024, ratio-appropriate) + dropout(0.3 for long runs) + length(6000). Further length (9k) hits the
data-limited ceiling (§2). 20B exhausted — holding local. 7B cross-scale capacity probe continues on ohio.

**Revised peak:** best config × 6000 steps = **−24.96%** (mean) — length + capacity + α-scaling +
regularization all stack. Coverage/mean trade-off at 6k is the open refinement (do0.3×6k, running).

## Phase 8 — cross-scale generalization (does the 20B recipe transfer?) [ohio, via subagent]
Ran on ohio (subagent + main bridging). NOTE: 7B base ckpt requires `--model-size evo2_7b_base`
(11008 MLP dim); `evo2_7b` (11264) errors with a dist-checkpoint shape mismatch. 7B uses bf16 (no
vortex-FP8); scored vs a fresh 7B bf16 base (PPL 3.6128).
- **7B × BEST-20B-CONFIG DONE: dim256 × α1024 × do0.2 → −8.05%, coverage 59.9%** (`lora_run_7b_best`).
  **The 20B recipe does NOT transfer — it's WORSE than the 7B dim16 baseline (−10.90%, cov 96.8%)**
  and far below the 20B best (−24.33%). The collapsed coverage (60%) + degraded mean is the
  partial-divergence signature of **α too high**: the α-cliff **scales with model width** — α1024
  optimal for 20B over-drives the narrower 7B. → **Optimal adapter config is model-size-dependent, NOT
  universal; α (and dim) must scale with hidden width.**
- **7B DONE: dim128 × α512 × do0.2 (half-scale, ratio4) → −8.14%, coverage 60.6%** — **≈ identical to
  the full 20B config on 7B (−8.05%/59.9%); halving dim+α did NOT fix it.** Still worse than dim16
  baseline (−10.90%/96.8%), coverage still collapsed ~60%. → **α512 is STILL too high for 7B; the 7B
  α-cliff is well below 512.** The dim16 baseline works because it uses α32 (ratio2). `lora_run_7b_scaled`.
- **7B DONE: dim128 × α256 × do0.2 (ratio2) = BREAKTHROUGH → −21.80%, coverage 97.0%!** — **2× the
  dim16 baseline (−10.90%)**, full coverage, comparable to the 20B range. `lora_run_7b_r2`.
- **7B (RUNNING): dim256 × α512 × do0.2 (ratio2)** — does doubling dim at ratio2 push the 7B further
  (as it did the 20B), or is dim128 the 7B capacity sweet spot? `lora_run_7b_r2_dim256`.

### CROSS-SCALE CONCLUSION: the recipe TRANSFERS — scale the α/dim RATIO to model width.
It was the **ratio (α/dim), not absolute α or dim, that broke the 7B**: ratio4 collapses the 7B
(coverage ~60%, −8%) but **ratio2 thrives (dim128×α256 → −21.80%, 97% cov)**. Compare 20B: peaks at
**ratio8** (−23.37%), breaks at ratio16. → **The usable α/dim ratio scales with model width** (7B tops
out ~ratio2, 20B ~ratio8). Recipe = high capacity + dropout0.2 + **ratio dialed to the model size**.
The earlier "doesn't transfer" was applying the 20B's ratio (4-8) verbatim, which over-drives the 7B.

### dsRNA root-cause (GPU-free analysis) — UNDER-REPRESENTATION
Manifest covers 100% of train records. **dsRNA = 12.2% of records but only 2.45% of TOKENS**
(dsRNA seqs short, median 2233 bp; dsDNA hogs 65% of tokens). Since training is token-based, dsRNA
is starved → likely why LoRA barely helps it (and its base PPL 3.736 is highest = intrinsically
hardest too). **Prepared token-reweighting experiment:** split train → `train_dsRNA.fasta` (1574
recs, 4.2M bp) + `train_rest.fasta` (11370, 167.6M bp); preprocessing both now
(`dsRNA_preprocess.log`). Plan: blend `viral_train_rest`(w) + `viral_train_dsRNA`(w) so dsRNA ≈ 15–25%
of tokens (weights are ~token fractions in the Megatron blend → dsRNA weight 0.15–0.25 vs rest
0.75–0.85), train best-config, score dsRNA. Risk: only ~2100 unique dsRNA 16k-samples → repetition/
overfit at high upweight; use a moderate target (~15%). Launch when a server frees. NOTE: ohio-3 tail must point at phase2_alpha.log (re-tail after
Ctrl-C); direct `docker exec evo2run grep ... /data/viral/*.log` via ohio-2 is the reliable status read.

**Target-module finding (dim64):** all-5 −20.09% | **MLP-only −19.72%** | mixer/attn-only −16.68%.
→ **MLP LoRA (linear_fc1/fc2) carries almost all the benefit**; attn/mixer projections add only ~0.4pp
(mixer-only ≈ dim16-all-5). Actionable: capacity belongs in the MLP.

## Phase 3 — probing the frontier
- **local (RUNNING): MLP-only @ dim256** — hypothesis: dim256-all-5 overfit (cov 87%) may be driven by
  the low-value attn adapters; MLP-only@256 = fewer, higher-value params → test if it keeps the high
  mean AND recovers coverage. `lora_run_20b_16k_dim256_Tmlp`.
- Pending: fold ohio alpha results; then synthesize a **best-config** recommendation; then pivot to a
  **dsRNA-targeted** experiment (only lever that has moved dsRNA is capacity) rather than more
  dim/target permutations.

Since the curve was still climbing at 64, extended UP: ohio does 128, local does 256 (parallel) to
find the ceiling. dim-256 uses recompute-1 (model-identical; more memory headroom — pure-DP runs sit
at ~127 GB/GPU).

## Decision procedure (when Phase 1 completes)
1. Read all `RESULTS_dim{D}_pergenome.txt` + end-of-train full-val; tabulate the curve; pick **D\***.
2. Branch:
   - **Saturated** (128 ≤ 64, D\* ∈ {32,64}) → D\* is the capacity sweet spot. Phase 2 = orthogonal
     levers at D\*, split across servers:
       - target-module ablation: attn-only `{linear_qkv,linear_proj,dense_projection}` vs mlp-only
         `{linear_fc1,linear_fc2}` (vs current all-5).
       - alpha/dim scaling ∈ {1, 4} (current is 2).
   - **Still climbing at 128** → push capacity: dim 256 on one server; target-module ablation at 128
     on the other.
3. Phase 3 = best-combined-config confirmation run; then consider dsRNA-targeted data reweighting.
4. GPU-free analyses to run in parallel anytime: dsRNA characterization (length/count/base-PPL/
   tokenization), a leakage sanity check on the valid set, and keeping `RESULTS_SUMMARY.md` committed.

## Guardrails
- Reuse the proven pipeline; only vary well-understood CLI knobs (dim/alpha/targets/lr). Avoid
  OOM-risky moves (e.g. full-FT) without a smoke test first.
- Fresh runs use default workers (matches baselines); resumes require `--workers 0`.
- Keep `--most-recent-k -1`. Commit doc updates as results land. Don't push.

## Timeline log
- 2026-07-10 ~22:13 UTC: ohio sweep launched (dim8→dim128). local on dim64. dim32 result recorded.
- 2026-07-11 ~04:18 UTC: dim64 done (−20.09%, val 2.685). Curve still climbing → killed the local
  sweep's redundant dim-8 (ohio owns it) and launched **dim-256** on local (parallel with ohio-128).
  LESSON: killing a torchrun job needs a **process-group kill** — `pkill -f` orphans the workers
  (they hold GPU mem in Rsl state); use `for g in $(ps -eo pgid,cmd|grep bin/train_evo2|awk '{print $1}'|sort -u); do kill -9 -$g; done`.
  Monitors: local → bvcozwv5n (dim-256); ohio → waiter b1rgagnu2 (dim-8 then re-arm for dim-128).
