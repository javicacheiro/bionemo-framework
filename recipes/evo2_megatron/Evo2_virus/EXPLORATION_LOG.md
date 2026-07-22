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
- **7B DONE: dim256 × α512 × do0.2 (ratio2) = 7B BEST → −23.32%, coverage 96.0%.** More capacity at
  ratio2 helps the 7B too (dim128→dim256: −21.80%→−23.32%, +1.52pp), just like on the 20B.
  `lora_run_7b_r2_dim256`.

## Phase 9 — 40B cross-scale point (two nodes) — the hypothesis breaks
Tested the 40B (Arc /data/evo2_40b_mbridge, TP4+vortex-FP8, dim256×do0.2, 1000 steps) to extend the
ratio-scaling law. FIRST pass picked high α by (wrong) extrapolation of "wider tolerates higher":
- **40B dim256×α2048 (local) → DIVERGED, in-train val PPL ~498.** (α2048 was only *mildly* broken on 20B, val ~3.6.)
- **40B dim256×α4096 (ohio) → DIVERGED, in-train val PPL ~507.**
→ **Both catastrophic — FAR more violent than the 20B at the same α. The 40B is MORE fragile to LoRA
scaling, not less.** The α-ceiling does NOT keep rising with width (7B~512, 20B~1024, but 40B < 2048 and
the divergence is explosive). Over-extrapolation corrected.
- **RELAUNCHED at conservative α to find the 40B healthy regime:** local **dim256×α1024** (20B-safe,
  ratio4) + ohio **dim256×α512** (7B-safe, ratio2). Brackets the real 40B ceiling. `lora_run_40b_a1024`,
  `lora_run_40b_a512`.
- **40B DONE (@1000 steps, vs 40B base 3.579):**
  - **α1024 (ratio4): −20.90%, cov 97.4%** (in-train val 2.631) ← 40B best
  - **α512 (ratio2): −20.23%, cov 97.9%** (in-train val 2.654)
  Both healthy; α1024 marginally better mean. vs 40B **dim16@1000 baseline −13.51%** → **+7.4pp** at
  matched steps (40B ran 1000 steps for time, not the 3000 of 7B/20B). → **40B ceiling ~α1024** (α2048
  explodes). `lora_run_40b_a1024` best.

## CROSS-SCALE SCALING LAW (3 points, all dim256 + do0.2 + all-5 targets; capped per-genome):
| model | α ceiling | best healthy config | overall | note |
|---|---|---|---:|---|
| 7B  | ~α512  | dim256×α512 (ratio2)  | −23.32% (@3000) | α1024 breaks |
| 20B | ~α1024-1536 | dim256×α1024 (ratio4-8) | −24.33% (@3000) | α2048 breaks (mild) |
| 40B | ~α1024 | dim256×α1024 (ratio4) | −20.90% (@1000) | α2048 EXPLODES (val ~500) |

**Law:** usable α **rises 7B→20B then plateaus ~α1024 by 40B** (NOT monotonic/unbounded with width);
**divergence past the ceiling gets MORE violent with scale** (7B/20B: gentle coverage collapse; 40B:
explosive val ~500). All scales want dim256 + dropout. **Practical recipe: dim256 + do0.2 + all-5, α
tuned per scale and capped ~α1024; be MORE conservative on bigger models** (opposite of the naive guess).

## Phase 10 — length across scales (autonomous, user away)
- **7B DONE: dim256×α512×do0.2 × 6000 steps → −24.44%, cov 93.8%, dsRNA −8.12%.** Length helps the 7B
  best (+1.12pp over @3000 −23.32%); coverage dips 96→93.8% (same length-overfit the 20B's do0.3 fixed).
  **7B@6k (−24.44%) now MATCHES the 20B best@3k (−24.33%)** — a well-tuned 7B + length ≈ the 20B.
  `lora_run_7b_best6k`.
- **ohio DONE: dim256×α512×do0.3 × 6000 → −15.84%, cov 80.7% — WORSE than do0.2@6k (−24.44%/93.8%)!**
  Dropout 0.3 HURTS the 7B (over-regularizes the smaller model), opposite of the 20B (where do0.3@6k
  helped → −25.44%). `lora_run_7b_best6k_do3`. → **Optimal dropout is scale-dependent: rises with model
  size** (7B wants do0.2, 20B wants do0.3 for long runs). **7B optimum = dim256×α512×do0.2×6000 = −24.44%.**
- **local DONE: 40B dim256×α1024×do0.2 × 3000 → DIVERGED** (in-train val ~3.9 from step500, never
  learned; scoring predict then errored on the broken adapter). **Same config was HEALTHY at 1000 steps
  (val 2.63, −20.90%)** — the difference is the SCHEDULE: decay over 3000 keeps LR higher longer, and
  the fragile 40B can't tolerate sustained higher LR at α1024. `lora_run_40b_a1024_3k`.
- **ohio (RUNNING): 40B dim256×α512×do0.2 × 3000 → HEALTHY** (val 2.94→2.80 dropping) — the more
  conservative α512 SURVIVES the 3000-step schedule → **this is the valid 40B step-match.** ETA ~23:20. `lora_run_40b_a512_3k`.
- **local (RUNNING): 40B dim256×α1024×do0.2 × 3000, lr 1.5e-4 (half)** — is the α1024@3000 divergence
  LR-driven (sustained high LR) or α-fundamental? If gentler LR rescues α1024@3000 → 40B just needs a
  lower LR for longer training (and may beat α512). `lora_run_40b_a1024_3k_lr15`.

## Phase 13 — B1: Arc-vs-NVIDIA 40B checkpoint (H200, vortex-FP8) [2 nodes]
Is the 40B "fragility" (α1024@3000 diverges) intrinsic or Arc-checkpoint-specific? Transferred the
NVIDIA (NeMo2) 40B mbridge to the H200s (B300→local→ohio via the claude-h200 key, authorized) and ran
the same configs the Arc (Savanna) 40B used.
- **NVIDIA-40B α1024×3000 (local) = HEALTHY → −24.14%** (base 3.587→2.721; in-train val 2.55→2.47 dropping).
  **The Arc-40B at the IDENTICAL config DIVERGED (val ~3.9).** → **The 40B fragility is Arc-checkpoint-
  specific, NOT architectural.** `lora_run_40b_nvH_a1024_3k`.
- NVIDIA-40B −24.14% (α1024) also EDGES Arc's best (α768 −23.95%) and nearly matches the 20B (−24.33%).
  The NVIDIA checkpoint is both **more robust AND slightly better** for viral LoRA.
- **ohio (RUNNING): NVIDIA-40B α768×3000** — identical-config head-to-head vs Arc α768 (−23.95%). `lora_run_40b_nvH_a768_3k`.
- **local DONE: NVIDIA-40B α1536×3000 = DIVERGED** (step-250 val 498.7, explosive). So the NVIDIA
  checkpoint, while stable at α1024 (Arc diverged there), STILL has a ceiling ~α1024 — α1536 explodes.
  → **NVIDIA-40B best = α1024 (−24.14%), nearly matches but does NOT beat the 20B (−24.33%).**
(B300 left off-limits throughout — busy with another agent's job.)

### B1 CONCLUSION: the NVIDIA (NeMo2) 40B is MORE ROBUST than the Arc (Savanna) 40B (stable at α1024
where Arc diverged) and slightly better (−24.14% vs Arc's −23.95%), but it has the same ~α1024 ceiling
(α1536 explodes) and still does not surpass the 20B (−24.33%). **20B remains the price/perf sweet spot;**
the Arc-40B's earlier fragility was partly a checkpoint artifact, but even the better NVIDIA-40B ≈ 20B, not >.
Remaining: ohio NVIDIA α768 (same-config vs Arc −23.95%) for the clean identical-config checkpoint delta.

## Phase 12 — context-length revisit at BEST config (dim256×α1024×do0.2, 20B) [2 H200 nodes]
Re-asked §3's "does long context help" at HIGH capacity (dim16 → dim256). Trained 32k-best (local,
3000 steps, TP2, val 2.406) and 128k-best (ohio, 1000 steps, TP4). Scored uncapped length-stratified
(base/16k-best/128k-best, PASS1 CP=1 short + PASS2 CP=8 long) on ohio; 32k-best capped on local.
- **32k-best capped = −24.39% ≈ 16k-best (−24.33%)** — context-neutral on the overall/capped metric (§3 holds at dim256).
- **16k-best uncapped = −24.72% overall, and STRONG on every length bucket incl. the long tail:**
  A≤8k −24.88 | B 8-16k −24.48 | C 16-32k −22.73 | **D 32-128k −24.06 | E >128k −21.75**. → **The 16k-trained
  adapter GENERALIZES to long genomes it never trained on** (−22 to −24% on >16k records at full length).
- **128k-best DIVERGED** (in-train val rose 3.49→3.84 > base 3.58; final loss 1.34 vs ~0.8 healthy; 0 NaN;
  uncapped +10% WORSE than base on every bucket incl. the ≤8192 control). dim256×α1024 is unstable at
  128k/TP4 — the same high-effective-magnitude fragility (α×LR / long-context). Not a usable adapter.

### CONTEXT CONCLUSION (dim256): 16k is the clear choice. The 16k-best adapter already handles the
long-genome tail (generalizes to >128k records), so long-context TRAINING adds no headroom — and at high
capacity it's counterproductive (128k training diverges). Stronger than §3b's dim16 verdict: not just
"context ≈ neutral" but "16k generalizes to long records AND long-context training is unstable at dim256."

## Phase 11 — B300 bf16 40B fragility test: BLOCKED by a stock-image bug (not a result)
Tried to test whether the 40B fragility is precision-specific by running the NVIDIA-40B in plain bf16
on a Blackwell B300 (α1024×3000). **Blocked:** the B300's STOCK `train_evo2` (image evo2:20260628,
without the modified Hopper source) throws `'Parameter' object has no attribute 'main_grad'` for
**dim≥128 LoRA in the plain-bf16 path** (dim16 works; dim128 AND dim256 fail, at both TP1 and TP4; not
memory — fails at 71 GB/GPU on TP4). The stock `train_evo2` also lacks the `--vortex-style-fp8` flag
(that's a Hopper-repo source modification). The Hopper env sidesteps the bug via the modified code +
vortex-FP8. Did NOT port the WIP code autonomously. B300 left provisioned (SSH key `claude-h200`,
container up, data + both 40B ckpts ready) for user-directed use. **The bf16-vs-FP8 40B fragility
question remains open** (needs the modified train_evo2 on the B300, or a Blackwell node with the repo).

### 40B fragility (key): the 40B is far more LR/α-sensitive than the 20B. α2048 explodes (val ~500);
α1024 is stable ONLY at 1000 steps (fast decay) and DIVERGES at 3000 (sustained LR); α512 is stable at
3000. → For the fragile 40B, step-matched/longer training needs LOWER α (and/or lower LR) than the 20B.

- **ohio DONE: 40B dim256×α512×do0.2 × 3000 = STEP-MATCH → −23.43%, coverage 94.7%** (val 2.544).
  Step-matched cross-scale @3000: **7B −23.32% | 20B −24.33% | 40B −23.43%.** → **The 40B does NOT beat
  the 20B** at matched steps — its fragility forces α512 (< 20B's α1024), capping it at ~20B/7B level.
  **20B remains the sweet spot.** `lora_run_40b_a512_3k`.
- **ohio DONE: 40B dim256×α768×do0.2 × 3000 = 40B OPTIMUM → −23.95%, cov 94.7%** (val 2.537). Stable
  where α1024 diverges; better than α512 (−23.43%). **40B step-matched optimum = α768, right at the
  stability edge.** Still < 20B (−24.33%) — nearly matches but doesn't beat it. `lora_run_40b_a768_3k`.
  40B α-ladder@3000: α512 −23.43% | **α768 −23.95% (peak, edge of stability)** | α1024 diverges.
- **local DONE: 40B dim256×α1024×do0.2 × 3000, lr1.5e-4 → −22.87%, cov 95.0%** (val 2.488, best in-train
  of any 40B). **LR-rescue CONFIRMED: half-LR makes α1024 stable at 3000** (vs diverged at lr3e-4) → the
  divergence is LR-driven, not α-fundamental. BUT the capped score (−22.87%) is WORSE than α512@lr3e-4
  (−23.43%) — the lower LR trades adaptation for stability. **40B best@3000 stays α512 = −23.43%; still
  < 20B.** `lora_run_40b_a1024_3k_lr15`. (auto-score hit a GPU-teardown race; re-scored clean.)
- **local DONE: 20B dim256×α2048×do0.2 × 3000, lr1.5e-4 → −24.01%, cov 94.7%** (val 2.529). Half-LR
  RESCUES the 20B's α2048 (broke to +2.77% at lr3e-4) — but the rescued config (−24.01%) does NOT beat
  the α1024@lr3e-4 peak (−24.33%). `lora_run_20b_a2048_lr15`.

## α×LR — UNIFIED CONCLUSION (no free lunch). The "α ceiling" is an α×LR (effective-update-magnitude)
ceiling: high α breaks at standard LR but is RESCUED by halving LR — yet the rescued high-α/low-LR
config NEVER beats the standard-α/standard-LR peak (lower LR trades away exactly the adaptation the
higher α buys). Same at both scales:
| scale | peak (std LR) | high-α rescued (½ LR) | high-α (std LR) |
|---|---|---|---|
| 20B | α1024@3e-4 **−24.33%** | α2048@1.5e-4 −24.01% | α2048@3e-4 BROKE (+2.77%) |
| 40B | α512@3e-4 **−23.43%** | α1024@1.5e-4 −22.87% | α1024@3e-4 BROKE (diverged) |
→ You cannot unlock a higher peak by pushing α + lowering LR. The peak is set by the standard-LR α ceiling.

## 7B THREAD COMPLETE — recipe transfers, both sizes want dim256, differ only in ratio.
7B best = dim256×α512 (ratio2) do0.2 = **−23.32%/96%**; 20B best (3k) = dim256×α1024 (ratio4-8) do0.2 =
−24.33%/94.6%. **Same dim256 capacity + dropout0.2; only the α/dim ratio scales with width (7B→2, 20B→8).**
7B (−23.32%) nearly matches 20B (−24.33%) at 3k. Both servers now idle — adapter frontier fully mapped
across two scales. Open: 40B scaling-law point (expensive, offered to user); downstream eval; leakage audit.

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
