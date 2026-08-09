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

## Phase 14 — downstream generality (influenza HA) + B300 full fine-tune [3 nodes]
- **Influenza HA DMS generality (local, `DOWNSTREAM_HA_EVAL.md`):** the SARS-CoV-2 downstream edge
  TRANSFERS to a 2nd virus family, weaker. LoRA beats base on both strains (all significant):
  **H1N1(WSN) Spearman 0.10→0.29, AUROC 0.54→0.65; H3N2(Perth) −0.01→0.10, AUROC 0.49→0.55.** Magnitude
  decays with divergence: SARS-CoV-2 RBD 0.37–0.44 > H1N1 0.29 > H3N2 0.10. → Real, transferable gain
  (not single-virus), but virus-dependent, not uniformly large.
- **B300 full 40B fine-tune (GPUs 1-4, bf16, no LoRA, lr 2e-6, 1000 steps):** in-train val bottomed
  ~3.09 (step 800) then ROSE (3.11→3.13) = mild overfit knee. Improves over base (~3.6) but FAR worse
  than LoRA (val ~2.4–2.5). → **Full fine-tuning is the WRONG tool for this 86M-token corpus** — it
  underperforms LoRA badly and overfits; LoRA (small adapter) is right for a small corpus.
  **Capped: −11.20%** (base 3.647→3.239) — WORSE than even the dim16 LoRA baseline (−13.51%) and <½ the
  LoRA best (−25.44%). Full-FT is decisively the wrong tool for the 86M-token corpus.
  (GPU0 = another agent's vLLM, untouched throughout.) `fullft_40b_nv_b300`.

## Phase 13 — B1: Arc-vs-NVIDIA 40B checkpoint (H200, vortex-FP8) [2 nodes]
Is the 40B "fragility" (α1024@3000 diverges) intrinsic or Arc-checkpoint-specific? Transferred the
NVIDIA (NeMo2) 40B mbridge to the H200s (B300→local→ohio via the claude-h200 key, authorized) and ran
the same configs the Arc (Savanna) 40B used.
- **NVIDIA-40B α1024×3000 (local) = HEALTHY → −24.14%** (base 3.587→2.721; in-train val 2.55→2.47 dropping).
  **The Arc-40B at the IDENTICAL config DIVERGED (val ~3.9).** → **The 40B fragility is Arc-checkpoint-
  specific, NOT architectural.** `lora_run_40b_nvH_a1024_3k`.
- NVIDIA-40B −24.14% (α1024) also EDGES Arc's best (α768 −23.95%) and nearly matches the 20B (−24.33%).
  The NVIDIA checkpoint is both **more robust AND slightly better** for viral LoRA.
- **ohio DONE: NVIDIA-40B α768×3000 = −24.10%** (base 3.5865→2.7220; val 2.559). **Identical-config
  head-to-head: NVIDIA −24.10% > Arc −23.95% (+0.15pp).** So at the SAME config the NVIDIA checkpoint is
  marginally better; plus it's stable at α1024 (−24.14%) where Arc diverges. `lora_run_40b_nvH_a768_3k`.

### B1 FINAL: NVIDIA (NeMo2) 40B is consistently slightly better AND more robust than Arc (Savanna) 40B
— same-config α768: −24.10% vs −23.95%; NVIDIA usable at α1024 (−24.14%, Arc diverges there). Both share
the ~α1024 ceiling (α1536 explodes) and both ≈ the 20B (−24.33%), neither beats it. **20B remains the
price/perf sweet spot.** The Arc-40B's earlier fragility/underperformance was partly a checkpoint artifact.
- **local DONE: NVIDIA-40B α1536×3000 = DIVERGED** (step-250 val 498.7, explosive). So the NVIDIA
  checkpoint, while stable at α1024 (Arc diverged there), STILL has a ceiling ~α1024 — α1536 explodes.
  → **NVIDIA-40B best = α1024 (−24.14%), nearly matches but does NOT beat the 20B (−24.33%).**
(B300 left off-limits throughout — busy with another agent's job.)

### B1 CONCLUSION: the NVIDIA (NeMo2) 40B is MORE ROBUST than the Arc (Savanna) 40B (stable at α1024
where Arc diverged) and slightly better (−24.14% vs Arc's −23.95%), but it has the same ~α1024 ceiling
(α1536 explodes) and still does not surpass the 20B (−24.33%). **20B remains the price/perf sweet spot;**
the Arc-40B's earlier fragility was partly a checkpoint artifact, but even the better NVIDIA-40B ≈ 20B, not >.
Identical-config delta now closed: ohio NVIDIA α768 = −24.10% vs Arc α768 = −23.95% (+0.15 pp) — see B1 FINAL above.

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

## Phase 14 — reverse-complement (RC) augmentation [2 H200 nodes, 2026-07-24/25]
The prior dsRNA lever (token-reweighting) BACKFIRED (+5.28%, memorised the tiny 1574-record set).
The writeup's own conclusion was "needs **more unique dsRNA data**, not reweighting." Untried lever:
**reverse-complement augmentation** — the opposite strand read 5'->3' is a biologically valid, DISTINCT
example (IUPAC-aware RC via `/data/viral/rc_augment.py`; palindrome + ambiguity-code self-tests pass).
Two complementary runs, both 20B dim256xa1024 seq16384 vortex-FP8, vs the standard capped-8192 base
(base 3.579), scored per-genome.

- **local: dsRNA-only RC-doubling** (add RC of the 1574 dsRNA recs -> corpus 12944->14518; dsRNA unique
  windows 2x, no artificial reweight). do0.2 x3k = **OVERALL -24.28% (== baseline -24.33%), dsRNA -8.28%**,
  cov 94.4%. dsRNA BEATS every plain-3k baseline (do0.1 -7.87%, do0.2/do0.3 ~-7.67%) at ZERO overall cost.
  KEY: RC augmentation (real unique data) HELPS dsRNA where reweighting (repetition) hurt it — validates
  the "more unique data" hypothesis. `lora_run_20b_16k_dsRNArc_d256a1024do2_3k`, `viral_dataset_dsRNArc.yaml`.
- **ohio: whole-corpus RC-doubling** (all 12944 recs + their RC -> 25888). do0.2 x3k = OVERALL -22.40%
  (< baseline -24.33%) but **cov 97.0%** (highest of any 3k run). UNDERFIT-but-broader: doubling the corpus
  halves epochs/step, so at a fixed 3k budget each window is fit less (broad, shallow). `..._fullrc_...`.

Follow-ups (running, ~15h each, results ~2026-07-25 PM):
- local: dsRNA-RC at CHAMPION config **do0.3 x6k** — best shot at beating -25.44% overall AND pushing
  dsRNA past the -9.69% (plain do0.3x6k) / -9.78% (plain do0.2x6k) ceiling.
- ohio: whole-corpus RC **do0.2 x6k** — same-compute vs plain do0.2x6k (-24.96%): does the extra unique
  data recover/surpass the plain run at matched steps, or does 2x-data underfit persist?

### Phase 14 — 6k results + conclusion (2026-07-25)
- **local: dsRNA-RC do0.3 x6k = NEW BEST -> OVERALL -25.51% (>= champion -25.44%), dsRNA -10.02%** (first
  sub--10%, vs plain champion -9.69%), cov 94.1%, in-train val PPL 2.489. Targeted RC of the thin class
  improves dsRNA at zero overall cost (+12% data, same compute). `lora_run_20b_16k_dsRNArc_d256a1024do3_6k`.
- **ohio: whole-corpus RC do0.2 x6k = -24.21%** (< plain-6k do0.2 -24.96%), dsRNA -6.56%, cov 96.5%.
  Confirms the underfit diagnosis: 2x corpus at 6k ~= 9 epochs ~= plain-3k (-24.33%). Matching plain-6k
  epochs needs ~12k steps (~30h) -> not worth it. Whole-corpus RC = broad-not-deep; NEGATIVE for the mean.
- CONCLUSION: **targeted RC (add unique strands only where the corpus is thin) is the right form of RC**;
  whole-corpus doubling only dilutes epochs. dsRNA-RC is a free recipe add-on (dsRNA -9.69->-10.02%).
- Round 3 (running, 2026-07-25 PM): does RC-doubled unique dsRNA now RESCUE token-upweighting (which
  backfired at +5.28% with 1x unique data)? local = dsRNAboth upweight->10%, ohio = ->6%, both do0.3x6k.
- OPEN (proposed code change, left for user review): no train-time per-epoch RC exists (both preprocess
  flags are static: `random_reverse_complement` bakes one fixed flip, `embed_reverse_complement` doubles
  on disk). A ~20-line stochastic RC in `Evo2Dataset` (byte-level tokenizer -> clean complement LUT
  65<->84, 67<->71, +lowercase/ambiguity, reversed) would give per-epoch RC WITHOUT corpus-doubling /
  epoch-dilution -- the theoretically cleanest win, but it touches the training data path (label/loss-mask
  alignment) so it needs a smoke test, not a blind overnight edit.

### Phase 14 — Round 3: RC RESCUES upweighting (2026-07-26)
The reweight-backfire was a UNIQUE-DATA shortage, not a flaw in reweighting. With dsRNA RC-doubled
(3148 unique windows via both strands), upweighting no longer memorises:
- **local: dsRNAboth up10 (dsRNA->10% tokens) do0.3x6k = NEW BEST -> OVERALL -25.56%, dsRNA -13.13%**,
  cov 93.1%. vs the FAILED 1x-unique up10 (dsRNA +5.28%): RC-doubling flips a +5.28% backfire into a
  -13.13% gain at the SAME 10% sampling rate. dsRNA jumps -10.02% (natural RC) -> -13.13% (RC+up10).
  `lora_run_20b_16k_dsRNAboth_up10_d256a1024do3_6k`.
- ohio: dsRNAboth up6 (dsRNA->6%) do0.3x6k -- running (low-end bracket).
- local: dsRNAboth up15 (dsRNA->15%) do0.3x6k -- running (high-end bracket; ~4.6 dsRNA-epochs, testing
  where upweight re-breaks now that unique data is 2x).
dsRNA ladder (do0.3x6k): plain -9.69% | RC-natural(~4.8%) -10.02% | RC+up10 **-13.13%** | up6/up15 pending.

- **ohio: dsRNAboth up6 (dsRNA->6%) do0.3x6k = BEST OVERALL -> -25.74%**, dsRNA -11.79%, cov 93.8%.
  Beats up10 (-25.56%) on overall and the prior champion (-25.44%) by +0.30pp (above noise).
  TRADEOFF: **6% maximizes OVERALL (-25.74%), 10% maximizes dsRNA (-13.13%)** -- harder dsRNA upweight
  helps dsRNA but slightly starves the other classes. `lora_run_20b_16k_dsRNAboth_up6_d256a1024do3_6k`.
  dsRNA ladder (do0.3x6k): plain -9.69 | RC-nat -10.02 | up6 -11.79 | up10 **-13.13** | up15 pending.
  overall ladder: champion -25.44 | RC-nat -25.51 | up10 -25.56 | **up6 -25.74** | up15 pending.

### Phase 14 — DOWNSTREAM caveat: dsRNA-upweight trades ssRNA(+) transfer (2026-07-26)
Scored up6 (best aggregate PPL, -25.74%) on the SARS-CoV-2 RBD DMS variant-effect benchmark (ssRNA(+))
vs the old -25.44% plain adapter (same base, same analyze.py/all.fasta -- base numbers match to the digit):
| metric | base | old -25.44% | up6 |
|---|---|---|---|
| Spearman bind (all) | -0.012 | 0.368 | 0.325 |
| Spearman bind (single-nt) | 0.017 | 0.437 | 0.361 |
| Spearman expr (all) | 0.004 | 0.351 | 0.347 |
| AUROC bind (all) | 0.473 | 0.659 | 0.635 |
| AUROC bind (single-nt) | 0.496 | 0.724 | 0.670 |
up6 is WORSE than the plain adapter on 7/8 downstream metrics despite BETTER aggregate held-out PPL.
INTERPRETATION: upweighting dsRNA (6% of sampling) diverts from the dominant ssRNA(+) class, so the
aggregate-PPL win masks a per-class trade that shows up on the ssRNA(+) downstream task. **Aggregate PPL
is not the whole story -- dsRNA-focused upweighting can hurt downstream generalization on other classes.**
NEXT: score natural-RC (dsRNArc, no upweight, dsRNA -10.02%) downstream -- does RC WITHOUT upweight keep
the dsRNA PPL gain WITHOUT the ssRNA(+) downstream cost? (That would make natural-RC the clean recipe add-on.)
`downstream_results_up6.json`.

### Phase 14 — DECISIVE: RC helps PPL but NOT downstream (PPL-orthogonal) (2026-07-26)
Scored natural-RC (dsRNArc, no upweight, dsRNA -10.02%) on the SARS-CoV-2 RBD DMS (ssRNA(+)) task:
| metric | base | plain -25.44% | up6 | natural-RC |
|---|---|---|---|---|
| Spearman bind (all) | -0.012 | 0.368 | 0.325 | 0.295 |
| Spearman expr (all) | 0.004 | 0.351 | 0.347 | 0.269 |
| AUROC bind (all) | 0.473 | 0.659 | 0.635 | 0.631 |
| AUROC expr (all) | 0.498 | 0.690 | 0.678 | 0.646 |
BOTH RC adapters are WORSE downstream than the PLAIN champion; upweighting is not the cause (natural-RC,
which upweights LEAST, is the worst). THE KEY: natural-RC and plain have **IDENTICAL ssRNA(+) held-out PPL**
(-21.53% vs -21.54%) yet natural-RC scores far lower downstream (0.295 vs 0.368) -> the regression is
**ORTHOGONAL to PPL** (natural-RC == plain on ssRNA(+) PPL yet worse downstream). (A "RC teaches
strand-invariance" guess was made here and LATER REFUTED in Phase 15.) **Held-out PPL (even per-class)
is NOT a sufficient proxy for downstream utility.** Caveat: single-seed downstream runs; gap is ~4 sampling-SE +
a training-seed component, but consistent across 8 metrics x 3 adapters.

### RC AUGMENTATION — FINAL VERDICT
- **In-distribution PPL:** RC is a real win. Targeted dsRNA-RC + light upweight -> dsRNA -9.69%->-13.13%,
  aggregate -25.44%->-25.74%. Whole-corpus RC-doubling underfits (negative). Reweighting alone backfires
  (+5.28%); RC rescues it (more UNIQUE data, not repetition).
- **Downstream transfer:** RC does NOT help and mildly HURTS (PPL-orthogonal, above). So the RC PPL gain
  is metric-local, not a generalization gain.
- **RECOMMENDATION: keep the PLAIN champion (dim256xa1024xdo0.3x6k, -25.44%) as the production recipe --
  it is the best downstream.** Use dsRNA-RC ONLY when in-distribution dsRNA likelihood is itself the
  deliverable (dsRNA generation/scoring), never as a blanket upgrade, and document the downstream caveat.
`downstream_results_natrc.json`.

### Phase 14 — upweight bracket complete (up15, 2026-07-27)
- **local: dsRNAboth up15 (dsRNA->15%) do0.3x6k = -25.33% overall (BELOW plain -25.44%!), dsRNA -13.43%**,
  cov 94.3%. Completes the bracket:
  | dsRNA upweight | overall | dsRNA |
  |---|---|---|
  | natural ~4.8% | -25.51% | -10.02% |
  | up6 | **-25.74%** (overall peak) | -11.79% |
  | up10 | -25.56% | -13.13% |
  | up15 | -25.33% (< plain -25.44%) | -13.43% (dsRNA saturates) |
  PATTERN: overall peaks ~6% then declines (heavy dsRNA upweight starves the other classes -> up15 drops
  below the plain champion); dsRNA improves monotonically but SATURATES ~-13.4% by up10-15 (only 3148
  unique dsRNA windows even with RC). Confirms up6 = overall-optimal upweight; and that (with the downstream
  caveat) aggressive dsRNA upweighting is counterproductive for generalization. **RC study CLOSED.**

## Phase 15 — PPL vs downstream QUANTIFIED + mechanism tested (2026-08-02) [2 H200 nodes]
Scored 11 adapters on the SARS-CoV-2 RBD DMS variant-effect task (ssRNA(+)); held-out PPL from the
per-genome capped scores. Downstream = Spearman(evo2_delta, DMS bind_avg, all 3802 muts):
| adapter | held-out PPL | ds Spearman | ds AUROC | family |
|---|---:|---:|---:|---|
| dim64 do0.1x3k | -20.09% | 0.329 | 0.646 | plain |
| dim256 do0.1x3k | -21.29% | 0.394 | 0.676 | plain |
| **dim256_a1024 do0.1x3k** | -23.48% | **0.415** | 0.684 | plain (BEST downstream) |
| do0.3x3k | -24.28% | 0.385 | 0.668 | plain |
| do0.2x6k | -24.96% | 0.376 | 0.684 | plain |
| CHAMPION do0.3x6k | -25.44% | 0.368 | 0.659 | plain |
| fullrc do0.2x6k | -24.21% | 0.329 | 0.652 | RC |
| up15 | -25.33% | 0.355 | 0.647 | RC |
| natRC | -25.51% | 0.295 | 0.631 | RC |
| up10 | -25.56% | 0.343 | 0.651 | RC |
| up6 | -25.74% | 0.325 | 0.635 | RC |

corr(held-out-PPL magnitude, downstream Spearman): **ALL 11 rho=-0.50; PLAIN past the -23.5% peak
(n=4) rho=-1.00** (perfectly monotonic decline). **BEST downstream = dim256_a1024 do0.1x3k (0.415), a
LIGHTLY-trained adapter -- the PPL champion (-25.44%) scores only 0.368.** Every PPL-lowering lever we
used (more steps, higher dropout, RC augmentation) HURT downstream past the peak.
**HEADLINE (strong form) [!! REFUTED BY RESEED -- see Phase 15-RESEED below !!]: single-seed scatter
suggested held-out PPL and downstream are ANTI-CORRELATED (lightly-trained best). This DID NOT REPLICATE
across seeds -- it was seed noise. Read the Phase 15-RESEED correction before citing this.**

MECHANISM TESTED + **REFUTED**: hypothesis was "RC teaches strand-invariance -> dilutes directional
sensitivity." Strand-symmetry probe = correlation of an adapter's per-window logprob(seq) vs
logprob(reverse-complement(seq)) over the 4020 DMS windows:
| adapter | pearson(fwd,rc) | mean|fwd-rc| |
|---|---:|---:|
| champion (plain) | -0.259 | 0.911 |
| natRC | -0.304 | 0.903 |
| up6 | -0.019 | 0.863 |
ALL adapters stay strongly strand-ASYMMETRIC; RC did NOT make them symmetric, and natRC (WORST
downstream) is the MOST asymmetric -- the OPPOSITE of the hypothesis. **Strand-invariance is NOT the
mechanism (withdrawn).** The robust result is purely empirical (the anti-correlation above), and it is
NOT RC-specific -- plain over-training hurts downstream too. Likely real mechanism (open): over-fitting
the corpus PPL objective washes out the per-position sensitivity variant-effect scoring relies on.
CAVEATS: single benchmark (ssRNA(+) RBD); single seed per adapter (reseed running to confirm the
peak-vs-champion gap survives seed noise). The plain-past-peak rho=-1.00 across 4 points is suggestive
even at one seed. Data: `/data/viral/downstream/ds_scatter_*.json`, `strand_sym.py`.

## Phase 15-RESEED — the anti-correlation DOES NOT REPLICATE (2026-08-03) [CORRECTION]
Reseeded the two do0.3 endpoints at seed 2345 (dropout controlled; both trained healthy, 0 nan, PPL
reproduced -- champion s2345 held-out -25.50% ~= -25.44%). Downstream Spearman(bind_all):
| config | seed 1234 | seed 2345 | mean | spread |
|---|---:|---:|---:|---:|
| champion do0.3x6k | 0.368 | 0.371 | 0.370 | 0.003 (STABLE) |
| light do0.3x3k | 0.385 | 0.289 | 0.337 | 0.096 (HUGE) |
The ordering FLIPS between seeds: at s1234 light(3k) 0.385 > champion(6k) 0.368; at s2345 champion 0.371 >
light 0.289. **The lightly-trained adapter has ~0.10 seed variance; the champion is seed-stable (~0.003).**
=> The Phase-15 single-seed scatter (rho -1.0 "past peak") was DRIVEN BY SEED NOISE in the under-trained
points, NOT a real PPL->downstream anti-correlation. Averaged over seeds the champion (0.370) actually
BEATS the 3k (0.337). CORRECTED CONCLUSIONS:
1. **NO reliable PPL<->downstream anti-correlation** -- the single-seed differences (range 0.29-0.42) are
   within the per-config seed noise (~+-0.05-0.10 Spearman). Single-seed downstream rankings of these
   adapters are UNDERPOWERED (this also weakens the earlier Phase-14 "RC hurts downstream" claim -- same
   single-seed caveat).
2. **What IS robust:** (a) the champion (well-trained do0.3x6k) is downstream-STABLE and competitive; more
   training -> MORE seed-reliable downstream, not worse. (b) low-dropout do0.1 is FRAGILE (diverged at
   seed 2345: loss jumped 1.06->1.37 at iter 200, downstream 0.003). (c) the plain champion remains the
   safe production recipe on BOTH held-out PPL and downstream stability.
LESSON: downstream variant-effect eval on a single benchmark + single seed is noisy; need >=3 seeds to
rank adapters. The exciting single-seed result did not survive replication -- reported honestly.

## Phase 16 — FULL-CORPUS production run: equal in-corpus, WORSE out-of-corpus (2026-08-04/05) [h200-2]
Trained the champion recipe on the **whole corpus** (train.fasta 12,944 + valid.fasta 1,348 = **14,292
records**), i.e. no held-out set — the standard "final production model" move once hyperparameters are
locked. **Hyperparameters byte-identical to `RECIPE.sh` §1** (20B vortex-FP8, dim256 × α1024 × do0.3,
seq16384, mbs1/gbs16, recompute-1, 6000 steps, lr 3e-4→3e-5, warmup 10, decay 6000); ONLY the train
split changed. Merge verified lossless: preprocessed `.bin` = 188,461,996 B = exactly
171,803,928 (train) + 16,658,068 (valid). Ran clean: **0 nan / 0 skipped over all 6000 steps**,
9.17 s/step, ~397 TFLOP/s/GPU, 15.6 h, LR landed exactly on the 3e-5 floor. Step-20 sanity vs the
champion's own step 20 was near-identical (lm loss 1.17769 vs 1.17844; params norm 2240.927 both).

**In-train PPL is NOT comparable and must not be quoted as a result.** valid is now inside train, so the
eval is a training-fit monitor: 2.654 (500) → 2.512 → 2.430 → 2.412 → 2.292 → 2.199 → 2.131 → 2.118 →
2.072 → 2.044 → 1.906 → **1.908 (6000)**; final validation-set 1.983 / test-set 1.976. For scale the
champion's *held-out* curve ended at 2.411 — the full-corpus run passes that value by step ~2000. Same
reason the −25.44% capped headline **cannot** be restated for this adapter: `valid_cap8192` is trained-on.

### Downstream (the only valid read) — base vs champion vs full-corpus
Scored with the SAME `pred_base` and the SAME settings as every other adapter (mbs 1, bf16_mixed +
vortex-style-fp8) so the numbers are directly comparable.

| benchmark | metric | base | champion | **full-corpus** |
|---|---|---:|---:|---:|
| SARS-CoV-2 RBD (**in-corpus**) | Spearman bind all | −0.012 | 0.368 / 0.371 | **0.372** |
|                                | Spearman bind single-nt | 0.017 | — | 0.460 |
|                                | Spearman expr all | 0.004 | — | 0.401 |
| HA WSN H1N1 (**out-of-corpus**) | Spearman all | 0.097 | **0.286** | 0.204 |
|                                 | Spearman single-nt | 0.091 | **0.291** | 0.229 |
|                                 | AUROC median / ≤−2 | 0.543/0.549 | **0.646/0.660** | 0.601/0.623 |
| HA Perth H3N2 (**out-of-corpus**) | Spearman all | −0.015 | **0.095** | 0.041 |
|                                   | Spearman single-nt | 0.008 | **0.128** | 0.063 |
|                                   | AUROC median / ≤−2 | 0.487/0.486 | **0.545/0.548** | 0.519/0.517 |

**RBD: does not distinguish the configurations at all** — and the story here changed twice as seeds
accumulated, which is itself the lesson. At n=1 it looked identical; at n=2 the full-corpus arm looked
uniquely noisy (spread 0.056 vs the champion's 0.003); at **n=3 the champion is the noisier of the two**
(0.368/0.371/**0.293**, spread **0.078**) and the means are indistinguishable (0.3440 vs 0.3442).
See "Phase 16 — RBD VARIANCE CORRECTION" below. Treat RBD as uninformative for this comparison.
**HA: worse on all 8 metrics, both strains** — −0.081 Spearman on WSN (−28% rel) and −0.054 on Perth
(−57% rel). Still far above base everywhere, so the adapter learned real signal; it just transfers less.

### In-corpus vs out-of-corpus is the axis (leakage audit extension)
Checked the benchmark organisms against the training corpus directly: **`MN908947` (SARS-CoV-2
Wuhan-Hu-1) IS in `train.fasta`** (1 record, absent from valid), while **influenza HA is absent from both
splits** — no accession hit (`NC_002023`, `NC_007366`, `CY147323`, `AF389118`) and no exact 90-mer match
from either the WSN or Perth HA reference CDS. Consequences:
1. **RBD was never truly zero-shot** — the champion's published 0.368/0.371 carries this caveat too. It is
   a *within-corpus* variant-effect readout.
2. **HA is the genuinely out-of-corpus generality test** — but see the k-mer correction below: only the
   **Perth (H3N2)** strain is clean; WSN (H1N1) shares exact 60-mers with a homologous H1 strain that IS in
   the corpus. Cite **Perth** for "generalizes to unseen viruses", not "HA" broadly.
3. Neither organism is in the 1,348 newly-added records → the champion-vs-full-corpus comparison is **not**
   confounded by differential benchmark leakage. (Caveat: exact-substring matching would miss a divergent
   HA strain; read as "no evidence of presence," not proof of absence.)

### Over-training is NOT the explanation (arithmetic rules it out)
Tempting first guess, but wrong: at a fixed 6000 steps × gbs 16 the run consumes 96,000 × 16,384 =
**1.573 B tokens** regardless of corpus size, so
- champion:     1.573 B / 171.80 M = **9.15 epochs**
- full-corpus:  1.573 B / 188.46 M = **8.35 epochs**

More data at fixed steps = **fewer** passes per record. The full-corpus model is *less* over-trained yet
transfers *worse*, so "it over-fit" cannot be the mechanism, and a checkpoint sweep over this run's
retained `iter_*` would be testing a hypothesis the arithmetic already disfavours.

### Working hypothesis: corpus specialization (TENTATIVE, single seed)
The pattern that fits is **specialization to this corpus**: unchanged on the benchmark whose organism is
*inside* the corpus, degraded on both whose organisms are *outside* it. Adding 10.4% more of the same
corpus appears to buy in-distribution fit at the cost of transfer to distant viruses.
**NOT ESTABLISHED — single seed.** Per Phase 15-RESEED the champion is seed-stable on RBD (sd 0.003) and
our full-corpus RBD reproduces that, but **HA has never been reseeded**, so its per-config seed variance is
unknown and the −0.05/−0.08 gaps are not demonstrably outside a plausible noise band. A full-corpus
**reseed (seed 2345) scored on HA** is the discriminating experiment and is what settles this.
Seed bookkeeping (verified in both config dumps): the champion's original and the full-corpus run BOTH
used `seed: 1234`, so the comparison above is **initialization-matched** — but not draw-matched, since a
different corpus necessarily yields a different data order. `seed: 42` appears as a separate unchanged
sub-config in every run.

### Phase 16-RESEED attempt 1 (seed 2345) — DIVERGED. Full-corpus is the LESS STABLE variant.
`reseed_full_corpus_s2345.sh` (identical to `RUN_full_corpus.sh` except `--seed 2345`) **collapsed at
~iter 100–300** and was killed at iter 300 rather than burn 15.6 h. Trajectory
(`train_prod_full_corpus_s2345.DIVERGED.log`):

| iter | lm loss | grad norm |
|---:|---:|---:|
| 50  | 1.156 | 0.097 |
| 100 | 1.111 | **0.758** |
| 150 | **1.378** | 0.801 |
| 200 | 1.369 | 0.019 |
| 250 | **3.211** | **0.000** |
| 300 | **6.220** | **0.000** |

Grad norm spiked ~8×, loss inverted, then grad norm pinned at exactly 0.000 while loss ran away; val PPL
at iter 250 = **502** vs base ~3.5, and by iter 300 train loss (6.2199) had converged to val loss (6.2187)
— the adapter collapsed to a constant output. **0 nan / 0 skipped throughout**, so this is a genuine
dead-adapter collapse, NOT a masked numerical fault. Healthy runs at the same point hold grad norm ~0.1
with loss declining (cf. `train_light_do3_3k_seed2345.log`: 1.163→1.015, gn 0.095→0.151).

**The configuration matrix is the finding:**
| data | seed 1234 | seed 2345 |
|---|---|---|
| train-only (champion) | healthy, −25.44% | healthy, −25.50% |
| **full corpus** | **healthy** | **DIVERGED** |

Train-only survives both seeds; full-corpus survives 1234 but not 2345. So **full-corpus is measurably
less stable**, which is a *second, independent* mark against it alongside the weaker HA transfer. This fits
the α×LR / effective-update-magnitude fragility theme (see "α×LR — UNIFIED CONCLUSION"): the recipe sits
near a stability edge and enlarging the corpus nudges it closer. Caveat: n=1 divergence — one unlucky draw
is not ruled out.

Consequence for the seed question: a diverged run settles NOTHING about the HA gap, so it does not count as
the second seed. Diverged run kept as evidence: `PROD_20b_full_corpus_s2345/` (9.6 GB) + `*.DIVERGED.log`.

### Phase 16-RESEED attempt 2 (seed 3456) — HEALTHY, and the HA gap REPLICATES (8/8 metrics disjoint)
`reseed_full_corpus_s3456.sh` ran clean: 0 nan / 0 skipped, 9.18 s/step, LR to the floor, final fit PPL
1.947 (val-set 1.912 / test-set 1.927) vs s1234's 1.908 (1.983/1.976) — **seed spread on the FIT metric is
only ~0.04–0.06 PPL**, and the two curves tracked within ~0.005 at 1500/2500/3000/4500. Grad norm held
~0.10–0.20 throughout; it sailed through iters 100–300 where s2345 died (gn 0.098/0.141/0.126/0.149/0.141),
confirming the s2345 divergence was **seed-specific, not inherent to the full-corpus config**.

**A NECESSARY FIX TO THE COMPARISON:** HA had only ever been scored for ONE champion seed (s1234), so two
full-corpus seeds vs one champion point would have compared a spread to a point. The champion s2345 adapter
already existed from Phase 15-RESEED, so it was scored on HA for ~50 min of predict, no training
(`QUEUE_ha_champion_s2345.sh` → `{wsn,perth}_results_champion_s2345.json`). That gives a true **2 v 2**.
(Its original RBD predictions did NOT survive the h200 decommission — only the published 0.371 did — a
small argument for retaining adapters, not just their scores.)

HA, 2 seeds per config (champion s1234/s2345 vs full-corpus s1234/s3456):

| metric | champion mean (spread) | full-corpus mean (spread) | gap | ranges | champ wins |
|---|---:|---:|---:|:--:|:--:|
| WSN spearman all      | 0.2687 (0.0336) | 0.2187 (0.0289) | +0.0500 | DISJOINT | 4/4 |
| WSN spearman single-nt| 0.2691 (0.0433) | 0.2362 (0.0136) | +0.0329 | DISJOINT | 4/4 |
| WSN auroc median      | 0.6362 (0.0186) | 0.6077 (0.0138) | +0.0285 | DISJOINT | 4/4 |
| WSN auroc ≤−2         | 0.6479 (0.0239) | 0.6269 (0.0086) | +0.0210 | DISJOINT | 4/4 |
| Perth spearman all    | 0.1142 (0.0381) | 0.0531 (0.0246) | +0.0611 | DISJOINT | 4/4 |
| Perth spearman single-nt| 0.1444 (0.0337) | 0.0778 (0.0306) | +0.0666 | DISJOINT | 4/4 |
| Perth auroc median    | 0.5555 (0.0215) | 0.5252 (0.0129) | +0.0304 | DISJOINT | 4/4 |
| Perth auroc ≤−2       | 0.5595 (0.0237) | 0.5240 (0.0134) | +0.0355 | DISJOINT | 4/4 |

**On all 8 HA metrics the two configs' seed ranges do not overlap and the champion wins every one of the 4
pairwise seed comparisons.** Within-config HA seed spread is ~0.01–0.04; the between-config gaps are
0.02–0.07, i.e. larger. Both configs move around between seeds (champion WSN 0.286→0.252, Perth
0.095→0.133 — note it went DOWN on WSN and UP on Perth), so this is not one config being noisy: the
**ordering is invariant**. => **The out-of-corpus transfer deficit REPLICATES.** As strong as n=2 permits.

RBD (in-corpus), 2 seeds per config — ranges OVERLAP, champion wins only 2/4 pairwise, so **RBD is
UNRESOLVED at n=2**. At the time the full-corpus arm looked uniquely noisy (spread 0.0557 vs the
champion's 0.0030, "~19×"). **THAT READING DID NOT SURVIVE A THIRD CHAMPION SEED — see the RBD
VARIANCE CORRECTION section below. Do not cite the 19× figure.**

### Phase 16 — RBD VARIANCE CORRECTION (2026-08-08): the champion is NOT RBD-stable either
Champion seed 3456 (trained clean: 0 nan, held-out PPL 2.450, entirely normal) scored RBD Spearman
bind_all = **0.2930**, far below s1234 (0.368) and s2345 (0.371):

| config | seeds | values | mean | spread |
|---|---:|---|---:|---:|
| champion | n=3 | 0.3680 / 0.3710 / **0.2930** | 0.3440 | **0.0780** |
| full-corpus | n=2 | 0.3720 / 0.3163 | 0.3442 | 0.0557 |

**The champion is now the NOISIER arm on RBD, and the two means are indistinguishable (0.3440 vs
0.3442).** Consequences:
1. **"Full-corpus is ~19× more seed-variable on RBD" is REFUTED.** It was an artifact of n=2 — the
   champion happened to draw two adjacent seeds first.
2. It also weakens **Phase 15-RESEED's** "the champion is downstream-STABLE (0.368→0.371, spread
   0.003)" — that too was n=2, and a third seed spreads it to 0.078. Champion downstream stability
   should no longer be claimed on RBD.
3. **RBD cannot separate these configurations.** Its between-config difference (~0.000) is far below
   its within-config seed noise (~0.06-0.08). Report RBD as uninformative here, not as weak support.
4. The instability case against full-corpus now rests on TWO signals, not three: the divergence
   (1 of 3 full-corpus seeds vs 0 of 3 champion seeds) and the lower HA means. The RBD-variance
   signal is withdrawn.
**The HA finding is UNAFFECTED and still holds at champion n=3 vs full n=2:** champion WSN
[0.2519, 0.2855] vs full [0.2042, 0.2331] and champion Perth [0.0951, 0.1332] vs full [0.0408, 0.0654]
— still fully DISJOINT, champion winning all 6 pairwise comparisons per strain. Note the contrast:
on HA the between-config gap exceeds the within-config noise; on RBD it does not. That is precisely
why the out-of-corpus benchmarks carry the argument.
METHODOLOGICAL LESSON: this claim flipped at n=1, again at n=2, and settled only at n=3 — exactly the
failure mode Phase 15-RESEED warned about. Do not report a variance comparison from two seeds.

### Phase 16-ABLATION — intermediate corpus (train + HALF of valid): DOSE-RESPONSE IS AMBIGUOUS
To separate "corpus specialization" (smooth: more corpus -> worse transfer) from "adding the valid split
perturbed a near-edge optimisation into a worse basin" (threshold: nothing until the full set), trained a
corpus exactly halfway between: `train.fasta + valid_halfA` = **13,618 records**. Seed **1234**, matching
champion-s1234 and full-s1234, so the three points differ ONLY in corpus size. Stride-2 split of
`valid.fasta` (it is GROUPED by source — all `ictv:*` then `ncbi_assembly:*` — so a first-half split would
be source-biased); the halves came out balanced 671 ictv / 3 ncbi each, zero overlap. Preprocess verified
lossless: 171,803,928 + 8,046,449 = 179,850,377, and halfA + halfB = 16,658,068 = the original valid `.bin`.
Unlike the full-corpus run this **keeps a real held-out set** (`valid_halfB`, 674 recs, never trained on):
held-out PPL 2.752 (500) -> 2.606 -> 2.529 -> 2.539 -> 2.484 -> 2.477 -> 2.421 -> 2.373 -> 2.413 -> 2.398
-> 2.383 -> **2.446 (6000)**; final val-set 2.431 / test-set 2.358. (NOT comparable to the champion's
-25.44%: different, smaller held-out set. Kept as an internal sanity signal only.) Clean run, 0 nan/0 skipped.

Seed-matched (all seed 1234) downstream:

| corpus | records | Mtok | epochs @6k steps | WSN | Perth | RBD |
|---|---:|---:|---:|---:|---:|---:|
| champion (train only) | 12,944 | 171.80 | 9.15 | 0.2855 | 0.0951 | 0.3680 |
| **half (train+674)**  | 13,618 | 179.85 | 8.75 | **0.2109** | **0.1035** | **0.3286** |
| full (train+1348)     | 14,292 | 188.46 | 8.35 | 0.2042 | 0.0408 | 0.3720 |

Placing the (n=1) half point against the 2-seed ranges of the other two configs:

| metric | half | champion range | full range | half behaves like |
|---|---:|---|---|---|
| WSN   | 0.2109 | [0.252, 0.285] | [0.204, 0.233] | **FULL** |
| Perth | 0.1035 | [0.095, 0.133] | [0.041, 0.065] | **CHAMPION** |
| RBD   | 0.3286 | [0.368, 0.371] | [0.316, 0.372] | FULL (weak: full's range is wide) |

**THE TWO HA STRAINS DISAGREE, so the mechanism is NOT resolved.** WSN says the deficit is already fully
present at +674 records (dose-like, saturating early); Perth says the deficit only appears at the full
+1,348 (threshold-like). RBD is the least informative (contaminated, and full's seed spread 0.056 spans
the half point). Tension worth noting: **Perth is the CLEANEST benchmark** and it favours the threshold
reading, but half is n=1 and HA seed spreads are ~0.03, which is the size of the effect being judged.
=> **Do not claim a dose-response.** Resolving it needs a second half-corpus seed (~15.6 h); the HIV-1 Env
benchmark (Phase 16-ENV) will also weigh in as a third, near-clean family.

WHAT IS ROBUST from this ablation: **half-corpus is <= champion on all three benchmarks and beats it on
none.** So there is no "add a little data for free" regime — the champion (train-only) remains the recipe,
and the practical guidance in RECIPE.sh is unchanged regardless of which mechanism turns out to be right.

### Phase 16 — k-mer CORRECTION to the leakage check (2026-08-06): rank the benchmarks by contamination
The original check used ONE 90-mer per HA strain, which only detects a near-identical strain. A proper sweep
(`kmer_probe.py`: non-overlapping probes across each full CDS, fwd + revcomp, vs all 188.46 Mbp of
`train_all.fasta`) shows **WSN is NOT clean**:

| probe | k=90 | k=60 | k=40 | k=30 | k=24 |
|---|---:|---:|---:|---:|---:|
| WSN HA (H1N1)   | 0/18 | **2/28** | **5/42** | **13/56** | **20/70** |
| Perth HA (H3N2) | 0/18 | 0/28 | 0/42 | 0/56 | **0/70** |

A 24-mer matching 1.9e8 bp by chance is ~7e-5, so every hit is real homology (revcomp hits: 0 everywhere).
=> An influenza A with a **related H1 HA is in the corpus**; WSN is only partially out-of-corpus. **Perth
(H3N2) is the single cleanest out-of-corpus benchmark in the study** (nothing detectable at k>=24).
Contamination ranking: **RBD (exact genome present) > WSN (homologous strain) > Perth (nothing)**.
TWO IMPLICATIONS, the second of which STRENGTHENS the Phase 16 result:
1. Champion Spearman tracks that ranking exactly — RBD 0.370 > WSN 0.269 > Perth 0.114 — so part of the
   RBD/WSN signal is plausibly homology-assisted rather than pure generalization.
2. The champion-vs-full-corpus deficit is **relatively LARGEST on the cleanest benchmark**: Perth
   0.114 vs 0.053 = **2.15x**, WSN 0.269 vs 0.219 = 1.23x. The specialization effect is most pronounced
   exactly where contamination cannot explain it.
LESSON (methodological): use a k-mer sweep, not a single long probe, to test corpus containment; and rank
benchmarks by contamination rather than treating "absent" as binary.

### Phase 16-ENV — third virus family (HIV-1 Env): the deficit REPLICATES (2026-08-08)
Built a third downstream benchmark because the k-mer sweep left only ONE clean out-of-corpus test
(Perth H3N2) — a single point of failure for every generality claim. HIV-1 Env (Haddox 2018,
BF520 + BG505, 25,310 windows) is *Retroviridae*, near-clean (0 hits at k>=40). ZIKV E was evaluated
and REJECTED: 12/16 exact 90-mers in the corpus, effectively memorised. Full construction, and the
HXB2-numbering problem solved by deriving offset=29 (argmax rate 54.7%/46.6% vs 5% chance, 12 shifted
controls all at chance), in `DOWNSTREAM_ENV_EVAL.md`.

| strain | base | champion mean (range, n=3) | full mean (range, n=2) | gap | ranges | champ wins |
|---|---:|---|---|---:|:--:|:--:|
| BF520 | 0.0280 | **0.2263** [0.2047, 0.2482] | 0.1911 [0.1906, 0.1916] | +0.0352 | DISJOINT | 6/6 |
| BG505 | 0.0433 | **0.2207** [0.2039, 0.2468] | 0.1734 [0.1600, 0.1868] | +0.0473 | DISJOINT | 6/6 |

1. **The LoRA transfers to HIV.** Base ~0.03/0.04 -> LoRA 0.16-0.25 on a family essentially absent
   from the corpus. Strongest evidence yet that the PPL gain buys real capability, not corpus recall.
2. **The champion-vs-full deficit REPLICATES on an independent family** — disjoint ranges, 6/6 both
   strains. Corpus specialization now rests on **2 out-of-corpus families / 4 strains**, not 1 strain.
3. **half stays ambiguous in the same split way**: champion-like on BF520, full-like on BG505 —
   mirroring HA (champion-like on Perth, full-like on WSN). Across all 4 clean strains half is 2-2.
   Dose vs threshold still unresolved; more half seeds are training.
4. **Weakens an earlier observation**: champion Spearman does NOT simply track contamination. Env is
   near-clean yet scores 0.226/0.221, well above clean Perth (0.109). Benchmark difficulty/biology
   dominates. Treat "signal tracks contamination" as weak and confounded; what survives is that RBD
   is not zero-shot and comparative claims must rest on out-of-corpus families.

### Phase 16-CLUSTER (2026-08-08) — n=3 per arm: DOSE-RESPONSE CONFIRMED, "disjoint" claim WEAKENED
Used a second 4-node H200 cluster (shared `/fsx`) to bring every arm to n=3: `half_s2345`,
`half_s3456`, `full_s4567` trained concurrently, all clean (0 nan, TRAIN_EXIT=0), each scored on all
five readouts. Full results per seed in the `*_results_*.json` files.

| benchmark | champion mean | half mean | full mean | gap c−f | champion-vs-full ranges | pairwise |
|---|---:|---:|---:|---:|:--:|:--:|
| HA WSN    | 0.2656 | 0.2332 | 0.2313 | +0.0343 | **OVERLAP** | 8/9 |
| HA Perth  | 0.1088 | 0.0981 | 0.0583 | +0.0505 | **DISJOINT** | 9/9 |
| Env BF520 | 0.2263 | 0.2129 | 0.2006 | +0.0257 | **OVERLAP** | 8/9 |
| Env BG505 | 0.2207 | 0.2000 | 0.1859 | +0.0348 | **OVERLAP** | 8/9 |
| RBD       | 0.3440 | 0.3344 | 0.3344 | +0.0096 | OVERLAP | 4/9 |

**CORRECTION — the "DISJOINT on 8/8 metrics" result does NOT survive n=3.** `full_s4567` drew high
across the board (WSN 0.2565, Env 0.2196/0.2108) and dissolves the separation on three of the four
out-of-corpus strains. Only **Perth** — the cleanest benchmark — remains fully disjoint. Any statement
of the form "the ranges never overlap" must now be qualified; that was an n=2/n=3 artifact and is the
THIRD claim in this phase overturned by adding seeds.

**WHAT SURVIVES, and is now better supported than the disjointness ever was:**
- champion mean > full mean on **all five** benchmarks;
- **33 of 36** pairwise seed comparisons favour the champion across the four out-of-corpus strains;
- Perth (cleanest) still fully disjoint, 9/9.
(Caveat: the 36 pairwise comparisons are not independent — they come from 3+3 seeds — so treat 33/36
as a consistent direction, not a p-value.)

**DOSE-RESPONSE IS CONFIRMED — this resolves the Phase 16-ABLATION ambiguity.** Ordering the three
corpora by size (12,944 → 13,618 → 14,292 records):

    champion  >  half  >  full     on 4 of 4 out-of-corpus strains, by mean

The half-corpus point sits *between* the other two everywhere, so transfer degrades **gradually with
added corpus**, not at a threshold. The earlier 2–2 split was `half_s1234` being a low draw: at n=3 the
half arm has the largest seed spread of any arm on HA (0.081 WSN, 0.078 Perth), which is exactly why
n=1 could not resolve this. Under a random ordering (given champion > full) the chance of half landing
in between on all four strains is (1/3)^4 ≈ 1.2%, so the monotonicity is unlikely to be coincidence —
though 4 strains from 2 families are not fully independent either.

**Mechanism implication:** a graded dose-response supports *corpus specialization* (more of this
corpus progressively trades away out-of-corpus transfer) over the "adding valid perturbed the run into
a worse basin" alternative, which predicted a threshold. Not proof, but the ablation now points one way.

### Phase 16-FINAL (2026-08-09) — n≈5 per arm on 3 virus families. Both surviving claims HOLD.
Second cluster round brought every arm to n=5 (full n=4 at time of writing; `full_s5678` still
training on h200). Every model scored on all five readouts. Mean ± sd across seeds:

| benchmark | champion (n=5) | half (n=5) | full (n=4) | gap c−f | ranges | pairwise | dose |
|---|---|---|---|---:|:--:|:--:|:--:|
| HA WSN    | 0.2801 ± 0.024 | 0.2396 ± 0.042 | 0.2346 ± 0.022 | +0.0455 | OVERLAP | 19/20 | ✔ |
| HA Perth  | 0.1113 ± 0.021 | 0.0933 ± 0.031 | 0.0655 ± 0.019 | +0.0458 | **DISJOINT** | **20/20** | ✔ |
| Env BF520 | 0.2311 ± 0.019 | 0.2167 ± 0.024 | 0.2051 ± 0.016 | +0.0260 | OVERLAP | 18/20 | ✔ |
| Env BG505 | 0.2285 ± 0.025 | 0.2039 ± 0.020 | 0.1929 ± 0.025 | +0.0356 | OVERLAP | 17/20 | ✔ |
| RBD       | 0.3454 ± 0.033 | 0.3275 ± 0.018 | 0.3250 ± 0.033 | +0.0204 | OVERLAP | 12/20 | ✔ |

**BOTH surviving claims hold at n=5, on 2 out-of-corpus families / 4 strains:**
1. **champion > full on every benchmark**, **74 of 80** pairwise seed comparisons out-of-corpus, and
   Perth (the cleanest benchmark) still fully DISJOINT at 20/20.
2. **Dose-response monotonic on 4/4 out-of-corpus strains** (champion > half > full), now with the
   half arm at n=5 rather than the n=1 that made this ambiguous for two days.

Effect sizes are modest and comparable to within-arm sd (gaps 0.026–0.046 vs sd 0.016–0.042), which is
why RANGES OVERLAP on 3 of 4 strains and why single-seed comparisons were so misleading here. State
this as **a consistent shift in means with a reproducible ordering**, never as separation of runs.
Seed sd is itself informative: the **half arm is the most variable** (up to 0.042 on WSN) — a middle
corpus size gives the least reproducible adapter, which is a practical argument against half-measures.
RBD reaches monotonicity too but at 12/20 pairwise with a 0.020 gap against 0.033 sd — still
uninformative on its own, exactly as recorded in the RBD VARIANCE CORRECTION.
CLAIM-STABILITY NOTE: three claims in this phase were overturned by adding seeds (RBD variance,
contamination-tracking, range disjointness). The two above are the ones that survived every increase
from n=1 to n=5 — a useful marker of which kind of statement is trustworthy in this study: statements
about MEANS and ORDERINGS across many strains, not about individual runs or ranges.

### Phase 16-BENCH4 (2026-08-09) — a 4th out-of-corpus benchmark is NOT AVAILABLE. Why, and the rule.
Attempted to add a fourth downstream benchmark. **Result: negative — do not retry without new data.**

Surveyed all **157 jbloomlab repos**. Every DMS dataset in the `site,A..Y` preference format belongs to
a virus family already used (influenza / HIV / SARS-CoV-2 / Zika) or is an antibody (2B06,
Ab-CGGnaive); the rest are software. Repos for genuinely new families — `MeV_SSPE_Dynamics` (measles),
`Herpesvirus-Glycoprotein-Analysis` (EBV) — contain reference genomes and alignments but **no DMS
preference data**, so they are not variant-effect benchmarks at all.

The two remaining same-family options were tested and **both are fully contaminated**:

| candidate | DMS strain | k=90 | k=60 | k=40 | k=30 | k=24 | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| PB2 (polymerase) | A/PR/8/1934 | **25/25** | 38/38 | 57/57 | 76/76 | 95/95 | 100% IN CORPUS |
| M1 (matrix) | A/PR/8/1934 | 5/8 | 9/12 | 15/18 | 22/25 | 28/31 | IN CORPUS |

**ROOT CAUSE, and the generalisable rule.** The corpus is the ICTV VMR (one exemplar per species), and
the exemplar for *Influenza A virus* is **A/Puerto Rico/8/1934** — present as its segment records
`V00603.1`, `J02151.1`, `V01099.1`, all `ictv:VMR1001046`. So every PR8-derived DMS is memorised. This
single fact explains the whole influenza pattern:

    PB2/M1  (PR8, 1934)        -> 100% contained
    HA WSN  (WSN, 1933)        -> partial (exact 60-mers shared with the PR8 exemplar)
    HA Perth (Perth, 2009)     -> clean (≈75 years of antigenic drift from the exemplar)

=> **Corpus containment is governed by how far the DMS strain has DRIFTED from the corpus exemplar,
NOT by taxonomy.** A "new virus family" is neither necessary nor sufficient for a clean benchmark.
The clean ones we have (Perth H3N2, HIV Env BF520/BG505) are both **fast-evolving RNA viruses with
modern DMS strains**; the contaminated ones (PR8 1934, ZIKV MR766 1947, SARS-CoV-2 Wuhan-Hu-1) are
canonical lab/reference strains that ARE the exemplars.

**SELECTION RULE for any future benchmark: pick a fast-evolving virus whose DMS strain is decades
diverged from the ICTV exemplar, then verify with `kmer_probe.py` BEFORE building anything.** Screening
costs minutes; building first wastes hours (as it nearly did here).
The study therefore stands on **3 virus families / 4 clean-or-near-clean strains** (HA Perth, HA WSN
partial, Env BF520, Env BG505), which is what Phase 16-FINAL rests on. Extending it needs a DMS from
outside the Bloom-lab preference corpus (e.g. a ProteinGym-style fitness dataset on a hypervariable
virus such as HCV NS5A), which is a different data format and a separate piece of work.

### Phase 16 — FINAL CONCLUSIONS
1. **Training on the full corpus does NOT produce a better model; it produces a worse-transferring one.**
   At n=3 per arm across 2 out-of-corpus virus families / 4 strains: champion mean > full mean on every
   benchmark, 33/36 pairwise seed comparisons favour the champion, and transfer degrades **monotonically
   with corpus size** (champion > half > full on 4/4 strains). Ranges overlap on 3 of 4 strains at n=3
   (only Perth stays disjoint), so state this as a consistent mean/ordering effect, NOT as separation.
   In-corpus RBD cannot distinguish the configurations at all.
2. **Two independent signals against the full-corpus config**: (a) 1 of 3 seeds diverged outright vs 0 of 3
   for the champion; (b) HA means lower on every metric, ranges disjoint. *(A third signal — "RBD seed
   spread 0.056 vs 0.003" — was claimed at n=2 and is WITHDRAWN: at n=3 the champion's RBD spread is
   0.078, larger than full's. See the RBD VARIANCE CORRECTION section.)*
3. **The champion (train-only, dim256 × α1024 × do0.3 × 6k) remains the production recipe** — on held-out
   PPL, on downstream transfer, and on run-to-run reliability. The full-corpus adapter is a **sidegrade**:
   defensible only if the deployment distribution closely matches the training corpus, and it forfeits any
   held-out metric by construction.
4. **Mechanism still open.** Over-training is ruled out arithmetically (fewer epochs, above). "Corpus
   specialization" fits but is not proven; distinguishing it from "adding the valid split perturbed a
   near-edge optimisation into a worse basin" would need more seeds and probably an intermediate-corpus
   ablation (e.g. train + half of valid).
5. **k-fold is not worth running** (see the earlier k-fold assessment): more data from this corpus made
   transfer worse, so fold-splitting the same corpus will not produce a better model, and at k=5 it costs
   ~78 h. Spend node time on additional *benchmarks* or *seeds* instead.
6. **Remaining caveat: n=2.** Phase 15-RESEED set ≥3 seeds as this study's standard. HA's ordering is
   invariant across all 8 metrics so a third seed is unlikely to flip it, but RBD genuinely needs one
   (champion s3456 does not exist: ~15.6 h) before any claim is made about in-corpus performance.

### Practical verdict
The full-corpus model is **not a strictly-better production artifact — it is a sidegrade.** Prefer it only
when the deployment target resembles the training corpus; prefer the **champion for novel/divergent
viruses**. This also further weakens the k-fold case: if "more data from this corpus" does not improve
transfer, fold-splitting the same corpus (~78 h for k=5) is unlikely to produce a better model either.
Assets: `PROD_20b_full_corpus/evo2/checkpoints/iter_0006000` (115 GB, 12 ckpts retained),
`train_all.fasta`, `viral_all_{preprocess,dataset}.yaml`, `RUN_full_corpus.sh`,
`QUEUE_downstream_full_corpus.sh`, `downstream/ds_prod_full_corpus.json`,
`downstream_ha/{wsn,perth}_results_prod_full_corpus.json`.
OPS GOTCHA (2026-08-08, NVLS — sharper trigger than previously recorded): the documented
`NCCL_NVLS_ENABLE=0` fix is needed not only after a `kill -9` of a torchrun, but after **ANY recycling
of the container on a node that has already run one** — `docker rm -f evo2run` followed by a fresh
`docker run` reproduces it exactly. All 3 round-2 cluster runs died within 60 s of launch with
`ncclUnhandledCudaError` / `Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA error 401`
on all 8 ranks. So: **set `NCCL_NVLS_ENABLE=0` whenever the container is recycled rather than freshly
booted.** `node_run.sh` also carries a detect-and-retry (grep the log for NVLS/Multicast/ncclUnhandled,
`rm -rf` the partial result dir, relaunch once with the flag) — that safety net is what saved three
15.6 h slots here, and is worth keeping for any unattended launch on these nodes.
NOTE ON HOST NAMES (2026-08-07): the box called `h200-2` throughout this log was RENAMED to **`h200`**
when a separate 4-node cluster (`h200-1`..`h200-4`, shared Lustre `/fsx`) was added. Same machine,
same data. `h200-2` now resolves to a DIFFERENT host. Read every historical "h200-2" below as `h200`.
OPS GOTCHA (2026-08-05): the `evo2run` container on h200-2 holds **~1,450 unreaped ZOMBIE `train_evo2`
processes** from earlier runs (`<defunct>`, reparented to ppid 1, etime ~20 d — the container's PID 1 does
not reap). A plain `pgrep -f train_evo2` matches those **forever**, so any "wait until training finishes"
gate built that way never unblocks (it stalled this phase's queued scoring until caught). Count only LIVE
processes: `ps -eo stat=,args= | grep -v defunct | grep -vE '^[[:space:]]*Z' | grep -c '[t]rain_evo2'`.
Note the container's `awk` lacks `!~`, so an awk-based zombie filter fails open (silently returns 0).
Also: never edit a shell script while it is executing — bash reads scripts incrementally and a mid-run
edit shifts byte offsets (produced one spurious `command not found` here; outputs verified unaffected).
