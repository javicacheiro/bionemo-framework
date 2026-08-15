# Third downstream benchmark: HIV-1 Env variant-effect prediction

**Status:** benchmark BUILT + VALIDATED 2026-08-06; scoring queued (`QUEUE_env_scoring.sh`), results in §5.

## 1. Why a third benchmark — the existing two are partly contaminated

Phase 16's leakage work found the study's downstream evidence is not as clean as assumed. A k-mer
sweep (`kmer_probe.py`: non-overlapping probes across each full CDS, forward + reverse-complement,
against all 188,461,996 bp of `train_all.fasta`; chance match at k=24 is ~7e-5, so any hit is real
homology) gives this contamination ranking:

| benchmark | family | k=90 | k=60 | k=40 | k=30 | k=24 | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| SARS-CoV-2 RBD | *Coronaviridae* | — | — | — | — | — | **exact genome `MN908947` in `train.fasta`** |
| HA WSN (H1N1) | *Orthomyxoviridae* | 0/18 | **2/28** | **5/42** | **13/56** | **20/70** | homologous H1 strain in corpus |
| HA Perth (H3N2) | *Orthomyxoviridae* | 0/18 | 0/28 | 0/42 | 0/56 | **0/70** | clean |
| **Env BF520** | *Retroviridae* | 0/28 | 0/42 | 0/63 | 3/85 | 5/106 | **near-clean** |
| **Env BG505** | *Retroviridae* | 0/28 | 0/43 | 2/64 | 2/86 | 6/107 | **near-clean** |

Before this, **Perth (H3N2) was the only clean out-of-corpus benchmark in the entire study** — a
single point of failure for every "generalizes to unseen viruses" claim. HIV-1 Env is a third virus
family (*Retroviridae*, ssRNA-RT), is near-clean (nothing at k>=40; a handful of stray 24/30-mers,
expected for conserved retroviral stretches), and roughly doubles the clean evidence base.

**Rejected candidate — Zika virus E.** `ZIKV_DMS_with_EvansLab` has data in exactly the right format,
but ZIKV E is **heavily contaminated**: 12/16 exact **90-mers** present in the corpus. Effectively
memorized; unusable as a generality test. Recorded here so nobody re-adds it.

## 2. Benchmark

**Haddox et al. 2018** (*eLife*), "Mapping mutational effects along the evolutionary landscape of HIV
envelope" — replicate-averaged site-specific amino-acid preferences for two transmitted/founder
HIV-1 Envs, from `jbloomlab/EnvMutationalShiftsPaper`:

| tag | strain | CDS | prefs sites | variants |
|---|---|---|---|---|
| `bf520` | BF520.W14M.C2 | 2,559 bp (853 codons) | 662 | **12,578** |
| `bg505` | BG505.T332N | 2,583 bp (861 codons) | 670 | **12,730** |

Same experimental readout as the HA scan (`pi[site][aa]`, 20 values per site summing to 1), so the
per-variant label is the identical transform `dms_effect = log2(pi[mut]/pi[wt])` (higher = more
tolerated). Median `dms_effect`: BF520 −1.338, BG505 −1.506 — negative as expected (most mutations
deleterious), but markedly less extreme than HA (WSN −6.25), consistent with Env's far greater
mutational tolerance under diversifying immune selection.

## 3. The numbering problem, and how it was solved (do not skip this)

Unlike HA, **Env preferences do NOT index the CDS**. They use HXB2-style numbering: sites 31–702,
non-contiguous, with insertion codes (`184a`, `184b`, `190a`, `317a`, `395a`; BG505 has `184a`–`184h`)
against 853/861 codons. Naively treating `site` as a codon index yields a silently meaningless
benchmark.

The CSV rows *are* sequential along the protein, so row *i* maps to codon *(OFFSET + i)*. **OFFSET was
not assumed — it was derived and validated:**

- `env_offset_scan.py` scans every offset, scoring each by mean `pi(residue the reference encodes)`.
- `env_validate.py` then applies two threshold-free checks against shifted negative controls.

**Result: OFFSET = 29 for both strains independently.**

| | argmax rate | mean pref-rank |
|---|---:|---:|
| BF520 @ 29 | **54.7%** | **0.885** |
| BG505 @ 29 | **46.6%** | **0.851** |
| 12 shifted controls | 4.5–6.6% | 0.49–0.52 |
| chance | 5.0% | 0.500 |

"argmax rate" = fraction of sites where the residue the reference CDS encodes is the *single most
preferred* residue in the DMS. ~11x and ~9x chance, with every negative control sitting exactly at
chance, and both strains peaking independently at the same offset. The mapping is correct.

`prep_variants_env.py` **re-runs this validation and aborts** (`argmax < 25%` or `rank < 0.70`) rather
than emit a benchmark on an unverified numbering. It also enforces ORF sanity: ATG start, single
terminal stop, **0 internal stops** — verified for both strains.

Note on a calibration trap: mean `pi_wt` is 0.123 (BF520) / 0.149 (BG505), *below* the 0.15 threshold
that would be natural to borrow from HA — yet the mapping is unambiguously right. `pi_wt` is not
comparable across viruses (HA WSN 0.488, Perth 0.198, Env ~0.13) because it reflects how peaked
selection is, not mapping quality. Use argmax-rate-vs-control instead.

## 4. Eval design (mirrors HA exactly)

Full-CDS windows — Env at ~2.5 kb fits well inside the 8,192 bp budget, and ref/variant windows are
identical in length, differing only at the mutated codon. Mutant codon = minimal-Hamming codon for
the target residue, deterministic lexicographic tie-break (`n_nt_changes` recorded: BF520
{1:3847, 2:6369, 3:2362}, BG505 {1:3892, 2:6460, 3:2378}). Scored with
`predict_evo2 --output-log-prob-seqs --log-prob-collapse-option mean --mixed-precision-recipe
bf16_mixed --vortex-style-fp8`, mbs 1 — **identical to every other adapter in the study**, so numbers
are directly comparable. `evo2_delta = mean_logprob(variant) − mean_logprob(ref)`; metrics are
Spearman (all + single-nt) and AUROC (median split + fixed `dms_effect <= -2`).

Because the meta columns match HA's, **`analyze_ha.py` is reused unchanged**.

## 5. Results (2026-08-08) — the LoRA transfers to HIV, and the champion-vs-full gap REPLICATES

> **⚠ SUPERSEDED IN PART (2026-08-10, Phase 16-EPOCH).** The champion-vs-full-corpus deficit
> below was measured at a FIXED 6000 steps, which confounds corpus size with epoch count
> (bigger corpus = fewer passes). Re-running the full corpus at MATCHED epochs recovers ~59%
> of the deficit on average and **100% on HA Perth** (0.0653 -> 0.1140 vs champion 0.1113).
> Treat the numbers below as a fixed-step comparison, not as a corpus-size effect.
> See `EXPLORATION_LOG.md` "Phase 16-EPOCH".

Seven models scored concurrently, one per GPU, ~1.5 h. Spearman (all substitutions):

| model | seeds | BF520 | BG505 |
|---|---|---|---|
| **base 20B** | — | **0.0280** | **0.0433** |
| champion | s1234 / s2345 / s3456 | 0.2482 / 0.2261 / 0.2047 | 0.2114 / 0.2468 / 0.2039 |
| full-corpus | s1234 / s3456 | 0.1916 / 0.1906 | 0.1600 / 0.1868 |
| half-corpus | s1234 | 0.2061 | 0.1793 |

| strain | champion mean (range) | full mean (range) | gap | ranges | champion wins |
|---|---|---|---:|:--:|:--:|
| BF520 | **0.2263** [0.2047, 0.2482] | 0.1911 [0.1906, 0.1916] | +0.0352 | **DISJOINT** | **6/6** |
| BG505 | **0.2207** [0.2039, 0.2468] | 0.1734 [0.1600, 0.1868] | +0.0473 | **DISJOINT** | **6/6** |

**Finding 1 — the viral LoRA transfers to a third virus family.** The base 20B is at ~0.03/0.04
(essentially no signal) while every LoRA reaches 0.16–0.25 on HIV-1 Env, a *Retroviridae* target that
is near-absent from the training corpus (nothing at k>=40). This is the cleanest evidence in the study
that the −25% PPL gain buys genuine, transferable variant-effect capability rather than corpus recall.

**Finding 2 — the champion-vs-full-corpus deficit REPLICATES on an independent family.** Disjoint
ranges on both strains, champion winning all 6 pairwise seed comparisons on each. Combined with
influenza HA, the corpus-specialization result now holds across **two independent out-of-corpus virus
families and four strains** (HA WSN + Perth, Env BF520 + BG505) — no longer resting on one strain.

**Finding 3 — the half-corpus point stays ambiguous, in the same split way.** Env repeats the HA
pattern exactly: half is champion-like on one strain (BF520 0.2061, inside the champion range) and
full-like on the other (BG505 0.1793, inside the full range). Across all four out-of-corpus strains
half is champion-like on two (Perth, BF520) and full-like on two (WSN, BG505). **Dose-response vs
threshold remains unresolved**, and n=1 for half is the obvious reason — additional half seeds are
training.

**Caveat that partially undercuts an earlier observation.** `DOWNSTREAM_HA_EVAL.md` §4c noted that
champion Spearman appeared to track corpus contamination (RBD 0.344 > WSN 0.266 > Perth 0.109). Env
does not fit that ordering: it is near-clean yet scores **0.226 / 0.221**, well above clean Perth. So
benchmark difficulty and biology clearly matter more than contamination alone, and the
"signal tracks contamination" reading should be treated as weak and partly confounded rather than as
an established pattern. What survives is the narrower, well-supported point: RBD (exact genome in
corpus) cannot be called zero-shot, and the *comparative* claims rest on the out-of-corpus families.

## 6. Assets (`/data/viral/downstream_env/`)

| file | what |
|---|---|
| `BF520_env.fasta`, `BG505_env.fasta` | reference CDS (= DMS wildtype) |
| `BF520_avgprefs.csv`, `BG505_avgprefs.csv` | replicate-averaged aa preferences (ground truth) |
| `prep_variants_env.py` | builds windows; re-validates the offset-29 mapping and aborts on failure |
| `{bf520,bg505}_ref.fasta`, `{bf520,bg505}_var.fasta`, `env_all.fasta` | windows; combined = 25,310 |
| `{bf520,bg505}_meta.csv` | variant → site, wt, mut, n_nt_changes, dms_effect, pi_wt, pi_mut |
| `../kmer_probe.py` | corpus-containment sweep used for the table in §1 |
| `../QUEUE_env_scoring.sh` | the scoring sweep |
