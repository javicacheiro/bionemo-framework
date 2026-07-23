# Downstream generality follow-up: influenza HA variant-effect prediction

**Question.** The viral LoRA (Evo2-20B fine-tuned on eukaryotic-host viral genomes)
gained a strong zero-shot variant-effect signal on the SARS-CoV-2 RBD DMS that the
base model entirely lacks (Spearman ≈ 0 → 0.37–0.44; see `DOWNSTREAM_EVAL.md`).
SARS-CoV-2 is one virus (ssRNA(+)). **Does that edge generalize to a SECOND virus?**
Here we test on an **influenza hemagglutinin (HA) deep-mutational scan** — a different
virus family, a different genome architecture (segmented, **ssRNA(−)**), and a
different DMS phenotype (functional tolerance from viral growth, not ACE2 binding).

This eval mirrors the SARS-CoV-2 pipeline (`/data/viral/downstream/`) exactly except
where influenza biology forces a change (window length, noted below).

---

## 1. Benchmark

**Bloom-lab influenza HA deep mutational scans**, two strains, from the reproducible
data in `jbloomlab/Perth2009-DMS-Manuscript` (`analysis_code/data`, `.../results/preferences`):

| tag | strain | subtype | source DMS |
|---|---|---|---|
| `wsn`   | A/WSN/1933       | **H1N1** | Doud & Bloom 2016, *Viruses* 8:155 ("Accurate measurement of the effects of all amino-acid mutations on influenza hemagglutinin") |
| `perth` | A/Perth/16/2009  | **H3N2** | Lee et al. 2018, *PNAS* 115:E8276 ("Deep mutational scanning of hemagglutinin helps predict evolutionary fates of human H3N2 influenza variants") |

Both DMS experiments mutagenized every codon of HA, passaged the mutant virus
libraries under functional selection (viral growth in cell culture), and deep-sequenced
before/after to infer **site-specific amino-acid preferences** `pi[site][aa]`
(replicate-averaged; each site's 20 values sum to 1). This is the canonical influenza
HA DMS readout and is directly analogous to the SARS-CoV-2 RBD DMS, but measures
**functional/replicative tolerance** rather than ACE2-binding affinity.

**Per-variant experimental label.** For a substitution wt→mut at a site we use the
standard mutational-effect transform
```
dms_effect = log2( pi[site][mut] / pi[site][wt] )   (higher = more tolerated)
```
which matches the "higher = more tolerated" sign of the SARS-CoV-2 `bind_avg`/`expr_avg`
labels. Most HA mutations are deleterious (HA is under strong purifying selection):
WSN median dms_effect = −6.25, Perth median = −2.45.

- WSN: 564 sites, **10,716** amino-acid substitutions.
- Perth: 566 sites, **10,754** amino-acid substitutions.

## 2. Eval design (mirrors the SARS-CoV-2 pipeline)

1. **Reference (= DMS wildtype).** The HA coding sequence the mutant library was built
   on: `WSN_HA_reference.fa` (1,698 nt = 566 codons) and `Perth09_HA_reference.fa`
   (1,701 nt = 567 codons). Sequential DMS site numbering maps directly to codon index
   (codon start = 3·(site−1)).
2. **Reading frame VERIFIED** (parallel to the SARS-CoV-2 "0-mismatch" check). Each
   reference CDS is a clean ORF and every reference codon translates to the DMS
   wildtype residue:
   - ATG start ✓, single terminal stop ✓, **0 internal stop codons** ✓ (both strains).
   - Translations begin with the textbook HA signal peptides — WSN (H1):
     `MKAKLLVLLYAFVATDADTICIGYHANNST…`, Perth (H3): `MKTIIALSYILCLVFAQKLPGNDNSTATLC…`.
   - **mean pi(wildtype) = 0.488 (WSN) / 0.198 (Perth)**, far above the 0.05
     uniform baseline — i.e. the residue the reference encodes at each site is the one
     the DMS strongly prefers. A frame error would randomize this to ≈0.05. This is the
     influenza equivalent of the SARS-CoV-2 "all reference codons translate to the DMS
     wildtype residue" check.
3. **Nucleotide mutant construction.** Replace the WT codon with the **minimal-Hamming
   codon** for the target residue (deterministic lexicographic tie-break), identical to
   the SARS-CoV-2 script. n_nt_changes recorded for a single-nt subset:
   WSN {1:3292, 2:5485, 3:1939}, Perth {1:3358, 2:5482, 3:1914}.
4. **Windows — the one necessary difference from SARS-CoV-2.** The HA gene sits on a
   short (~1.7 kb) genome segment, so there is **no room for the 8,192-bp *centered*
   window** used for the 30 kb SARS-CoV-2 genome. Instead the window is the **entire HA
   CDS**: `*_ref.fasta` = 1 WT window; `*_var.fasta` = one window per substitution, all
   the same length as the reference and differing only at the mutated codon. Because
   `predict_evo2` collapses log-probs by **mean** (per-token), ref/var deltas remain
   directly comparable across variants and to the SARS-CoV-2 run; if anything the
   single-codon change is *less* diluted (3/1698 vs 3/8192).
5. **Scoring.** `predict_evo2 --output-log-prob-seqs --log-prob-collapse-option mean
   --mixed-precision-recipe bf16_mixed --vortex-style-fp8`, once with the **base 20B**
   (`/data/evo2_20b_mbridge/iter_0000001`) and once with the **best LoRA**
   (`/data/viral/lora_run_20b_16k_dim256_a1024_do3_6k/evo2/checkpoints/iter_0006000`,
   the −25.44% adapter). Both strains scored in one combined `ha_all.fasta` (21,472
   windows); TP=1, one 20B per H200, GPU 0 (base) + GPU 1 (LoRA), concurrently.
6. **Variant score & metrics** (base vs LoRA):
   - `evo2_delta = mean_logprob(variant) − mean_logprob(ref)`.
   - **Spearman** ρ of `evo2_delta` vs `dms_effect`, all substitutions and single-nt subset.
   - **AUROC** for a "deleterious" label at the per-strain **median split** (balanced)
     and at a fixed strongly-deleterious threshold (dms_effect ≤ −2).

## 3. Assets (all under `/data/viral/downstream_ha/`)

| file | what |
|---|---|
| `WSN_HA_reference.fa`, `Perth09_HA_reference.fa` | HA reference CDS (= DMS wildtype) |
| `WSN_avgprefs_seqnumbering.csv`, `Perth_summary_avgprefs.csv` | replicate-averaged amino-acid preferences (ground truth) |
| `prep_variants_ha.py` | builds windows + verifies reading frame |
| `wsn_ref.fasta`/`wsn_var.fasta`, `perth_ref.fasta`/`perth_var.fasta`, `ha_all.fasta` | 8,192→full-CDS windows; combined = 21,472 seqs |
| `wsn_meta.csv`, `perth_meta.csv` | variant → (ref_name, var_name, site, wt, mut, n_nt_changes, dms_effect, pi_wt, pi_mut) |
| `analyze_ha.py` | predict outputs → Spearman + AUROC, base vs LoRA |
| `predict_ha.sh` | 2 predict jobs (base + LoRA), TP=1, one 20B per H200 |

## 4. Results (base 20B vs best LoRA dim256×α1024×do0.3×6000, −25.44% PPL)

Zero-shot HA variant-effect: `evo2_delta = mean_logprob(var) − mean_logprob(ref)` over full-CDS
windows, correlated with replicate-averaged DMS preference effect. AUROC target = deleterious.

| strain | metric | base | LoRA |
|---|---|---:|---:|
| **WSN (H1N1)** | Spearman all (n=10716)      | 0.097 | **0.285** (p≈5e-200) |
|                | Spearman single-nt (n=3292) | 0.091 | **0.291** |
|                | AUROC median / ≤−2         | 0.543 / 0.549 | **0.646 / 0.660** |
| **Perth (H3N2)** | Spearman all (n=10754)    | −0.015 | **0.095** (p≈5e-23) |
|                | Spearman single-nt (n=3358) | 0.008 | **0.128** |
|                | AUROC median / ≤−2         | 0.487 / 0.486 | **0.545 / 0.548** |

## 5. Conclusion — the downstream edge GENERALIZES to influenza, but weaker than SARS-CoV-2
On **both** influenza HA strains the LoRA improves over base and every LoRA correlation is highly
significant, so the SARS-CoV-2 finding is **not virus-specific** — the −25% PPL LoRA adds genuine
variant-effect signal on a second, distant virus family. But the **effect size is virus-dependent and
smaller than SARS-CoV-2**: Spearman SARS-CoV-2 RBD **0.37–0.44** > H1N1 **0.29** > H3N2 **0.10**. The
base has ~no signal on SARS-CoV-2/H3N2 (ρ≈0) and weak signal on H1N1 (ρ≈0.10); the LoRA lifts all three.
So: the improvement generalizes directionally and significantly, but its magnitude decays on more
divergent/harder targets (H3N2 weakest). A real, transferable gain — not a single-virus artifact —
though not uniformly large. (Single model; no seeds/CIs.)
