# Downstream functional evaluation of the viral LoRA

**Question.** The viral LoRA (Evo2-20B fine-tuned on eukaryotic-host viral genomes)
improves held-out **perplexity** by up to −25%. PPL is a proxy. Does that gain
translate to a **real downstream task** — zero-shot viral variant-effect prediction?

Status: **pipeline built and validated end-to-end without GPU; awaiting GPU time to
score.** (This file is updated as results land.)

---

## 1. Benchmark

**Starr et al. 2020, "Deep mutational scanning of SARS-CoV-2 receptor binding domain
reveals constraints on folding and ACE2 binding"** (Cell 182:1295). SARS-CoV-2 is a
**eukaryotic-host ssRNA(+)** virus — squarely in the LoRA's training scope (ssRNA(+)
is the largest class in the viral corpus).

- Source file: `single_mut_effects.csv` from `jbloomlab/SARS-CoV-2-RBD_DMS`
  (fetched via the git-LFS media URL). 4,221 rows.
- Every single amino-acid mutation of the RBD (spike residues **331–531**, 201 sites)
  with two experimental phenotype scores:
  - **`bind_avg`** — change in ACE2-binding affinity (Δlog10 K_D, averaged over two
    libraries). Primary label.
  - **`expr_avg`** — change in RBD surface expression / folding stability. Secondary
    label (a stability/fitness proxy, often closer to what a genome LM captures).
- After dropping WT self-measurements (201) and stop codons (201), **3,819 amino-acid
  substitutions** remain (3,802 with `bind_avg`, 3,798 with `expr_avg`).

Why this benchmark: it is the canonical, cleanest, most-cited viral DMS; single
substitutions map cleanly onto the genome; and it directly parallels the repo's
own BRCA1 zero-shot notebook (`examples/zeroshot_brca1.ipynb`), so the scoring
method is already blessed in-repo.

**Candidates considered / rejected for the first pass:**
- *Bloom & Neher 2023 SARS-CoV-2 genome-wide fitness* — genome-scale and excellent,
  but the "ground truth" is itself derived from natural-sequence counts, partly
  circular with an LM likelihood; keep as a possible follow-up.
- *ProteinGym viral DMS* — protein-level mutant sequences; mapping to genomic
  nucleotide context is less exact than using the actual reference genome. Good as a
  breadth follow-up.
- *Influenza HA DMS (Doud & Bloom)* — in scope (ssRNA(−)); reserved as a second
  virus to test generality if the SARS-CoV-2 result is positive.

## 2. Eval design (mirrors the in-repo BRCA1 zero-shot VEP)

1. **Reference genome:** SARS-CoV-2 `NC_045512.2` (Wuhan-Hu-1, 29,903 bp).
   Spike CDS 21563–25384 (+ strand, no introns). Reading frame **verified**: all 201
   RBD reference codons translate to the DMS `wildtype` residue (0 mismatches;
   N501, E484, K417 all correct).
2. **Nucleotide mutant construction:** for each amino-acid substitution, replace the
   WT codon with the **minimal-Hamming codon** encoding the target residue
   (deterministic lexicographic tie-break). 1,192 variants are single-nt-accessible,
   1,982 two-nt, 645 three-nt (recorded as `n_nt_changes` for a clean single-nt
   subset analysis).
3. **Windows:** an **8,192-bp genome window centered on the mutated codon**
   (identical window size to the BRCA1 notebook). `ref.fasta` = 201 unique WT windows
   (one per site); `var.fasta` = 3,819 mutant windows. All windows are exactly 8,192 bp
   and in-bounds. Combined `all.fasta` = 4,020 sequences (unique names).
4. **Scoring:** `predict_evo2 --output-log-prob-seqs --log-prob-collapse-option mean
   --mixed-precision-recipe bf16_mixed --vortex-style-fp8`, once with the **base 20B**
   (`/data/evo2_20b_mbridge/iter_0000001`) and once with the **best LoRA**
   (`/data/viral/lora_run_20b_16k_dim256_a1024_do3_6k/evo2/checkpoints/iter_0006000`,
   the −25.44% adapter).
5. **Variant score:** `evo2_delta = mean_logprob(variant_window) −
   mean_logprob(ref_window)` (higher = more tolerated), exactly as in the BRCA1
   notebook.
6. **Metrics (base vs LoRA):**
   - **Spearman** ρ of `evo2_delta` vs `bind_avg` and vs `expr_avg` (standard DMS
     metric), on all substitutions and on the single-nt subset.
   - **AUROC** for a binarized "strongly deleterious" label (`bind_avg ≤ −1`,
     `expr_avg ≤ −1`).
   - **Headline:** does the LoRA's ρ / AUROC exceed the base model's? That is the
     test of whether the PPL gain transfers downstream.

## 3. Assets (all under `/data/viral/downstream/`)

| file | what |
|---|---|
| `single_mut_effects.csv` | Starr 2020 RBD DMS (ground truth) |
| `NC_045512.2.fasta` | SARS-CoV-2 reference genome |
| `prep_variants.py` | builds windows from DMS + genome (frame-checked) |
| `ref.fasta` / `var.fasta` / `all.fasta` | 201 / 3,819 / 4,020 windows (8,192 bp) |
| `metadata.csv` | variant→(ref_name, var_name, site, wt, mut, n_nt_changes, bind_avg, expr_avg) |
| `analyze.py` | loads predict outputs → Spearman + AUROC, base vs LoRA (loader smoke-tested on an existing predict dir) |
| `predict_downstream.sh` | 2 predict jobs (base + LoRA), TP=1, one 20B per H200 |

**Validated without GPU:** genome download, DMS download, reading-frame translation
(201/201), window construction + bounds + codon assertions, prediction-loader parsing
against a real existing `predictions__rank_*.pt`, and `predict_evo2` CLI flags.

## 4. GPU needed

Two `predict_evo2` runs (base, LoRA), each 4,020 × 8,192-bp sequences, 20B vortex-FP8,
TP=1 → one H200 each; can run concurrently on 2 GPUs (`BASE_GPU`/`LORA_GPU`). Run:
`bash /data/viral/downstream/predict_downstream.sh` inside the container (it also runs
`analyze.py` at the end). Optional: shard `all.fasta` across more GPUs for speed.

## 5. Results

_Pending GPU run._

---

## RESULTS (base 20B vs best LoRA dim256×α1024×do0.3×6000, −25.44% PPL)

Zero-shot variant-effect prediction on the Starr 2020 SARS-CoV-2 RBD DMS. `evo2_delta =
mean_logprob(variant_window) − mean_logprob(ref_window)`, correlated with experimental scores.
"all" = all reachable substitutions; "single_nt" = variants reachable by one nucleotide change
(cleanest for a nucleotide LM). AUROC target = deleterious (score ≤ −1.0).

| metric | base | LoRA |
|---|---:|---:|
| Spearman bind_avg (all, n=3802)        | −0.012 (p=0.45, n.s.) | **+0.368** (p≈5e-122) |
| Spearman bind_avg (single-nt, n=1190)  |  0.017 (n.s.)         | **+0.437** (p≈1e-56)  |
| Spearman expr_avg (all, n=3798)        |  0.004 (n.s.)         | **+0.351** (p≈2e-110) |
| Spearman expr_avg (single-nt, n=1190)  |  0.014 (n.s.)         | **+0.397** (p≈3e-46)  |
| AUROC bind_avg≤−1 (all)                |  0.473                | **0.659** |
| AUROC bind_avg≤−1 (single-nt)          |  0.496                | **0.724** |
| AUROC expr_avg≤−1 (all)                |  0.498                | **0.690** |
| AUROC expr_avg≤−1 (single-nt)          |  0.508                | **0.721** |

## CONCLUSION — the PPL gain transfers to real downstream biology
The **base Evo2 has no signal** on SARS-CoV-2 RBD variant effects (Spearman ≈ 0, all non-significant;
AUROC ≈ 0.5 = chance). The **viral-LoRA gains a strong, highly-significant signal** (Spearman 0.35–0.44;
AUROC 0.66–0.72), consistent across both binding and expression and strongest on single-nt variants.
So the −25% held-out perplexity improvement is **not just a proxy** — it buys genuine zero-shot
variant-effect-prediction capability the base model entirely lacks, on a canonical experimental
benchmark, for an in-scope virus (SARS-CoV-2, ssRNA(+)). This is the study's strongest external validation.

Caveat: single-model (no seeds/CIs); one virus; nucleotide-window mapping (8192 bp centered on the
mutated codon). A generality follow-up (influenza HA DMS) is a natural next step.
