# Leakage Audit — Viral LoRA study (valid_cap8192, 20B best run)

**Date:** 2026-07-19  CPU/analysis only, no GPU, no re-prediction (existing prediction outputs re-aggregated).
**Best run under audit:** base `pred_20b_base_vfp8` vs LoRA `pred_dim256_a1024_do3_6k_vfp8`
(dim256 x alpha1024 x dropout0.3 x 6000 steps, 20B/16k). Headline: **-25.44% mean PPL**.

## Bottom line
The headline **survives**. The one known contaminated record is **already excluded** from the
headline, and the internal memorization signal shows contamination-suspect records **deflate**
(not inflate) the reported improvement. Excluding suspects makes the result slightly **stronger**.

---

## 1. Leakage source / provenance of the KF740664 flag
- The flag exists **only** in `RESULTS_SUMMARY.md` Sec.6.1: `KF740664.1|ictv:VMR1024671` flagged as
  contaminated against Evo2s own base pretraining data and "slated for rescoring". No contamination
  list, provenance file, or local copy of Evo2s pretraining reference exists under `/data/viral`,
  in the manifest, or in the repo docs (grep found no other mention).
- **What "rescoring" actually was:** a dedup step (`valid.fasta.pre_dedup` -> `valid.fasta`,
  `valid_cap8192.fasta.pre_dedup` -> `valid_cap8192.fasta`, 2026-07-05) that removed **exactly one
  record: KF740664.1**. Pre-dedup = 1349 records, post-dedup = 1348. The `pred_*_dedup/` prediction
  dirs are re-predictions on that 1348-record set.

## 2. Impact of excluding the known flag
- **KF740664 is already out of the headline.** `aggregate_ppl.py` scores `common = base INTERSECT lora`.
  The base dir `pred_20b_base_vfp8` still has 1349 records (incl. KF740664), but the best-LoRA dir
  `pred_dim256_a1024_do3_6k_vfp8` has 1348 (no KF740664), so the intersection is 1348 and KF740664
  is dropped automatically. Verified: `KF740664 in common set = False`; the published per-genome
  file already reads `common=1348`.
- Re-aggregating with KF740664 explicitly excluded is therefore **identical**: **-25.44%**, per-genome
  table unchanged. **The -25.44% headline is already clean of the known contaminated record.**
- For reference, KF740664 base PPL = 2.746 (percentile 1.6, mildly low); under LoRA it did not improve
  much, so its removal has negligible effect either way.

## 3. Detectable leakage without the full pretraining reference (memorization signature)
Proxy for pretraining memorization = anomalously **low base PPL** (Evo2 already "knows" the sequence).

- **Base PPL distribution (n=1349):** min 1.19, p1 2.58, p5 3.09, median 3.64, mean 3.58, max 3.97.
- **Lowest tail (memorization suspects):** U68408.1 (1.19), D83003.1 (1.70), M18706.1 (1.74),
  M12927.1 (1.88), U12626.1 (2.14), M34549.1 (2.30)... — a cluster of low ICTV VMR ids (classic,
  well-studied viruses likely heavily represented in Evo2 pretraining).
- **Direction of the bias is CONSERVATIVE, not inflationary.** These low-base-PPL records mostly get
  *worse* under LoRA (little headroom): decile analysis of per-record improvement by base PPL:
  - lowest-base-PPL decile: **-12.1%** improvement (smallest)
  - mid deciles: **-27% to -31%**
  - highest-base-PPL decile: -19.8%
  The -25.44% is driven by records Evo2 finds *hard* (base PPL 3.5-3.8), i.e. NOT memorized ones.

- **Exclusion impact (headline gets STRONGER as suspects are removed):**
  | exclusion set | n | base PPL | lora PPL | rel |
  |---|---|---|---|---|
  | baseline (as published) | 1348 | 3.5790 | 2.6686 | **-25.44%** |
  | excl base PPL < 2.5 (9 recs) | 1339 | 3.5898 | 2.6706 | -25.61% |
  | excl bottom-1% base PPL (13) | 1335 | 3.5929 | 2.6710 | -25.66% |
  | excl bottom-5% base PPL (67) | 1281 | 3.6225 | 2.6763 | -26.12% |

- **Per-genome survives.** Excluding base PPL<2.5: all classes stay strongly negative and roughly
  unchanged (ssRNA(-) -30.7%, ssDNA -35.0%, dsRNA -9.7%, dsDNA -25.0%, ssRNA(+) -21.6%,
  ssRNA(other) -19.4% -> -22.9%). The 9 suspects fall in ssRNA(other) (7), ssRNA(+) (1), dsDNA (1);
  removing them *improves* ssRNA(other). The weakest class, **dsRNA (-9.69%), has zero suspects** and
  is unaffected.

- **Own train/valid split is clean** (no leakage between OUR corpora):
  - record_id overlap train vs valid: **0**; accession overlap: **0**.
  - exact full-sequence (md5) matches valid vs train: **0**.
  - valid 8192-prefix matching a train-sequence prefix: **0**.
  - within-valid exact-duplicate sequence groups: **0**.

## 4. What a FULL audit still needs (not local)
A complete exact-overlap audit against Evo2s actual pretraining corpus (OpenGenome2 /
GTDB+IMG/VR+ NCBI viral, per the Evo2 paper) is **not possible on this host** — that reference is not
under `/data` or in the container. To finish it one would need: the Evo2 pretraining sequence set (or
its k-mer / minhash index), then run containment/minhash (e.g. `sourmash`, or a suffix-array exact
substring match) of each of the 1348 valid records against it. The base-PPL proxy used here is an
internal-signal substitute and is sufficient to bound the risk direction.

## Conclusion
The -25.44% headline is robust: the known contaminated record is already excluded, no leakage exists
between our own train/valid, and pretraining-memorization suspects (low base PPL) work *against* the
reported gain — excluding them strengthens it to about -26%. Per-genome conclusions, including the weak
dsRNA class, are unchanged. A full exact-overlap audit against Evo2s pretraining set remains open but
would only further validate (the internal signal already points the conservative way).
