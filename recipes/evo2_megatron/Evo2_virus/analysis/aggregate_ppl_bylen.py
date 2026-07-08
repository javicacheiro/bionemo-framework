#!/usr/bin/env python3
"""Length-stratified base-vs-LoRA perplexity (context-benefit test).

Compares base, 16k-trained LoRA, and 128k-trained LoRA on the SAME (uncapped,
up to 131072 bp) valid records, stratified by ORIGINAL record length. The whole
point: records <=8192 are a control (all models see identical context); records
>8192 are where a 128k-trained adapter *could* beat a 16k-trained one if long
context actually helps this corpus.

Usage:
  aggregate_ppl_bylen.py <base_dir> <lora16k_dir> <lora128k_dir> \
      <record_lengths.tsv> <manifest.tsv.gz>
"""
import glob, gzip, json, math, os, sys
import torch


def _load_one(pred_dir):
    pt = sorted(glob.glob(os.path.join(pred_dir, "predictions__rank_*.pt")))
    assert pt, f"no predictions__rank_*.pt in {pred_dir}"
    idx2lp = {}
    for p in pt:
        d = torch.load(p, weights_only=False)
        for i, lp in zip(d["seq_idx"].tolist(), d["log_probs_seqs"].tolist()):
            idx2lp[int(i)] = float(lp)
    with open(os.path.join(pred_dir, "seq_idx_map.json")) as f:
        smap = json.load(f)
    rec2idx = {}
    for k, v in smap.items():
        if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
            rec2idx[k] = int(v)
        else:
            rec2idx[str(v)] = int(k)
    return {rec: idx2lp[i] for rec, i in rec2idx.items() if i in idx2lp}


def load_pred(pred_dirs):
    """Accept one dir or a comma-separated list of dirs (short+long passes); merge."""
    merged = {}
    for d in pred_dirs.split(","):
        d = d.strip()
        if d:
            merged.update(_load_one(d))
    return merged


def load_lengths(path):
    rec2len = {}
    with open(path) as f:
        next(f)
        for line in f:
            r, l = line.rstrip("\n").split("\t")
            rec2len[r] = int(l)
    return rec2len


def genome_bucket(g):
    g0 = g.split(";")[0].strip()
    for pre in ["ssRNA(+)", "ssRNA(-)"]:
        if g0.startswith(pre):
            return pre
    if g0.startswith("ssRNA"):
        return "ssRNA(other)"
    for pre in ["dsRNA", "dsDNA", "ssDNA"]:
        if g0.startswith(pre):
            return pre
    return g0 or "unknown"


def load_manifest(path):
    rec2g = {}
    with gzip.open(path, "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ri, gi = hdr.index("record_id"), hdr.index("genome")
        for line in f:
            c = line.rstrip("\n").split("\t")
            rec2g[c[ri]] = c[gi]
    return rec2g


def len_bucket(n):
    if n <= 8192:      return "A_<=8192 (control)"
    if n <= 16384:     return "B_8192-16k"
    if n <= 32768:     return "C_16k-32k"
    if n <= 131072:    return "D_32k-128k"
    return "E_>128k (capped 131072)"


BUCKETS = ["A_<=8192 (control)", "B_8192-16k", "C_16k-32k", "D_32k-128k", "E_>128k (capped 131072)"]
ppl = lambda lp: math.exp(-lp)


def main():
    base_d, l16_d, l128_d, len_path, man_path = sys.argv[1:6]
    base, l16, l128 = load_pred(base_d), load_pred(l16_d), load_pred(l128_d)
    rec2len = load_lengths(len_path)
    rec2g = load_manifest(man_path)
    # records scored by ALL three models (fair comparison)
    common = set(base) & set(l16) & set(l128) & set(rec2len)
    print(f"records scored by all 3 models: {len(common)} "
          f"(base {len(base)}, 16k {len(l16)}, 128k {len(l128)})\n")

    def stats(recs):
        recs = [r for r in recs if r in common]
        if not recs:
            return None
        b = sum(ppl(base[r]) for r in recs) / len(recs)
        a16 = sum(ppl(l16[r]) for r in recs) / len(recs)
        a128 = sum(ppl(l128[r]) for r in recs) / len(recs)
        return len(recs), b, a16, a128

    hdr = f"{'length bucket':<26} {'n':>5} {'base':>7} {'16kLoRA':>9} {'128kLoRA':>10} " \
          f"{'16k %':>8} {'128k %':>8} {'128k-16k':>9}"
    print("=== BY RECORD LENGTH (the context test) ===")
    print(hdr); print("-" * len(hdr))
    by_bucket = {}
    for r in common:
        by_bucket.setdefault(len_bucket(rec2len[r]), []).append(r)
    for bk in BUCKETS:
        s = stats(by_bucket.get(bk, []))
        if not s:
            print(f"{bk:<26} {'0':>5}")
            continue
        n, b, a16, a128 = s
        p16 = 100 * (a16 - b) / b
        p128 = 100 * (a128 - b) / b
        edge = 100 * (a128 - a16) / a16  # negative => 128k better than 16k
        print(f"{bk:<26} {n:>5} {b:>7.3f} {a16:>9.3f} {a128:>10.3f} "
              f"{p16:>7.2f}% {p128:>7.2f}% {edge:>8.2f}%")
    s = stats(list(common))
    n, b, a16, a128 = s
    print("-" * len(hdr))
    print(f"{'OVERALL':<26} {n:>5} {b:>7.3f} {a16:>9.3f} {a128:>10.3f} "
          f"{100*(a16-b)/b:>7.2f}% {100*(a128-b)/b:>7.2f}% {100*(a128-a16)/a16:>8.2f}%")
    print("\n(16k %/128k % = LoRA vs base; 128k-16k = 128k-LoRA vs 16k-LoRA, "
          "negative => longer-context training helps. Buckets C/D/E are the ones "
          "that need >16k context.)\n")

    # genome-class breakdown restricted to long records (>8192), where context can matter
    longrecs = [r for r in common if rec2len[r] > 8192]
    print(f"=== GENOME CLASS, LONG RECORDS ONLY (>8192 bp; n={len(longrecs)}) ===")
    print(hdr); print("-" * len(hdr))
    by_g = {}
    for r in longrecs:
        by_g.setdefault(genome_bucket(rec2g.get(r, "unknown")), []).append(r)
    for g in sorted(by_g, key=lambda k: -len(by_g[k])):
        s = stats(by_g[g])
        if not s:
            continue
        n, b, a16, a128 = s
        print(f"{g:<26} {n:>5} {b:>7.3f} {a16:>9.3f} {a128:>10.3f} "
              f"{100*(a16-b)/b:>7.2f}% {100*(a128-b)/b:>7.2f}% {100*(a128-a16)/a16:>8.2f}%")


if __name__ == "__main__":
    main()
