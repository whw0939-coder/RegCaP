#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, json, os, re
from typing import Dict, List, Tuple

def load_pred_jsonl(path: str) -> Dict[str, dict]:
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            key = r.get("file")
            if key is None:
                key = f"__idx_{len(d)}"
            d[key] = r
    return d

def binarize_prob(prob: float, thr: float) -> int:
    return 1 if prob >= thr else 0

def confusion_counts(records: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    tp = fp = fn = tn = 0
    for y, p in records:
        if y==1 and p==1: tp+=1
        elif y==0 and p==1: fp+=1
        elif y==1 and p==0: fn+=1
        else: tn+=1
    return tp, fp, fn, tn

def f1_from_counts(tp:int, fp:int, fn:int) -> float:
    denom = 2*tp + fp + fn
    return (2*tp/denom) if denom>0 else 0.0

# Supports three naming methods:
#   ..._api_s020.jsonl
#   ..._token_s060.jsonl
#   ..._both_s020x040.jsonl
def parse_mode_strength(fname: str):
    base = os.path.basename(fname)
    m1 = re.search(r'_(api|token)_s(\d{3})', base, re.I)
    if m1:
        mode = m1.group(1).lower()
        s = int(m1.group(2))/100.0
        return {"mode": mode, "s": round(s,2), "s_token": (s if mode=="token" else 0.0), "s_api": (s if mode=="api" else 0.0)}
    m2 = re.search(r'_both_s(\d{3})x(\d{3})', base, re.I)
    if m2:
        s_tok = int(m2.group(1))/100.0
        s_api = int(m2.group(2))/100.0
        return {"mode":"both","s":None,"s_token":round(s_tok,2),"s_api":round(s_api,2)}
    return {"mode":"unknown","s":None,"s_token":None,"s_api":None}

def compute_metrics(base_path: str, pert_paths: List[str], outdir: str, threshold: float, only_correct_flag: bool):
    os.makedirs(outdir, exist_ok=True)

    base = load_pred_jsonl(base_path)
    base_pairs = [(r["y_true"], r.get("pred", binarize_prob(r["prob"], threshold))) for r in base.values()]
    tp_b, fp_b, fn_b, tn_b = confusion_counts(base_pairs)
    F1_base = f1_from_counts(tp_b, fp_b, fn_b)

    results = []
    for p in pert_paths:
        pert = load_pred_jsonl(p)
        keys = sorted(set(base.keys()) & set(pert.keys()))
        if not keys:
            print(f"[WARN] no overlap with base: {p}")
            continue

        agree = 0
        agree_on_base_correct = 0
        correct_base = 0
        pert_pairs = []

        for k in keys:
            rb = base[k]; rp = pert[k]
            y  = int(rb["y_true"])
            pb = int(rb.get("pred", binarize_prob(rb["prob"], threshold)))
            pp = int(rp.get("pred", binarize_prob(rp["prob"], threshold)))
            pert_pairs.append((y, pp))
            if pb == pp: agree += 1
            if pb == y:
                correct_base += 1
                if pb == pp: agree_on_base_correct += 1

        tp, fp, fn, tn = confusion_counts(pert_pairs)
        F1_pert = f1_from_counts(tp, fp, fn)
        deltaF1 = F1_base - F1_pert
        IS = agree/len(keys) if keys else 0.0
        IS_correct = (agree_on_base_correct/correct_base) if correct_base>0 else 0.0

        md = parse_mode_strength(p)
        row = {
            "file": os.path.basename(p),
            "mode": md["mode"],
            "s": md["s"],
            "s_token": md["s_token"],
            "s_api": md["s_api"],
            "N_overlap": len(keys),
            "F1_base": round(F1_base,4),
            "F1_pert": round(F1_pert,4),
            "DeltaF1": round(deltaF1,4),
            "IS": round(IS,4),
            "IS_correct": round(IS_correct,4),
            "threshold": threshold,
        }
        results.append(row)

    def sort_key(r):
        m = r["mode"]
        if m=="both":
            st = r["s_token"]; sa = r["s_api"]
            st = 999 if st is None else st
            sa = 999 if sa is None else sa
            return (0, st, sa, r["file"])
        elif m in ("api","token"):
            s = r["s"]; s = 999 if s is None else s
            return (1, m, s, r["file"])
        else:
            return (2, r["file"])
    results.sort(key=sort_key)

    import csv
    cols = ["mode","s","s_token","s_api","file","N_overlap","F1_base","F1_pert","DeltaF1","IS","IS_correct","threshold"]
    csv_path = os.path.join(outdir, "delta_is_summary.csv")
    with open(csv_path,"w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results: w.writerow(r)
    json_path = os.path.join(outdir, "delta_is_summary.json")
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {json_path}")
    return results

def plot_curves_and_heatmaps(results: List[dict], outdir: str):
    from collections import defaultdict
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    for mode in ("api","token"):
        rows = by_mode.get(mode, [])
        rows = [r for r in rows if isinstance(r.get("s"), float)]
        if not rows: continue
        rows.sort(key=lambda r: r["s"])
        svals = [r["s"] for r in rows]
        dF1   = [r["DeltaF1"] for r in rows]
        IS    = [r["IS"] for r in rows]
        ISc   = [r["IS_correct"] for r in rows]

        # ΔF1 vs s
        plt.figure()
        plt.plot(svals, dF1, marker='o')
        plt.xlabel("s")
        plt.ylabel("ΔF1 (F1_base - F1_pert)")
        plt.title(f"ΔF1 vs s — {mode}")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"DeltaF1_vs_s_{mode}.png"), dpi=200)
        plt.close()

        # IS vs s
        plt.figure()
        plt.plot(svals, IS, marker='o')
        plt.xlabel("s")
        plt.ylabel("IS (prediction consistency)")
        plt.title(f"IS vs s — {mode}")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"IS_vs_s_{mode}.png"), dpi=200)
        plt.close()

        # IS@correct vs s
        plt.figure()
        plt.plot(svals, ISc, marker='o')
        plt.xlabel("s")
        plt.ylabel("IS@correct")
        plt.title(f"IS@correct vs s — {mode}")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"IS_correct_vs_s_{mode}.png"), dpi=200)
        plt.close()

    # （ΔF1 / IS / IS@correct）
    rows = by_mode.get("both", [])
    rows = [r for r in rows if isinstance(r.get("s_token"), float) and isinstance(r.get("s_api"), float)]
    if rows:
        xs = sorted(sorted({r["s_token"] for r in rows}))
        ys = sorted(sorted({r["s_api"] for r in rows}))
        X_idx = {v:i for i,v in enumerate(xs)}
        Y_idx = {v:i for i,v in enumerate(ys)}

        def make_grid(key: str):
            grid = np.full((len(xs), len(ys)), np.nan, dtype=float)
            for r in rows:
                i = X_idx[r["s_token"]]; j = Y_idx[r["s_api"]]
                grid[i,j] = r[key]
            return grid

        grids = {
            "DeltaF1": make_grid("DeltaF1"),
            "IS": make_grid("IS"),
            "IS_correct": make_grid("IS_correct"),
        }

        for title, M in grids.items():
            plt.figure()
            plt.imshow(M, origin='lower', aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.xticks(range(len(ys)), [f"{v:.2f}" for v in ys], rotation=0)
            plt.yticks(range(len(xs)), [f"{v:.2f}" for v in xs])
            plt.xlabel("s_api")
            plt.ylabel("s_token")
            plt.title(f"{title} heatmap — both")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{title}_heatmap_both.png"), dpi=200)
            plt.close()

def main():
    ap = argparse.ArgumentParser(description="Compute ΔF1/IS for api, token, and both (2D) perturbations; optionally plot.")
    ap.add_argument("--base", required=True, help="Base jsonl (unperturbed)")
    ap.add_argument("--inputs", nargs="+", required=True, help="Perturbed jsonl list or globs")
    ap.add_argument("--outdir", default="./delta_is", help="Output dir")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold if 'pred' missing")
    ap.add_argument("--plot", action="store_true", help="Generate line charts and heatmaps")
    args = ap.parse_args()

    # expand globs
    files = []
    for pat in args.inputs:
        matched = sorted(glob.glob(pat))
        if matched: files.extend(matched)
        elif os.path.isfile(pat): files.append(pat)
        else: print(f"[WARN] no match: {pat}")
    # dedup
    seen=set(); pert_paths=[]
    for p in files:
        if p not in seen:
            seen.add(p); pert_paths.append(p)

    os.makedirs(args.outdir, exist_ok=True)
    results = compute_metrics(args.base, pert_paths, args.outdir, args.threshold, only_correct_flag=False)
    if args.plot and results:
        plot_curves_and_heatmaps(results, args.outdir)

if __name__ == "__main__":
    main()
