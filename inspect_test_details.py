import argparse, os, glob, pickle, gzip, json, re
from typing import Any, Dict, List, Optional
from pprint import pprint

def load_pkl_any(path: str):
    with open(path, "rb") as f:
        sig = f.read(2)
    if sig == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)

def human_yes(b: bool) -> str:
    return "✓" if b else "·"

def trim_list(lst, max_items=10, show_full=False):
    if show_full or not isinstance(lst, list):
        return lst
    if len(lst) <= max_items:
        return lst
    return lst[:max_items] + [f"...(+{len(lst)-max_items} more)"]

def to_table_row(idx: int,
                 mask_row: List[bool],
                 pred_scores_full: List[float],
                 true_scores_full: List[float],
                 line_lists: List[List[int]],
                 ast_lists: List[List[int]],
                 cfg_lists: List[List[int]],
                 pdg_lists: List[List[int]],
                 topk_idx: Optional[List[int]]):
    m = bool(mask_row[idx]) if (mask_row and idx < len(mask_row)) else False
    sp = pred_scores_full[idx] if (pred_scores_full and idx < len(pred_scores_full)) else None
    st = true_scores_full[idx] if (true_scores_full and idx < len(true_scores_full)) else None
    L = len(line_lists[idx]) if (line_lists and idx < len(line_lists)) else 0
    A = len(ast_lists[idx])  if (ast_lists  and idx < len(ast_lists))  else 0
    C = len(cfg_lists[idx])  if (cfg_lists  and idx < len(cfg_lists))  else 0
    P = len(pdg_lists[idx])  if (pdg_lists  and idx < len(pdg_lists))  else 0
    is_topk = (topk_idx is not None and idx in set(topk_idx))
    return [idx, human_yes(m), is_topk, sp, st, L, A, C, P]

def print_region_table(item: Dict[str, Any], show_full=False, max_regions_to_show=999):
    mask      = item.get("region_mask", [])
    pred_full = item.get("region_pred_scores_full", [])
    true_full = item.get("region_true_scores_full", [])
    topk_idx  = item.get("topk_idx", None)

    line_lists = item.get("region_line_indices", [])
    ast_lists  = item.get("region_ast_nodes", [])
    cfg_lists  = item.get("region_cfg_nodes", [])
    pdg_lists  = item.get("region_pdg_nodes", [])

    K = max(len(mask), len(pred_full), len(true_full), len(line_lists), len(ast_lists), len(cfg_lists), len(pdg_lists))
    K = min(K, max_regions_to_show)

    header = ["rid", "valid", "topK", "pred", "true", "#lines", "#ast", "#cfg", "#pdg"]
    print("\n[Region overview]")
    print("  " + " | ".join(header))
    print("  " + "-" * (len(" | ".join(header)) + 2))

    for r in range(K):
        row = to_table_row(r, mask, pred_full, true_full, line_lists, ast_lists, cfg_lists, pdg_lists, topk_idx)
        print("  {:>3} | {} | {} | {:>6} | {:>6} | {:>6} | {:>5} | {:>5} | {:>5}".format(
            row[0], row[1], "★" if row[2] else " ",
            f"{row[3]:.3f}" if row[3] is not None else "  -  ",
            f"{row[4]:.3f}" if row[4] is not None else "  -  ",
            row[5], row[6], row[7], row[8]
        ))

    print("\n[Per-region details]")
    for r in range(K):
        print(f"  - region {r}:")
        if "region_line_indices_full" in item and "region_line_mask" in item:
            print("      line_indices_full (with pad):", trim_list(item["region_line_indices_full"][r], 20, show_full))
            print("      line_mask_row:               ", trim_list(item["region_line_mask"][r], 20, show_full))
        if "region_line_indices" in item:
            print("      line_indices_filtered:       ", trim_list(item["region_line_indices"][r], 20, show_full))

        if "region_ast_nodes_full" in item and "region_ast_node_mask" in item:
            print("      ast_nodes_full:              ", trim_list(item["region_ast_nodes_full"][r], 20, show_full))
            print("      ast_node_mask_row:           ", trim_list(item["region_ast_node_mask"][r], 20, show_full))
        if "region_ast_nodes" in item:
            print("      ast_nodes_global:            ", trim_list(item["region_ast_nodes"][r], 20, show_full))
        if "region_ast_nodes_local" in item:
            print("      ast_nodes_local:             ", trim_list(item["region_ast_nodes_local"][r], 20, show_full))

        if "region_cfg_nodes_full" in item and "region_cfg_node_mask" in item:
            print("      cfg_nodes_full:              ", trim_list(item["region_cfg_nodes_full"][r], 20, show_full))
            print("      cfg_node_mask_row:           ", trim_list(item["region_cfg_node_mask"][r], 20, show_full))
        if "region_cfg_nodes" in item:
            print("      cfg_nodes_global:            ", trim_list(item["region_cfg_nodes"][r], 20, show_full))
        if "region_cfg_nodes_local" in item:
            print("      cfg_nodes_local:             ", trim_list(item["region_cfg_nodes_local"][r], 20, show_full))

        if "region_pdg_nodes_full" in item and "region_pdg_node_mask" in item:
            print("      pdg_nodes_full:              ", trim_list(item["region_pdg_nodes_full"][r], 20, show_full))
            print("      pdg_node_mask_row:           ", trim_list(item["region_pdg_node_mask"][r], 20, show_full))
        if "region_pdg_nodes" in item:
            print("      pdg_nodes_global:            ", trim_list(item["region_pdg_nodes"][r], 20, show_full))
        if "region_pdg_nodes_local" in item:
            print("      pdg_nodes_local:             ", trim_list(item["region_pdg_nodes_local"][r], 20, show_full))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", nargs="+", help="test_details_*.pkl (supports wildcards)")
    ap.add_argument("--name", required=True, help="Filename keywords (default substring matching), or add `--exact` for exact matching.")
    ap.add_argument("--exact", action="store_true", help="Should we use exact matching (==)?")
    ap.add_argument("--show-full", action="store_true", help="Print the complete array (it may be very long).")
    ap.add_argument("--limit", type=int, default=999999, help="How many matching samples can be displayed at most?")
    ap.add_argument("--max-regions", type=int, default=999, help="How many regions can be displayed per sample at most?")
    ap.add_argument("--save-json", type=str, default=None, help="Save the complete matching results to a JSON file.")
    args = ap.parse_args()

    paths = []
    for p in args.pkl:
        paths.extend(glob.glob(p))
    if not paths:
        raise FileNotFoundError("No PKL matched.")
    all_items: List[Dict[str, Any]] = []
    for p in paths:
        items = load_pkl_any(p)
        if not isinstance(items, list):
            print(f"[WARN] {p} is not a list. Skipped.")
            continue
        all_items.extend(items)

    if args.exact:
        filt = [it for it in all_items if str(it.get("file_name", "")) == args.name]
    else:
        filt = [it for it in all_items if args.name in str(it.get("file_name", ""))]

    print(f"Matched items: {len(filt)}")
    if not filt:
        return

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(filt, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] full matches dumped to: {args.save_json}")

    for i, it in enumerate(filt[: args.limit]):
        fname = it.get("file_name", "<unknown>")
        y_true = it.get("y_true", None)
        y_prob = it.get("y_prob", None)
        y_pred = it.get("y_pred", None)
        print("\n" + "="*100)
        print(f"[{i+1}/{min(len(filt), args.limit)}] file_name: {fname}")
        print(f"  y_true={y_true}  y_pred={y_pred}  y_prob={y_prob:.6f}" if isinstance(y_prob, (int,float)) else
              f"  y_true={y_true}  y_pred={y_pred}  y_prob={y_prob}")

        if "topk_idx" in it and it["topk_idx"] is not None:
            print(f"  topk_idx: {it['topk_idx']}")

        print_region_table(it, show_full=args.show_full, max_regions_to_show=args.max_regions)

    print("\nDone.")

if __name__ == "__main__":
    main()
