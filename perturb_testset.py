#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, random
from typing import Optional, List, Dict, Tuple
from collections import Counter

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "int", "long", "register", "return",
    "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned",
    "void", "volatile", "while", "inline", "restrict", "_Bool", "_Complex", "_Imaginary",
    "bool", "true", "false", "nullptr"
}
NON_API_CALLEES = C_KEYWORDS | {"while", "for", "if", "switch", "return", "sizeof"}

IDENT_RE = re.compile(r"[A-Za-z_]\w+")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT_RE = re.compile(r"//.*?$", re.M)
STRING_RE = re.compile(r"\"([^\"\\]|\\.)*\"|\'([^\'\\]|\\.)*\'", re.S)


def strip_comments_and_strings(code: str) -> str:
    code = BLOCK_COMMENT_RE.sub(" ", code)
    code = LINE_COMMENT_RE.sub(" ", code)
    code = STRING_RE.sub(" ", code)
    return code


def read_text(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("latin-1", errors="ignore")


def write_text(path: str, s: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def load_list(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def label_from_name(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except Exception:
        return 0


def guess_c_path(c_dir: str, name: str) -> Optional[str]:
    p1 = os.path.join(c_dir, f"{name}.c")
    if os.path.isfile(p1):
        return p1
    base = name.split("_", 1)[0]
    p2 = os.path.join(c_dir, f"{base}.c")
    if os.path.isfile(p2):
        return p2
    return None


def extract_calls(code_clean: str) -> Counter:
    c = Counter()
    for m in CALL_RE.finditer(code_clean):
        callee = m.group(1)
        if callee and (callee not in NON_API_CALLEES):
            c[callee] += 1
    return c


def extract_identifiers(code_clean: str) -> Counter:
    c = Counter()
    for tok in IDENT_RE.findall(code_clean):
        if tok not in C_KEYWORDS:
            c[tok] += 1
    return c


def load_train_stats(stats_json: str) -> Dict[str, Dict[str, int]]:
    with open(stats_json, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- TOKEN 干预 ----------
def choose_replaceable_idents(code_clean: str, calls_in_file: set) -> List[str]:
    """
    Conservative choices for replaceable identifiers:

    - Use ordinary identifiers (non-keywords)

    - Do not appear as a caller (avoid function names/macro-style usage)

    - Note: No scope-level alpha-rename is performed; only text-based replacement is used (for a more conservative approach).
    """
    all_idents = [tok for tok in IDENT_RE.findall(code_clean) if tok not in C_KEYWORDS]
    cand = []
    for t in set(all_idents):
        if t not in calls_in_file:
            cand.append(t)
    return cand


def replace_idents(code: str, mapping: Dict[str, str]) -> Tuple[str, int]:
    """
    Perform a uniform replacement of "whole word boundaries" on the code; count the number of replacements (total number of occurrences).
    """
    if not mapping:
        return code, 0
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    replaced = 0
    for src, tgt in items:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])")
        code, n = pattern.subn(tgt, code)
        replaced += n
    return code, replaced


def token_perturb(code: str, code_clean: str, s: float, vocab: List[str],
                  rng: random.Random, max_ratio: float = 1.0) -> Tuple[str, int, int]:
    """
    s∈[0,1]: Intensity. Select the names of e "replaceable identifiers" according to e = floor(s * E) and replace them with words from the vocab.

    max_ratio: The maximum number of replacements of E (default 100%)

    Returns: New code, number of "distinct identifiers" replaced, total number of replacements
    """
    calls_set = set(extract_calls(code_clean).keys())
    candidates = choose_replaceable_idents(code_clean, calls_set)
    if not candidates:
        return code, 0, 0

    E = len(candidates)
    e = max(0, min(E, int(round(s * E * max_ratio))))
    if e == 0:
        return code, 0, 0

    rng.shuffle(candidates)
    src_idents = candidates[:e]

    mapping = {}
    for src in src_idents:
        for _ in range(10):
            tgt = rng.choice(vocab)
            if tgt != src and tgt not in C_KEYWORDS:
                mapping[src] = tgt
                break

    new_code, total_occ = replace_idents(code, mapping)
    return new_code, len(mapping), total_occ

FUNC_DEF_RE = re.compile(r"""
    (^|\n)                              
    ([A-Za-z_]\w*[\s\*]+)+              
    ([A-Za-z_]\w*)\s*                   
    \(([^;{}()]|\([^)]*\))*\)\s*        
    \{                                  
""", re.X)


def insert_dead_calls_in_functions(code: str, apis: List[str], m_budget: int, rng: random.Random) -> Tuple[str, int]:
    """
    After the opening curly brace of each function body, insert at most m unreachable API calls, according to the budget.

    Template: if(0){ api(...); }

    Returns: New code, total number of insertions
    """
    if m_budget <= 0 or not apis:
        return code, 0

    lines = code.splitlines(keepends=True)
    text = "".join(lines)

    inserts = []  # (insert_pos, text_to_insert)
    total_inserted = 0
    for m in FUNC_DEF_RE.finditer(text):
        brace_pos = m.end()
        ins_this = rng.randint(0, m_budget)
        for _ in range(ins_this):
            api = rng.choice(apis)
            snippet = f"\n#if 0\n(void){api}();\n#endif\n"
            inserts.append((brace_pos, snippet))
            total_inserted += 1

    if not inserts:
        return code, 0

    inserts.sort(key=lambda x: x[0], reverse=True)
    buf = text
    for pos, snip in inserts:
        buf = buf[:pos] + snip + buf[pos:]
    return buf, total_inserted


def main():
    ap = argparse.ArgumentParser(description="Perturb C files in test set by strength s (0..1).")
    ap.add_argument("--mode", choices=["token", "api"], required=True, help="Intervention type: token or api")
    ap.add_argument("--direction", choices=["wash_out", "contam_in", "auto"], default="auto",
                    help="token: wash_out = introduces a non-vulnerable vocabulary into the vulnerability sample; contam_in = introduces a vulnerability vocabulary into the non-vulnerable sample; auto = both follow their respective rules.")
    ap.add_argument("--strength", type=float, required=True, help="Intensity s ∈ [0,1]")
    ap.add_argument("--topk", type=int, default=50, help="Size of the top-k vocabulary taken from the training distribution")
    ap.add_argument("--api-budget", type=int, default=5, help="API insertion budget (maximum number of insertions per function)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c-dir", required=True, help="Test suite C source code directory")
    ap.add_argument("--test-list", required=True, help="List of test set filenames (*.txt)")
    ap.add_argument("--train-stats", required=True, help="Training set statistics JSON (train_token_api_freq.json)")
    ap.add_argument("--out-dir", required=True, help="Output the perturbated C file directory")
    ap.add_argument("--log-jsonl", default="perturb_log.jsonl", help="Output log (JSONL)")
    args = ap.parse_args()

    assert 0.0 <= args.strength <= 1.0, "strength s must be in [0,1]"

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    stats = load_train_stats(args.train_stats)
    token_vuln = list(stats["token_vuln"].keys())[:args.topk]
    token_nonv = list(stats["token_nonvuln"].keys())[:args.topk]
    api_vuln = list(stats["api_vuln"].keys())[:args.topk]
    api_nonv = list(stats["api_nonvuln"].keys())[:args.topk]

    names = load_list(args.test_list)
    logf = open(args.log_jsonl, "w", encoding="utf-8")

    total_done, total_changed = 0, 0

    for name in names:
        y = label_from_name(name)  # 1=vuln, 0=nonvuln
        c_path = guess_c_path(args.c_dir, name)
        if not c_path or not os.path.isfile(c_path):
            logf.write(json.dumps({"id": name, "status": "missing"}) + "\n")
            continue

        code = read_text(c_path)
        code_clean = strip_comments_and_strings(code)

        changed = False
        rec = {"id": name, "label": y, "mode": args.mode, "s": args.strength, "seed": args.seed}

        if args.mode == "token":
            direction = args.direction
            if direction == "auto":
                direction = "wash_out" if y == 1 else "contam_in"
            vocab = token_nonv if direction == "wash_out" else token_vuln

            new_code, n_idents, n_total = token_perturb(
                code=code,
                code_clean=code_clean,
                s=args.strength,
                vocab=vocab,
                rng=rng,
                max_ratio=1.0
            )
            rec.update({"direction": direction, "idents_changed": n_idents, "occurrences": n_total})
            if n_total > 0:
                code = new_code
                changed = True

        elif args.mode == "api":
            apis = api_nonv if y == 1 else api_vuln
            m_budget = int(round(args.strength * args.api_budget))
            new_code, n_ins = insert_dead_calls_in_functions(code, apis, m_budget, rng)
            rec.update({"per_func_budget": m_budget, "api_inserted": n_ins})
            if n_ins > 0:
                code = new_code
                changed = True

        out_path = os.path.join(args.out_dir, f"{name}.c")
        if changed:
            write_text(out_path, code)
            rec["status"] = "ok"
            total_changed += 1
        else:
            write_text(out_path, code)
            rec["status"] = "no_change"

        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total_done += 1

    logf.close()
    print(f"[Perturb] done={total_done}, changed={total_changed}, out_dir={args.out_dir}, log={args.log_jsonl}")
    print("Hint: The next step is to use this .c code to run Joern → Preprocessing → Inference, and calculate ΔF1/IS.")


if __name__ == "__main__":
    main()
