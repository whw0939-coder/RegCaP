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

# ---------- TOKEN  ----------
def choose_replaceable_idents(code_clean: str, calls_in_file: set) -> List[str]:
    all_idents = [tok for tok in IDENT_RE.findall(code_clean) if tok not in C_KEYWORDS]
    cand = []
    for t in set(all_idents):
        if t not in calls_in_file:
            cand.append(t)
    return cand

def replace_idents(code: str, mapping: Dict[str, str]) -> Tuple[str, int]:
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

# ---------- API ----------
FUNC_DEF_RE = re.compile(r"""
    (^|\n)
    ([A-Za-z_]\w*[\s\*]+)+
    ([A-Za-z_]\w*)\s*
    \(([^;{}()]|\([^)]*\))*\)\s*
    \{
""", re.X)

def insert_dead_calls_in_functions(code: str, apis: List[str], m_budget: int, rng: random.Random) -> Tuple[str, int]:
    if m_budget <= 0 or not apis:
        return code, 0

    text = code  # keep line endings
    inserts = []
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

# ---------- （ token / api / both） ----------
def process_one_file(name: str,
                     y: int,
                     c_dir: str,
                     mode: str,
                     s_token: float,
                     s_api: float,
                     direction_token: str,
                     api_budget: int,
                     rng: random.Random,
                     token_nonv: List[str],
                     token_vuln: List[str],
                     api_nonv: List[str],
                     api_vuln: List[str]) -> Tuple[Optional[str], dict]:
    rec = {"id": name, "label": y, "mode": mode, "s_token": s_token, "s_api": s_api}

    c_path = guess_c_path(c_dir, name)
    if not c_path or not os.path.isfile(c_path):
        rec["status"] = "missing"
        return None, rec

    code = read_text(c_path)
    code_clean = strip_comments_and_strings(code)
    changed = False

    # === token ===
    if mode in ("token", "both"):
        direction = direction_token
        if direction == "auto":
            direction = "wash_out" if y == 1 else "contam_in"
        vocab = token_nonv if direction == "wash_out" else token_vuln

        new_code, n_idents, n_total = token_perturb(
            code=code,
            code_clean=code_clean,
            s=s_token,
            vocab=vocab,
            rng=rng,
            max_ratio=1.0
        )
        rec.update({"token_direction": direction, "token_idents_changed": n_idents, "token_occurrences": n_total})
        if n_total > 0:
            code = new_code
            code_clean = strip_comments_and_strings(code)
            changed = True

    # === api ===
    if mode in ("api", "both"):
        apis = api_nonv if y == 1 else api_vuln
        m_budget = int(round(s_api * api_budget))
        new_code, n_ins = insert_dead_calls_in_functions(code, apis, m_budget, rng)
        rec.update({"api_per_func_budget": m_budget, "api_inserted": n_ins})
        if n_ins > 0:
            code = new_code
            changed = True

    rec["status"] = "ok" if changed else "no_change"
    return code, rec

def run_single(args,
               s_token: float,
               s_api: float,
               out_dir: str,
               rng_seed: int):
    rng = random.Random(rng_seed)
    os.makedirs(out_dir, exist_ok=True)

    stats = load_train_stats(args.train_stats)
    token_vuln = list(stats["token_vuln"].keys())[:args.topk]
    token_nonv = list(stats["token_nonvuln"].keys())[:args.topk]
    api_vuln = list(stats["api_vuln"].keys())[:args.topk]
    api_nonv = list(stats["api_nonvuln"].keys())[:args.topk]

    names = load_list(args.test_list)
    log_path = os.path.join(out_dir, args.log_jsonl)
    with open(log_path, "w", encoding="utf-8") as logf:
        total_done, total_changed = 0, 0
        for name in names:
            y = label_from_name(name)
            new_code, rec = process_one_file(
                name=name,
                y=y,
                c_dir=args.c_dir,
                mode=args.mode,
                s_token=s_token,
                s_api=s_api,
                direction_token=args.direction,
                api_budget=args.api_budget,
                rng=rng,
                token_nonv=token_nonv, token_vuln=token_vuln,
                api_nonv=api_nonv, api_vuln=api_vuln
            )
            if new_code is None:
                logf.write(json.dumps(rec) + "\n")
                continue

            out_path = os.path.join(out_dir, f"{name}.c")
            write_text(out_path, new_code)
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_done += 1
            if rec["status"] == "ok":
                total_changed += 1

    print(f"[Perturb {args.mode}] s_token={s_token:.2f} s_api={s_api:.2f} -> done={total_done}, changed={total_changed}, out_dir={out_dir}, log={log_path}")

def main():
    ap = argparse.ArgumentParser(description="Perturb C files in test set by strength s (0..1).")
    ap.add_argument("--mode", choices=["token", "api", "both"], required=True, help="Intervention type: token / api / both (both combined)")
    ap.add_argument("--direction", choices=["wash_out", "contam_in", "auto"], default="auto",
                    help="token: wash_out = introduces a non-vulnerability terminology into the vulnerability sample; contam_in = introduces a vulnerability terminology into the non-vulnerability sample; auto = automatically selects based on the sample.")
    ap.add_argument("--strength", type=float, default=None,
                    help="Single-axis/single-value strength (use in token or api mode; for both mode, it is recommended to use --strength-token / --strength-api)")
    ap.add_argument("--strength-token", type=float, default=None, help="In both modes, the token strength s_token ∈ [0,1]")
    ap.add_argument("--strength-api", type=float, default=None, help="In both modes, the API strength s_api ∈ [0,1]")

    # 5×5 网格批量
    ap.add_argument("--grid", action="store_true", help="In both mode: Generate a set of intensity combinations at once by grid.")
    ap.add_argument("--grid-strengths", type=str, default="0.2,0.4,0.6,0.8,1.0",
                    help="A comma-separated list of intensity levels (5 levels by default). Use only with both+--grid.")

    ap.add_argument("--topk", type=int, default=50, help="Size of the top-k vocabulary taken from the training distribution")
    ap.add_argument("--api-budget", type=int, default=5, help="API Insertion Budget (Maximum number of inserts per function)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--c-dir", required=True, help="Test suite C source code directory")
    ap.add_argument("--test-list", required=True, help="List of test set filenames (*.txt)")
    ap.add_argument("--train-stats", required=True, help="Training set statistics JSON (train_token_api_freq.json)")
    ap.add_argument("--out-dir", required=True, help="Output the perturbed C file in the **root directory** (the script will create subdirectories under it).")
    ap.add_argument("--log-jsonl", default="perturb_log.jsonl", help="Output log file names (written in each subdirectory).")
    args = ap.parse_args()

    if args.mode in ("token", "api"):
        if args.strength is None:
            raise ValueError("--strength must be specified when using --mode token/api")
        assert 0.0 <= args.strength <= 1.0, "--strength must be in the range [0,1]"
    else:
        # both
        if args.grid:
            strengths = [float(x.strip()) for x in args.grid_strengths.split(",") if x.strip()]
            if not strengths:
                raise ValueError("--grid-strengths resolves to empty")
            for s in strengths:
                assert 0.0 <= s <= 1.0, "The `--grid-strengths` values must all be in the range [0,1]."
        else:
            if args.strength_token is None or args.strength_api is None:
                raise ValueError("When using `--mode both` instead of `--grid`, both `--strength-token` and `--strength-api` must be provided.")
            assert 0.0 <= args.strength_token <= 1.0, "The `--strength-token` must be in the range [0,1]."
            assert 0.0 <= args.strength_api <= 1.0, "--strength-api must be in [0,1]"

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode in ("token", "api"):
        s = args.strength
        sub = f"{args.mode}_s{int(round(s*100)):03d}"
        out_dir = os.path.join(args.out_dir, sub)
        s_token = s if args.mode == "token" else 0.0
        s_api   = s if args.mode == "api"   else 0.0
        run_single(args, s_token=s_token, s_api=s_api, out_dir=out_dir, rng_seed=args.seed)
        return

    if args.grid:
        strengths = [float(x.strip()) for x in args.grid_strengths.split(",") if x.strip()]
        strengths = sorted(set(strengths))
        for i, s_tok in enumerate(strengths):
            for j, s_api in enumerate(strengths):
                sub = f"both_s{int(round(s_tok*100)):03d}x{int(round(s_api*100)):03d}"
                out_dir = os.path.join(args.out_dir, sub)
                rng_seed = args.seed + i*100 + j
                run_single(args, s_token=s_tok, s_api=s_api, out_dir=out_dir, rng_seed=rng_seed)
        print(f"[GRID] both-mode complete, root directory:{args.out_dir}")
    else:
        s_tok = args.strength_token
        s_api = args.strength_api
        sub = f"both_s{int(round(s_tok*100)):03d}x{int(round(s_api*100)):03d}"
        out_dir = os.path.join(args.out_dir, sub)
        run_single(args, s_token=s_tok, s_api=s_api, out_dir=out_dir, rng_seed=args.seed)

if __name__ == "__main__":
    main()
