#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
from collections import Counter, defaultdict
from typing import Optional, List

C_KEYWORDS = {
    "auto","break","case","char","const","continue","default","do","double","else",
    "enum","extern","float","for","goto","if","int","long","register","return",
    "short","signed","sizeof","static","struct","switch","typedef","union","unsigned",
    "void","volatile","while","inline","restrict","_Bool","_Complex","_Imaginary",
    "bool","true","false","nullptr",
}

NON_API_CALLEES = C_KEYWORDS | {"while","for","if","switch","return","sizeof"}

IDENT_RE = re.compile(r"[A-Za-z_]\w+")

CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT_RE  = re.compile(r"//.*?$", re.M)

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

def guess_c_path(c_dir: str, name: str) -> Optional[str]:
    p1 = os.path.join(c_dir, f"{name}.c")
    if os.path.isfile(p1):
        return p1
    base = name.split("_", 1)[0]
    p2 = os.path.join(c_dir, f"{base}.c")
    if os.path.isfile(p2):
        return p2
    return None

def is_identifier_token(tok: str) -> bool:
    return bool(IDENT_RE.fullmatch(tok)) and (tok not in C_KEYWORDS)

def tokenize_identifiers(code: str):
    for tok in IDENT_RE.findall(code):
        if tok not in C_KEYWORDS:
            yield tok

def extract_call_names(code: str):
    for m in CALL_RE.finditer(code):
        callee = m.group(1)
        if callee and (callee not in NON_API_CALLEES):
            yield callee

def load_filename_list(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def label_from_name(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-dir", required=True, help="C source code directory, such as /data1/.../devign/c/")
    ap.add_argument("--train-list", required=True, help="A list of training set filenames, such as train_filenames.txt")
    ap.add_argument("--outdir", default="./train_stats", help="Output directory")
    ap.add_argument("--topk", type=int, default=50, help="Print preview of the top-k number")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_names = load_filename_list(args.train_list)

    token_freq = {0: Counter(), 1: Counter()}
    api_freq   = {0: Counter(), 1: Counter()}

    missing, total = 0, 0

    for name in train_names:
        total += 1
        y = label_from_name(name)
        path = guess_c_path(args.c_dir, name)
        if not path:
            missing += 1
            continue

        code = read_text(path)
        code_clean = strip_comments_and_strings(code)

        for tok in tokenize_identifiers(code_clean):
            token_freq[y][tok] += 1

        for callee in extract_call_names(code_clean):
            api_freq[y][callee] += 1

    out = {
        "token_vuln": token_freq[1],
        "token_nonvuln": token_freq[0],
        "api_vuln": api_freq[1],
        "api_nonvuln": api_freq[0],
        "meta": {
            "total_files": total,
            "missing_files": missing,
            "c_dir": args.c_dir,
            "train_list": args.train_list,
        }
    }

    def counter_to_dict(c: Counter) -> dict:
        return {k: int(v) for k, v in c.most_common()}

    json_payload = {
        "token_vuln":     counter_to_dict(token_freq[1]),
        "token_nonvuln":  counter_to_dict(token_freq[0]),
        "api_vuln":       counter_to_dict(api_freq[1]),
        "api_nonvuln":    counter_to_dict(api_freq[0]),
        "meta": out["meta"]
    }

    json_path = os.path.join(args.outdir, "train_token_api_freq.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    for key in ("token_vuln","token_nonvuln","api_vuln","api_nonvuln"):
        with open(os.path.join(args.outdir, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(json_payload[key], f, ensure_ascii=False, indent=2)

    k = args.topk
    def preview(title, c: Counter):
        print(f"\n== {title} (top-{k}) ==")
        for tok, cnt in c.most_common(k):
            print(f"{tok}\t{cnt}")

    print(f"\n[Done] Parsed {total} names, missing {missing}. Results saved to {args.outdir}")
    preview("TOKEN (VULN)", token_freq[1])
    preview("TOKEN (NONVULN)", token_freq[0])
    preview("API (VULN)", api_freq[1])
    preview("API (NONVULN)", api_freq[0])

if __name__ == "__main__":
    main()
