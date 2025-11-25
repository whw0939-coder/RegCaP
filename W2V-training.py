#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List
from nltk.tokenize import RegexpTokenizer
import gensim

def iter_json_files(root: str) -> List[str]:
    """Iterate all JSON files under the given root directory (case-insensitive)."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)

def build_corpus(json_dir: str, tokenizer: RegexpTokenizer) -> List[List[str]]:
    """Read all JSON files and extract tokenized AST node codes."""
    corpus = []
    for fpath in iter_json_files(json_dir):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = data.get("ast_nodes", [])
            for node in nodes:
                code = node.get("code", "")
                if code:
                    toks = tokenizer.tokenize(code)
                    if toks:
                        corpus.append(toks)
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON format: {fpath}")
        except Exception as e:
            print(f"[WARN] Error processing {fpath}: {e}")
    return corpus

def save_word_vectors(model, out_dir: str, model_name: str):
    """Save word vectors to the specified output directory."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, model_name)
    model.wv.save(out_path)
    print(f"[OK] Word vectors saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec model on AST node codes extracted from JSON files."
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Input JSON root directory (recursively search for *.json files).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the trained model.",
    )
    parser.add_argument(
        "--model-name",
        default="w2v-128-20.wordvectors",
        help="Output filename (default: w2v-128-20.wordvectors).",
    )
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension (default: 128).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum word frequency (default: 1).")
    parser.add_argument("--window", type=int, default=5, help="Context window size (default: 5).")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4).")

    args = parser.parse_args()

    # C-style tokenizer pattern
    c_regexp = (
        r"\w+|->|\+\+|--|<=|>=|==|!=|"
        r"<<|>>|&&|\|\||-=|\+=|\*=|/=|%=|"
        r"&=|<<=|>>=|\^=|\|=|::|"
        r"[!@#$%^&*()_+\-=\[\]{};':\"\|,.<>/?]"
    )
    tokenizer = RegexpTokenizer(c_regexp)

    print(f"[INFO] Collecting corpus from: {args.json_dir}")
    corpus = build_corpus(args.json_dir, tokenizer)
    print(f"[INFO] Total sentences: {len(corpus)}")

    if len(corpus) == 0:
        print("[ERROR] Empty corpus. Please check --json-dir.")
        return 1

    # Support both Gensim 3.x and 4.x
    major_ver = int(gensim.__version__.split(".")[0])
    if major_ver >= 4:
        print(f"[INFO] Using Gensim {gensim.__version__} (v4 parameter naming).")
        model = gensim.models.Word2Vec(
            sentences=corpus,
            vector_size=args.dim,
            epochs=args.epochs,
            min_count=args.min_count,
            window=args.window,
            workers=args.workers,
        )
    else:
        print(f"[INFO] Using Gensim {gensim.__version__} (v3 parameter naming).")
        model = gensim.models.Word2Vec(
            corpus,
            size=args.dim,
            iter=args.epochs,
            min_count=args.min_count,
            window=args.window,
            workers=args.workers,
        )

    save_word_vectors(model, args.out_dir, args.model_name)
    return 0

if __name__ == "__main__":
    main()
