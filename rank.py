#!/usr/bin/env python3
"""
rank.py — Main entry point for the Redrob Hackathon Ranker.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Produces a valid submission CSV (top-100 candidates ranked best-fit first)
for the Senior AI Engineer – Founding Team role at Redrob AI.

Constraints:
    ≤ 5 minutes wall-clock
    ≤ 16 GB RAM
    CPU only
    No external API calls

Architecture:
    1. Load & parse 100K candidates (streaming, low memory)
    2. Honeypot detection — eliminate impossible profiles
    3. Feature engineering — 8 scoring dimensions
    4. Semantic matching — sentence-transformers (MiniLM, CPU-friendly)
    5. Score composition — weighted combination
    6. Select top-100, generate reasoning, write CSV
"""

import argparse
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ranker import CandidateRanker
from writer import write_submission


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Senior AI Engineer role."
    )
    parser.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        help="Path to candidates.jsonl (or .jsonl.gz)",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output path for the submission CSV",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to include (default: 100)",
    )
    parser.add_argument(
        "--embeddings-cache",
        default=None,
        help="Path to pre-computed embeddings pickle (from precompute_embeddings.py). "
             "If omitted, embeddings are computed live for the top-K candidates.",
    )
    parser.add_argument(
        "--phase1-top-k",
        type=int,
        default=2000,
        help="Number of candidates to carry into semantic scoring after rule-based "
             "pre-filtering (default: 2000). Lower = faster, higher = more thorough.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics",
    )
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}")
        print("Please place candidates.jsonl (or .jsonl.gz) at the specified path.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Redrob Hackathon — Senior AI Engineer Ranker")
    print("=" * 60)
    print(f"Candidates file  : {candidates_path}")
    print(f"Output           : {out_path}")
    print(f"Top-N            : {args.top_n}")
    print(f"Embeddings cache : {args.embeddings_cache or '(none — live compute)'}")
    print(f"Phase-1 top-K    : {args.phase1_top_k}")
    print()

    ranker = CandidateRanker(
        debug=args.debug,
        embeddings_cache=args.embeddings_cache,
        phase1_top_k=args.phase1_top_k,
    )
    ranked = ranker.rank(str(candidates_path), top_n=args.top_n)

    write_submission(ranked, str(out_path))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — wrote {len(ranked)} candidates to {out_path}")
    print(f"Top candidate: {ranked[0]['candidate_id']} (score={ranked[0]['score']:.4f})")
    print(f"  {ranked[0]['reasoning']}")


if __name__ == "__main__":
    main()
