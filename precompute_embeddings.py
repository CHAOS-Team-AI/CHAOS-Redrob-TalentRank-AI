#!/usr/bin/env python3
"""
precompute_embeddings.py — Pre-compute candidate embeddings offline.

This is run ONCE before the ranking step and saves embeddings to disk.
The ranking step then loads pre-computed embeddings (fast).

Why:
- Embedding 100K candidates takes ~200s on CPU
- Pre-computation is allowed to exceed the 5-minute window
- The RANKING step (which must complete in 5 min) just loads pre-computed embeddings

Usage:
    python precompute_embeddings.py \
        --candidates ./data/candidates.jsonl \
        --output ./outputs/embeddings.pkl

This script is optional — if no cache exists, rank.py will compute
embeddings inline for the top-2000 candidates (~30-60s).
"""

import argparse
import time
import json
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings")
    parser.add_argument("--candidates", default="./data/candidates.jsonl")
    parser.add_argument("--output", default="./outputs/embeddings.pkl")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    
    from sentence_transformers import SentenceTransformer
    from semantic import candidate_to_text, MODEL_NAME, JD_TEXT
    
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Load all candidates
    print(f"Loading candidates from: {args.candidates}")
    candidates = []
    path = Path(args.candidates)
    
    opener = gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    
    print(f"  Loaded {len(candidates):,} candidates")
    
    # Convert to text
    print("Converting candidates to text...")
    texts = [candidate_to_text(c) for c in candidates]
    
    # Embed
    print(f"Embedding {len(texts):,} candidates in batches of {args.batch_size}...")
    t_embed = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    embed_time = time.time() - t_embed
    print(f"  Embedded in {embed_time:.1f}s ({len(texts)/embed_time:.0f} candidates/sec)")
    
    # Save as dict: candidate_id -> embedding
    print("Saving embeddings...")
    cache = {c["candidate_id"]: emb for c, emb in zip(candidates, embeddings)}
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {out_path} ({size_mb:.1f} MB)")
    print(f"  Total time: {time.time()-t0:.1f}s")
    
    # Also embed JD for verification
    jd_emb = model.encode([JD_TEXT], normalize_embeddings=True)[0]
    print(f"\nJD embedding shape: {jd_emb.shape}")
    
    # Quick sanity check: top-5 by cosine similarity
    import numpy as np
    sims = embeddings @ jd_emb
    top5_idx = np.argsort(sims)[-5:][::-1]
    print("\nTop 5 by semantic similarity:")
    for idx in top5_idx:
        c = candidates[idx]
        print(f"  {c['candidate_id']}: {c['profile']['current_title']} ({c['profile']['years_of_experience']:.1f}y) — sim={sims[idx]:.4f}")


if __name__ == "__main__":
    main()
