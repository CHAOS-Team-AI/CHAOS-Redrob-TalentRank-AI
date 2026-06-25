"""
ranker.py — Main orchestration pipeline for the Redrob candidate ranker.

Pipeline (fits 5-min CPU budget):
  Phase 1 — Rule-based scoring of all 100K candidates     (~25s)
  Phase 2 — Semantic embedding of top-2000 only           (~90s with MiniLM)
  Phase 3 — Final score combination + top-100 selection   (~2s)
  Phase 4 — Reasoning generation                          (~1s)
  Total                                                   ~2-2.5 minutes
"""

import json
import gzip
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from features import compute_all_features
from honeypot_detector import detect_honeypot, score_honeypot_penalty
from semantic import compute_semantic_scores
from reasoning import generate_reasoning
from jd_config import WEIGHTS


# ─── Loader ─────────────────────────────────────────────────────────────────

def _load(path: str):
    p = Path(path)
    opener = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ─── Score combination ───────────────────────────────────────────────────────

def _combine(feats: Dict[str, float], sem: float, hp_penalty: float) -> float:
    """
    Weighted combination with behavioral soft-multiplier.

    Behavioral score acts as multiplier (0.60–1.0) on the skill/career base,
    implementing the JD insight: inactive perfect-on-paper < active imperfect.
    """
    base = (
        feats["skill_match"]       * WEIGHTS["skill_match"]
        + feats["title_career"]    * WEIGHTS["title_career"]
        + feats["experience_fit"]  * WEIGHTS["experience_fit"]
        + feats["location_fit"]    * WEIGHTS["location_fit"]
        + feats["education_quality"] * WEIGHTS["education_quality"]
        + feats["github_activity"] * WEIGHTS["github_activity"]
        + sem                      * WEIGHTS["semantic_similarity"]
    )

    beh = feats["behavioral_signals"]
    beh_mult = 0.60 + beh * 0.40   # [0.60, 1.0]

    score = base * beh_mult + beh * WEIGHTS["behavioral_signals"]
    return min(1.0, max(0.0, score * hp_penalty))


# ─── Main ranker ─────────────────────────────────────────────────────────────

class CandidateRanker:
    def __init__(
        self,
        debug: bool = False,
        embeddings_cache: Optional[str] = None,
        phase1_top_k: int = 2000,
    ):
        self.debug = debug
        self.embeddings_cache = embeddings_cache
        self.phase1_top_k = phase1_top_k

    def rank(self, candidates_path: str, top_n: int = 100) -> List[Dict[str, Any]]:
        t0 = time.time()

        # ── Phase 1: rule-based scoring of all 100K ──────────────────────────
        print("[Phase 1] Scoring 100K candidates with rule-based features...")
        all_scored = []
        total = 0
        honeypot_count = 0

        for candidate in _load(candidates_path):
            total += 1
            feats = compute_all_features(candidate)
            _, hp_conf, hp_reasons = detect_honeypot(candidate)
            hp_penalty = score_honeypot_penalty(hp_conf)
            if hp_conf >= 0.5:
                honeypot_count += 1

            # Phase-1 score (no semantic yet)
            rule = _combine(feats, 0.0, hp_penalty)

            all_scored.append({
                "candidate": candidate,
                "feats": feats,
                "hp_conf": hp_conf,
                "hp_reasons": hp_reasons,
                "hp_penalty": hp_penalty,
                "rule_score": rule,
            })

            if total % 20000 == 0:
                print(f"  {total:,} / 100,000  ({time.time()-t0:.0f}s)")

        t1 = time.time()
        print(f"  Phase 1 done: {total:,} candidates in {t1-t0:.1f}s  |  honeypots={honeypot_count}")

        # ── Phase 2: semantic scoring of top-K ───────────────────────────────
        all_scored.sort(key=lambda x: -x["rule_score"])
        top_k = all_scored[: self.phase1_top_k]

        print(f"\n[Phase 2] Semantic scoring top {len(top_k)}...")
        t2 = time.time()
        try:
            sem_scores = compute_semantic_scores(
                [x["candidate"] for x in top_k],
                cache_path=self.embeddings_cache,
                batch_size=256,
            )
        except Exception as e:
            print(f"  Semantic failed ({e}), using zeros")
            sem_scores = np.zeros(len(top_k))
        print(f"  Phase 2 done in {time.time()-t2:.1f}s")

        # ── Phase 3: final scoring + select top-N ────────────────────────────
        print("\n[Phase 3] Final scores & selection...")
        for i, item in enumerate(top_k):
            item["sem_score"] = float(sem_scores[i])
            item["final_score"] = _combine(
                item["feats"], item["sem_score"], item["hp_penalty"]
            )

        top_k.sort(key=lambda x: (-x["final_score"], x["candidate"]["candidate_id"]))
        top_100 = top_k[:top_n]

        # ── Phase 4: reasoning ───────────────────────────────────────────────
        print("[Phase 4] Generating reasoning...")
        results = []
        for rank_idx, item in enumerate(top_100):
            rank = rank_idx + 1
            c = item["candidate"]
            reasoning = generate_reasoning(
                candidate=c,
                features=item["feats"],
                rank=rank,
                honeypot_flags=item["hp_reasons"],
            )
            results.append({
                "candidate_id": c["candidate_id"],
                "rank": rank,
                "score": round(item["final_score"], 6),
                "reasoning": reasoning,
                # extra — for demo / debug only
                "candidate": c,
                "features": item["feats"],
                "semantic_score": item.get("sem_score", 0.0),
                "rule_score": item["rule_score"],
                "hp_conf": item["hp_conf"],
                "hp_reasons": item["hp_reasons"],
            })

        total_time = time.time() - t0
        print(f"\n✅ Pipeline complete in {total_time:.1f}s")

        if self.debug:
            print("\n=== TOP 10 ===")
            for r in results[:10]:
                f = r["features"]
                print(
                    f"  #{r['rank']} {r['candidate_id']}  score={r['score']:.4f}"
                    f"  skill={f['skill_match']:.3f}  title={f['title_career']:.3f}"
                    f"  sem={r['semantic_score']:.3f}  beh={f['behavioral_signals']:.3f}"
                )
                print(f"     {r['reasoning']}")

        return results
