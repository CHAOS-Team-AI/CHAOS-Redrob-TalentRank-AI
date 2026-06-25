"""
semantic.py — Semantic similarity scoring.

PRIMARY: sentence-transformers (all-MiniLM-L6-v2)
  - 22M params, 384-dim, ~500 cands/sec on CPU
  - L2-normalised cosine similarity via dot product
  - Install: pip install sentence-transformers

FALLBACK (auto-used when sentence-transformers is absent):
  - TF-IDF + cosine similarity (sklearn, always available)
  - Slower to scale but deterministic and good enough for top-2000

The ranker calls compute_semantic_scores() and gets a numpy array [0,1]
regardless of which backend is active.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"


def _try_load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except ImportError:
        return None


_SBERT_MODEL = None
_TFIDF_FITTED = None  # (vectorizer, jd_vector)


def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = _try_load_sbert()
    return _SBERT_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# JD TEXT  (expanded with synonyms and alternate phrasings for TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────

JD_TEXT = """
Senior AI Engineer Redrob AI founding team Pune Noida India hybrid.
Five to nine years applied machine learning product companies.
Sweet spot six eight years shipped production systems real users.

Must have production embeddings retrieval systems sentence transformers
openai embeddings bge e5 dense retrieval semantic search.
Embedding drift index refresh retrieval quality regression production.
Vector databases vector search pinecone weaviate qdrant milvus faiss
opensearch elasticsearch pgvector hybrid search.
Strong python production code quality.
Evaluation frameworks ranking systems ndcg mrr map offline online
ab testing learning to rank information retrieval bm25.

Nice to have lora qlora peft fine tuning llms rlhf instruction tuning.
Xgboost lightgbm learning to rank neural ranking reranking.
Hr tech recruiting marketplace recommendation systems.
Rag haystack llamaindex retrieval augmented generation.
Distributed systems inference optimization triton onnx.
Open source github contributions ai ml community.
Mlops mlflow hugging face transformers pytorch tensorflow nlp deep learning.

Role own intelligence layer ranking retrieval matching candidate job matching.
Ship ranking system embeddings hybrid retrieval llm reranking.
Evaluation infrastructure benchmarks ab testing.
Mentor engineers. Product company background shipped ranking search recommendation.
Strong opinions retrieval evaluation llm fine tune prompt engineering.
Located willing relocate noida pune india open to work short notice period.
Active platform responds recruiters available.
"""


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE → TEXT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def candidate_to_text(candidate: Dict[str, Any]) -> str:
    """
    Convert a candidate profile to a single text blob for embedding/TF-IDF.

    Includes: headline, summary, title, top-15 skills (sorted by proficiency),
    top-4 career descriptions.
    Excludes: dates, locations, education names (handled by dedicated features).
    """
    p = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    parts = []

    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"][:500])

    current = f"{p.get('current_title','')} {p.get('current_industry','')}"
    parts.append(current)

    # Top-15 skills sorted by proficiency desc then duration desc
    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    top_skills = sorted(
        skills,
        key=lambda s: (prof_order.get(s.get("proficiency","beginner"), 1), s.get("duration_months", 0)),
        reverse=True
    )[:15]

    skill_strs = []
    for s in top_skills:
        name = s["name"]
        prof = s.get("proficiency", "intermediate")
        dur = s.get("duration_months", 0)
        if dur >= 12:
            skill_strs.append(f"{name} {prof} {dur//12} years")
        else:
            skill_strs.append(f"{name} {prof}")
    if skill_strs:
        parts.append("Skills: " + ". ".join(skill_strs))

    # Top-4 career descriptions
    for role in career[:4]:
        t  = role.get("title", "")
        co = role.get("company", "")
        d  = role.get("description", "")[:400]
        parts.append(f"{t} at {co}: {d}")

    # Assessment scores as text
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessments:
        top_assess = sorted(assessments.items(), key=lambda x: x[1], reverse=True)[:3]
        parts.append("Assessed: " + " ".join(f"{k} score {v:.0f}" for k, v in top_assess))

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# TFIDF BACKEND  (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_semantic_scores(texts: List[str]) -> np.ndarray:
    """
    TF-IDF cosine similarity fallback.
    Fit on JD + all candidate texts so vocabulary is complete.
    Returns scores in [0, 1].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_texts = [JD_TEXT] + texts
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        max_features=50000,
    )
    mat = vec.fit_transform(all_texts)
    sims = cosine_similarity(mat[0:1], mat[1:]).flatten()

    # Normalise to [0, 1]: TF-IDF cosines are already [0, 1] for L2 norms
    # but practically range 0–0.4 for this task — rescale for full dynamic range
    if sims.max() > 0:
        sims = sims / sims.max()

    return sims.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SBERT BACKEND  (primary)
# ─────────────────────────────────────────────────────────────────────────────

def _sbert_semantic_scores(texts: List[str], model, batch_size: int) -> np.ndarray:
    all_texts = [JD_TEXT] + texts
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    jd_emb  = embeddings[0]
    cand_emb = embeddings[1:]
    sims = (cand_emb @ jd_emb)            # dot product of L2-normalised = cosine
    sims = (sims + 1.0) / 2.0             # shift [-1,1] → [0,1]
    return sims.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def compute_semantic_scores(
    candidates: List[Dict[str, Any]],
    cache_path: Optional[str] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute semantic similarity scores for all candidates vs the JD.

    Tries sentence-transformers first; falls back to TF-IDF if unavailable.
    Returns numpy array of shape (len(candidates),) with values in [0, 1].
    """
    texts = [candidate_to_text(c) for c in candidates]

    # ── Try cache first ───────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            ids = [c["candidate_id"] for c in candidates]
            if all(cid in cache for cid in ids):
                print(f"  Loaded cached embeddings from {cache_path}")
                cand_emb = np.array([cache[cid] for cid in ids], dtype=np.float32)
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(MODEL_NAME)
                jd_emb = model.encode([JD_TEXT], normalize_embeddings=True)[0]
                sims = (cand_emb @ jd_emb)
                return ((sims + 1.0) / 2.0).astype(np.float32)
        except Exception as e:
            print(f"  Cache load failed ({e}), recomputing...")

    # ── Try sentence-transformers ─────────────────────────────────────────────
    model = _get_sbert()
    if model is not None:
        print(f"  Using sentence-transformers ({MODEL_NAME})")
        scores = _sbert_semantic_scores(texts, model, batch_size)

        # Save cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_path, "rb") as f:
                    existing = pickle.load(f)
            except Exception:
                existing = {}
            for c, emb in zip(candidates, scores):
                existing[c["candidate_id"]] = emb
            with open(cache_path, "wb") as f:
                pickle.dump(existing, f)
            print(f"  Saved embeddings cache → {cache_path}")

        return scores

    # ── TF-IDF fallback ───────────────────────────────────────────────────────
    print("  sentence-transformers unavailable → using TF-IDF fallback")
    return _tfidf_semantic_scores(texts)
