# Architecture

## System Overview

```
candidates.jsonl (465MB, 100K records)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Rule-Based Scoring  (~14s, ALL 100K)                  │
│                                                                 │
│  For each candidate:                                            │
│    ┌──────────────────────┐   ┌─────────────────────────────┐  │
│    │  features.py         │   │  honeypot_detector.py       │  │
│    │  • skill_match       │   │  • timeline check           │  │
│    │  • title_career      │   │  • expert+zero_duration     │  │
│    │  • experience_fit    │   │  • endorsement contradiction│  │
│    │  • behavioral_signals│   │  • perfect+zero engagement  │  │
│    │  • location_fit      │   │                             │  │
│    │  • education_quality │   │  → honeypot_penalty [0,1]   │  │
│    │  • github_activity   │   └─────────────────────────────┘  │
│    └──────────────────────┘                                     │
│              │                           │                      │
│              └───────── rule_score ──────┘                      │
└─────────────────────────────────────────────────────────────────┘
        │
        │  Sort by rule_score DESC, keep top-2000
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Semantic Scoring  (~4s cached / ~90s live, TOP 2000)  │
│                                                                 │
│  semantic.py                                                    │
│  • candidate_to_text() → structured text representation        │
│  • MiniLM-L6-v2 → 384-dim L2-normalized embeddings            │
│  • cosine_sim(candidate_embedding, jd_embedding) → [0, 1]     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Final Score Combination                               │
│                                                                 │
│  base = Σ(feature_i × weight_i)  [7 rule-based dims + semantic]│
│  behavior_mult = 0.60 + behavioral × 0.40                       │
│  final = base × behavior_mult                                   │
│        + behavioral × weight_behavioral                         │
│        × honeypot_penalty                                       │
│                                                                 │
│  Sort by (final_score DESC, candidate_id ASC)                   │
│  Select top-100                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Reasoning Generation                                  │
│                                                                 │
│  reasoning.py                                                   │
│  • Factual claims only (no hallucination)                       │
│  • Tier-based tone (top-10 = strong positive, 61-100 = honest) │
│  • References actual profile values                             │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
   submission.csv  (100 rows, validated)
```

## Module Responsibilities

| Module | Responsibility | Key Design Choice |
|---|---|---|
| `jd_config.py` | Single source of truth for JD interpretation | Exact skill names from dataset |
| `features.py` | 7 independent scoring functions | Each returns [0,1], composable |
| `honeypot_detector.py` | Profile integrity checks | Returns confidence + penalty multiplier |
| `semantic.py` | Embedding similarity | Lazy model load, cache-aware |
| `ranker.py` | Pipeline orchestration | Phase-1 pre-filter avoids 100K embedding cost |
| `reasoning.py` | Human-readable explanations | Rank-tier determines tone |
| `writer.py` | CSV output | 6dp to avoid false ties |

## Data Flow

```
candidates.jsonl
    │
    ├─► json.loads() line-by-line (streaming, not full load)
    │
    ├─► compute_all_features()  ← jd_config.py (weights, skill sets)
    │
    ├─► detect_honeypot()       → (is_hp, confidence, reasons)
    │
    ├─► rule_score = Σ(feat × weight) × honeypot_penalty
    │
    ├─► [after all 100K] sort desc, keep top-2000
    │
    ├─► compute_semantic_scores()  → numpy array [2000,]
    │
    ├─► final_score = combine(feats, semantic, honeypot_penalty)
    │
    ├─► sort top-100, assign ranks 1-100
    │
    ├─► generate_reasoning() per candidate
    │
    └─► write_submission() → submission.csv
```

## Memory Profile

| Phase | Peak RAM |
|---|---|
| Loading + scoring 100K | ~200MB (streaming JSON) |
| Top-2000 candidate objects | ~50MB |
| MiniLM model weights | ~90MB |
| 2000 × 384 embeddings | ~3MB |
| **Total** | **~350MB** |

Well within 16GB constraint.

## Performance Profile

| Phase | Time (CPU, 4-core) |
|---|---|
| Phase 1: 100K rule-based | ~14s |
| Phase 2: 2000 embeddings | ~4s (cached) / ~90s (live) |
| Phase 3+4: combine + reason | ~1s |
| **Total** | **~15s (cached) / ~2.5min (live)** |

Both well within the 5-minute constraint.
