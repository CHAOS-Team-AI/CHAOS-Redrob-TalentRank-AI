# Methodology

## JD Analysis: What the Role Actually Needs

### Explicit Requirements (easy part)
The JD lists: Sentence Transformers, vector databases (Pinecone/Weaviate/Qdrant/etc.), NDCG/MRR evaluation, Python, and LoRA/QLoRA as a nice-to-have.

### Implicit Requirements (the hard part)
The JD contains deliberate traps and hidden signals:

**Trap 1 — Keyword stuffers**: A candidate can list FAISS, Pinecone, NDCG as skills while being a Marketing Manager. These are not fits. We solve this with `title_career` hard-zeroing any disqualifying title regardless of skills.

**Trap 2 — Research-only candidates**: Someone with 10 publications on vector search but zero production deployment experience is not a fit for a "Founding Team" role that explicitly requires shipping to real users.

**Trap 3 — Consulting-only backgrounds**: The JD says "not pure research or pure consulting." We identify companies in `CONSULTING_COMPANIES` and penalize careers spent entirely there.

**Trap 4 — Inactive candidates**: The JD explicitly states that a candidate with a 5% recruiter response rate is "not actually available." We implement this as a behavioral multiplier, not just an additive signal.

**Trap 5 — Honeypot candidates**: ~80 fabricated profiles designed to fool keyword-based rankers. Detected via timeline impossibilities and signal contradictions.

---

## Feature Engineering Decisions

### Skill Match (30%)

**Why exact names?**  
Early versions used substring matching (`"embedding"` matches `"Embeddings"`). This produced wildly inaccurate scores because:
- `"embedding"` also matches `"Not an embedding specialist"`  
- `"python"` matches `"Python developer"` but also `"Monty Python fan"` in summaries  

The final version uses exact set-membership lookup against verified names from the dataset.

**Why quality-weight each skill?**  
A candidate who listed "FAISS" as a beginner skill used for 0 months with 0 endorsements is fundamentally different from one with expert-level FAISS for 36 months with 40 endorsements. We multiply:
```
quality = proficiency_mult × duration_mult × endorsement_mult
```

**Why a normalization denominator of 60%?**  
We don't expect a real candidate to have ALL ~20 must-have skills. The normalization assumes ~60% coverage = score of 1.0, preventing score compression where every realistic candidate scores 0.1–0.3.

### Title + Career (25%)

**Why hard-zero on disqualifying titles?**  
The JD is explicit. A Marketing Manager is not a fit regardless of what skills they claim. The hard zero prevents career history signals from rescuing a fundamentally wrong candidate.

**Why 50/50 title vs career?**  
The JD says both matter. A perfect title with pure consulting career should score around 0.5. A slightly suboptimal title (Backend Engineer) with strong ML career descriptions should score around 0.55.

### Behavioral as Multiplier (10%)

**Why a multiplier instead of pure additive?**  
Additive: `score = skill×0.30 + title×0.25 + ... + behavioral×0.10`  
A candidate with skill=0.95, title=0.90, but behavioral=0.05 gets: `0.285 + 0.225 + ... + 0.005 = ~0.55`. Still ranks top-20.

With multiplier: `behavioral_mult = 0.60 + 0.05 × 0.40 = 0.62`  
That same candidate gets: `0.55 × 0.62 = 0.34`. Falls out of top-100.

This correctly implements the JD's insight that availability is a prerequisite, not a bonus.

**The multiplier formula:**
```python
behavior_multiplier = 0.60 + behavioral_score * 0.40   # range [0.60, 1.00]
```
- `behavioral=1.0` → no penalty (×1.0)
- `behavioral=0.5` → mild penalty (×0.80)  
- `behavioral=0.0` → 40% reduction (×0.60, not ×0.0 — we don't fully disqualify on behavior alone)

### Semantic Similarity (15%)

**Why MiniLM and not larger models?**  
- `all-MiniLM-L6-v2`: 22M params, 384 dims, ~500 candidates/sec on CPU
- `all-mpnet-base-v2`: 109M params, 768 dims, ~100 candidates/sec on CPU  
- For 2,000 candidates: MiniLM takes 4s, mpnet takes 20s. Both fit the budget but MiniLM gives more headroom.
- Quality difference is small for this task (matching against a single JD text).

**Why only embed top-2,000?**  
At 500 cands/sec, embedding all 100K would take 200 seconds — over half our budget, and most of those 100K have already been correctly scored low by rule-based features. The 0.5% at the boundary between 2,000th and 2,001st place is where semantic might help; further down it cannot change the ranking.

**Candidate text construction:**  
We include headline, summary, top-15 skills (sorted by proficiency), and top-4 career descriptions. We exclude raw dates, location, education institutions — those are handled by dedicated features and would add noise to semantic matching.

---

## Honeypot Detection Strategy

The spec states: "honeypot rate > 10% in top-100 = disqualification."  
With 38 detected honeypots across 100K candidates, the expected rate in an unprotected ranker's top-100 would be near 0 anyway. But some honeypots are designed to look like ideal AI candidates.

### The dangerous honeypots
Honeypots with `title=NLP Engineer`, `yoe=7`, listing all the right skills but with `duration_months=0` for every expert skill. A pure keyword ranker would rank these #1.

### Detection cascade
1. **Timeline check** (highest precision): `career_history_total > yoe + 4 years` → confidence += 0.6  
2. **Zero-duration experts** (catches skill fabricators): `proficiency=expert AND duration=0` → confidence += 0.2 per occurrence  
3. **Endorsement contradiction**: all skills 50+ endorsements but `endorsements_received=0` → confidence += 0.5  
4. **Engagement impossibility**: `completeness=100 AND response_rate=0 AND interview_rate=0` → confidence += 0.35  

Penalty function: `confidence < 0.2 → penalty=1.0 (no change)` · `confidence ≥ 0.5 → penalty=0.0 (hard disqualify)`

---

## Evaluation Metric Implications

```
Final score = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**NDCG@10 is 50% of the score** — getting the top-10 right is everything.  

Implications for our design:
- We invest heavily in the features that discriminate the very best candidates (skill quality × career quality × behavioral)
- We use semantic similarity specifically for the top-2,000 to improve top-10 precision
- We hard-zero honeypots because even one honeypot in top-10 drops NDCG@10 dramatically
- Behavioral multiplier matters most for top-10: it prevents high-skill inactive candidates from crowding out active ones

**MAP (15%)** rewards finding relevant candidates throughout all 100 ranks — our broad feature coverage (7 dimensions) ensures we don't systematically miss entire categories of valid candidates.

---

## What We Explicitly Do NOT Do

1. **No external API calls** — fully offline, no OpenAI embeddings during ranking
2. **No web scraping** — candidate data only
3. **No model training** — no labelled data → no learning-to-rank model. Rule-based features + semantic similarity only
4. **No LangChain / agent frameworks** — direct Python, no abstraction overhead
5. **No MongoDB / Firebase / Kubernetes** — flat files, simple Python
