"""
features.py — Feature engineering for the Redrob candidate ranker.

All skill matching uses EXACT names from the dataset (verified against candidates.jsonl).
Every function returns a normalized score in [0.0, 1.0].

Scoring dimensions:
1. skill_match      — exact skill name matching against JD taxonomy
2. title_career     — current title tier + career history quality
3. experience_fit   — YoE in 5-9 year window
4. behavioral       — engagement, availability, notice period, recency
5. location_fit     — Pune/Noida preference
6. education        — institution tier
7. github           — open-source signal
"""

from datetime import datetime, date
from typing import Dict, Any

from jd_config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, MINOR_POSITIVE_SKILLS, NEGATIVE_SKILLS,
    IDEAL_TITLES, STRONG_TITLES, ADJACENT_TITLES, DISQUALIFYING_TITLES,
    CONSULTING_COMPANIES,
    YOE_IDEAL_MIN, YOE_IDEAL_MAX, YOE_SWEET_SPOT_MIN, YOE_SWEET_SPOT_MAX,
    TIER1_LOCATIONS, TIER2_LOCATIONS,
    RECENCY_FRESH_DAYS, RECENCY_STALE_DAYS, RECENCY_DEAD_DAYS,
    NOTICE_IDEAL_MAX, NOTICE_ACCEPTABLE_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    if not date_str:
        return 999
    try:
        d = datetime.fromisoformat(date_str).date()
        return (date.today() - d).days
    except (ValueError, TypeError):
        return 999


def _prof_weight(proficiency: str) -> float:
    return {"beginner": 0.35, "intermediate": 0.65, "advanced": 0.90, "expert": 1.0}.get(
        proficiency, 0.65
    )


def _duration_weight(duration_months: int) -> float:
    """Convert skill duration (months) to a credibility multiplier."""
    if duration_months == 0:
        return 0.05   # near-zero — strong honeypot signal too
    if duration_months < 3:
        return 0.25
    if duration_months < 6:
        return 0.45
    if duration_months < 12:
        return 0.65
    if duration_months < 24:
        return 0.82
    if duration_months < 48:
        return 0.95
    return 1.0


def _endorse_weight(endorsements: int) -> float:
    if endorsements == 0:
        return 0.70   # unendorsed — lower trust but not zero
    if endorsements < 5:
        return 0.85
    if endorsements < 20:
        return 0.95
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. SKILL MATCH
# ─────────────────────────────────────────────────────────────────────────────

def score_skill_match(candidate: Dict[str, Any]) -> float:
    """
    Exact-name skill matching against JD taxonomy.

    Uses set-based lookup (O(1) per skill) against MUST_HAVE_SKILLS and
    NICE_TO_HAVE_SKILLS which contain EXACT skill names from the dataset.

    Each matched skill is weighted by proficiency × duration × endorsement quality.

    Must-haves contribute 70% of the raw score; nice-to-haves 30%.
    Negative skills (Marketing, Sales, etc.) apply a small penalty.
    """
    skills = candidate.get("skills", [])
    sigs = candidate.get("redrob_signals", {})
    assessments = sigs.get("skill_assessment_scores", {})

    if not skills:
        return 0.0

    must_hits = 0.0
    nice_hits = 0.0
    neg_count = 0

    for s in skills:
        name = s["name"]
        prof = _prof_weight(s.get("proficiency", "intermediate"))
        dur = _duration_weight(s.get("duration_months", 12))
        end = _endorse_weight(s.get("endorsements", 0))

        # Assessment bonus (0–0.15 extra multiplier)
        assess_bonus = min(0.15, assessments.get(name, 0) / 100.0 * 0.15)

        quality = prof * dur * end + assess_bonus   # 0–1.15

        if name in MUST_HAVE_SKILLS:
            must_hits += quality
        elif name in NICE_TO_HAVE_SKILLS:
            nice_hits += quality * 0.5   # nice-to-haves count half
        elif name in NEGATIVE_SKILLS:
            neg_count += 1

    # Normalize against theoretical max
    # max must-hit score: len(MUST_HAVE_SKILLS) * 1.15 (all perfect)
    must_max = len(MUST_HAVE_SKILLS)     # ~20
    nice_max = len(NICE_TO_HAVE_SKILLS) * 0.5  # ~10

    must_norm = min(1.0, must_hits / (must_max * 0.60))  # expect ~60% coverage = 1.0
    nice_norm = min(1.0, nice_hits / (nice_max * 0.50))  # expect ~50% coverage = 1.0

    # Negative skill penalty
    neg_penalty = max(0.0, 1.0 - neg_count * 0.08)

    raw = (must_norm * 0.70 + nice_norm * 0.30) * neg_penalty
    return min(1.0, max(0.0, raw))


# ─────────────────────────────────────────────────────────────────────────────
# 2. TITLE + CAREER TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def score_title_career(candidate: Dict[str, Any]) -> float:
    """
    Score title fit AND career trajectory quality.

    JD insight: "A candidate who has all the AI keywords listed as skills
    but whose title is 'Marketing Manager' is not a fit."
    JD insight: "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone'
    but if their career history shows they built a recommendation system at a
    product company, they're a fit."

    Two sub-scores:
    a) Title tier (0-1)
    b) Career history: ML relevance + product company signals - consulting penalty
    """
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_title = p.get("current_title", "")
    title_lower = current_title.lower()

    # ── (a) Title tier ────────────────────────────────────────────────────────
    if current_title in DISQUALIFYING_TITLES:
        title_score = 0.0
    elif current_title in IDEAL_TITLES:
        title_score = 1.0
    elif current_title in STRONG_TITLES:
        title_score = 0.72
    elif current_title in ADJACENT_TITLES:
        title_score = 0.42
    else:
        # Soft match on key words for unlisted titles
        if any(kw in title_lower for kw in ["machine learning", "ml engineer", "ai engineer", "nlp engineer", "data scientist", "applied ml", "applied ai", "recommendation", "search engineer"]):
            title_score = 0.88
        elif any(kw in title_lower for kw in ["data", "software", "backend", "engineer", "developer"]):
            title_score = 0.38
        else:
            # Unknown / generic — could be anything
            title_score = 0.20

    # Hard disqualify on problematic keywords
    if any(bad in title_lower for bad in [
        "marketing", "graphic", "design", "account", "civil", "mechanical",
        "hr ", "human resource", "sales", "customer support", "content writ",
        "mobile dev", "ios dev", "android dev", "frontend dev", "frontend eng", "qa engineer",
    ]):
        title_score = 0.0

    # ── (b) Career history quality ────────────────────────────────────────────
    if not career:
        career_score = 0.30  # no history = neutral
    else:
        # ML/AI relevance keywords in roles
        ml_desc_kws = [
            "retrieval", "ranking", "recommendation", "search", "embedding",
            "vector", "nlp", "machine learning", "deep learning", "llm", "bert",
            "transformer", "pytorch", "tensorflow", "inference", "model training",
            "fine-tun", "a/b test", "evaluation", "ndcg", "mrr", "ranker",
            "similarity", "semantic", "index", "feature engineer",
        ]

        product_kws = [
            "launched", "shipped", "deployed", "production", "real users",
            "at scale", "live", "improved", "reduced latency", "revenue",
            "customer", "platform",
        ]

        ml_roles = 0
        product_signals = 0
        consulting_roles = 0
        title_chaser_penalty = 0.0
        total = len(career)

        # Detect title chasing: many short stints
        short_stints = sum(1 for r in career if r.get("duration_months", 24) < 18)
        if total >= 4 and short_stints / total > 0.6:
            title_chaser_penalty = 0.15   # JD explicitly penalizes this

        for role in career:
            desc = role.get("description", "").lower()
            rtitle = role.get("title", "").lower()
            company = role.get("company", "").lower()

            is_consulting = any(c in company for c in CONSULTING_COMPANIES)
            if is_consulting:
                consulting_roles += 1

            has_ml = any(kw in desc or kw in rtitle for kw in ml_desc_kws)
            has_product = any(kw in desc for kw in product_kws)

            if has_ml:
                ml_roles += 1
            if has_product:
                product_signals += 1

        ml_ratio = ml_roles / total
        product_ratio = product_signals / total
        consulting_ratio = consulting_roles / total

        # Entire career consulting = bad (JD explicit)
        all_consulting = consulting_ratio == 1.0

        career_score = (
            ml_ratio * 0.50
            + product_ratio * 0.30
            + (1.0 - consulting_ratio) * 0.20
        )
        if all_consulting:
            career_score *= 0.50
        career_score = max(0.0, career_score - title_chaser_penalty)
        career_score = min(1.0, career_score)

    # ── Combine title + career ────────────────────────────────────────────────
    # Disqualifying title overrides ALL career signals — a Marketing Manager
    # is not a fit regardless of ML career descriptions.
    if title_score == 0.0:
        return 0.0
    combined = title_score * 0.50 + career_score * 0.50

    # Industry signal
    industry = p.get("current_industry", "").lower()
    if any(ind in industry for ind in ["software", "technology", "ai", "ml", "saas", "fintech", "startup", "marketplace", "e-commerce"]):
        combined *= 1.05
    elif industry in ("it services", "consulting", "bpo"):
        combined *= 0.90

    return min(1.0, max(0.0, combined))


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPERIENCE FIT
# ─────────────────────────────────────────────────────────────────────────────

def score_experience_fit(candidate: Dict[str, Any]) -> float:
    """YoE scoring with sweet-spot centering on 6-8 years."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if yoe < 2:
        return 0.10
    if yoe > 20:
        return 0.25
    if YOE_SWEET_SPOT_MIN <= yoe <= YOE_SWEET_SPOT_MAX:
        return 1.00
    if YOE_IDEAL_MIN <= yoe < YOE_SWEET_SPOT_MIN:
        return 0.70 + (yoe - YOE_IDEAL_MIN) / (YOE_SWEET_SPOT_MIN - YOE_IDEAL_MIN) * 0.30
    if YOE_SWEET_SPOT_MAX < yoe <= YOE_IDEAL_MAX:
        return 0.70 + (YOE_IDEAL_MAX - yoe) / (YOE_IDEAL_MAX - YOE_SWEET_SPOT_MAX) * 0.30
    if 3.0 <= yoe < YOE_IDEAL_MIN:
        return 0.30 + (yoe - 3.0) / (YOE_IDEAL_MIN - 3.0) * 0.40
    if YOE_IDEAL_MAX < yoe <= 15:
        return 0.60 - (yoe - YOE_IDEAL_MAX) / (15 - YOE_IDEAL_MAX) * 0.30
    return 0.30


# ─────────────────────────────────────────────────────────────────────────────
# 4. BEHAVIORAL SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def score_behavioral_signals(candidate: Dict[str, Any]) -> float:
    """
    Engagement and availability composite.

    Key insight from JD and redrob_signals_doc:
    'These behavioral signals are often more predictive of whether a candidate
    can actually be hired than their static profile.'

    This is used as a SOFT MULTIPLIER in ranker.py:
    inactive perfect-on-paper < active imperfect
    """
    sigs = candidate.get("redrob_signals", {})

    # 1. Recency: last_active_date
    days_ago = _days_since(sigs.get("last_active_date", ""))
    if days_ago <= RECENCY_FRESH_DAYS:
        recency = 1.00
    elif days_ago <= RECENCY_STALE_DAYS:
        recency = 1.00 - (days_ago - RECENCY_FRESH_DAYS) / (RECENCY_STALE_DAYS - RECENCY_FRESH_DAYS) * 0.30
    elif days_ago <= RECENCY_DEAD_DAYS:
        recency = 0.70 - (days_ago - RECENCY_STALE_DAYS) / (RECENCY_DEAD_DAYS - RECENCY_STALE_DAYS) * 0.45
    else:
        recency = max(0.08, 0.25 - (days_ago - RECENCY_DEAD_DAYS) / 180 * 0.17)

    # 2. Open to work flag (strong signal)
    otw = 0.20 if sigs.get("open_to_work_flag", False) else 0.0

    # 3. Recruiter response rate
    rr = sigs.get("recruiter_response_rate", 0.0)
    response = min(1.0, rr * 1.25)   # 0.8 rate → 1.0 score

    # 4. Avg response time
    art = sigs.get("avg_response_time_hours", 72)
    if art <= 4:
        resp_time = 1.00
    elif art <= 24:
        resp_time = 0.85
    elif art <= 72:
        resp_time = 0.65
    elif art <= 168:
        resp_time = 0.42
    else:
        resp_time = 0.20

    # 5. Interview completion rate
    icr = sigs.get("interview_completion_rate", 0.5)

    # 6. Notice period
    notice = sigs.get("notice_period_days", 60)
    if notice <= NOTICE_IDEAL_MAX:
        notice_score = 1.00
    elif notice <= NOTICE_ACCEPTABLE_MAX:
        notice_score = 0.70 - (notice - NOTICE_IDEAL_MAX) / (NOTICE_ACCEPTABLE_MAX - NOTICE_IDEAL_MAX) * 0.25
    else:
        notice_score = max(0.25, 0.45 - (notice - NOTICE_ACCEPTABLE_MAX) / 90 * 0.20)

    # 7. Social proof: saved by recruiters in last 30d
    saved = min(1.0, sigs.get("saved_by_recruiters_30d", 0) / 8.0)

    # 8. Verification signals
    verif = sum([
        sigs.get("verified_email", False) * 0.04,
        sigs.get("verified_phone", False) * 0.04,
        sigs.get("linkedin_connected", False) * 0.04,
    ])

    # 9. Profile completeness
    completeness = sigs.get("profile_completeness_score", 50) / 100.0

    # 10. Applications submitted recently (active job seeker)
    apps = min(1.0, sigs.get("applications_submitted_30d", 0) / 5.0) * 0.05

    # Weighted combination
    combined = (
        recency       * 0.22
        + otw                  # direct bonus
        + response    * 0.18
        + resp_time   * 0.08
        + icr         * 0.10
        + notice_score * 0.16
        + saved        * 0.05
        + verif                # direct bonus
        + completeness * 0.09
        + apps                 # direct bonus
    )

    return min(1.0, max(0.0, combined))


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOCATION FIT
# ─────────────────────────────────────────────────────────────────────────────

def score_location_fit(candidate: Dict[str, Any]) -> float:
    p = candidate.get("profile", {})
    sigs = candidate.get("redrob_signals", {})

    loc = p.get("location", "").lower()
    country = p.get("country", "").lower()
    relocate = sigs.get("willing_to_relocate", False)
    work_mode = sigs.get("preferred_work_mode", "")

    if any(t in loc for t in TIER1_LOCATIONS):
        return 1.0

    if any(t in loc for t in TIER2_LOCATIONS):
        return 0.95 if relocate else 0.80

    if country in ("india", "in"):
        return 0.82 if relocate else 0.62

    # Outside India
    return 0.45 if relocate else 0.18


# ─────────────────────────────────────────────────────────────────────────────
# 6. EDUCATION
# ─────────────────────────────────────────────────────────────────────────────

def score_education(candidate: Dict[str, Any]) -> float:
    """Minor signal — IIT/IISc boost, others neutral."""
    edu = candidate.get("education", [])
    if not edu:
        return 0.50

    tier_map = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.65, "tier_4": 0.55, "unknown": 0.50}
    return max(tier_map.get(e.get("tier", "unknown"), 0.50) for e in edu)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GITHUB ACTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def score_github(candidate: Dict[str, Any]) -> float:
    """Open-source signal. -1 = no GitHub (neutral)."""
    g = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if g == -1:
        return 0.35  # No GitHub — slight negative vs active contributor
    return g / 100.0


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    return {
        "skill_match":       score_skill_match(candidate),
        "title_career":      score_title_career(candidate),
        "experience_fit":    score_experience_fit(candidate),
        "behavioral_signals": score_behavioral_signals(candidate),
        "location_fit":      score_location_fit(candidate),
        "education_quality": score_education(candidate),
        "github_activity":   score_github(candidate),
    }
