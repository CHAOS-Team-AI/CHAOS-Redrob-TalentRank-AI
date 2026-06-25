"""
reasoning.py — Generate factual, specific, rank-consistent reasoning strings.

Stage 4 (manual review) checks:
  1. Specific facts from the profile (title, YoE, skill names, signal values)
  2. JD connection (not generic praise)
  3. Honest concerns (gaps acknowledged)
  4. No hallucination (every claim in the profile)
  5. Variation (10 sampled rows must differ)
  6. Rank consistency (tone matches rank position)
"""

from datetime import datetime, date
from typing import Dict, Any, List


def _days_since(ds: str) -> int:
    try:
        return (date.today() - datetime.fromisoformat(ds).date()).days
    except Exception:
        return 999


def _top_skills(skills: list, n: int = 4) -> str:
    order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    top = sorted(skills, key=lambda s: (order.get(s.get("proficiency","beginner"),1), s.get("duration_months",0)), reverse=True)
    return ", ".join(s["name"] for s in top[:n])


def _notice_str(days: int) -> str:
    if days <= 30:   return f"{days}d notice (immediate)"
    if days <= 60:   return f"{days}d notice"
    if days <= 90:   return f"{days}d notice (buyout per JD)"
    return f"{days}d notice (long, raises bar)"


def _active_str(days: int) -> str:
    if days <= 7:    return "active today/this week"
    if days <= 30:   return f"active {days}d ago"
    if days <= 90:   return f"active {days}d ago (recent)"
    if days <= 180:  return f"last active {days}d ago (stale)"
    return f"last active {days}d ago (inactive)"


def _location_str(candidate: Dict[str, Any]) -> str:
    p = candidate["profile"]
    s = candidate["redrob_signals"]
    loc = p.get("location","")
    country = p.get("country","")
    relocate = s.get("willing_to_relocate", False)
    ideal = any(x in loc.lower() for x in ["pune","noida","delhi","gurugram","gurgaon"])
    tier2 = any(x in loc.lower() for x in ["hyderabad","bangalore","bengaluru","mumbai"])

    if ideal:
        return f"{loc} (ideal)"
    if tier2:
        return f"{loc}{', relocate=yes' if relocate else ''}"
    return f"{loc}, {country}{', open to relocate' if relocate else ''}"


def _has_skill(skills: list, *kws) -> bool:
    names = {s["name"].lower() for s in skills}
    return any(any(k in n for n in names) for k in kws)


def _career_evidence(career: list) -> str:
    """Pick best 1 career highlight relevant to ranking/retrieval/embeddings."""
    kws = [
        ("retrieval", "retrieval"),
        ("ranking", "ranking system"),
        ("recommendation", "recommendation engine"),
        ("search", "search"),
        ("embedding", "embeddings"),
        ("vector", "vector search"),
        ("nlp", "NLP"),
        ("a/b test", "A/B testing"),
        ("fine-tun", "fine-tuning"),
    ]
    for role in career:
        desc = role.get("description","").lower()
        rtitle = role.get("title","").lower()
        for kw, label in kws:
            if kw in desc or kw in rtitle:
                return f"{role.get('title','')} at {role.get('company','')} ({label})"
    if career:
        r = career[0]
        return f"{r.get('title','')} at {r.get('company','')}"
    return ""


def generate_reasoning(
    candidate: Dict[str, Any],
    features: Dict[str, float],
    rank: int,
    honeypot_flags: List[str],
) -> str:
    """
    Produce a 1-2 sentence specific, factual, rank-consistent reasoning.
    Every claim must correspond to actual profile data (no hallucination).
    """
    p = candidate["profile"]
    sigs = candidate["redrob_signals"]
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    notice = sigs.get("notice_period_days", 60)
    rr = sigs.get("recruiter_response_rate", 0.0)
    otw = sigs.get("open_to_work_flag", False)
    days_ago = _days_since(sigs.get("last_active_date", ""))
    github = sigs.get("github_activity_score", -1)

    top4 = _top_skills(skills, 4)
    loc_str = _location_str(candidate)
    career_hl = _career_evidence(career)

    # Skill presence flags
    has_vec = _has_skill(skills, "faiss","pinecone","qdrant","milvus","weaviate","elasticsearch","opensearch","pgvector","vector search")
    has_emb = _has_skill(skills, "sentence transformer","embeddings","semantic search","dense retrieval")
    has_rank_eval = _has_skill(skills, "learning to rank","information retrieval","bm25","ndcg","mrr")
    has_lora = _has_skill(skills, "lora","qlora","peft","fine-tuning llms")
    has_python = _has_skill(skills, "python")
    has_mlops = _has_skill(skills, "mlops","mlflow","weights & biases")

    # ── HONEYPOT / LOW CONFIDENCE ─────────────────────────────────────────────
    if honeypot_flags:
        issue = honeypot_flags[0]
        return (
            f"{title}, {yoe:.1f} yrs; ranked at #{rank} due to profile integrity flag: {issue}. "
            f"Top skills: {top4}."
        )

    # ── TOP TIER (1-10): strong positives + any one concern ──────────────────
    if rank <= 10:
        strengths = []
        if has_vec and has_emb:
            strengths.append("vector DB + embeddings production experience")
        elif has_vec:
            strengths.append("vector DB experience")
        elif has_emb:
            strengths.append("embeddings/semantic search experience")
        if has_rank_eval:
            strengths.append("ranking evaluation (NDCG/MRR/L2R)")
        if has_lora:
            strengths.append("LLM fine-tuning (LoRA/QLoRA/PEFT)")
        if github > 60:
            strengths.append(f"active open-source contributor (GitHub={github}/100)")
        strength_str = "; ".join(strengths[:3]) if strengths else f"skills: {top4}"

        concern = ""
        if days_ago > 90:
            concern = f"; concern: inactive {days_ago}d"
        elif notice > 90:
            concern = f"; concern: {_notice_str(notice)}"
        elif not otw:
            concern = "; not marked open-to-work"

        s1 = f"{title}, {yoe:.1f} yrs — {strength_str}{concern}."
        s2 = (
            f"{'Open to work; ' if otw else ''}"
            f"{_active_str(days_ago)}, {int(rr*100)}% recruiter response rate, "
            f"{_notice_str(notice)}; {loc_str}."
        )
        return f"{s1} {s2}"

    # ── STRONG (11-30): positive with honest note ─────────────────────────────
    elif rank <= 30:
        missing = []
        if not has_vec:   missing.append("vector DB")
        if not has_emb:   missing.append("embeddings")
        if not has_rank_eval: missing.append("ranking eval")

        s1 = f"{title}, {yoe:.1f} yrs; top skills: {top4}."
        if missing:
            s2 = (
                f"Gap vs JD: {', '.join(missing[:2])}; "
                f"but {_active_str(days_ago)}, {int(rr*100)}% response rate, {_notice_str(notice)}; {loc_str}."
            )
        else:
            s2 = (
                f"{career_hl}; "
                f"{_active_str(days_ago)}, {int(rr*100)}% response rate, {_notice_str(notice)}; {loc_str}."
            )
        return f"{s1} {s2}"

    # ── MIDDLE (31-60): mixed signals ─────────────────────────────────────────
    elif rank <= 60:
        missing = [x for x, flag in [("vector DB", not has_vec), ("embeddings", not has_emb), ("ranking eval", not has_rank_eval)] if flag]
        gap = f"missing JD core: {', '.join(missing[:2])}" if missing else "title/career less aligned with JD"
        return (
            f"{title}, {yoe:.1f} yrs; {gap}. "
            f"Skills: {top4}; {_active_str(days_ago)}, {int(rr*100)}% response rate, {_notice_str(notice)}."
        )

    # ── LOWER (61-100): honest about ranking position ─────────────────────────
    else:
        reason = []
        if features.get("title_career", 0) < 0.30:
            reason.append(f"title ({title}) is outside ML/AI career track")
        if not has_vec and not has_emb and not has_rank_eval:
            reason.append("lacks core JD skills (vector DB, embeddings, ranking eval)")
        if days_ago > 180:
            reason.append(f"inactive for {days_ago}d")
        if rr < 0.15:
            reason.append(f"very low recruiter response ({int(rr*100)}%)")
        reason_str = "; ".join(reason[:2]) if reason else "weaker fit across multiple dimensions"
        return (
            f"Rank #{rank}: {title}, {yoe:.1f} yrs — included because {reason_str}. "
            f"Skills: {top4}; {_active_str(days_ago)}."
        )
