"""
honeypot_detector.py — Detect and penalize impossible/fabricated candidate profiles.

The challenge spec says ~80 honeypot candidates exist with "subtly impossible profiles".
Honeypot rate > 10% in top-100 causes disqualification.

Detection logic covers:
1. Impossible timelines (career history exceeds stated YoE)
2. Expert proficiency with zero months of usage
3. Skill-endorsement contradictions
4. Behavioral signal impossibilities
5. Experience inflation (many skills at expert with no endorsements)
"""

from datetime import datetime, date
from typing import Dict, Any, List, Tuple


def detect_honeypot(candidate: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
    """
    Analyze a candidate for honeypot signals.
    
    Returns:
        (is_honeypot, confidence, reasons)
        - is_honeypot: True if almost certainly fabricated
        - confidence: 0.0-1.0 honeypot confidence score
        - reasons: list of detected issues
    """
    issues = []
    score = 0.0
    
    p = candidate["profile"]
    sigs = candidate["redrob_signals"]
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    
    yoe = p.get("years_of_experience", 0)
    
    # ─── Check 1: Career history exceeds stated YoE ───────────────────────────
    career_total_months = sum(r.get("duration_months", 0) for r in career)
    career_years = career_total_months / 12.0
    
    if career_years > yoe + 4:
        # More than 4 years over claimed experience — strongly impossible
        issues.append(
            f"career_history_total={career_years:.1f}y far exceeds yoe={yoe}y"
        )
        score += 0.6
    elif career_years > yoe + 2:
        # 2-4 years over — suspicious
        issues.append(
            f"career_history_total={career_years:.1f}y exceeds yoe={yoe}y by >{2}y"
        )
        score += 0.3
    
    # ─── Check 2: Expert proficiency with 0 months of usage ──────────────────
    expert_zero = [
        s for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months") == 0
    ]
    if expert_zero:
        count = len(expert_zero)
        issues.append(
            f"{count} skills at advanced/expert with 0 months duration: "
            f"{[s['name'] for s in expert_zero[:3]]}"
        )
        score += min(0.7, count * 0.2)
    
    # ─── Check 3: All skills with very high endorsements but 0 total ──────────
    if len(skills) >= 5:
        all_high_endorsements = all(s.get("endorsements", 0) > 50 for s in skills)
        if all_high_endorsements and sigs.get("endorsements_received", 0) == 0:
            issues.append("all skills have 50+ endorsements but profile shows 0 endorsements_received")
            score += 0.5
    
    # ─── Check 4: Multiple expert skills all with suspiciously low duration ───
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 5:
        expert_low_duration = [s for s in expert_skills if s.get("duration_months", 99) < 6]
        if len(expert_low_duration) >= 3:
            issues.append(
                f"{len(expert_low_duration)}/{len(expert_skills)} expert skills have <6 months duration"
            )
            score += 0.4
    
    # ─── Check 5: Current role start date implies impossible tenure ───────────
    current_roles = [r for r in career if r.get("is_current", False)]
    for role in current_roles:
        try:
            start = datetime.fromisoformat(role["start_date"]).date()
            # If role started in the future
            if start > date.today():
                issues.append(f"current role starts in future: {start}")
                score += 0.5
        except (KeyError, ValueError, TypeError):
            pass
    
    # ─── Check 6: Impossibly high GitHub + zero connections ──────────────────
    github = sigs.get("github_activity_score", -1)
    connections = sigs.get("connection_count", 0)
    if github > 90 and connections == 0 and sigs.get("endorsements_received", 0) == 0:
        issues.append(f"github_score={github} but zero connections and zero endorsements")
        score += 0.4
    
    # ─── Check 7: Perfect completeness + zero engagement ─────────────────────
    completeness = sigs.get("profile_completeness_score", 0)
    response_rate = sigs.get("recruiter_response_rate", 0)
    interview_rate = sigs.get("interview_completion_rate", 0)
    if (
        completeness == 100
        and response_rate == 0
        and interview_rate == 0
        and sigs.get("applications_submitted_30d", 0) == 0
    ):
        issues.append("perfect completeness + zero recruiter response + zero interview completion")
        score += 0.35
    
    # ─── Decision ─────────────────────────────────────────────────────────────
    # Cap at 1.0
    score = min(score, 1.0)
    
    # Honeypot if score >= 0.5 (high confidence of fabrication)
    is_honeypot = score >= 0.5
    
    return is_honeypot, score, issues


def score_honeypot_penalty(confidence: float) -> float:
    """
    Convert honeypot confidence to a score multiplier.
    
    0.0 confidence → multiplier 1.0 (no penalty)
    0.5+ confidence → multiplier 0.0 (full disqualification)
    
    Returns a value in [0, 1] to multiply the final score by.
    """
    if confidence < 0.2:
        return 1.0
    elif confidence < 0.5:
        # Graduated penalty
        return 1.0 - (confidence - 0.2) / 0.3 * 0.5
    else:
        # Hard disqualification
        return 0.0
