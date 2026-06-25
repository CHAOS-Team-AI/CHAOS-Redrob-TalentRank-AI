"""
test_honeypot.py — Unit tests for honeypot detection.
Run: python3 -m unittest tests/test_honeypot.py -v
"""
import sys, unittest
from pathlib import Path
from datetime import date, timedelta
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from honeypot_detector import detect_honeypot, score_honeypot_penalty


def _base_signals():
    return {
        "open_to_work_flag": True, "recruiter_response_rate": 0.7,
        "notice_period_days": 30, "last_active_date": (date.today() - timedelta(days=10)).isoformat(),
        "avg_response_time_hours": 12, "interview_completion_rate": 0.8,
        "saved_by_recruiters_30d": 5, "willing_to_relocate": True,
        "profile_completeness_score": 85, "github_activity_score": 60,
        "verified_email": True, "verified_phone": True, "linkedin_connected": True,
        "endorsements_received": 20, "connection_count": 250,
        "applications_submitted_30d": 2, "skill_assessment_scores": {},
        "preferred_work_mode": "hybrid",
    }


def _make_candidate(career_months=60, yoe=6.0, skills=None, signals=None):
    return {
        "candidate_id": "CAND_0000001",
        "profile": {"current_title": "ML Engineer", "years_of_experience": yoe,
                    "location": "Pune", "country": "India", "headline": "ML",
                    "summary": "Test", "current_company": "Co", "current_industry": "Tech"},
        "skills": skills or [
            {"name": "Python", "proficiency": "expert", "duration_months": 24, "endorsements": 10}
        ],
        "career_history": [{
            "title": "ML Engineer", "company": "Co",
            "start_date": "2018-01-01", "end_date": None, "is_current": True,
            "duration_months": career_months, "description": "Built ML.",
            "industry": "Technology", "company_size": "201-500",
        }],
        "education": [{"tier": "tier_2", "degree": "B.Tech", "field": "CS", "institution": "NIT"}],
        "redrob_signals": signals or _base_signals(),
    }


class TestHoneypotDetector(unittest.TestCase):

    def test_clean_candidate_not_flagged(self):
        c = _make_candidate(career_months=72, yoe=6.0)
        is_hp, conf, reasons = detect_honeypot(c)
        self.assertFalse(is_hp, f"Clean candidate wrongly flagged: {reasons}")
        self.assertLess(conf, 0.5)

    def test_career_far_exceeds_yoe(self):
        # 10 years career history but claims 4 years experience
        c = _make_candidate(career_months=120, yoe=4.0)
        is_hp, conf, reasons = detect_honeypot(c)
        self.assertGreater(conf, 0.25, f"Impossible timeline not detected (conf={conf:.2f})")

    def test_expert_zero_duration_flagged(self):
        skills = [
            {"name": "FAISS",    "proficiency": "expert", "duration_months": 0, "endorsements": 10},
            {"name": "Pinecone", "proficiency": "expert", "duration_months": 0, "endorsements": 10},
        ]
        c = _make_candidate(skills=skills)
        is_hp, conf, reasons = detect_honeypot(c)
        self.assertGreater(conf, 0.25, f"Expert+zero_duration not detected (conf={conf:.2f})")

    def test_penalty_zero_at_low_confidence(self):
        self.assertEqual(score_honeypot_penalty(0.0), 1.0)
        self.assertEqual(score_honeypot_penalty(0.1), 1.0)

    def test_penalty_zero_at_high_confidence(self):
        self.assertEqual(score_honeypot_penalty(0.5), 0.0)
        self.assertEqual(score_honeypot_penalty(1.0), 0.0)

    def test_penalty_graduated_in_middle(self):
        p_low  = score_honeypot_penalty(0.25)
        p_mid  = score_honeypot_penalty(0.40)
        self.assertGreater(p_low, p_mid)
        self.assertGreater(p_mid, 0.0)
        self.assertLess(p_low, 1.0)

    def test_massive_timeline_gap_hard_zeroed(self):
        # 200 months career vs 5 years claimed — completely impossible
        c = _make_candidate(career_months=200, yoe=5.0)
        _, conf, _ = detect_honeypot(c)
        penalty = score_honeypot_penalty(conf)
        self.assertEqual(penalty, 0.0, f"Confirmed honeypot not zeroed (conf={conf:.2f}, penalty={penalty:.2f})")

    def test_clean_candidate_no_penalty(self):
        c = _make_candidate(career_months=60, yoe=7.0)
        _, conf, _ = detect_honeypot(c)
        self.assertEqual(score_honeypot_penalty(conf), 1.0)


if __name__ == "__main__":
    unittest.main()
