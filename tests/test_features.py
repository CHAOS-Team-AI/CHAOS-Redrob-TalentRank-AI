"""
test_features.py — Unit tests for all feature scoring functions.
Run: python3 -m unittest tests/test_features.py -v
"""
import sys, unittest
from pathlib import Path
from datetime import date, timedelta
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    score_skill_match, score_title_career, score_experience_fit,
    score_behavioral_signals, score_location_fit, score_education,
    score_github, compute_all_features,
)

_DEFAULT_SKILLS = [
    {"name": "FAISS",                 "proficiency": "expert",   "duration_months": 24, "endorsements": 15},
    {"name": "Sentence Transformers", "proficiency": "advanced", "duration_months": 18, "endorsements": 10},
    {"name": "Embeddings",            "proficiency": "expert",   "duration_months": 30, "endorsements": 20},
    {"name": "Learning to Rank",      "proficiency": "advanced", "duration_months": 12, "endorsements": 8},
    {"name": "Python",                "proficiency": "expert",   "duration_months": 60, "endorsements": 50},
    {"name": "Information Retrieval", "proficiency": "advanced", "duration_months": 24, "endorsements": 12},
]

_DEFAULT_CAREER = [{
    "title": "ML Engineer", "company": "ProductCo",
    "start_date": "2018-01-01", "end_date": None, "is_current": True,
    "duration_months": 72, "industry": "Technology", "company_size": "201-500",
    "description": "Built retrieval and ranking systems, deployed embeddings to production, A/B tested ranking models.",
}]

_SENTINEL = object()  # use instead of None to allow explicit [] override


def _make_candidate(
    title="Senior AI Engineer", yoe=7.0,
    skills=_SENTINEL, career=_SENTINEL,
    location="Pune", country="India", industry="Technology",
    otw=True, rr=0.8, notice=30, days_ago=10, github=-1,
    willing_to_relocate=True,
):
    last_active = (date.today() - timedelta(days=days_ago)).isoformat()
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": title, "years_of_experience": yoe,
            "location": location, "country": country,
            "current_industry": industry, "headline": f"{title}",
            "summary": "Experienced ML engineer.", "current_company": "TestCo",
        },
        "skills":          _DEFAULT_SKILLS if skills  is _SENTINEL else skills,
        "career_history":  _DEFAULT_CAREER if career  is _SENTINEL else career,
        "education": [{"tier": "tier_1", "degree": "B.Tech", "field": "CS", "institution": "IIT"}],
        "redrob_signals": {
            "open_to_work_flag": otw, "recruiter_response_rate": rr,
            "notice_period_days": notice, "last_active_date": last_active,
            "avg_response_time_hours": 12, "interview_completion_rate": 0.85,
            "saved_by_recruiters_30d": 5, "willing_to_relocate": willing_to_relocate,
            "profile_completeness_score": 90, "github_activity_score": github,
            "verified_email": True, "verified_phone": True, "linkedin_connected": True,
            "endorsements_received": 25, "connection_count": 300,
            "applications_submitted_30d": 3, "skill_assessment_scores": {},
            "preferred_work_mode": "hybrid",
        },
    }


class TestSkillMatch(unittest.TestCase):

    def test_ideal_ai_candidate_scores_above_25pct(self):
        c = _make_candidate()
        score = score_skill_match(c)
        self.assertGreater(score, 0.25, f"Ideal AI candidate should score >0.25, got {score:.3f}")

    def test_no_skills_returns_zero(self):
        c = _make_candidate(skills=[])
        self.assertEqual(score_skill_match(c), 0.0)

    def test_must_have_beats_nice_to_have(self):
        c_must = _make_candidate(skills=[
            {"name": "FAISS",   "proficiency": "expert", "duration_months": 24, "endorsements": 15},
            {"name": "Pinecone","proficiency": "expert", "duration_months": 24, "endorsements": 15},
        ])
        c_nice = _make_candidate(skills=[
            {"name": "PyTorch",    "proficiency": "expert", "duration_months": 24, "endorsements": 15},
            {"name": "TensorFlow", "proficiency": "expert", "duration_months": 24, "endorsements": 15},
        ])
        self.assertGreater(score_skill_match(c_must), score_skill_match(c_nice))

    def test_zero_duration_expert_heavily_penalized(self):
        c_good = _make_candidate(skills=[
            {"name": "FAISS", "proficiency": "expert", "duration_months": 24, "endorsements": 10}
        ])
        c_bad = _make_candidate(skills=[
            {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 10}
        ])
        self.assertGreater(score_skill_match(c_good), score_skill_match(c_bad) * 2)

    def test_negative_skills_penalized(self):
        c_clean = _make_candidate(skills=[
            {"name": "FAISS", "proficiency": "expert", "duration_months": 24, "endorsements": 10}
        ])
        c_neg = _make_candidate(skills=[
            {"name": "FAISS",      "proficiency": "expert", "duration_months": 24, "endorsements": 10},
            {"name": "Marketing",  "proficiency": "expert", "duration_months": 24, "endorsements": 10},
            {"name": "Accounting", "proficiency": "expert", "duration_months": 24, "endorsements": 10},
        ])
        self.assertGreater(score_skill_match(c_clean), score_skill_match(c_neg))

    def test_score_always_in_0_1(self):
        for title in ["Senior AI Engineer", "Marketing Manager", "HR Manager"]:
            c = _make_candidate(title=title)
            s = score_skill_match(c)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class TestTitleCareer(unittest.TestCase):

    def test_ideal_title_scores_high(self):
        for title in ["Senior AI Engineer", "ML Engineer", "Senior Machine Learning Engineer", "NLP Engineer"]:
            c = _make_candidate(title=title)
            s = score_title_career(c)
            self.assertGreater(s, 0.55, f"'{title}' should score >0.55, got {s:.3f}")

    def test_disqualifying_titles_score_zero(self):
        for bad in ["Marketing Manager", "HR Manager", "Graphic Designer", "Accountant", "Sales Executive"]:
            c = _make_candidate(title=bad)
            s = score_title_career(c)
            self.assertEqual(s, 0.0, f"'{bad}' should score exactly 0.0, got {s:.3f}")

    def test_ai_engineer_far_beats_marketing_manager(self):
        c_ai  = _make_candidate(title="Senior AI Engineer")
        c_mkt = _make_candidate(title="Marketing Manager")
        self.assertGreater(score_title_career(c_ai), 0.7)
        self.assertEqual(score_title_career(c_mkt), 0.0)

    def test_ml_career_descriptions_help(self):
        good_career = [{"title": "Backend Engineer", "company": "ProductCo",
            "start_date": "2019-01-01", "end_date": None, "is_current": True,
            "duration_months": 60, "industry": "Technology", "company_size": "201-500",
            "description": "Built recommendation and ranking systems using embeddings and vector search. A/B tested rankers."}]
        blank_career = [{"title": "Backend Engineer", "company": "ProductCo",
            "start_date": "2019-01-01", "end_date": None, "is_current": True,
            "duration_months": 60, "industry": "Technology", "company_size": "201-500",
            "description": "Maintained internal CRUD APIs."}]
        c_ml    = _make_candidate(career=good_career)
        c_blank = _make_candidate(career=blank_career)
        self.assertGreater(score_title_career(c_ml), score_title_career(c_blank))


class TestExperienceFit(unittest.TestCase):

    def test_sweet_spot_scores_one(self):
        for yoe in [6.0, 7.0, 8.0]:
            self.assertEqual(score_experience_fit(_make_candidate(yoe=yoe)), 1.0)

    def test_under_2_very_low(self):
        self.assertLess(score_experience_fit(_make_candidate(yoe=1.0)), 0.15)

    def test_over_20_low(self):
        self.assertLess(score_experience_fit(_make_candidate(yoe=22.0)), 0.30)

    def test_monotone_around_sweet_spot(self):
        yoes  = [3, 5, 6, 7, 8, 9, 11, 15]
        scores = [score_experience_fit(_make_candidate(yoe=y)) for y in yoes]
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[4], scores[5])
        self.assertGreater(scores[5], scores[6])

    def test_5_to_9_all_above_0_6(self):
        for yoe in [5, 5.5, 6, 7, 8, 8.5, 9]:
            s = score_experience_fit(_make_candidate(yoe=yoe))
            self.assertGreater(s, 0.60, f"yoe={yoe} should >0.60, got {s:.3f}")


class TestBehavioralSignals(unittest.TestCase):

    def test_active_otw_high_rr_scores_high(self):
        c = _make_candidate(otw=True, rr=0.9, notice=15, days_ago=5)
        self.assertGreater(score_behavioral_signals(c), 0.65)

    def test_inactive_200d_low_rr_scores_below_active(self):
        c_active   = _make_candidate(days_ago=5,   otw=True,  rr=0.9)
        c_inactive = _make_candidate(days_ago=200, otw=False, rr=0.05)
        self.assertGreater(score_behavioral_signals(c_active), score_behavioral_signals(c_inactive) + 0.15)

    def test_long_notice_scores_below_short_notice(self):
        c_short = _make_candidate(notice=15,  otw=False, rr=0.5, days_ago=10)
        c_long  = _make_candidate(notice=180, otw=False, rr=0.5, days_ago=10)
        self.assertGreater(score_behavioral_signals(c_short), score_behavioral_signals(c_long))

    def test_otw_flag_adds_bonus(self):
        c_otw    = _make_candidate(otw=True,  days_ago=10, rr=0.7)
        c_no_otw = _make_candidate(otw=False, days_ago=10, rr=0.7)
        self.assertGreater(score_behavioral_signals(c_otw), score_behavioral_signals(c_no_otw))

    def test_score_in_0_1(self):
        for days in [1, 30, 90, 200, 365]:
            c = _make_candidate(days_ago=days, otw=False, rr=0.1)
            s = score_behavioral_signals(c)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class TestLocationFit(unittest.TestCase):

    def test_pune_scores_1(self):
        self.assertEqual(score_location_fit(_make_candidate(location="Pune", country="India")), 1.0)

    def test_noida_scores_1(self):
        self.assertEqual(score_location_fit(_make_candidate(location="Noida", country="India")), 1.0)

    def test_outside_india_no_relocate_low(self):
        c = _make_candidate(location="San Francisco", country="USA", willing_to_relocate=False)
        self.assertLess(score_location_fit(c), 0.25)

    def test_india_with_relocate_decent(self):
        c = _make_candidate(location="Chennai", country="India", willing_to_relocate=True)
        self.assertGreaterEqual(score_location_fit(c), 0.80)


class TestComputeAllFeatures(unittest.TestCase):

    def test_returns_correct_keys(self):
        feats = compute_all_features(_make_candidate())
        expected = {"skill_match","title_career","experience_fit","behavioral_signals",
                    "location_fit","education_quality","github_activity"}
        self.assertEqual(set(feats.keys()), expected)

    def test_all_values_in_0_1(self):
        feats = compute_all_features(_make_candidate())
        for k, v in feats.items():
            self.assertGreaterEqual(v, 0.0, f"{k} < 0")
            self.assertLessEqual(v,   1.0, f"{k} > 1")

    def test_ideal_candidate_scores_high(self):
        feats = compute_all_features(_make_candidate())
        self.assertGreater(feats["title_career"],      0.60)
        self.assertEqual(  feats["experience_fit"],    1.0)
        self.assertGreater(feats["behavioral_signals"],0.55)
        self.assertEqual(  feats["location_fit"],      1.0)

    def test_disqualified_title_score_zero(self):
        c = _make_candidate(title="Marketing Manager", skills=[], career=[])
        feats = compute_all_features(c)
        self.assertEqual(feats["title_career"], 0.0)


if __name__ == "__main__":
    unittest.main()


class TestSemanticScoring(unittest.TestCase):
    """Tests for the semantic.py module (TF-IDF fallback, no sentence-transformers needed)."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from semantic import compute_semantic_scores, candidate_to_text
        self.compute = compute_semantic_scores
        self.to_text  = candidate_to_text

    def _make(self, title, skills, summary=""):
        return {
            "candidate_id": f"CAND_{title[:5].upper()}",
            "profile": {"current_title": title, "years_of_experience": 7,
                        "headline": title, "summary": summary,
                        "current_industry": "Technology", "current_company": "Co",
                        "location": "Pune", "country": "India"},
            "skills": [{"name": s, "proficiency": "expert",
                        "duration_months": 24, "endorsements": 10} for s in skills],
            "career_history": [{"title": title, "company": "ProductCo",
                "start_date": "2018-01-01", "end_date": None, "is_current": True,
                "duration_months": 60, "industry": "Technology", "company_size": "201-500",
                "description": f"Built {' '.join(skills[:3]).lower()} systems in production."}],
            "education": [],
            "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.7,
                "notice_period_days": 30, "last_active_date": "2026-06-01",
                "avg_response_time_hours": 12, "interview_completion_rate": 0.8,
                "saved_by_recruiters_30d": 2, "willing_to_relocate": True,
                "profile_completeness_score": 85, "github_activity_score": 60,
                "verified_email": True, "verified_phone": True, "linkedin_connected": True,
                "endorsements_received": 10, "connection_count": 150,
                "applications_submitted_30d": 1, "skill_assessment_scores": {},
                "preferred_work_mode": "hybrid"},
        }

    def test_returns_array_correct_length(self):
        cands = [
            self._make("ML Engineer", ["FAISS","Embeddings","Python"]),
            self._make("Marketing Manager", ["Marketing","SEO","Photoshop"]),
        ]
        scores = self.compute(cands)
        self.assertEqual(len(scores), 2)

    def test_scores_in_0_1_range(self):
        cands = [self._make("ML Engineer", ["FAISS","Embeddings","Python","NDCG","Pinecone"])]
        scores = self.compute(cands)
        self.assertGreaterEqual(float(scores[0]), 0.0)
        self.assertLessEqual(float(scores[0]), 1.0)

    def test_ai_candidate_beats_marketing_manager(self):
        ai  = self._make("ML Engineer",       ["FAISS","Embeddings","Sentence Transformers","NDCG","Python"])
        mkt = self._make("Marketing Manager", ["Marketing","SEO","Content Writing","Photoshop"])
        scores = self.compute([ai, mkt])
        self.assertGreater(float(scores[0]), float(scores[1]),
            f"AI candidate sem={scores[0]:.3f} should > Marketing sem={scores[1]:.3f}")

    def test_candidate_to_text_not_empty(self):
        c = self._make("ML Engineer", ["FAISS","Embeddings"])
        text = self.to_text(c)
        self.assertGreater(len(text), 50)
        self.assertIn("ML Engineer", text)
