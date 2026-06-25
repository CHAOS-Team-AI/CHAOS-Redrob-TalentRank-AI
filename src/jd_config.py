"""
jd_config.py — Job Description signal weights and EXACT keyword taxonomy.

All skill names here use the EXACT casing from the dataset.
Verified against actual skill distribution in candidates.jsonl.
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXACT skill names from the dataset (verified against candidates.jsonl)
# ─────────────────────────────────────────────────────────────────────────────

# MUST-HAVE: production retrieval / vector / embeddings / ranking eval
# These map 1:1 to skills that appear hundreds of times in the dataset
MUST_HAVE_SKILLS = {
    # Vector databases
    "FAISS", "Pinecone", "Qdrant", "Milvus", "Weaviate",
    "OpenSearch", "Elasticsearch", "pgvector",
    "Vector Search",
    # Embeddings / retrieval
    "Sentence Transformers", "Embeddings", "Semantic Search",
    "Dense Retrieval", "Hybrid Search",
    # Ranking evaluation
    "Learning to Rank", "Information Retrieval", "BM25",
    "NDCG", "MRR",
    # Core engineering
    "Python", "MLOps",
}

# NICE-TO-HAVE: LLM fine-tuning, broader ML, search tooling
NICE_TO_HAVE_SKILLS = {
    # LLM fine-tuning
    "LoRA", "QLoRA", "PEFT", "Fine-tuning LLMs",
    "RLHF",
    # ML frameworks
    "PyTorch", "TensorFlow", "Hugging Face Transformers", "JAX",
    "scikit-learn",
    # RAG & tooling
    "RAG", "Haystack", "LlamaIndex",
    "LangChain",  # slight negative actually but present in dataset
    # ML platforms
    "MLflow", "Weights & Biases", "Kubeflow", "BentoML",
    # General ML
    "NLP", "Deep Learning", "Machine Learning", "Feature Engineering",
    # LLMs broadly
    "LLMs", "Prompt Engineering",
    # Recommendation / search
    "Recommendation Systems",
}

# MINOR POSITIVE (present but not core to the JD)
MINOR_POSITIVE_SKILLS = {
    "Computer Vision", "Reinforcement Learning",
    "Docker", "Kubernetes", "GCP", "Azure",
    "SQL", "Databricks", "Kafka", "Airflow",
}

# NEGATIVE skills — indicate wrong specialty
NEGATIVE_SKILLS = {
    "Marketing", "Sales", "Accounting", "Tally", "SEO",
    "Content Writing", "Photoshop", "Excel", "Project Management",
}

# ─────────────────────────────────────────────────────────────────────────────
# TITLE TAXONOMY — exact strings from current_title field
# ─────────────────────────────────────────────────────────────────────────────

# Score = 1.0 — perfect title match
IDEAL_TITLES = {
    "Senior AI Engineer", "Lead AI Engineer", "Staff AI Engineer",
    "Senior Machine Learning Engineer", "Staff Machine Learning Engineer",
    "Principal ML Engineer", "Lead ML Engineer",
    "Senior NLP Engineer", "Lead NLP Engineer",
    "Senior Applied Scientist", "Applied ML Engineer",
    "Senior AI/ML Engineer", "AI/ML Engineer",
    "Senior Search Engineer", "Search Engineer",
    "Senior Recommendation Engineer",
    "ML Engineer",  # strong but slightly junior
    "AI Engineer",
    "NLP Engineer",
}

# Score = 0.75 — strong but not ideal
STRONG_TITLES = {
    "Junior ML Engineer",
    "Senior Software Engineer",
    "Senior Backend Engineer",
    "Senior Data Scientist",
    "Data Scientist",
    "Research Engineer",
    "AI Research Engineer",
    "Computer Vision Engineer",  # adjacent but valid
    "Machine Learning Engineer",  # duplicate spelling
}

# Score = 0.45 — adjacent, might work
ADJACENT_TITLES = {
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer",
    "Backend Engineer", "Software Engineer",
    "Full Stack Developer", "Senior Full Stack Developer",
    "Cloud Engineer", "DevOps Engineer",
    "Data Analyst", "Senior Data Analyst",
}

# Score = 0.0 — hard disqualification
DISQUALIFYING_TITLES = {
    "Marketing Manager", "Marketing Executive", "Digital Marketing Manager",
    "SEO Specialist", "Content Writer", "Content Strategist",
    "Graphic Designer", "UI Designer", "UX Designer",
    "Accountant", "Financial Analyst", "CA", "Chartered Accountant",
    "Civil Engineer", "Mechanical Engineer", "Electrical Engineer",
    "HR Manager", "HR Executive", "Recruiter", "Talent Acquisition",
    "Customer Support", "Customer Success Manager",
    "Sales Executive", "Sales Manager", "Business Development",
    "Operations Manager", "Supply Chain Manager",
    "Project Manager",  # unless ML background
    "Mobile Developer", "iOS Developer", "Android Developer",
    "Frontend Engineer", "Frontend Developer",  # unless strong ML skills
    "QA Engineer", "Test Engineer",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSULTING COMPANIES — entire career there = weak product signal
# ─────────────────────────────────────────────────────────────────────────────

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "niit technologies", "l&t infotech", "l&t technology",
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE RANGE
# ─────────────────────────────────────────────────────────────────────────────

YOE_IDEAL_MIN = 5.0
YOE_IDEAL_MAX = 9.0
YOE_SWEET_SPOT_MIN = 6.0
YOE_SWEET_SPOT_MAX = 8.0

# ─────────────────────────────────────────────────────────────────────────────
# LOCATION PREFERENCES
# ─────────────────────────────────────────────────────────────────────────────

TIER1_LOCATIONS = {
    "pune", "noida", "delhi", "new delhi", "delhi ncr", "ncr",
    "gurugram", "gurgaon", "faridabad", "greater noida",
}

TIER2_LOCATIONS = {
    "hyderabad", "bangalore", "bengaluru", "mumbai", "bombay",
    "india",
}

# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL SIGNAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

RECENCY_FRESH_DAYS = 30      # Active within 30d = full score
RECENCY_STALE_DAYS = 90      # 90d = moderate penalty
RECENCY_DEAD_DAYS = 180      # 180d = heavy penalty

NOTICE_IDEAL_MAX = 30        # ≤30d = ideal
NOTICE_ACCEPTABLE_MAX = 90   # ≤90d = acceptable (buyout possible per JD)

# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS — sum to 1.0
# NDCG@10 is 50% of final score so top-10 quality is paramount
# Skill + title/career dominate = 55% of composite
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skill_match":          0.30,   # Core: skills vs JD
    "title_career":         0.25,   # Career trajectory
    "semantic_similarity":  0.15,   # Embedding cosine to JD
    "experience_fit":       0.10,   # YoE window
    "behavioral_signals":   0.10,   # Engagement & availability
    "location_fit":         0.05,   # Pune/Noida preference
    "education_quality":    0.03,   # Institution tier
    "github_activity":      0.02,   # Open-source signal
}

# ─────────────────────────────────────────────────────────────────────────────
# JD TEXT for semantic embedding
# ─────────────────────────────────────────────────────────────────────────────

JD_TEXT = """
Senior AI Engineer at Redrob AI, Series A startup, Pune or Noida India. Hybrid.
5 to 9 years experience. Sweet spot 6 to 8 years applied ML at product companies.

Must have:
Production embeddings-based retrieval systems using Sentence Transformers, OpenAI embeddings, BGE, E5.
Handled embedding drift, index refresh, retrieval quality regression in production.
Production vector databases: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, pgvector.
Strong Python, production code quality.
Evaluation frameworks for ranking: NDCG, MRR, MAP, offline-to-online correlation, A/B testing, BM25, Information Retrieval.
Learning to Rank, hybrid search, dense retrieval, semantic search.

Nice to have:
LLM fine-tuning: LoRA, QLoRA, PEFT, Fine-tuning LLMs.
XGBoost learning-to-rank or neural ranking models.
HR-tech, recruiting, marketplace product experience.
RAG, Haystack, LlamaIndex, recommendation systems.
Open source AI ML contributions, GitHub activity.
MLOps, MLflow, Hugging Face Transformers, PyTorch, NLP, Deep Learning.

Role: Own intelligence layer, ranking retrieval matching systems.
Ship v2 ranking with embeddings, hybrid retrieval, LLM re-ranking.
Evaluation infrastructure offline benchmarks online A/B testing.
Candidate-JD matching at scale. Mentor engineers.
Product company background, shipped end-to-end ranking search recommendation to real users.
Strong opinions on retrieval hybrid dense, evaluation offline online, LLM fine-tune vs prompt.
Located in or willing to relocate to Noida or Pune, India.
Active on platform, responds to recruiters, short notice period.
"""
