#!/usr/bin/env python3
"""
notebooks/eda.py — Exploratory Data Analysis on the candidate dataset.

Run with: python3 notebooks/eda.py
Outputs a text report to stdout.

If you have Jupyter: jupyter notebook notebooks/eda.ipynb
(Convert this script to notebook with: jupytext --to notebook notebooks/eda.py)
"""
import sys, json
from pathlib import Path
from collections import Counter
import statistics
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from jd_config import MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS

CHALLENGE_DIR = Path(__file__).parent.parent / "data"
JSONL_PATHS = [
    Path(__file__).parent.parent / "data" / "candidates.jsonl",
    Path("data/candidates.jsonl"),
    Path("../data/candidates.jsonl"),
]
JSONL = next((p for p in JSONL_PATHS if p.exists()), None)

if JSONL is None:
    print("ERROR: candidates.jsonl not found. Checked:")
    for p in JSONL_PATHS:
        print(f"  {p}")
    sys.exit(1)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def bar(label, value, max_val, width=40):
    filled = int(value / max_val * width) if max_val else 0
    return f"  {label:<30} {'█'*filled}{'░'*(width-filled)} {value:,}"


print("Redrob Hackathon — Dataset EDA Report")
print(f"Generated: {date.today()}")
print(f"Dataset: data/candidates.jsonl")

# ── Pass 1: collect all stats ────────────────────────────────────────────────
titles = Counter()
yoe_list = []
locations = Counter()
countries = Counter()
industries = Counter()
all_skill_names = Counter()
ai_skill_names = Counter()
response_rates = []
notice_periods = []
otw_count = 0
total = 0
honeypot_count = 0

AI_TITLES_KW = ["ai engineer","ml engineer","machine learning","nlp engineer","data scientist",
                "applied ml","search engineer","recommendation","applied scientist","senior ai"]

with open(JSONL) as f:
    for line in f:
        c = json.loads(line)
        total += 1
        p = c["profile"]
        sigs = c["redrob_signals"]
        title_lower = p["current_title"].lower()
        is_ai = any(kw in title_lower for kw in AI_TITLES_KW)

        titles[p["current_title"]] += 1
        yoe_list.append(p["years_of_experience"])
        locations[p.get("location", "Unknown")] += 1
        countries[p.get("country", "Unknown")] += 1
        industries[p.get("current_industry", "Unknown")] += 1
        response_rates.append(sigs["recruiter_response_rate"])
        notice_periods.append(sigs["notice_period_days"])
        if sigs["open_to_work_flag"]:
            otw_count += 1

        for s in c["skills"]:
            all_skill_names[s["name"]] += 1
            if is_ai:
                ai_skill_names[s["name"]] += 1

        # Quick honeypot check
        career_months = sum(r.get("duration_months", 0) for r in c["career_history"])
        expert_zero = sum(1 for s in c["skills"] if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0)
        if career_months / 12 > p["years_of_experience"] + 4 or expert_zero >= 2:
            honeypot_count += 1

section("DATASET OVERVIEW")
print(f"  Total candidates    : {total:,}")
print(f"  File size           : {JSONL.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Honeypots detected  : {honeypot_count}")
print(f"  Open to work        : {otw_count:,} ({otw_count/total*100:.1f}%)")

section("YEARS OF EXPERIENCE")
print(f"  Min     : {min(yoe_list):.1f}")
print(f"  Max     : {max(yoe_list):.1f}")
print(f"  Mean    : {statistics.mean(yoe_list):.1f}")
print(f"  Median  : {statistics.median(yoe_list):.1f}")
buckets = Counter()
for y in yoe_list:
    if y < 2: buckets["<2y"] += 1
    elif y < 5: buckets["2-5y"] += 1
    elif y < 9: buckets["5-9y (JD target)"] += 1
    elif y < 15: buckets["9-15y"] += 1
    else: buckets["15y+"] += 1
print()
max_b = max(buckets.values())
for k in ["<2y","2-5y","5-9y (JD target)","9-15y","15y+"]:
    print(bar(k, buckets[k], max_b))

section("TITLE DISTRIBUTION (top 20)")
max_t = titles.most_common(1)[0][1]
for title, cnt in titles.most_common(20):
    print(bar(title[:30], cnt, max_t))

section("COUNTRY DISTRIBUTION (top 10)")
max_c = countries.most_common(1)[0][1]
for country, cnt in countries.most_common(10):
    print(bar(country[:30], cnt, max_c))

section("INDUSTRY DISTRIBUTION (top 10)")
max_i = industries.most_common(1)[0][1]
for ind, cnt in industries.most_common(10):
    print(bar(ind[:30], cnt, max_i))

section("BEHAVIORAL SIGNALS")
print(f"  Avg recruiter response rate : {statistics.mean(response_rates):.2f}")
print(f"  Avg notice period (days)    : {statistics.mean(notice_periods):.0f}")
notice_bins = Counter()
for n in notice_periods:
    if n <= 30: notice_bins["≤30d (ideal)"] += 1
    elif n <= 90: notice_bins["31-90d (acceptable)"] += 1
    else: notice_bins["90d+ (long)"] += 1
print()
max_n = max(notice_bins.values())
for k in ["≤30d (ideal)","31-90d (acceptable)","90d+ (long)"]:
    print(bar(k, notice_bins[k], max_n))

section("TOP SKILLS IN AI-TITLED CANDIDATES")
print("  (candidates with AI/ML engineer titles)")
print()
max_sk = ai_skill_names.most_common(1)[0][1] if ai_skill_names else 1
for skill, cnt in ai_skill_names.most_common(30):
    in_jd = "⭐" if skill in MUST_HAVE_SKILLS else ("·" if skill in NICE_TO_HAVE_SKILLS else " ")
    print(f"  {in_jd} {bar(skill[:28], cnt, max_sk)}")

section("JD SKILL COVERAGE IN DATASET")
print("  (how many candidates have each JD must-have skill)")
print()
for skill in sorted(MUST_HAVE_SKILLS):
    cnt = all_skill_names.get(skill, 0)
    print(f"  MUST  {skill:<35} {cnt:5,}  ({cnt/total*100:.1f}%)")
print()
for skill in sorted(list(NICE_TO_HAVE_SKILLS)[:10]):
    cnt = all_skill_names.get(skill, 0)
    print(f"  NICE  {skill:<35} {cnt:5,}  ({cnt/total*100:.1f}%)")

print(f"\n{'='*60}")
print("  EDA complete.")
print(f"{'='*60}\n")
