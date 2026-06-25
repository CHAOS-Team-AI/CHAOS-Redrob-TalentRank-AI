"""
demo/app.py — Streamlit demo application for the Redrob Candidate Ranker.

Run with:
    streamlit run demo/app.py

The demo loads sample_candidates.json by default (500 candidates).
Upload any .jsonl file via the sidebar for custom data.
For the full 100K dataset, ensure data/candidates.jsonl is present.
"""

import sys
import json
import csv
from pathlib import Path
from datetime import date, timedelta, datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

if not HAS_STREAMLIT:
    print("ERROR: streamlit and plotly are required.")
    print("Install with: pip install streamlit plotly pandas")
    sys.exit(1)

from features import compute_all_features
from honeypot_detector import detect_honeypot
from reasoning import generate_reasoning
from jd_config import WEIGHTS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_since(ds):
    try:
        return (date.today() - datetime.fromisoformat(ds).date()).days
    except Exception:
        return 999


@st.cache_data(show_spinner="Loading candidates…")
def load_candidates_from_bytes(file_bytes: bytes, is_json_array: bool) -> list:
    text = file_bytes.decode("utf-8")
    if is_json_array:
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@st.cache_data(show_spinner="Loading default sample data…")
def load_default_candidates(path: str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        content = f.read(1)
        f.seek(0)
        if content == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data(show_spinner="Scoring candidates…")
def score_candidates(data_key: str, candidates_json: str) -> list:
    candidates = json.loads(candidates_json)
    results = []
    for c in candidates:
        feats = compute_all_features(c)
        is_hp, hp_conf, hp_reasons = detect_honeypot(c)
        hp_penalty = 0.0 if hp_conf >= 0.5 else (1.0 - (hp_conf - 0.2) / 0.3 * 0.5 if hp_conf > 0.2 else 1.0)
        beh = feats["behavioral_signals"]
        beh_mult = 0.60 + beh * 0.40
        base = sum(feats[k] * WEIGHTS[k] for k in feats)
        score = (base * beh_mult + beh * WEIGHTS["behavioral_signals"]) * hp_penalty
        score = min(1.0, max(0.0, score))
        p = c["profile"]
        sigs = c["redrob_signals"]
        results.append({
            "candidate_id":  c["candidate_id"],
            "title":         p.get("current_title", ""),
            "yoe":           p.get("years_of_experience", 0),
            "location":      p.get("location", ""),
            "country":       p.get("country", ""),
            "score":         score,
            "skill_match":   feats["skill_match"],
            "title_career":  feats["title_career"],
            "exp_fit":       feats["experience_fit"],
            "behavioral":    feats["behavioral_signals"],
            "location_fit":  feats["location_fit"],
            "edu":           feats["education_quality"],
            "github":        feats["github_activity"],
            "is_honeypot":   is_hp,
            "hp_conf":       hp_conf,
            "hp_reasons":    "; ".join(hp_reasons[:2]),
            "otw":           sigs.get("open_to_work_flag", False),
            "rr":            sigs.get("recruiter_response_rate", 0),
            "notice":        sigs.get("notice_period_days", 60),
            "last_active":   sigs.get("last_active_date", ""),
            "github_score":  sigs.get("github_activity_score", -1),
            "_candidate":    c,
            "_features":     feats,
            "_hp_reasons":   hp_reasons,
        })
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["reasoning"] = generate_reasoning(
            candidate=r["_candidate"],
            features=r["_features"],
            rank=i + 1,
            honeypot_flags=r["_hp_reasons"],
        )
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎯 Redrob Ranker")
st.sidebar.markdown("**Senior AI Engineer — Founding Team**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Leaderboard", "🔍 Inspector", "🚨 Honeypots", "📈 Analytics"],
)

st.sidebar.divider()
st.sidebar.markdown("**Data Source**")
uploaded = st.sidebar.file_uploader("Upload .jsonl or .json", type=["jsonl","json"])

# Resolve data path
DEFAULT_PATHS = [
    Path(__file__).parent.parent / "data" / "sample_candidates.json",
    Path(__file__).parent.parent / "data" / "candidates.jsonl",
    Path("data/sample_candidates.json"),
    Path("data/candidates.jsonl"),
]
default_path = next((str(p) for p in DEFAULT_PATHS if p.exists()), None)

if uploaded:
    raw = uploaded.read()
    is_json = uploaded.name.endswith(".json") and raw.lstrip()[:1] == b"["
    candidates = load_candidates_from_bytes(raw, is_json)
    data_key = uploaded.name
    st.sidebar.success(f"Loaded {len(candidates):,} candidates from upload")
elif default_path:
    candidates = load_default_candidates(default_path)
    data_key = default_path
    fname = Path(default_path).name
    n = len(candidates)
    st.sidebar.info(f"Using: {fname} ({n:,} candidates)")
else:
    candidates = []
    data_key = ""

if not candidates:
    st.warning("⚠️ No candidate data found.")
    st.markdown("""
    **To get started:**
    1. Upload a `.jsonl` or `.json` file via the sidebar, **or**
    2. Place `sample_candidates.json` in the `data/` folder, **or**
    3. Place `candidates.jsonl` in the `data/` folder for the full 100K dataset
    """)
    st.stop()

# Score
results = score_candidates(data_key, json.dumps(candidates))
df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
n_total = len(results)
n_show = min(n_total, 100)


# ── Overview ──────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🎯 Redrob — Senior AI Engineer Ranker")
    st.caption("Ranking candidates for the founding-team AI Engineer role · Pune/Noida, India")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Candidates", f"{n_total:,}")
    c2.metric("Honeypots Detected", str(int(df["is_honeypot"].sum())))
    c3.metric("Open to Work", str(int(df["otw"].sum())))
    c4.metric("In Pune/Noida", str(int(df["location"].str.lower().str.contains("pune|noida", na=False).sum())))
    c5.metric("Top Score", f"{df['score'].max():.3f}")

    st.divider()
    st.subheader("🏆 Top 10 Candidates")

    for r in results[:10]:
        icon = "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "🥉" if r["rank"] == 3 else f"#{r['rank']}"
        with st.expander(
            f"{icon} **{r['candidate_id']}** — {r['title']} · {r['yoe']:.1f} yrs · Score: **{r['score']:.3f}**",
            expanded=r["rank"] <= 3,
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall", f"{r['score']:.3f}")
            col2.metric("Skill Match", f"{r['skill_match']:.3f}")
            col3.metric("Title/Career", f"{r['title_career']:.3f}")
            col4.metric("Behavioral", f"{r['behavioral']:.3f}")
            st.info(f"**Reasoning:** {r['reasoning']}")

    st.divider()
    st.subheader("⚖️ Score Component Weights")
    wdf = pd.DataFrame(
        [(k.replace("_"," ").title(), v*100) for k, v in WEIGHTS.items()],
        columns=["Component","Weight (%)"]
    ).sort_values("Weight (%)", ascending=True)
    fig = px.bar(wdf, x="Weight (%)", y="Component", orientation="h",
                 color="Weight (%)", color_continuous_scale="viridis",
                 title="How the composite score is built")
    fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Leaderboard ───────────────────────────────────────────────────────────────
elif page == "📊 Leaderboard":
    st.title("📊 Candidate Leaderboard")

    col1, col2, col3 = st.columns(3)
    with col1: max_rank = st.slider("Show top N", 10, n_show, min(50, n_show))
    with col2: hide_hp  = st.checkbox("Hide honeypots", value=True)
    with col3: only_otw = st.checkbox("Open to work only", value=False)

    view = df.head(max_rank).copy()
    if hide_hp:  view = view[~view["is_honeypot"]]
    if only_otw: view = view[view["otw"]]

    display = view[["rank","candidate_id","title","yoe","location","score",
                     "skill_match","title_career","behavioral","otw","notice","rr"]].copy()
    display.columns = ["Rank","ID","Title","YoE","Location","Score",
                        "Skill","Title/Career","Behavioral","OTW","Notice(d)","Resp%"]
    for col in ["Score","Skill","Title/Career","Behavioral"]:
        display[col] = display[col].map("{:.3f}".format)
    display["Resp%"] = display["Resp%"].map("{:.0%}".format)
    display["OTW"] = display["OTW"].map({True:"✅",False:"❌"})

    st.dataframe(display, use_container_width=True, height=600)

    st.subheader("Score Distribution")
    fig = px.histogram(df, x="score", nbins=40,
                       title=f"Score distribution across all {n_total:,} candidates",
                       color_discrete_sequence=["#6366f1"])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── Inspector ─────────────────────────────────────────────────────────────────
elif page == "🔍 Inspector":
    st.title("🔍 Candidate Inspector")

    options = [
        f"#{r['rank']}  {r['candidate_id']}  —  {r['title']}  ({r['yoe']:.1f}y)  score={r['score']:.3f}"
        for r in results[:n_show]
    ]
    sel = st.selectbox("Select a candidate:", options)
    cid = sel.split()[1]

    row = next(r for r in results if r["candidate_id"] == cid)
    c   = row["_candidate"]
    p   = c["profile"]
    sigs = c["redrob_signals"]

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"## {p.get('current_title','')}  ·  {cid}")
        st.markdown(f"*{p.get('headline','')}*")
        st.caption(f"📍 {p.get('location','')} · {p.get('country','')} · {p.get('current_industry','')}")
    with col2:
        st.metric("Rank",  f"#{row['rank']}")
        st.metric("Score", f"{row['score']:.3f}")
    with col3:
        st.metric("YoE",    f"{p.get('years_of_experience',0):.1f} yrs")
        st.metric("Notice", f"{sigs.get('notice_period_days',0)} days")

    st.divider()

    # Radar chart
    dims = {
        "Skill Match":   row["skill_match"],
        "Title/Career":  row["title_career"],
        "Experience":    row["exp_fit"],
        "Behavioral":    row["behavioral"],
        "Location":      row["location_fit"],
        "Education":     row["edu"],
        "GitHub":        row["github"],
    }
    fig = go.Figure(go.Scatterpolar(
        r=list(dims.values()) + [list(dims.values())[0]],
        theta=list(dims.keys()) + [list(dims.keys())[0]],
        fill="toself", line_color="#6366f1",
        fillcolor="rgba(99,102,241,0.20)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False, height=380, title="Score component breakdown",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Reasoning:** {row['reasoning']}")

    if row["is_honeypot"]:
        st.error(f"⚠️ HONEYPOT DETECTED (conf={row['hp_conf']:.2f}): {row['hp_reasons']}")

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Open to Work",  "✅" if sigs.get("open_to_work_flag") else "❌")
    col2.metric("Response Rate", f"{sigs.get('recruiter_response_rate',0):.0%}")
    col3.metric("Interview Rate",f"{sigs.get('interview_completion_rate',0):.0%}")
    col4.metric("GitHub Score",  str(sigs.get("github_activity_score",-1)))

    st.markdown(f"**Summary:** {p.get('summary','N/A')}")

    st.subheader("Skills")
    skills_df = pd.DataFrame(c.get("skills", []))
    if not skills_df.empty:
        skills_df = skills_df.sort_values("endorsements", ascending=False)
        st.dataframe(skills_df[["name","proficiency","duration_months","endorsements"]], use_container_width=True)

    st.subheader("Career History")
    for role in c.get("career_history", []):
        end = role.get("end_date") or "Present"
        with st.expander(f"**{role.get('title','')}** at {role.get('company','')}  ({role.get('start_date','')[:7]} – {str(end)[:7]})"):
            st.text(role.get("description",""))


# ── Honeypots ─────────────────────────────────────────────────────────────────
elif page == "🚨 Honeypots":
    st.title("🚨 Honeypot Detector")
    st.markdown("""
    The dataset contains ~80 fabricated profiles. A **honeypot rate > 10% in top-100 causes
    disqualification**. Our detector identifies impossible profiles using 5 checks:
    - Career history duration > stated YoE + 4 years
    - Expert proficiency with 0 months of usage
    - All skills show 50+ endorsements but `endorsements_received = 0`
    - GitHub score = 100 with zero connections
    - Perfect completeness + zero recruiter engagement
    """)

    suspicious = df[df["hp_conf"] > 0.10].sort_values("hp_conf", ascending=False)
    confirmed  = df[df["is_honeypot"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Suspicious (conf > 0.10)", len(suspicious))
    c2.metric("Confirmed Honeypots (conf ≥ 0.50)", len(confirmed))
    c3.metric("Honeypots in Top-100", str(int(df.head(100)["is_honeypot"].sum())))

    st.divider()
    st.subheader("Suspicious Profiles")
    if len(suspicious):
        disp = suspicious[["candidate_id","title","yoe","score","hp_conf","hp_reasons"]].head(20).copy()
        disp.columns = ["ID","Title","YoE","Score","Confidence","Reasons"]
        disp["Confidence"] = disp["Confidence"].map("{:.2f}".format)
        disp["Score"]      = disp["Score"].map("{:.4f}".format)
        st.dataframe(disp, use_container_width=True)
    else:
        st.info("No suspicious profiles found in this dataset.")


# ── Analytics ─────────────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.title("📈 Score Analytics")

    top100 = df.head(min(100, n_total))

    # Feature scores vs rank
    st.subheader("Feature Scores by Rank (Top 100)")
    fig = go.Figure()
    for col, color in [("skill_match","#6366f1"),("title_career","#22c55e"),
                        ("behavioral","#f59e0b"),("exp_fit","#ec4899")]:
        fig.add_trace(go.Scatter(x=top100["rank"], y=top100[col], mode="lines",
                                  name=col.replace("_"," ").title(), line_color=color))
    fig.update_layout(height=400, yaxis_range=[0,1], xaxis_title="Rank", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Title Distribution — Top 50")
        title_counts = top100.head(50)["title"].value_counts().reset_index()
        title_counts.columns = ["Title","Count"]
        fig2 = px.bar(title_counts.head(12), x="Count", y="Title", orientation="h",
                      color="Count", color_continuous_scale="viridis")
        fig2.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("YoE Distribution — Top 100")
        fig3 = px.histogram(top100, x="yoe", nbins=15, color_discrete_sequence=["#6366f1"])
        fig3.add_vrect(x0=5, x1=9, fillcolor="green", opacity=0.10,
                       annotation_text="JD ideal range")
        fig3.update_layout(height=380, xaxis_title="Years of Experience")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Notice Period vs Rank")
    fig4 = px.scatter(top100, x="rank", y="notice", color="score",
                      color_continuous_scale="viridis",
                      labels={"rank":"Rank","notice":"Notice Period (days)"})
    fig4.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Ideal ≤30d")
    fig4.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Acceptable ≤90d")
    fig4.update_layout(height=380)
    st.plotly_chart(fig4, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Redrob Hackathon 2026 · Senior AI Engineer Ranker")
st.sidebar.caption("Demo uses rule-based scoring (no GPU required).")
