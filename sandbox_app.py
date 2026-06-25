"""
NeuroRecruit — Sandbox Demo
Redrob Hackathon — Streamlit Cloud
Run: streamlit run sandbox_app.py
"""

import streamlit as st
import json
import io
import time
import pandas as pd

st.set_page_config(
    page_title="NeuroRecruit — Intelligent Candidate Discovery",
    page_icon="🧠",
    layout="wide"
)

JD_REQUIRED_SKILLS = {
    "embeddings": 10,"embedding": 10,"sentence-transformers": 10,
    "pinecone": 9,"weaviate": 9,"qdrant": 9,"milvus": 9,"faiss": 9,
    "opensearch": 8,"elasticsearch": 8,"vector database": 10,"vector search": 9,
    "information retrieval": 10,"semantic search": 10,"hybrid search": 10,
    "retrieval": 9,"ranking": 9,"re-ranking": 9,"bm25": 8,
    "ndcg": 9,"mrr": 8,"evaluation framework": 9,"a/b test": 7,
    "learning to rank": 8,"ltr": 7,"lambdamart": 7,"python": 6,
}
JD_PREFERRED_SKILLS = {
    "llm": 5,"rag": 5,"fine-tuning": 5,"lora": 5,"qlora": 5,"peft": 5,
    "xgboost": 6,"lightgbm": 6,"recommendation": 5,"nlp": 4,
    "pytorch": 4,"transformers": 4,"huggingface": 4,"bert": 4,
    "hr-tech": 4,"recruiting": 4,
}
JD_NEGATIVE_SKILLS = {
    "langchain": -4,"computer vision": -3,"opencv": -3,"yolo": -2,
    "speech recognition": -3,"robotics": -4,"solidworks": -3,
}

INTERVIEW_QUESTIONS = {
    "ndcg": "Walk me through how you have measured ranking quality in production. What does your NDCG measurement pipeline look like?",
    "information retrieval": "Describe a retrieval system you shipped end-to-end. What was the hardest scaling challenge?",
    "semantic search": "How would you approach building a hybrid search system that combines dense and sparse retrieval?",
    "embeddings": "How do you handle embedding drift in production? What monitoring have you put in place?",
    "faiss": "When would you choose FAISS over a managed vector database like Pinecone? What are the tradeoffs?",
    "pinecone": "Describe your experience managing a vector database at scale. How did you handle index updates?",
    "weaviate": "What factors did you consider when selecting a vector database for your use case?",
    "learning to rank": "Walk me through a learning-to-rank model you have trained. What features did you use?",
    "hybrid search": "How do you combine BM25 and dense retrieval scores? What fusion strategy did you use?",
    "a/b test": "Describe how you have run an A/B test on a search or ranking system. What was your primary metric?",
    "evaluation framework": "How do you build an offline evaluation framework for retrieval? How do you collect relevance labels?",
    "lambdamart": "Why did you choose LambdaMART over a neural ranking approach? What were the latency/quality tradeoffs?",
    "lora": "How have you used LoRA/QLoRA for fine-tuning? What was the task and what did you measure?",
    "rag": "Describe the retrieval component of a RAG system you have built. How did you handle retrieval failures?",
    "recommendation": "How did you evaluate your recommendation system? What offline and online metrics did you use?",
    "python": "Show me a piece of Python code you are proud of from a production ML system.",
}

BIAS_FLAGS = {
    "ninja":          "Masculine-coded language — statistically discourages women from applying",
    "rockstar":       "Masculine-coded language — replace with specific skill requirements",
    "superhero":      "Masculine-coded language — replace with specific skill requirements",
    "dominant":       "Aggressive language that skews male in research studies",
    "aggressive":     "Aggressive language that skews male in research studies",
    "young":          "Age-coded language — potentially discriminatory in many jurisdictions",
    "digital native": "Age-coded language — implies preference for younger candidates",
    "energetic":      "Can be used as a proxy for age — older candidates self-select out",
    "fast-paced":     "Can discourage candidates with caregiving responsibilities",
}

def check_jd_bias(jd_text):
    found = []
    jd_lower = jd_text.lower()
    for term, reason in BIAS_FLAGS.items():
        if term in jd_lower:
            found.append((term, reason))
    return found

def build_gap_analysis(candidate, jd_required):
    skills = candidate.get("skills", [])
    skill_names_lower = {s["name"].lower() for s in skills}
    profile_text = (
        candidate["profile"].get("headline", "") + " " +
        candidate["profile"].get("summary", "") + " " +
        " ".join(j.get("description", "") for j in candidate.get("career_history", []))
    ).lower()
    present, missing = [], []
    for token, weight in sorted(jd_required.items(), key=lambda x: -x[1])[:12]:
        if any(token in s for s in skill_names_lower) or token in profile_text:
            present.append(token)
        else:
            missing.append(token)
    questions = []
    for gap in missing[:3]:
        if gap in INTERVIEW_QUESTIONS:
            questions.append((gap, INTERVIEW_QUESTIONS[gap]))
    return present, missing, questions

def load_candidates(uploaded_file):
    """
    Stream-reads candidates line by line to avoid RAM overflow.
    Works for both .json (sample) and .jsonl (full 100K dataset).
    Never holds more data in memory than needed.
    """
    filename = uploaded_file.name
    content = uploaded_file.read()

    if filename.endswith(".json"):
        # Small sample file — load normally
        candidates = json.loads(content)
        return candidates, len(candidates)

    # Large JSONL — stream line by line
    candidates = []
    errors = 0
    lines = content.decode("utf-8", errors="ignore").splitlines()
    total = len(lines)

    progress = st.progress(0, text="Reading candidates file...")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError:
            errors += 1
            continue
        # Update progress bar every 5000 lines
        if (i + 1) % 5000 == 0:
            pct = min(int((i + 1) / total * 100), 99)
            progress.progress(pct, text=f"Reading... {len(candidates):,} candidates loaded")

    progress.progress(100, text=f"✅ Loaded {len(candidates):,} candidates")

    if errors > 0:
        st.warning(f"{errors} lines could not be parsed and were skipped.")

    return candidates, len(candidates)

def rank_candidates_streamlit(candidates, top_n=20):
    """Full ranking logic — same as rank.py, runs on all candidates passed in."""
    from datetime import date, datetime

    REFERENCE_DATE = date(2026, 6, 18)
    CONSULTING = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "tech mahindra", "hcl", "mphasis", "hexaware", "mindtree"
    }
    PRODUCT_COS = {
        "swiggy", "zomato", "razorpay", "paytm", "phonepe", "flipkart",
        "meesho", "ola", "uber", "cred", "groww", "freshworks", "browserstack",
        "sarvam", "krutrim", "google", "microsoft", "amazon", "meta", "netflix",
        "apple", "openai", "anthropic", "cohere", "databricks",
        "mad street den", "observe.ai", "verloop", "yellow.ai"
    }

    scored = []
    progress = st.progress(0, text="Scoring candidates...")
    total = len(candidates)

    for idx, c in enumerate(candidates):
        profile = c["profile"]
        signals = c["redrob_signals"]
        yoe = profile["years_of_experience"]

        # Hard filter
        if yoe < 3:
            continue

        skill_names = {s["name"].lower() for s in c.get("skills", [])}
        profile_text = (
            profile.get("headline", "") + " " +
            profile.get("summary", "") + " " +
            " ".join(j.get("description", "") for j in c.get("career_history", []))
        ).lower()

        # Skills score
        req_score, req_max = 0.0, 0.0
        for token, w in JD_REQUIRED_SKILLS.items():
            req_max += w
            if any(token in s for s in skill_names):
                req_score += w * 0.9
            elif token in profile_text:
                req_score += w * 0.4
        pref_score, pref_max = 0.0, 0.0
        for token, w in JD_PREFERRED_SKILLS.items():
            pref_max += w
            if any(token in s for s in skill_names):
                pref_score += w * 0.8
            elif token in profile_text:
                pref_score += w * 0.3
        neg = sum(p for t, p in JD_NEGATIVE_SKILLS.items()
                  if any(t in s for s in skill_names) or t in profile_text)
        skills_score = min(1.0, max(0.0,
            (0.7 * (req_score / max(req_max * 0.5, 1)) +
             0.3 * (pref_score / max(pref_max * 0.4, 1))) *
            max(0.3, 1 + neg * 0.05)
        ))

        # Career score
        narrative_hits = sum(1 for p in [
            "embedding", "ranking", "retrieval", "semantic search", "ndcg",
            "evaluation", "recommendation", "vector", "learning to rank", "hybrid"
        ] if p in profile_text)
        career_score = min(1.0, narrative_hits / 5)
        if 5 <= yoe <= 9:
            career_score = min(1.0, career_score * 1.1)
        all_cos = [j["company"].lower() for j in c.get("career_history", [])]
        if any(p in co for co in all_cos for p in PRODUCT_COS):
            career_score = min(1.0, career_score * 1.15)
        if all(any(cc in co for cc in CONSULTING) for co in all_cos):
            career_score *= 0.4

        # Behavioral score
        try:
            la = datetime.strptime(
                signals.get("last_active_date", "2020-01-01"), "%Y-%m-%d"
            ).date()
            days_ago = (REFERENCE_DATE - la).days
        except:
            days_ago = 365
        recency = 1.0 if days_ago <= 14 else 0.85 if days_ago <= 60 else 0.65 if days_ago <= 90 else 0.3
        rr = signals.get("recruiter_response_rate", 0.5)
        resp = 1.0 if rr >= 0.7 else 0.85 if rr >= 0.5 else 0.65 if rr >= 0.3 else 0.3
        gh = signals.get("github_activity_score", -1)
        gh_s = min(1.0, gh / 70) if gh >= 0 else 0.4
        otw = 1.0 if signals.get("open_to_work_flag") else 0.65
        beh = (0.3 * recency + 0.25 * resp + 0.2 * otw +
               0.15 * gh_s + 0.1 * (signals.get("profile_completeness_score", 70) / 100))

        base = (0.4 * career_score + 0.35 * skills_score +
                0.15 * beh + 0.1 * (yoe / 9 if yoe <= 9 else 0.7))
        composite = base * (0.4 + 0.6 * beh)

        scored.append({
            "candidate": c,
            "composite": composite,
            "skills": skills_score,
            "career": career_score,
            "behavioral": beh
        })

        # Update progress every 5000 candidates
        if (idx + 1) % 5000 == 0:
            pct = min(int((idx + 1) / total * 100), 99)
            progress.progress(pct, text=f"Scoring... {idx+1:,} / {total:,} candidates")

    progress.progress(100, text=f"✅ Scored {len(scored):,} candidates")
    scored.sort(key=lambda x: -x["composite"])
    return scored[:top_n]

# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.tag-match{background:#E1F5EE;color:#085041;padding:2px 8px;border-radius:10px;font-size:12px;margin:2px;display:inline-block}
.tag-gap{background:#FCEBEB;color:#791F1F;padding:2px 8px;border-radius:10px;font-size:12px;margin:2px;display:inline-block}
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("### 🧠")
with col_title:
    st.markdown("## NeuroRecruit — Intelligent Candidate Discovery")
    st.caption("Redrob Hackathon Demo · Semantic ranking · Behavioral signals · Career intelligence")

st.divider()

col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown("### Job description")
    jd_text = st.text_area(
        "Paste or type JD here",
        value="Senior AI/ML Engineer with expertise in retrieval systems, vector databases (Pinecone/Weaviate/FAISS), semantic search, learning to rank, and evaluation frameworks (NDCG/MRR). Must have Python and 5+ years in production ML.",
        height=140,
        label_visibility="collapsed"
    )

    # Bias check runs live as JD is typed
    bias_issues = check_jd_bias(jd_text)
    if bias_issues:
        st.warning(f"⚠️ Bias check: {len(bias_issues)} flag(s) found in JD")
        for term, reason in bias_issues:
            st.markdown(f"- **'{term}'** — {reason}")
    else:
        st.success("✅ Bias check passed — no flagged language detected")

    st.markdown("### Upload candidates")
    st.caption("Supports sample_candidates.json (50 candidates) or the full candidates.jsonl (100K)")
    uploaded = st.file_uploader(
        "Upload file",
        type=["jsonl", "json"],
        label_visibility="collapsed"
    )
    top_n = st.slider("Candidates to show in results", 5, 50, 20)
    run_btn = st.button("Rank candidates", type="primary", use_container_width=True)

with col_right:
    if run_btn and uploaded:
        t_start = time.time()

        # ── Load candidates ──────────────────────────────────────────────────
        st.markdown("**Step 1 — Loading candidates**")
        candidates, total_loaded = load_candidates(uploaded)

        if not candidates:
            st.error("No candidates found in the file. Check the file format.")
            st.stop()

        # ── Rank candidates ──────────────────────────────────────────────────
        st.markdown("**Step 2 — Ranking candidates**")
        results = rank_candidates_streamlit(candidates, top_n=top_n)

        t_elapsed = time.time() - t_start

        # ── Summary metrics ──────────────────────────────────────────────────
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates loaded", f"{total_loaded:,}")
        m2.metric("Shortlisted", len(results))
        m3.metric("Top score", f"{results[0]['composite']:.3f}" if results else "—")
        m4.metric("Total time", f"{t_elapsed:.1f}s")

        # ── Results ──────────────────────────────────────────────────────────
        st.markdown("### Ranked shortlist")
        for i, item in enumerate(results):
            c = item["candidate"]
            p = c["profile"]
            sig = c["redrob_signals"]
            composite = item["composite"]

            with st.expander(
                f"#{i+1}  {p['current_title']} @ {p['current_company']}  "
                f"|  Score: {composite:.3f}  |  {p['location']}",
                expanded=(i < 3)
            ):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Overall fit", f"{composite*100:.0f}%")
                col_b.metric("Skills", f"{item['skills']*100:.0f}%")
                col_c.metric("Career", f"{item['career']*100:.0f}%")
                col_d.metric("Availability", f"{item['behavioral']*100:.0f}%")

                rr      = sig.get("recruiter_response_rate", 0)
                gh      = sig.get("github_activity_score", -1)
                notice  = sig.get("notice_period_days", 90)
                otw     = sig.get("open_to_work_flag", False)
                last_active = sig.get("last_active_date", "unknown")
                skills_list = [s["name"] for s in c.get("skills", [])[:10]]

                col_info, col_signals = st.columns(2)
                with col_info:
                    st.markdown(f"**YOE:** {p['years_of_experience']} years")
                    st.markdown(f"**Skills:** {', '.join(skills_list[:6])}")
                    edu = c.get("education", [])
                    if edu:
                        st.markdown(f"**Education:** {edu[0].get('degree','')} — {edu[0].get('field_of_study','')}")
                with col_signals:
                    st.markdown(f"**Response rate:** {rr:.0%} {'✅' if rr >= 0.5 else '⚠️'}")
                    st.markdown(f"**GitHub:** {gh if gh >= 0 else 'not linked'} {'✅' if gh >= 40 else '—'}")
                    st.markdown(f"**Notice:** {notice}d {'✅' if notice <= 30 else '⚠️' if notice <= 60 else '🔴'}")
                    st.markdown(f"**Open to work:** {'✅' if otw else '—'} | Last active: {last_active}")

                present, missing, questions = build_gap_analysis(c, JD_REQUIRED_SKILLS)
                st.markdown("**JD skill alignment:**")
                tag_html = ""
                for s in present[:6]:
                    tag_html += f'<span class="tag-match">✓ {s}</span> '
                for s in missing[:4]:
                    tag_html += f'<span class="tag-gap">✗ {s}</span> '
                st.markdown(tag_html, unsafe_allow_html=True)

                if questions:
                    st.markdown("**Suggested interview questions for skill gaps:**")
                    for gap, q in questions:
                        st.markdown(f"- **{gap}:** _{q}_")

                career_snippets = [
                    (j["title"], j["company"], j.get("duration_months", 0))
                    for j in c.get("career_history", [])[:3]
                ]
                st.markdown(
                    "**Career:** " +
                    " → ".join(f"{t} @ {co} ({d}mo)" for t, co, d in career_snippets)
                )

        # ── Download button ──────────────────────────────────────────────────
        if results:
            rows = []
            for i, item in enumerate(results):
                c   = item["candidate"]
                p   = c["profile"]
                sig = c["redrob_signals"]
                matching = [
                    s["name"] for s in c.get("skills", [])
                    if any(t in s["name"].lower() for t in JD_REQUIRED_SKILLS)
                ][:3]
                reasoning = (
                    f"{p['years_of_experience']:.0f}-year {p['current_title']} at {p['current_company']}; "
                    f"skills: {', '.join(matching) if matching else 'ML/AI background'}. "
                    f"Response rate {sig.get('recruiter_response_rate', 0):.0%}, "
                    f"{sig.get('notice_period_days', 90)}d notice."
                )
                rows.append({
                    "candidate_id": c["candidate_id"],
                    "rank": i + 1,
                    "score": round(item["composite"], 6),
                    "reasoning": reasoning
                })
            df = pd.DataFrame(rows)
            st.download_button(
                "⬇️ Download ranked CSV",
                df.to_csv(index=False).encode(),
                f"neurorecruit_top{len(results)}.csv",
                "text/csv",
                use_container_width=True
            )

    elif run_btn and not uploaded:
        st.warning("Please upload a candidates file first.")

    else:
        st.markdown("### How it works")
        st.markdown("""
**5-layer intelligent ranking engine:**

1. **Deep JD understanding** — reads required/preferred/negative signals from the job description
2. **Semantic career fit** — searches role descriptions for production evidence (not just skill keywords)
3. **Behavioral signals** — 8 engagement signals weighted and applied as a composite multiplier
4. **Career trajectory** — product company experience, YOE sweet spot, ML narrative density
5. **Gap analysis** — per-candidate skill delta with suggested interview questions

Supports both `sample_candidates.json` (50 candidates, instant) and `candidates.jsonl` (100K, ~60s).
        """)
        st.info("💡 The full 100K ranking runs locally via: `python rank.py --candidates candidates.jsonl --out submission.csv`")
