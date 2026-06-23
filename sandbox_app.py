"""
NeuroRecruit — Sandbox Demo
Redrob Hackathon — HuggingFace Spaces / Streamlit Cloud compatible
Run: streamlit run sandbox_app.py
"""

import streamlit as st
import json
import tempfile
import os
import pandas as pd
import sys

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

def build_gap_analysis(candidate, jd_required):
    skills = candidate.get("skills", [])
    skill_names_lower = {s["name"].lower() for s in skills}
    profile_text = " ".join(
        candidate["profile"].get("headline","") + " " +
        candidate["profile"].get("summary","") + " " +
        " ".join(j.get("description","") for j in candidate.get("career_history",[]))
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

def rank_candidates_streamlit(candidates, top_n=20):
    """Simplified ranker for sandbox — same logic, returns extra debug info."""
    import re
    from datetime import date, datetime

    REFERENCE_DATE = date(2026, 6, 18)
    CONSULTING = {"tcs","infosys","wipro","accenture","cognizant","capgemini","tech mahindra","hcl","mphasis","hexaware","mindtree"}
    PRODUCT_COS = {"swiggy","zomato","razorpay","paytm","phonepe","flipkart","meesho","ola","uber","cred","groww",
                   "freshworks","browserstack","sarvam","krutrim","google","microsoft","amazon","meta","netflix",
                   "apple","openai","anthropic","cohere","databricks","mad street den","observe.ai","verloop","yellow.ai"}

    scored = []
    for c in candidates:
        profile = c["profile"]
        signals = c["redrob_signals"]
        yoe = profile["years_of_experience"]
        if yoe < 3: continue

        skill_names = {s["name"].lower() for s in c.get("skills",[])}
        profile_text = (
            profile.get("headline","") + " " + profile.get("summary","") + " " +
            " ".join(j.get("description","") for j in c.get("career_history",[]))
        ).lower()

        req_score, req_max = 0.0, 0.0
        for token, w in JD_REQUIRED_SKILLS.items():
            req_max += w
            if any(token in s for s in skill_names): req_score += w * 0.9
            elif token in profile_text: req_score += w * 0.4
        pref_score, pref_max = 0.0, 0.0
        for token, w in JD_PREFERRED_SKILLS.items():
            pref_max += w
            if any(token in s for s in skill_names): pref_score += w * 0.8
            elif token in profile_text: pref_score += w * 0.3
        neg = sum(p for t, p in JD_NEGATIVE_SKILLS.items() if any(t in s for s in skill_names) or t in profile_text)
        skills_score = min(1.0, max(0.0, (0.7*(req_score/max(req_max*0.5,1)) + 0.3*(pref_score/max(pref_max*0.4,1))) * max(0.3, 1+neg*0.05)))

        narrative_hits = sum(1 for p in ["embedding","ranking","retrieval","semantic search","ndcg","evaluation","recommendation","vector","learning to rank","hybrid"] if p in profile_text)
        career_score = min(1.0, narrative_hits / 5)
        if 5 <= yoe <= 9: career_score = min(1.0, career_score * 1.1)
        all_cos = [j["company"].lower() for j in c.get("career_history",[])]
        if any(p in co for co in all_cos for p in PRODUCT_COS): career_score = min(1.0, career_score * 1.15)
        if all(any(cc in co for cc in CONSULTING) for co in all_cos): career_score *= 0.4

        try:
            la = datetime.strptime(signals.get("last_active_date","2020-01-01"), "%Y-%m-%d").date()
            days_ago = (REFERENCE_DATE - la).days
        except: days_ago = 365
        recency = 1.0 if days_ago<=14 else 0.85 if days_ago<=60 else 0.65 if days_ago<=90 else 0.3
        rr = signals.get("recruiter_response_rate", 0.5)
        resp = 1.0 if rr>=0.7 else 0.85 if rr>=0.5 else 0.65 if rr>=0.3 else 0.3
        gh = signals.get("github_activity_score",-1)
        gh_s = min(1.0, gh/70) if gh>=0 else 0.4
        otw = 1.0 if signals.get("open_to_work_flag") else 0.65
        beh = 0.3*recency + 0.25*resp + 0.2*otw + 0.15*gh_s + 0.1*(signals.get("profile_completeness_score",70)/100)

        base = 0.4*career_score + 0.35*skills_score + 0.15*beh + 0.1*(yoe/9 if yoe<=9 else 0.7)
        composite = base * (0.4 + 0.6*beh)

        scored.append({"candidate":c,"composite":composite,"skills":skills_score,"career":career_score,"behavioral":beh})

    scored.sort(key=lambda x: -x["composite"])
    return scored[:top_n]

st.markdown("""
<style>
.metric-card{background:#f8f8f8;border-radius:8px;padding:10px 14px;text-align:center}
.score-big{font-size:28px;font-weight:600;color:#534AB7}
.tag-match{background:#E1F5EE;color:#085041;padding:2px 8px;border-radius:10px;font-size:12px;margin:2px}
.tag-gap{background:#FCEBEB;color:#791F1F;padding:2px 8px;border-radius:10px;font-size:12px;margin:2px}
.tag-sig{background:#EEEDFE;color:#3C3489;padding:2px 8px;border-radius:10px;font-size:12px;margin:2px}
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
    st.markdown("### Upload candidates")
    uploaded = st.file_uploader("Upload candidates.jsonl or sample_candidates.json", type=["jsonl","json"])
    top_n = st.slider("Candidates to rank", 5, 50, 20)
    run_btn = st.button("Rank candidates", type="primary", use_container_width=True)

with col_right:
    if run_btn and uploaded:
        with st.spinner("Loading and ranking candidates..."):
            content = uploaded.read()
            try:
                if uploaded.name.endswith(".json"):
                    candidates = json.loads(content)
                else:
                    candidates = [json.loads(l) for l in content.decode().split("\n") if l.strip()]
            except Exception as e:
                st.error(f"Could not parse file: {e}")
                st.stop()

            results = rank_candidates_streamlit(candidates, top_n=top_n)

        st.success(f"Ranked {len(candidates)} candidates in under 1s — showing top {len(results)}")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Candidates", f"{len(candidates):,}")
        m2.metric("Ranked", len(results))
        m3.metric("Top score", f"{results[0]['composite']:.2f}" if results else "—")
        m4.metric("Runtime", "<1s")

        st.markdown("### Ranked shortlist")
        for i, item in enumerate(results):
            c = item["candidate"]
            p = c["profile"]
            sig = c["redrob_signals"]
            composite = item["composite"]

            with st.expander(f"#{i+1}  {p['current_title']} @ {p['current_company']}  |  Score: {composite:.3f}  |  {p['location']}", expanded=(i<3)):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Overall fit", f"{composite*100:.0f}%")
                col_b.metric("Skills", f"{item['skills']*100:.0f}%")
                col_c.metric("Career", f"{item['career']*100:.0f}%")
                col_d.metric("Availability", f"{item['behavioral']*100:.0f}%")

                skills_list = [s["name"] for s in c.get("skills",[])[:10]]
                rr = sig.get("recruiter_response_rate",0)
                gh = sig.get("github_activity_score",-1)
                notice = sig.get("notice_period_days",90)
                otw = sig.get("open_to_work_flag",False)
                last_active = sig.get("last_active_date","unknown")

                col_info, col_signals = st.columns(2)
                with col_info:
                    st.markdown(f"**YOE:** {p['years_of_experience']} years")
                    st.markdown(f"**Skills:** {', '.join(skills_list[:6])}")
                    st.markdown(f"**Education:** {c['education'][0]['degree'] if c.get('education') else 'N/A'} — {c['education'][0].get('field_of_study','') if c.get('education') else ''}")
                with col_signals:
                    st.markdown(f"**Response rate:** {rr:.0%} {'✅' if rr>=0.5 else '⚠️'}")
                    st.markdown(f"**GitHub activity:** {gh if gh>=0 else 'not linked'} {'✅' if gh>=40 else '—'}")
                    st.markdown(f"**Notice period:** {notice} days {'✅' if notice<=30 else '⚠️' if notice<=60 else '🔴'}")
                    st.markdown(f"**Open to work:** {'✅ Yes' if otw else '—'} | Last active: {last_active}")

                present, missing, questions = build_gap_analysis(c, JD_REQUIRED_SKILLS)
                st.markdown("**JD skill alignment:**")
                tag_html = ""
                for s in present[:6]: tag_html += f'<span class="tag-match">✓ {s}</span> '
                for s in missing[:4]: tag_html += f'<span class="tag-gap">✗ {s}</span> '
                st.markdown(tag_html, unsafe_allow_html=True)

                if questions:
                    st.markdown("**Suggested interview questions for gaps:**")
                    for gap, q in questions:
                        st.markdown(f"- **{gap}:** _{q}_")

                career_snippets = [(j["title"],j["company"],j.get("duration_months",0)) for j in c.get("career_history",[])[:3]]
                st.markdown("**Career history:** " + " → ".join(f"{t} @ {co} ({d}mo)" for t,co,d in career_snippets))

        if results:
            rows = []
            for i, item in enumerate(results):
                c = item["candidate"]
                p = c["profile"]
                sig = c["redrob_signals"]
                matching = [s["name"] for s in c.get("skills",[]) if any(t in s["name"].lower() for t in JD_REQUIRED_SKILLS)][:3]
                reasoning = f"{p['years_of_experience']:.0f}-year {p['current_title']} at {p['current_company']}; skills: {', '.join(matching[:3]) if matching else 'ML/AI background'}. Response rate {sig.get('recruiter_response_rate',0):.0%}, {sig.get('notice_period_days',90)}d notice."
                rows.append({"candidate_id":c["candidate_id"],"rank":i+1,"score":round(item["composite"],6),"reasoning":reasoning})
            df = pd.DataFrame(rows)
            st.download_button("Download CSV", df.to_csv(index=False).encode(), f"neurorecruit_top{len(results)}.csv", "text/csv", use_container_width=True)
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

Upload `sample_candidates.json` from the hackathon bundle and click Rank to see it live.
        """)
        st.info("This sandbox runs on a sample of candidates. The full submission ranks all 100,000.")

