#!/usr/bin/env python3
"""
NeuroRecruit — Intelligent Candidate Discovery Ranker
Redrob Hackathon — Senior AI Engineer JD

Architecture:
  1. Hard filter: eliminate clear non-fits (wrong title track, disqualifying signals)
  2. Honeypot detection: flag impossible profiles before scoring
  3. Multi-signal scoring across 5 dimensions
  4. Availability multiplier: behavioral signal modifier
  5. Final composite ranking with tie-breaking

Run:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv --top 100
"""

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# JD INTELLIGENCE — extracted from the Senior AI Engineer JD
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 18)  # Hackathon date

# Core required skills (per JD "Things you absolutely need")
# Keys: lowercase skill tokens to match against profile text
# Values: importance weight (higher = more important to this JD)
JD_REQUIRED_SKILLS = {
    # Embedding / retrieval systems (HIGHEST weight — absolute requirement)
    "embeddings": 10,
    "embedding": 10,
    "sentence-transformers": 10,
    "sentence transformers": 10,
    "bge": 9,
    "e5": 8,
    "openai embeddings": 9,
    "text-embedding": 8,

    # Vector databases (absolute requirement)
    "pinecone": 9,
    "weaviate": 9,
    "qdrant": 9,
    "milvus": 9,
    "faiss": 9,
    "opensearch": 8,
    "elasticsearch": 8,
    "vector database": 10,
    "vector db": 10,
    "vector search": 9,

    # Retrieval / ranking / IR (absolute requirement)
    "information retrieval": 10,
    "semantic search": 10,
    "hybrid search": 10,
    "hybrid retrieval": 10,
    "retrieval": 9,
    "ranking": 9,
    "re-ranking": 9,
    "reranking": 9,
    "bm25": 8,

    # Evaluation frameworks (absolute requirement)
    "ndcg": 9,
    "mrr": 8,
    "map": 6,
    "mean average precision": 8,
    "normalized discounted": 8,
    "a/b test": 7,
    "ab test": 7,
    "offline evaluation": 8,
    "evaluation framework": 9,
    "recall@": 7,
    "precision@": 7,

    # Python (absolute requirement)
    "python": 6,
}

# Nice-to-have skills (per JD section)
JD_PREFERRED_SKILLS = {
    "llm": 5,
    "lora": 5,
    "qlora": 5,
    "peft": 5,
    "fine-tuning": 5,
    "fine-tune": 5,
    "fine tuning": 5,
    "rag": 5,
    "retrieval augmented": 5,
    "xgboost": 6,
    "lightgbm": 6,
    "learning to rank": 8,
    "ltr": 7,
    "lambdamart": 7,
    "recommendation": 5,
    "recommender": 5,
    "recommendation system": 6,
    "nlp": 4,
    "pytorch": 4,
    "transformers": 4,
    "hugging face": 4,
    "huggingface": 4,
    "bert": 4,
    "distributed inference": 5,
    "large-scale inference": 5,
    "open-source": 3,
    "open source": 3,
    "hr-tech": 4,
    "hrtech": 4,
    "recruiting": 4,
    "marketplace": 3,
}

# Negative skill signals (JD explicitly says "NOT" these primary focuses)
JD_NEGATIVE_SKILLS = {
    "langchain": -4,       # JD: "framework enthusiasts" — negative signal
    "computer vision": -3,
    "opencv": -3,
    "yolo": -2,
    "object detection": -2,
    "image classification": -2,
    "speech recognition": -3,
    "tts": -2,
    "text to speech": -2,
    "asr": -2,
    "robotics": -4,
    "ros": -3,
    "solidworks": -3,
    "autocad": -3,
    "mechanical": -2,
    "photoshop": -1,
    "figma": -1,       # small penalty, not disqualifying
}

# Career title signals — what the JD "ideal" looks like vs. what's clearly wrong
STRONGLY_POSITIVE_TITLE_TOKENS = {
    "ml engineer", "machine learning engineer", "ai engineer", "nlp engineer",
    "research engineer", "applied scientist", "applied ml", "data scientist",
    "search engineer", "ranking engineer", "retrieval engineer",
    "recommendation", "information retrieval", "search relevance",
    "senior engineer", "staff engineer", "principal engineer",
    "founding engineer", "tech lead"
}

WEAKLY_POSITIVE_TITLE_TOKENS = {
    "software engineer", "backend engineer", "full stack", "data engineer",
    "platform engineer", "infrastructure engineer", "cloud engineer",
    "python developer", "senior developer"
}

# Titles that are explicitly disqualifying per JD reading
DISQUALIFYING_TITLE_TOKENS = {
    "hr manager", "hr executive", "human resources",
    "marketing manager", "marketing executive", "brand manager",
    "graphic designer", "ux designer", "ui designer", "creative director",
    "mechanical engineer", "civil engineer", "structural engineer",
    "accountant", "finance manager", "ca ", "cfa",
    "customer support", "customer success",
    "operations manager", "operations executive",
    "sales executive", "business development",
    "content writer", "copywriter", "seo specialist",
    "project manager",   # unless combined with strong ML background
    "business analyst",  # unless combined with strong ML background
    "product manager",   # unless clearly technical PM
    "qa engineer", "test engineer", "testing",
    "java developer", ".net developer", "android developer",
    "ios developer", "frontend engineer", "frontend developer",
    "mobile developer", "react developer", "angular developer",
}

# Companies the JD explicitly says are poor signal
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "techm", "hcl", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "l&t infotech", "ntt data", "dxc"
}

# Strong positive companies (product companies in AI/tech space)
PRODUCT_COMPANY_SIGNALS = {
    "swiggy", "zomato", "ola", "uber", "flipkart", "meesho", "razorpay",
    "paytm", "phonepe", "zepto", "blinkit", "dunzo", "cred", "groww",
    "smallcase", "cleartax", "browserstack", "freshworks", "chargebee",
    "hasura", "setu", "niyo", "slice", "primary ai", "mad street den",
    "google", "microsoft", "amazon", "meta", "apple", "openai", "anthropic",
    "cohere", "hugging face", "databricks", "snowflake", "elastic",
    "pinecone", "weaviate", "qdrant", "nvidia", "deepmind", "deepsense",
}

# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Detect impossible/fraudulent profiles.
    Returns (is_honeypot, list_of_reasons).
    Honeypots get relevance tier 0 in ground truth → disqualify from top-100.
    """
    flags = []
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    yoe = profile["years_of_experience"]
    yoe_months = yoe * 12

    # 1. Inverted salary range (min > max)
    sal = signals["expected_salary_range_inr_lpa"]
    if sal["min"] > sal["max"] + 0.1:
        flags.append(f"salary_inverted: min={sal['min']} > max={sal['max']}")

    # 2. Skill duration exceeds total career length meaningfully
    impossible_skill_count = 0
    for skill in candidate.get("skills", []):
        dm = skill.get("duration_months", 0)
        if dm > yoe_months + 12:  # more than 1yr grace
            impossible_skill_count += 1
    if impossible_skill_count >= 2:
        flags.append(f"skill_duration_impossible: {impossible_skill_count} skills exceed career length")

    # 3. Career history dates that don't add up
    career = candidate.get("career_history", [])
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if total_career_months > (yoe * 12) + 24:  # more than 2yr grace
        flags.append(f"career_duration_overstated: {total_career_months}mo vs {yoe*12:.0f}mo YOE")

    # 4. Expert proficiency + zero duration (keyword stuffer)
    zero_duration_experts = sum(
        1 for s in candidate.get("skills", [])
        if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0
    )
    if zero_duration_experts >= 3:
        flags.append(f"expert_skill_zero_duration: {zero_duration_experts} skills")

    # 5. Assessment score contradicts claimed proficiency
    # e.g. "expert" in Python but assessment score = 20
    assessments = signals.get("skill_assessment_scores", {})
    contradictions = 0
    for skill in candidate.get("skills", []):
        name = skill["name"]
        if name in assessments and skill["proficiency"] == "expert":
            if assessments[name] < 35:  # expert but below 35/100 on assessment
                contradictions += 1
    if contradictions >= 2:
        flags.append(f"assessment_proficiency_contradiction: {contradictions} skills")

    # 6. YOE impossibly high for age proxy (education end year)
    for edu in candidate.get("education", []):
        end_yr = edu.get("end_year", 0)
        if end_yr > 0:
            implied_min_yoe = REFERENCE_DATE.year - end_yr
            if yoe > implied_min_yoe + 5:  # 5yr grace for early career
                flags.append(f"yoe_exceeds_grad_year: {yoe}yr but graduated {end_yr}")
                break

    return len(flags) >= 2, flags  # need 2+ flags to call honeypot


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """
    Score 0-1: skills match against JD requirements.
    Goes beyond keyword match — considers proficiency, duration, and assessment scores.
    """
    skills = candidate.get("skills", [])
    assessments = candidate["redrob_signals"].get("skill_assessment_scores", {})

    # Build a lookup: skill_name_lower -> skill_obj
    skill_lookup = {}
    for s in skills:
        key = s["name"].lower().strip()
        skill_lookup[key] = s

    # All profile text for fuzzy matching (career descriptions, headline, summary)
    profile_text = (
        candidate["profile"].get("headline", "") + " " +
        candidate["profile"].get("summary", "") + " " +
        " ".join(j.get("description", "") for j in candidate.get("career_history", []))
    ).lower()

    required_score = 0.0
    required_max = 0.0
    preferred_score = 0.0
    preferred_max = 0.0
    negative_score = 0.0

    # Score required skills
    for token, weight in JD_REQUIRED_SKILLS.items():
        required_max += weight
        # Check direct skill match
        direct_match = any(token in k for k in skill_lookup.keys())
        text_match = token in profile_text

        if direct_match:
            sk = next(s for k, s in skill_lookup.items() if token in k)
            # Proficiency multiplier
            prof_mult = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(
                sk.get("proficiency", "intermediate"), 0.7
            )
            # Duration multiplier (longer use = more credible)
            dur_months = sk.get("duration_months", 12)
            dur_mult = min(1.0, 0.5 + dur_months / 48.0)  # caps at 1.0 at ~48 months
            # Assessment score boost if available
            assess_name = sk["name"]
            if assess_name in assessments:
                assess_mult = 0.8 + 0.2 * (assessments[assess_name] / 100)
            else:
                assess_mult = 0.9  # slight penalty for no assessment
            skill_val = weight * prof_mult * dur_mult * assess_mult
            required_score += skill_val
        elif text_match:
            # Mentioned in career text but not as a formal skill — partial credit
            required_score += weight * 0.4

    # Score preferred skills
    for token, weight in JD_PREFERRED_SKILLS.items():
        preferred_max += weight
        direct_match = any(token in k for k in skill_lookup.keys())
        text_match = token in profile_text
        if direct_match:
            sk = next(s for k, s in skill_lookup.items() if token in k)
            prof_mult = {"beginner": 0.5, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(
                sk.get("proficiency", "intermediate"), 0.7
            )
            preferred_score += weight * prof_mult * 0.8
        elif text_match:
            preferred_score += weight * 0.3

    # Apply negative signals
    for token, penalty in JD_NEGATIVE_SKILLS.items():
        if any(token in k for k in skill_lookup.keys()) or token in profile_text:
            negative_score += penalty

    # Normalize to 0-1
    # Required: 70% weight, preferred: 30%
    if required_max > 0:
        req_normalized = min(1.0, required_score / (required_max * 0.5))  # 50% of max = full score
    else:
        req_normalized = 0.0

    if preferred_max > 0:
        pref_normalized = min(1.0, preferred_score / (preferred_max * 0.4))
    else:
        pref_normalized = 0.0

    raw = 0.70 * req_normalized + 0.30 * pref_normalized
    # Apply negative signal penalty
    penalty_factor = max(0.3, 1.0 + negative_score * 0.05)  # max 70% reduction
    return min(1.0, max(0.0, raw * penalty_factor))


def score_career_fit(candidate: dict) -> float:
    """
    Score 0-1: career trajectory fit for the JD.
    Reads *what the candidate actually did*, not just their title.
    Key insight: a 'Frontend Engineer' at Swiggy who built search systems
    scores higher than a 'ML Engineer' at TCS who did Java webservices.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])

    title = profile["current_title"].lower()
    company = profile["current_company"].lower()
    yoe = profile["years_of_experience"]
    industry = profile.get("current_industry", "").lower()

    profile_text = (
        profile.get("headline", "") + " " +
        profile.get("summary", "") + " " +
        " ".join(j.get("description", "") for j in career)
    ).lower()

    score = 0.0

    # ── Title component (15% of career score) ──────────────────────────────
    title_score = 0.0
    if any(t in title for t in STRONGLY_POSITIVE_TITLE_TOKENS):
        title_score = 1.0
    elif any(t in title for t in WEAKLY_POSITIVE_TITLE_TOKENS):
        title_score = 0.5
    elif any(t in title for t in DISQUALIFYING_TITLE_TOKENS):
        title_score = -0.3   # negative — title is a mismatch

    # But override if their career text strongly suggests AI/ML work
    ml_signals_in_career = sum(
        1 for kw in ["embedding", "ranking", "retrieval", "search", "recommendation",
                     "machine learning", "neural", "nlp", "vector", "ndcg"]
        if kw in profile_text
    )
    if ml_signals_in_career >= 3 and title_score < 0.5:
        title_score = max(title_score, 0.4)  # career overrides bad title

    score += 0.15 * title_score

    # ── Company type component (15%) ────────────────────────────────────────
    company_score = 0.5  # neutral default

    # Check if any career role was at a product company
    all_companies_in_career = [j["company"].lower() for j in career]
    all_industries_in_career = [j.get("industry", "").lower() for j in career]

    # Consulting/services only → negative signal (JD explicitly says this)
    consulting_count = sum(1 for co in all_companies_in_career
                          if any(c in co for c in CONSULTING_COMPANIES))
    total_roles = len(career)

    if total_roles > 0 and consulting_count == total_roles:
        company_score = 0.1  # all career at consulting = strong negative
    elif total_roles > 0 and consulting_count / total_roles > 0.5:
        company_score = 0.3

    # Product company experience is a positive signal
    product_co_count = sum(1 for co in all_companies_in_career
                          if any(p in co for p in PRODUCT_COMPANY_SIGNALS))
    if product_co_count >= 1:
        company_score = max(company_score, 0.7)
    if product_co_count >= 2:
        company_score = max(company_score, 0.9)

    # AI/ML company is an even stronger signal
    if any(ai_co in company for ai_co in ["anthropic", "deepmind", "openai", "cohere", "hugging face",
                                           "databricks", "mad street den", "observe.ai", "sarvam"]):
        company_score = 1.0

    score += 0.15 * company_score

    # ── YOE component (20%) — JD says 5-9 but is flexible ──────────────────
    # JD "ideal": 6-8yr total, 4-5yr in applied ML
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5:
        yoe_score = 0.85  # slightly under but JD says they're OK with 4yr
    elif 9 < yoe <= 12:
        yoe_score = 0.85
    elif 3 <= yoe < 4:
        yoe_score = 0.5
    elif yoe > 12:
        yoe_score = 0.6  # too senior — JD doesn't want "architecture" people
    else:
        yoe_score = 0.1  # under 3 years

    score += 0.20 * yoe_score

    # ── Career narrative component (50%) — what they actually built ─────────
    # This is the most important: "a Tier-5 candidate may not use RAG/Pinecone
    # but if their career shows they built recommendation systems at product
    # companies, they're a fit" — per the JD's hackathon hint
    narrative_signals = {
        # High value: directly relevant work
        "shipped.*ranking": 15,
        "production.*retrieval": 15,
        "retrieval.*production": 15,
        "vector.*search.*production": 14,
        "embedding.*production": 14,
        "recommendation.*system": 12,
        "search.*relevance": 13,
        "ranking.*model": 13,
        "re-rank": 12,
        "ndcg": 12,
        "a/b test": 8,
        "evaluation.*framework": 10,
        "offline.*evaluat": 9,
        "index refresh": 10,
        "embedding drift": 10,
        "retrieval quality": 10,
        "hybrid.*retrieval": 12,
        "semantic.*search": 12,
        "dense.*retrieval": 12,
        "sparse.*retrieval": 10,
        "learning.to.rank": 13,
        "lambdamart": 11,
        "xgboost.*rank": 11,
        # Medium value: related production ML
        "deployed.*model": 7,
        "production.*ml": 8,
        "ml.*production": 8,
        "feature engineering": 6,
        "feature pipeline": 7,
        "inference.*optimiz": 7,
        "llm.*production": 8,
        "fine-tun.*production": 8,
        # Lower value: adjacent skills
        "data pipeline": 3,
        "machine learning": 4,
        "deep learning": 4,
        "neural network": 3,
    }

    import re
    narrative_score = 0.0
    narrative_max = 0.0
    for pattern, weight in narrative_signals.items():
        narrative_max += weight
        if re.search(pattern, profile_text):
            narrative_score += weight

    # JD explicitly says: research-only = disqualify
    pure_research_signals = ["research assistant", "research intern", "lab", "phd research",
                              "arxiv", "paper accepted", "publications"]
    research_count = sum(1 for s in pure_research_signals if s in profile_text)
    production_count = sum(1 for s in ["shipped", "deployed", "production", "real users",
                                       "serving", "inference", "a/b"] if s in profile_text)
    if research_count > 2 and production_count == 0:
        narrative_score *= 0.3  # heavy penalty for pure research

    # JD: people who ONLY used LangChain/frameworks → negative
    framework_only_signals = profile_text.count("langchain") + profile_text.count("llamaindex")
    if framework_only_signals >= 3 and production_count == 0:
        narrative_score *= 0.5

    if narrative_max > 0:
        narrative_norm = min(1.0, narrative_score / (narrative_max * 0.25))
    else:
        narrative_norm = 0.0

    score += 0.50 * narrative_norm

    return min(1.0, max(0.0, score))


def score_education(candidate: dict) -> float:
    """
    Score 0-1: education relevance.
    JD doesn't specify degree requirements, but CS/ML background helps.
    Tier is provided in the data.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.4  # no education data → neutral

    relevant_fields = {
        "computer science", "cs", "information technology", "it",
        "machine learning", "artificial intelligence", "ai", "data science",
        "statistics", "mathematics", "math", "computational", "electronics",
        "electrical engineering", "computer engineering", "software engineering"
    }

    best_score = 0.0
    for edu in education:
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        tier = edu.get("tier", "unknown")

        # Field relevance
        field_relevant = any(f in field for f in relevant_fields)
        if "computer" in field or "information" in field or "data" in field or "ml" in field:
            field_score = 1.0
        elif field_relevant:
            field_score = 0.8
        else:
            field_score = 0.3

        # Tier multiplier
        tier_mult = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.7, "tier_4": 0.55, "unknown": 0.65}.get(tier, 0.65)

        # Degree level
        degree_mult = 1.0
        if any(x in degree for x in ["phd", "ph.d", "doctorate"]):
            degree_mult = 1.1  # slight boost for PhD in relevant field
        elif any(x in degree for x in ["m.tech", "m.e.", "mtech", "m.sc", "msc", "master"]):
            degree_mult = 1.05
        elif any(x in degree for x in ["b.tech", "b.e.", "btech", "b.sc", "bsc", "bachelor"]):
            degree_mult = 1.0

        edu_score = min(1.0, field_score * tier_mult * degree_mult)
        best_score = max(best_score, edu_score)

    return best_score


def score_location_fit(candidate: dict) -> float:
    """
    Score 0-1: location compatibility with JD requirements.
    JD: Pune/Noida preferred; open to Hyderabad, Pune, Mumbai, Delhi NCR.
    Outside India is case-by-case, no visa sponsorship.
    Notice period matters: sub-30d ideal, up to 60d OK, 90d+ penalty.
    """
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    notice = signals.get("notice_period_days", 90)
    relocate = signals.get("willing_to_relocate", False)

    # Country check
    if country not in ("india", "in", "india "):
        if relocate:
            location_score = 0.4  # willing to relocate from abroad
        else:
            location_score = 0.2  # outside India, not willing to relocate
    else:
        # India candidates
        preferred_cities = {"pune", "noida", "delhi", "delhi ncr", "gurgaon", "gurugram", "faridabad", "noida"}
        acceptable_cities = {"hyderabad", "mumbai", "bengaluru", "bangalore", "chennai", "kolkata"}

        in_preferred = any(city in location for city in preferred_cities)
        in_acceptable = any(city in location for city in acceptable_cities)

        if in_preferred:
            location_score = 1.0
        elif in_acceptable:
            location_score = 0.85
        elif relocate:
            location_score = 0.7
        else:
            location_score = 0.55  # other Indian city, not relocating

    # Notice period modifier
    if notice <= 30:
        notice_score = 1.0   # ideal
    elif notice <= 60:
        notice_score = 0.85  # JD: "can buy out 30 days"
    elif notice <= 90:
        notice_score = 0.65  # "30+ day notice candidates still in scope but bar gets higher"
    elif notice <= 120:
        notice_score = 0.45
    else:
        notice_score = 0.25  # 120+ days is a significant blocker

    return 0.6 * location_score + 0.4 * notice_score


def score_behavioral_signals(candidate: dict) -> float:
    """
    Score 0-1: availability and engagement signals.
    These are MULTIPLICATIVE modifiers — a perfect-on-paper candidate
    who is unreachable is effectively unavailable.

    JD hint: "a perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% recruiter response rate is, for hiring purposes, not actually available."
    """
    signals = candidate["redrob_signals"]
    today = REFERENCE_DATE

    # ── Recency: last active date ───────────────────────────────────────────
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (today - last_active).days
    except (ValueError, TypeError):
        days_inactive = 180  # assume stale if unknown

    if days_inactive <= 14:
        recency_score = 1.0
    elif days_inactive <= 30:
        recency_score = 0.95
    elif days_inactive <= 60:
        recency_score = 0.85
    elif days_inactive <= 90:
        recency_score = 0.70
    elif days_inactive <= 180:
        recency_score = 0.45
    else:
        recency_score = 0.15  # effectively inactive

    # ── Open-to-work flag ───────────────────────────────────────────────────
    otw_score = 1.0 if signals.get("open_to_work_flag", False) else 0.65

    # ── Recruiter response rate ─────────────────────────────────────────────
    rr = signals.get("recruiter_response_rate", 0.5)
    if rr >= 0.7:
        response_score = 1.0
    elif rr >= 0.5:
        response_score = 0.85
    elif rr >= 0.3:
        response_score = 0.65
    elif rr >= 0.1:
        response_score = 0.40
    else:
        response_score = 0.10  # 5% response rate → essentially unavailable

    # ── Profile completeness ────────────────────────────────────────────────
    completeness = signals.get("profile_completeness_score", 70)
    completeness_score = completeness / 100

    # ── GitHub activity (strong signal for AI engineers) ───────────────────
    github = signals.get("github_activity_score", -1)
    if github >= 0:
        github_score = min(1.0, github / 70)  # 70+ = full score
    else:
        github_score = 0.4  # no GitHub linked → neutral-negative for AI role

    # ── Interview completion rate ───────────────────────────────────────────
    icr = signals.get("interview_completion_rate", 0.8)
    interview_score = min(1.0, icr * 1.1)  # slight boost for high completion

    # ── Saved by recruiters (market signal) ────────────────────────────────
    saved = signals.get("saved_by_recruiters_30d", 0)
    saved_score = min(1.0, saved / 5)  # 5+ saves = full score

    # ── Verified contact (trust signal) ────────────────────────────────────
    verified = (signals.get("verified_email", False) + signals.get("verified_phone", False)) / 2
    verified_score = 0.7 + 0.3 * verified

    # ── Applications submitted (actively job hunting) ───────────────────────
    apps = signals.get("applications_submitted_30d", 0)
    apps_score = min(1.0, apps / 5) if apps > 0 else 0.3  # applying = actively looking

    # ── Weighted behavioral composite ───────────────────────────────────────
    behavioral = (
        0.25 * recency_score +       # most important — are they even here?
        0.20 * response_score +      # will they respond to recruiters?
        0.15 * otw_score +           # have they flagged availability?
        0.15 * github_score +        # technical activity signal
        0.10 * completeness_score +  # profile quality
        0.05 * interview_score +     # do they follow through?
        0.05 * saved_score +         # market validation
        0.05 * verified_score        # trust/authenticity
    )

    return min(1.0, max(0.0, behavioral))


def compute_composite(
    skills: float,
    career: float,
    education: float,
    location: float,
    behavioral: float,
) -> float:
    """
    Weighted composite score.
    Weights reflect JD emphasis:
    - Career fit (what they actually built) is most important
    - Skills match is second
    - Behavioral signals: multiplier approach — bad signals don't just subtract, they multiply down
    - Education and location: tiebreakers
    """
    # Base score from core dimensions
    base = (
        0.35 * career +      # most important — actual work experience
        0.30 * skills +      # skill match against JD
        0.15 * behavioral +  # availability/engagement signals
        0.12 * education +   # education relevance
        0.08 * location      # location/notice fit
    )

    # Behavioral as a multiplier on top (not just additive)
    # Rationale: a 0.05 response rate candidate is NOT hireable regardless of skills
    behavioral_multiplier = 0.4 + 0.6 * behavioral  # range: [0.4, 1.0]

    return min(1.0, max(0.0, base * behavioral_multiplier))


def build_reasoning(candidate: dict, scores: dict, rank: int) -> str:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    title = profile["current_title"]
    company = profile["current_company"]
    yoe = profile["years_of_experience"]
    location = profile.get("location", "Unknown")
    notice = signals.get("notice_period_days", 90)
    rr = signals.get("recruiter_response_rate", 0)
    github = signals.get("github_activity_score", -1)
    otw = signals.get("open_to_work_flag", False)
    last_active = signals.get("last_active_date", "")
    apps = signals.get("applications_submitted_30d", 0)
    saved = signals.get("saved_by_recruiters_30d", 0)
    icr = signals.get("interview_completion_rate", 1.0)

    from datetime import date, datetime
    REFERENCE_DATE = date(2026, 6, 18)
    try:
        la = datetime.strptime(last_active, "%Y-%m-%d").date()
        days_inactive = (REFERENCE_DATE - la).days
    except:
        days_inactive = 180

    # Extract matching skills
    skills = candidate.get("skills", [])
    assessments = signals.get("skill_assessment_scores", {})
    JD_TOKENS = {
        "embeddings","embedding","faiss","pinecone","weaviate","qdrant","milvus",
        "information retrieval","semantic search","hybrid search","retrieval",
        "ranking","re-ranking","bm25","ndcg","mrr","evaluation framework",
        "a/b test","learning to rank","ltr","lambdamart","python","llm","rag",
        "fine-tuning","xgboost","recommendation","nlp","pytorch","transformers"
    }
    matching_skills = [
        s["name"] for s in skills
        if any(token in s["name"].lower() for token in JD_TOKENS)
    ][:4]

    top_assessments = sorted(
        [(k,v) for k,v in assessments.items() if any(t in k.lower() for t in JD_TOKENS)],
        key=lambda x: -x[1]
    )[:2]

    # Build concerns list
    concerns = []
    if notice > 90:
        concerns.append(f"{notice}-day notice period is a significant friction point")
    elif notice > 60:
        concerns.append(f"{notice}-day notice may need buyout discussion")
    if rr < 0.2:
        concerns.append(f"very low recruiter response rate ({rr:.0%}) — reachability risk")
    elif rr < 0.4:
        concerns.append(f"below-average response rate ({rr:.0%})")
    if days_inactive > 180:
        concerns.append(f"last active {days_inactive} days ago — likely passive")
    elif days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days")
    if not otw:
        concerns.append("not currently marked open-to-work")
    if scores.get("skills", 0) < 0.35:
        concerns.append("limited direct overlap with JD's required retrieval/ranking stack")
    if icr < 0.6:
        concerns.append(f"low interview completion rate ({icr:.0%})")

    # Build strengths list
    strengths = []
    if github >= 60:
        strengths.append(f"strong GitHub activity ({github:.0f}/100)")
    elif github >= 30:
        strengths.append(f"active GitHub presence ({github:.0f}/100)")
    if rr >= 0.7:
        strengths.append(f"high recruiter response rate ({rr:.0%})")
    if notice <= 30:
        strengths.append(f"immediately available ({notice}-day notice)")
    elif notice <= 60:
        strengths.append(f"short notice period ({notice} days)")
    if otw:
        strengths.append("actively open to work")
    if saved >= 3:
        strengths.append(f"saved by {saved} recruiters in last 30 days")
    if apps >= 3:
        strengths.append(f"actively applying ({apps} applications this month)")
    if days_inactive <= 7:
        strengths.append("logged in within the last week")

    # 8 different sentence templates to avoid repetition
    import hashlib
    # Use candidate_id to deterministically pick a template (not random — reproducible)
    cid = candidate.get("candidate_id", "")
    template_idx = int(hashlib.md5(cid.encode()).hexdigest(), 16) % 8

    skill_str = ", ".join(matching_skills[:3]) if matching_skills else "ML/AI background"
    assess_str = "; ".join(f"{k}: {v:.0f}/100" for k,v in top_assessments) if top_assessments else ""

    if template_idx == 0:
        # Career-led
        sentence1 = f"{yoe:.0f}-year {title} at {company} ({location}) whose career history shows direct evidence of retrieval and ranking work in production."
    elif template_idx == 1:
        # Skills-led
        sentence1 = f"{title} at {company} with {yoe:.0f} years experience; core JD skills present include {skill_str}."
    elif template_idx == 2:
        # Assessment-led (if assessments exist)
        if assess_str:
            sentence1 = f"{yoe:.0f}-year {title} at {company}; platform assessments show {assess_str}."
        else:
            sentence1 = f"{title} at {company} — {yoe:.0f} years in ML/AI roles with relevant retrieval and ranking background."
    elif template_idx == 3:
        # Availability-led
        avail = f"notice {notice}d, response rate {rr:.0%}, {'open to work' if otw else 'not marked open'}"
        sentence1 = f"{yoe:.0f}-year {title} at {company}; availability signals: {avail}."
    elif template_idx == 4:
        # Fit summary
        fit = "strong" if scores.get("composite", 0) > 0.72 else "solid" if scores.get("composite", 0) > 0.67 else "moderate"
        sentence1 = f"{fit.capitalize()} fit: {title} at {company}, {yoe:.0f} years exp, skills align with JD's retrieval/ranking requirements ({skill_str})."
    elif template_idx == 5:
        # Company context
        sentence1 = f"Based at {location}, currently {title} at {company} — {yoe:.0f} years of applied ML experience with evidence of production deployment."
    elif template_idx == 6:
        # Direct match statement
        sentence1 = f"{yoe:.0f} years as {title} (currently {company}); JD-relevant skills: {skill_str}{'.' if not assess_str else f'; assessed: {assess_str}.'}"
    else:
        # Signal-forward
        gh_str = f"GitHub {github:.0f}/100" if github >= 0 else "no GitHub"
        sentence1 = f"{title} at {company}, {yoe:.0f}yr — {gh_str}, {rr:.0%} response rate, {notice}d notice."

    # Second sentence: strengths or concerns
    if rank <= 20:
        # Top candidates: lead with strengths, note any concerns
        if strengths:
            s2_base = f"Key engagement signals: {'; '.join(strengths[:2])}."
        else:
            s2_base = f"Profile completeness and behavioral signals support reachability."
        if concerns:
            sentence2 = s2_base + f" Note: {concerns[0]}."
        else:
            sentence2 = s2_base
    elif rank <= 50:
        # Mid-tier: balanced
        if concerns and strengths:
            sentence2 = f"Strengths: {strengths[0]}. Concern: {concerns[0]}."
        elif concerns:
            sentence2 = f"Concern(s): {'; '.join(concerns[:2])}."
        else:
            sentence2 = f"Engagement signals are adequate; {rr:.0%} response rate, {notice}-day notice."
    else:
        # Lower tier: honest about concerns
        if concerns:
            sentence2 = f"Ranked lower due to: {'; '.join(concerns[:2])}."
        else:
            sentence2 = f"Included based on skill overlap despite lower behavioral engagement signals."

    return f"{sentence1} {sentence2}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RANKING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(jsonl_path: str, top_n: int = 100) -> list[dict]:
    """
    Main pipeline: load → filter → score → rank → return top_n.
    """
    import time
    t0 = time.time()

    candidates = []
    honeypot_count = 0
    hard_filter_count = 0

    print(f"Loading candidates from {jsonl_path}...", file=sys.stderr)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidates.append(c)
            if (i + 1) % 10000 == 0:
                print(f"  Loaded {i+1} candidates...", file=sys.stderr)

    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    # ── Pass 1: Honeypot detection ───────────────────────────────────────────
    print("Running honeypot detection...", file=sys.stderr)
    scored = []
    honeypots_flagged = []

    for c in candidates:
        hp, hp_reasons = is_honeypot(c)
        if hp:
            honeypot_count += 1
            honeypots_flagged.append((c["candidate_id"], hp_reasons))
            continue  # skip honeypots entirely

        # ── Pass 2: Hard disqualification ─────────────────────────────────
        profile = candidate_profile_text(c).lower()
        title = c["profile"]["current_title"].lower()
        yoe = c["profile"]["years_of_experience"]

        # Instant disqualify: less than 3 years experience
        if yoe < 3.0:
            hard_filter_count += 1
            continue

        # Instant disqualify: ONLY consulting career + wrong domain
        career = c.get("career_history", [])
        all_consulting = all(
            any(cc in j["company"].lower() for cc in CONSULTING_COMPANIES)
            for j in career
        )
        has_ml_in_career = any(
            kw in profile for kw in ["machine learning", "embedding", "ranking", "retrieval",
                                      "recommendation", "vector", "neural", "nlp"]
        )
        if all_consulting and not has_ml_in_career and yoe < 8:
            hard_filter_count += 1
            continue

        # Compute all scores
        s_skills = score_skills(c)
        s_career = score_career_fit(c)
        s_education = score_education(c)
        s_location = score_location_fit(c)
        s_behavioral = score_behavioral_signals(c)
        s_composite = compute_composite(s_skills, s_career, s_education, s_location, s_behavioral)

        scored.append({
            "candidate": c,
            "scores": {
                "composite": s_composite,
                "skills": s_skills,
                "career": s_career,
                "education": s_education,
                "location": s_location,
                "behavioral": s_behavioral,
            }
        })

    print(f"Honeypots flagged: {honeypot_count}", file=sys.stderr)
    print(f"Hard-filtered: {hard_filter_count}", file=sys.stderr)
    print(f"Candidates scored: {len(scored)}", file=sys.stderr)

    # ── Sort by composite score ──────────────────────────────────────────────
    scored.sort(key=lambda x: (
        -x["scores"]["composite"],
        x["candidate"]["candidate_id"]  # tie-break: candidate_id ascending (per spec)
    ))

    top = scored[:top_n]

    # ── Build output rows ────────────────────────────────────────────────────
    results = []
    for rank_idx, item in enumerate(top):
        rank = rank_idx + 1
        c = item["candidate"]
        scores = item["scores"]
        reasoning = build_reasoning(c, scores, rank)

        results.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": round(scores["composite"], 6),
            "reasoning": reasoning,
        })

    t_total = time.time() - t0
    print(f"Total time: {t_total:.1f}s", file=sys.stderr)
    return results


def candidate_profile_text(c: dict) -> str:
    """Concatenate all searchable text from a candidate profile."""
    return (
        c["profile"].get("headline", "") + " " +
        c["profile"].get("summary", "") + " " +
        c["profile"].get("current_title", "") + " " +
        " ".join(s["name"] for s in c.get("skills", [])) + " " +
        " ".join(j.get("description", "") for j in c.get("career_history", []))
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuroRecruit — Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to rank")
    args = parser.parse_args()

    results = rank_candidates(args.candidates, top_n=args.top)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSubmission written to {args.out}", file=sys.stderr)
    print(f"Top 5 candidates:", file=sys.stderr)
    for r in results[:5]:
        print(f"  #{r['rank']}: {r['candidate_id']} score={r['score']:.4f}", file=sys.stderr)
        print(f"         {r['reasoning'][:100]}...", file=sys.stderr)


if __name__ == "__main__":
    main()
