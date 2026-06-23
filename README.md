# NeuroRecruit — Intelligent Candidate Discovery
### Redrob Hackathon · Senior AI Engineer JD · 100,000 candidates → top 100 in 60 seconds

---

## Quickstart (reproduce submission in one command)

```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
python3 validate_submission.py submission.csv   # → "Submission is valid."
```

**Requirements:** Python 3.9+, zero pip installs. Pure stdlib.  
**Runtime:** ~60 seconds on CPU, <1 GB RAM, no network calls.

---

## Architecture

Five-layer intelligent ranking engine:

```
100,000 candidates
       ↓
[1] Honeypot detection      — 6 consistency checks, 2,213 flagged & removed
       ↓
[2] Hard filter             — <3yr YOE, consulting-only with no ML signal removed
       ↓
[3] 5-signal scoring        — per candidate across all dimensions
       ↓
[4] Behavioral multiplier   — suppresses unreachable candidates structurally
       ↓
[5] Ranked CSV + reasoning  — top 100 with per-candidate briefing notes
```

### Scoring weights

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Career narrative | 35% | Production ML signals in role descriptions |
| Skills match | 30% | JD skill overlap × proficiency × duration × assessment |
| Behavioral signals | 15% + multiplier | Availability, engagement, reachability |
| Education | 12% | CS/ML field relevance × institution tier |
| Location/notice | 8% | Pune/Noida preferred, sub-30d notice ideal |

**Behavioral is also a multiplier** (0.4–1.0×) on the composite score. A 5% response-rate candidate cannot rank highly regardless of skills — by design.

### Key design decisions

1. **Career narrative over skill keywords.** We search role descriptions for production evidence: "shipped ranking model", "embedding drift", "hybrid retrieval", "evaluation framework". These signals are harder to fake than a skill list.

2. **Negative skill signals.** LangChain-heavy profiles, CV/robotics focus — the JD explicitly calls these out. We apply score penalties.

3. **Honeypot detection before scoring.** Six checks including salary inversion, skill duration vs career length, assessment contradictions.

4. **Gap analysis.** For top candidates, we compute JD skill deltas and generate specific interview questions for each gap.

---

## Files

```
neurorecruit/
├── rank.py                    # Main ranker — all logic, 992 lines, no dependencies
├── sandbox_app.py             # Streamlit demo app
├── colab_demo.py              # Google Colab demo cells
├── README.md                  # This file
├── requirements.txt           # Empty (stdlib only)
├── requirements_sandbox.txt   # streamlit + pandas (sandbox only)
└── submission_metadata.yaml   # Hackathon submission metadata
```

---

## Sandbox

**HuggingFace Spaces (Streamlit):**
1. Create new Space at huggingface.co/new-space → Streamlit
2. Upload `sandbox_app.py` as `app.py`
3. Upload `rank.py` 
4. Upload `requirements_sandbox.txt` as `requirements.txt`
5. The Space builds automatically — share the URL

**Google Colab (alternative):**
1. Open colab.research.google.com → new notebook
2. Follow cells in `colab_demo.py`
3. Upload `sample_candidates.json` when prompted
4. Share as "Anyone with link can view"

---

## What makes this different from a keyword filter

Most systems rank by counting skill matches. NeuroRecruit:

- **Reads career descriptions** for production ML evidence (not just skill lists)
- **Detects fraudulent profiles** before they can contaminate rankings  
- **Models hiring outcomes** — an unreachable candidate scores low regardless of profile quality
- **Generates per-candidate reasoning** that references actual profile facts
- **Gap analysis + interview questions** for the top candidates
