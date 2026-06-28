# 🧠 NeuroRecruit — Intelligent Candidate Discovery
### Redrob Hackathon · Senior AI Engineer JD · 100,000 candidates → top 100 in 46 seconds

[![Sandbox](https://img.shields.io/badge/sandbox-neurorecruit.streamlit.app-red?logo=streamlit)](https://neurorecruit.streamlit.app/)
[![GitHub](https://img.shields.io/badge/repo-niraj2122%2Fneurorecruit-black?logo=github)](https://github.com/niraj2122/neurorecruit)
[![Validator](https://img.shields.io/badge/submission-valid-brightgreen)]()

---

## Quickstart — reproduce submission in one command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
# → Submission is valid.
```

**Requirements:** Python 3.9+, zero pip installs, pure stdlib.
**Runtime:** ~46 seconds on CPU · <1 GB RAM · zero network calls · no GPU.

---

## Live sandbox

**[neurorecruit.streamlit.app](https://neurorecruit.streamlit.app/)**

Upload `sample_candidates.json` (50 candidates, instant) or the full `candidates.jsonl`
(100K candidates, ~90s). The sandbox uses heap-based streaming — it never loads all
candidates into memory simultaneously, so it handles the full 487MB file without crashing.

Features visible in the sandbox:
- Real-time JD bias detection (flags masculine-coded and age-coded language)
- Per-candidate skill alignment tags (present vs missing vs JD requirements)
- Gap analysis with specific interview questions for each skill gap
- Download ranked CSV directly from the browser

---

## Architecture

```
100,000 candidates
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: Honeypot detection                        │
│  6 consistency checks → 2,213 flagged & removed     │
│  (inverted salary, impossible skill durations,       │
│   expert claims contradicted by assessments)         │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: Hard filter                               │
│  19,231 removed (<3yr YOE, consulting-only           │
│  careers with zero ML signal in descriptions)        │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3: Five-dimension scoring (78,556 candidates) │
│                                                     │
│  Career narrative    35%  production ML evidence    │
│                           in role descriptions      │
│  Skills match        30%  proficiency × duration    │
│                           × assessment score        │
│  Behavioral signals  15%  8 platform signals        │
│  Education           12%  field × institution tier  │
│  Location / notice    8%  Pune/Noida pref, 30d notice│
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: Behavioral multiplier                     │
│  composite = base × (0.4 + 0.6 × behavioral)        │
│  Range: 0.4× (5% response rate) to 1.0× (engaged)  │
│  Models hiring outcomes, not just profile quality   │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5: Ranked CSV + reasoning                    │
│  Top 100 candidates · 8 structural reasoning        │
│  variations · references real profile facts         │
└─────────────────────────────────────────────────────┘
```

---

## Scoring in detail

### Career narrative (35%) — the core insight

We search every word of every role description for production ML signals using
regex patterns. A candidate who wrote "deployed LambdaMART-based ranking to 10M
users and measured NDCG improvements via A/B testing" scores near-perfect even
if their skill list is sparse.

High-value patterns (weight 12–15 each):
`shipped.*ranking` · `embedding.*production` · `hybrid.*retrieval` ·
`ndcg` in career text · `evaluation.*framework` · `learning.to.rank` ·
`lambdamart` · `semantic.*search`

Penalty signals: research-only careers (0.3×), LangChain-heavy without
production deployment (0.5×), all-consulting with no ML narrative (0.4×).

### Skills match (30%)

Each skill match is multiplied by three sub-factors:

```
skill_value = base_weight × proficiency_mult × duration_mult × assessment_mult

proficiency_mult:  beginner=0.4  intermediate=0.7  advanced=0.9  expert=1.0
duration_mult:     min(1.0, 0.5 + duration_months / 48)
assessment_mult:   0.8 + 0.2 × (score / 100)  if assessed, else 0.9
```

### Behavioral signals (15% + multiplier)

Eight signals from the `redrob_signals` object:

| Signal | Weight |
|--------|--------|
| Last active date (recency) | 25% |
| Recruiter response rate | 20% |
| Open-to-work flag | 15% |
| GitHub activity score | 15% |
| Profile completeness | 10% |
| Interview completion rate | 5% |
| Saved by recruiters (30d) | 5% |
| Verified contact details | 5% |

The behavioral score also acts as a **multiplier** on the entire composite:
`composite = base × (0.4 + 0.6 × behavioral)`

A candidate with 5% response rate gets a 0.43× multiplier — their score is
nearly halved regardless of how strong their profile looks. This directly
models the JD's stated requirement: "a perfect-on-paper candidate who hasn't
logged in for 6 months is not actually available."

### Honeypot detection

Six consistency checks, requiring 2+ flags to call a honeypot (prevents
false positives from data entry errors):

1. Salary `min > max` — impossible range
2. Skill `duration_months > career_length_months + 12` — used skill longer than career
3. Total career months far exceeds stated YOE
4. 3+ skills claimed as "expert" with `duration_months = 0`
5. Assessment score < 35 on a skill claimed as "expert"
6. YOE exceeds what's possible given graduation year + 5yr grace

Result: **2,213 honeypots removed** before scoring begins.

---

## File structure

```
neurorecruit/
├── rank.py                    # Main ranker — zero dependencies, pure stdlib
├── sandbox_app.py             # Streamlit app — heap-based streaming, bias detection
├── colab_demo.py              # Google Colab demo
├── README.md                  # This file
├── requirements.txt           # Empty — stdlib only
├── requirements_sandbox.txt   # streamlit>=1.28.0, pandas>=1.5.0
├── submission_metadata.yaml   # Hackathon metadata
├── submission.csv             # Final submission — 100 ranked candidates
├── inspect.py                 # Inspect test_output.csv (sample run)
├── full_check.py              # Inspect submission.csv (full run)
├── .streamlit/
│   └── config.toml            # maxUploadSize = 1024 (1GB)
└── .gitignore                 # Excludes candidates.jsonl (487MB)
```

---

## What makes this different from a keyword filter

| What most systems do | What NeuroRecruit does |
|---------------------|----------------------|
| Count skill keyword matches | Read career descriptions for production evidence |
| Add behavioral scores additively | Apply behavioral as a structural multiplier |
| Rank all 100K equally | Remove 2,213 honeypots before scoring begins |
| Generic reasoning strings | 8 structural templates × real profile facts |
| No bias awareness | Real-time JD bias detection in sandbox |
| No gap visibility | Per-candidate skill delta + interview questions |

---

## Top 5 results

| Rank | Candidate | Role | Company | Score |
|------|-----------|------|---------|-------|
| #1 | CAND_0018499 | Senior ML Engineer | Zomato | 0.7816 |
| #2 | CAND_0081846 | Lead AI Engineer | Razorpay | 0.7602 |
| #3 | CAND_0077337 | Staff ML Engineer | Paytm | 0.7592 |
| #4 | CAND_0079387 | AI Engineer | Microsoft | 0.7329 |
| #5 | CAND_0071974 | Senior AI Engineer | Netflix | 0.7202 |

Score range across top 100: **0.6176 – 0.7816**

---

## Reproduce command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Single command. No setup. No installs. Runs in ~46 seconds on any CPU.
