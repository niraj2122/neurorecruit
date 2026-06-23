# NeuroRecruit — Google Colab Demo Notebook
# Redrob Hackathon Sandbox

import subprocess, json, csv, time

CANDIDATE_FILE = "sample_candidates.json"

print("=" * 60)
print("  NeuroRecruit — Intelligent Candidate Discovery")
print("  Redrob Hackathon Demo")
print("=" * 60)

t0 = time.time()
data = json.load(open(CANDIDATE_FILE))
with open("test_candidates.jsonl", "w") as f:
    for c in data:
        f.write(json.dumps(c) + "\n")

print(f"Loaded {len(data)} candidates")
result = subprocess.run(
    ["python", "rank.py", "--candidates", "test_candidates.jsonl",
     "--out", "demo_output.csv", "--top", "20"],
    capture_output=True, text=True
)
print(result.stderr)
print(f"Done in {time.time()-t0:.1f}s")

print("\n" + "=" * 70)
print(f"{'RANK':<6} {'CANDIDATE':<15} {'SCORE':<8} REASONING")
print("=" * 70)
with open("demo_output.csv") as f:
    for row in csv.DictReader(f):
        print(f"#{row['rank']:<5} {row['candidate_id']:<15} {float(row['score']):.4f}  {row['reasoning'][:60]}...")

JD_REQUIRED = {
    "embeddings": 10, "faiss": 9, "pinecone": 9, "weaviate": 9,
    "information retrieval": 10, "semantic search": 10, "hybrid search": 10,
    "ndcg": 9, "learning to rank": 8, "python": 6, "evaluation framework": 9,
}
INTERVIEW_Qs = {
    "ndcg": "Walk me through your NDCG measurement pipeline in production.",
    "information retrieval": "Describe the hardest scaling challenge in a retrieval system you shipped.",
    "semantic search": "How would you build hybrid search combining dense and sparse retrieval?",
    "embeddings": "How do you handle embedding drift in production?",
    "faiss": "When would you choose FAISS vs a managed vector DB?",
    "learning to rank": "Walk me through an LTR model you trained end to end.",
    "evaluation framework": "How do you build offline evaluation for retrieval?",
    "hybrid search": "How do you combine BM25 and dense scores? What fusion strategy?",
}

with open("test_candidates.jsonl") as f:
    all_cands = {json.loads(l)["candidate_id"]: json.loads(l) for l in f if l.strip()}
with open("demo_output.csv") as f:
    rows = list(csv.DictReader(f))

print("\n" + "=" * 70)
print("GAP ANALYSIS - TOP 3 CANDIDATES")
print("=" * 70)
for row in rows[:3]:
    c = all_cands.get(row["candidate_id"])
    if not c:
        continue
    p = c["profile"]
    skill_names = {s["name"].lower() for s in c.get("skills", [])}
    profile_text = (p.get("summary","") + " " + " ".join(
        j.get("description","") for j in c.get("career_history",[]))).lower()
    present = [t for t in JD_REQUIRED if any(t in s for s in skill_names) or t in profile_text]
    missing = [t for t in JD_REQUIRED if t not in present]
    print(f"\n#{row['rank']}: {p['current_title']} @ {p['current_company']}")
    print(f"  Present: {', '.join(present)}")
    print(f"  Missing: {', '.join(missing)}")
    for gap in missing[:2]:
        if gap in INTERVIEW_Qs:
            print(f"  Q [{gap}]: {INTERVIEW_Qs[gap]}")
print("\nDemo complete.")
