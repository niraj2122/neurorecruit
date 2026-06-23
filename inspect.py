import csv

print(f"{'RANK':<6} {'CANDIDATE_ID':<15} {'SCORE':<8} REASONING")
print("-" * 90)

with open("test_output.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rank     = row["rank"]
        cid      = row["candidate_id"]
        score    = float(row["score"])
        reason   = row["reasoning"][:70]
        print(f"#{rank:<5} {cid:<15} {score:.4f}  {reason}")
