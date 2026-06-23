import csv

print(f"{'RANK':<6} {'CANDIDATE_ID':<15} {'SCORE':<8} REASONING")
print("-" * 90)

with open("submission.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows: {len(rows)}")
print()
print("TOP 20:")
for row in rows[:20]:
    print(f"#{row['rank']:<5} {row['candidate_id']:<15} {float(row['score']):.4f}  {row['reasoning'][:70]}")

print()
print("BOTTOM 5:")
for row in rows[-5:]:
    print(f"#{row['rank']:<5} {row['candidate_id']:<15} {float(row['score']):.4f}  {row['reasoning'][:70]}")

scores = [float(r["score"]) for r in rows]
print()
print(f"Score range: {min(scores):.4f} to {max(scores):.4f}")
print(f"Avg score:   {sum(scores)/len(scores):.4f}")
