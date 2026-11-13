"""
Section 3.2 - blind human evaluation on 30 posts.

Step 1 (after evaluate.py):   python human_eval.py make
    -> results/human_eval.csv   : the sheet you both fill in (A/B/C, no model names)
    -> results/human_eval_key.json : which letter is which model (DON'T open it before judging)

Step 2: each member fills their own column (member1_pick / member2_pick) with
        A, B or C - independently, no discussing.

Step 3:                        python human_eval.py score
    -> results/table4_human_eval.md
"""
import csv
import json
import random
import sys

from common import read_jsonl, RESULTS

N, SEED = 30, 2026
SHEET = f"{RESULTS}/human_eval.csv"
KEY = f"{RESULTS}/human_eval_key.json"


def make():
    rows = read_jsonl(f"{RESULTS}/eval_summaries.jsonl")
    rng = random.Random(SEED)
    chosen = rng.sample(range(len(rows)), N)
    key = {}
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["post_id", "post", "A", "B", "C", "member1_pick", "member2_pick"])
        for pid in chosen:
            r = rows[pid]
            models = ["sft", "ppo", "dpo"]
            rng.shuffle(models)
            key[str(pid)] = dict(zip("ABC", models))
            w.writerow([pid, r["post"], *[r[f"{m}_summary"].strip() for m in models], "", ""])
    json.dump(key, open(KEY, "w"), indent=1)
    print(f"wrote {SHEET} - fill member1_pick / member2_pick with A, B or C")


def score():
    key = json.load(open(KEY))
    counts = {m: {"sft": 0, "ppo": 0, "dpo": 0} for m in ("member1", "member2")}
    agree, n = 0, 0
    with open(SHEET, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = key[row["post_id"]]
            p1, p2 = row["member1_pick"].strip().upper(), row["member2_pick"].strip().upper()
            if p1 not in "ABC" or p2 not in "ABC" or not p1 or not p2:
                print("skipping unfilled row", row["post_id"])
                continue
            counts["member1"][k[p1]] += 1
            counts["member2"][k[p2]] += 1
            agree += p1 == p2
            n += 1
    lines = ["# Table 4 - Human evaluation (30 posts)\n",
             "| | SFT picked | PPO picked | DPO picked |", "|---|---|---|---|"]
    for mem, c in counts.items():
        lines.append(f"| {mem.title()} | {c['sft']} | {c['ppo']} | {c['dpo']} |")
    lines.append(f"\nAgreement between members: **{agree} / {n}** ({agree / max(n, 1) * 100:.0f}%)")
    text = "\n".join(lines) + "\n"
    print(text)
    open(f"{RESULTS}/table4_human_eval.md", "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
