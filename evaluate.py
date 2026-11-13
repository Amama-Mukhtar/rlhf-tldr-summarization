"""
Task 6 - compare SFT, PPO and DPO on the 300 held-out evaluation posts.

For every policy: one greedy summary per post (max 48 new tokens), then
  - mean reward-model score
  - win rate vs SFT (share of posts where its r_phi beats SFT's r_phi)
  - KL to SFT on its own greedy summary  (sum over tokens of log pi - log pi_sft)
  - mean length in tokens
  - ROUGE-L F against the human reference

Writes results/eval_summaries.jsonl, results/table3_comparison.md,
results/table2_training.md and results/samples_candidates.md.

    python evaluate.py
"""
import json

import numpy as np
import torch
from rouge_score import rouge_scorer

from common import (load_tokenizer, load_causal_lm, sequence_logprob, generate_summaries,
                    freeze, read_jsonl, set_seed, DEVICE, CKPT, RESULTS)
from data import load_eval_posts
from batching import collate_prompts
from train_reward import RewardModel

set_seed()
tok = load_tokenizer()
posts = load_eval_posts(tok)
pad = tok.pad_token_id
rm = freeze(RewardModel.load(f"{CKPT}/reward"))
sft_ref = freeze(load_causal_lm(f"{CKPT}/sft"))
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

POLICIES = ["sft", "ppo", "dpo"]
out = {name: [] for name in POLICIES}

for name in POLICIES:
    model = sft_ref if name == "sft" else freeze(load_causal_lm(f"{CKPT}/{name}"))
    for i in range(0, len(posts), 16):
        chunk = posts[i:i + 16]
        batch = collate_prompts(chunk, pad)
        P = batch["input_ids"].shape[1]
        full_ids, full_mask, gen_mask, texts = generate_summaries(model, tok, batch, greedy=True)
        full_mask = full_mask.to(DEVICE)
        with torch.no_grad():
            score = rm(full_ids, full_mask)
            plen = torch.full((len(chunk),), P)
            kl = sequence_logprob(model, full_ids, full_mask, plen) - sequence_logprob(sft_ref, full_ids, full_mask, plen)
        for e, t, s, k, n in zip(chunk, texts, score.tolist(), kl.tolist(), gen_mask.sum(1).tolist()):
            out[name].append({"summary": t, "score": s, "kl": k, "length": n,
                              "rougeL": scorer.score(e["summary"].strip(), t.strip())["rougeL"].fmeasure})
    print(name, "done")
    if name != "sft":
        del model
        torch.cuda.empty_cache()

# ---- save every summary ------------------------------------------------------
with open(f"{RESULTS}/eval_summaries.jsonl", "w", encoding="utf-8") as f:
    for i, e in enumerate(posts):
        row = {"id": i, "post": e["prompt"], "reference": e["summary"].strip()}
        for name in POLICIES:
            for k, v in out[name][i].items():
                row[f"{name}_{k}"] = v
        f.write(json.dumps(row) + "\n")

# ---- Table 3 -----------------------------------------------------------------
sft_scores = np.array([r["score"] for r in out["sft"]])
lines = ["# Table 3 - Final comparison (300 evaluation posts)\n",
         "| Policy | Mean r_phi | Win rate vs SFT | KL to SFT | Length | ROUGE-L |",
         "|---|---|---|---|---|---|"]
for name in POLICIES:
    s = np.array([r["score"] for r in out[name]])
    win = "-" if name == "sft" else f"{(s > sft_scores).mean() * 100:.1f}%"
    kl = np.mean([r["kl"] for r in out[name]])
    ln = np.mean([r["length"] for r in out[name]])
    rl = np.mean([r["rougeL"] for r in out[name]])
    lines.append(f"| {name.upper()} | {s.mean():.3f} | {win} | {kl:.3f} | {ln:.1f} | {rl:.3f} |")
table3 = "\n".join(lines) + "\n"
print(table3)
open(f"{RESULTS}/table3_comparison.md", "w", encoding="utf-8").write(table3)

# ---- Table 2 (built from what each training script logged) ------------------
try:
    times = {r["stage"]: r for r in read_jsonl(f"{RESULTS}/stage_times.jsonl")}
    sft_val = read_jsonl(f"{RESULTS}/sft_log.jsonl")[-1]["val_loss"]
    rows = [("SFT", f"val. loss: {sft_val:.3f}"),
            ("Reward model", f"pairwise acc.: {times['Reward model']['val_acc'] * 100:.1f}%"),
            ("PPO", f"final mean r_phi: {times['PPO']['final_mean_score']:.3f}"),
            ("DPO", f"implicit reward acc.: {times['DPO']['val_implicit_acc'] * 100:.1f}%")]
    t2 = ["# Table 2 - Training\n", "| Stage | Final train loss | Validation metric | Time / GPU |", "|---|---|---|---|"]
    for stage, metric in rows:
        r = times[stage]
        t2.append(f"| {stage} | {r['final_train_loss']:.4f} | {metric} | {r['minutes']:.0f} min / {r['gpu']} |")
    open(f"{RESULTS}/table2_training.md", "w", encoding="utf-8").write("\n".join(t2) + "\n")
    print("\n".join(t2))
except (FileNotFoundError, KeyError) as err:
    print("could not build Table 2 yet:", err)

# ---- candidate posts for the 10 qualitative samples ---------------------------
# Biggest reward gains and biggest losses vs SFT. YOU still read them and pick
# the 10 for results/samples.md - "clearly better" is a human call, not a number.
gain = [(max(out["ppo"][i]["score"], out["dpo"][i]["score"]) - out["sft"][i]["score"], i) for i in range(len(posts))]
loss = [(min(out["ppo"][i]["score"], out["dpo"][i]["score"]) - out["sft"][i]["score"], i) for i in range(len(posts))]
picks = [i for _, i in sorted(gain, reverse=True)[:8]] + [i for _, i in sorted(loss)[:8]]
with open(f"{RESULTS}/samples_candidates.md", "w", encoding="utf-8") as f:
    f.write("# Candidate posts for samples.md (top reward gains, then top drops)\n\n")
    for i in picks:
        f.write(f"## Post #{i}\n\n```\n{posts[i]['prompt'][-700:]}\n```\n\n"
                f"- **Human:** {posts[i]['summary'].strip()}\n")
        for name in POLICIES:
            r = out[name][i]
            f.write(f"- **{name.upper()}** (r={r['score']:.2f}, len={r['length']}): {r['summary'].strip()}\n")
        f.write("\n")
print("wrote results/eval_summaries.jsonl, table3_comparison.md, samples_candidates.md")
