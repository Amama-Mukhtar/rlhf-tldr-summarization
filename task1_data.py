"""
Task 1 - run the starter code, report data statistics, and check the
summary mask on one real batch.

    python task1_data.py
"""
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (load_tokenizer, load_causal_lm, token_logprobs, summary_mask,
                    RESULTS, DEVICE, set_seed)
from data import load_sft_splits, load_preference_splits, load_eval_posts
from batching import collate_sft

set_seed()
tok = load_tokenizer()
sft_train, sft_val = load_sft_splits(tok)
pref_train, pref_val = load_preference_splits(tok)
eval_posts = load_eval_posts(tok)

# ---- 1.2  sizes and lengths ------------------------------------------------
p_len = np.array([len(x["prompt_ids"]) for x in sft_train])
s_len = np.array([len(x["summary_ids"]) for x in sft_train])  # includes the EOS

table = f"""# Table 1 - Data

| Item | Value |
|---|---|
| SFT train / validation pairs | {len(sft_train)} / {len(sft_val)} |
| Preference train / validation pairs | {len(pref_train)} / {len(pref_val)} |
| Evaluation posts | {len(eval_posts)} |
| Mean / max prompt length, SFT train (tokens) | {p_len.mean():.1f} / {p_len.max()} |
| Mean / max summary length, SFT train (tokens, incl. EOS) | {s_len.mean():.1f} / {s_len.max()} |
"""
print(table)
with open(f"{RESULTS}/table1_data.md", "w", encoding="utf-8") as f:
    f.write(table)

# ---- 1.3  token_logprobs + summary_mask check -------------------------------
batch = next(iter(DataLoader(sft_train[:4], batch_size=4,
                             collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))))
model = load_causal_lm()
ids = batch["input_ids"].to(DEVICE)
att = batch["attention_mask"].to(DEVICE)
with torch.no_grad():
    lp = token_logprobs(model, ids, att)
m = summary_mask(att, batch["prompt_lens"])
print("token_logprobs shape:", tuple(lp.shape), "  summary_mask shape:", tuple(m.shape))

lines = ["# Summary-mask check (Task 1.3 / correctness check 1)\n"]
for i in range(ids.shape[0]):
    # mask column j refers to token j+1, so shift back by one to decode
    kept = ids[i, 1:][m[i].bool().cpu()]
    decoded = tok.decode(kept)
    expected = sft_train[i]["summary"] + tok.eos_token
    ok = decoded == expected
    lines.append(f"row {i}: match={ok}\n  decoded : {decoded!r}\n  expected: {expected!r}\n")
    print(lines[-1])
with open(f"{RESULTS}/check_summary_mask.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
