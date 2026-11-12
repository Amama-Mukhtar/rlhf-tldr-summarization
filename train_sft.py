"""
Task 2 - Stage 1: Supervised fine-tuning (SFT).

Idea in one line: show the model 20,000 (post, human summary) pairs and
make it more likely to write the human summary. We only grade it on the
summary tokens - the post is given, so we don't ask it to "predict" the post.

    L_SFT = - mean over summary tokens of log pi(x_t | x_<t)

    python train_sft.py                 # full run
    python train_sft.py --max_steps 20  # quick smoke test
"""
import argparse
import time
from functools import partial

import torch
from torch.utils.data import DataLoader

from common import (load_tokenizer, load_causal_lm, token_logprobs, summary_mask,
                    generate_summaries, append_jsonl, set_seed, DEVICE, CKPT, RESULTS)
from data import load_sft_splits, load_eval_posts
from batching import collate_sft, collate_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--micro_batch", type=int, default=8, help="split each batch to save GPU memory")
ap.add_argument("--log_every", type=int, default=250)
ap.add_argument("--max_steps", type=int, default=None)
args = ap.parse_args()

set_seed()
tok = load_tokenizer()
train, val = load_sft_splits(tok)
eval_posts = load_eval_posts(tok)
pad = tok.pad_token_id
LOG = f"{RESULTS}/sft_log.jsonl"
open(LOG, "w").close()


def sft_loss_sum(model, batch):
    """Returns (sum of -log p over summary tokens, number of summary tokens).
    We return the sum, not the mean, so micro-batches add up correctly."""
    ids = batch["input_ids"].to(DEVICE)
    att = batch["attention_mask"].to(DEVICE)
    lp = token_logprobs(model, ids, att)
    m = summary_mask(att, batch["prompt_lens"])
    return -(lp * m).sum(), m.sum()


@torch.no_grad()
def validation_loss(model):
    model.eval()
    tot, n = 0.0, 0.0
    for b in DataLoader(val, batch_size=16, collate_fn=partial(collate_sft, pad_id=pad)):
        s, c = sft_loss_sum(model, b)
        tot, n = tot + s.item(), n + c.item()
    model.train()
    return tot / n


def five_samples(model, tag):
    model.eval()
    b = collate_prompts(eval_posts[:5], pad)
    _, _, _, texts = generate_summaries(model, tok, b, greedy=True)
    model.train()
    return [{"when": tag, "post": e["prompt"][-400:], "reference": e["summary"], "summary": t}
            for e, t in zip(eval_posts[:5], texts)]


model = load_causal_lm()
samples = five_samples(model, "before SFT (base model)")
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                    collate_fn=partial(collate_sft, pad_id=pad))

start = time.time()
step = 0
for batch in loader:  # exactly one epoch
    # --- one optimizer step, possibly in micro-batches ---
    pieces = [{k: v[i:i + args.micro_batch] for k, v in batch.items()}
              for i in range(0, batch["input_ids"].shape[0], args.micro_batch)]
    n_tokens = sum(summary_mask(p["attention_mask"], p["prompt_lens"]).sum() for p in pieces)
    total = 0.0
    for p in pieces:
        s, _ = sft_loss_sum(model, p)
        (s / n_tokens.to(DEVICE)).backward()  # mean over ALL summary tokens in the batch
        total += s.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    train_loss = total / n_tokens.item()

    if step % args.log_every == 0 or step == len(loader) or step == args.max_steps:
        v = validation_loss(model)
        rec = {"step": step, "train_loss": train_loss, "val_loss": v, "minutes": (time.time() - start) / 60}
        append_jsonl(LOG, rec)
        print(rec)
    if args.max_steps and step >= args.max_steps:
        break

minutes = (time.time() - start) / 60
model.save_pretrained(f"{CKPT}/sft")
tok.save_pretrained(f"{CKPT}/sft")

samples += five_samples(model, "after SFT")
with open(f"{RESULTS}/sft_samples.md", "w", encoding="utf-8") as f:
    f.write("# SFT: 5 evaluation posts, greedy, before and after\n\n")
    for i in range(5):
        before, after = samples[i], samples[i + 5]
        f.write(f"## Post {i + 1}\n\n```\n{before['post']}\n```\n\n"
                f"- **Human reference:** {before['reference']}\n"
                f"- **Base model:** {before['summary']}\n"
                f"- **After SFT:** {after['summary']}\n\n")
append_jsonl(f"{RESULTS}/stage_times.jsonl", {"stage": "SFT", "minutes": minutes,
                                              "final_train_loss": train_loss,
                                              "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"})
print(f"done in {minutes:.1f} min -> checkpoints/sft/")
