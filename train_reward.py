"""
Task 3 - Stage 2: the reward model r_phi(x, y).

Idea in one line: a "judge" network that reads post + summary and outputs
ONE number. We train it so the human-preferred summary gets a bigger number.

Architecture:  SFT transformer body (no LM head)  ->  Linear(hidden, 1)
               read at the LAST REAL token of prompt + summary.

Loss (Bradley-Terry, lecture Eq. 40):
    L = - mean log sigmoid( r(x, y_w) - r(x, y_l) )

    python train_reward.py
"""
import argparse
import os
import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (load_tokenizer, load_causal_lm, lm_forward, append_jsonl,
                    set_seed, save_resume, load_resume, clear_resume, DEVICE, CKPT, RESULTS, SEED)


class RewardModel(nn.Module):
    def __init__(self, lm_path):
        super().__init__()
        lm = load_causal_lm(lm_path)
        self.body = lm.model                       # transformer only, LM head dropped
        self.config = lm.config
        self.head = nn.Linear(lm.config.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=1.0 / (lm.config.hidden_size ** 0.5))
        nn.init.zeros_(self.head.bias)
        self._lm = [lm]                            # kept only so we can save_pretrained; not a submodule
        self.to(DEVICE)

    def forward(self, input_ids, attention_mask):
        h = lm_forward(self.body, input_ids, attention_mask).last_hidden_state  # (B, L, H)
        # Index of the last real token. Works for left AND right padding.
        # We use the attention mask, never input_ids != pad (pad == EOS here!).
        L = attention_mask.shape[1]
        pos = torch.arange(L, device=attention_mask.device).unsqueeze(0)
        last = (pos * attention_mask).argmax(dim=1)
        h_last = h[torch.arange(h.shape[0], device=h.device), last]
        return self.head(h_last.float()).squeeze(-1).float()  # (B,)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self._lm[0].save_pretrained(path)          # body weights live inside the LM object
        torch.save(self.head.state_dict(), f"{path}/reward_head.pt")

    @classmethod
    def load(cls, path):
        rm = cls(path)
        rm.head.load_state_dict(torch.load(f"{path}/reward_head.pt", map_location=DEVICE))
        return rm


def pair_scores(rm, batch):
    rw = rm(batch["chosen_ids"].to(DEVICE), batch["chosen_mask"].to(DEVICE))
    rl = rm(batch["rejected_ids"].to(DEVICE), batch["rejected_mask"].to(DEVICE))
    return rw, rl


@torch.no_grad()
def pairwise_accuracy(rm, pairs, pad, return_rows=False):
    rm.eval()
    correct, rows = 0, []
    from batching import collate_pref
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        rw, rl = pair_scores(rm, collate_pref(chunk, pad))
        correct += (rw > rl).sum().item()
        if return_rows:
            for p, a, b in zip(chunk, rw.tolist(), rl.tolist()):
                rows.append({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"],
                             "r_chosen": a, "r_rejected": b, "margin": a - b})
    rm.train()
    acc = correct / len(pairs)
    return (acc, rows) if return_rows else acc


if __name__ == "__main__":
    from data import load_preference_splits
    from batching import collate_pref

    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    set_seed()
    tok = load_tokenizer()
    train, val = load_preference_splits(tok)
    pad = tok.pad_token_id
    LOG = f"{RESULTS}/reward_log.jsonl"

    rm = RewardModel(f"{CKPT}/sft")
    rm.train()
    opt = torch.optim.AdamW(rm.parameters(), lr=args.lr)
    done, prev_min = load_resume("reward", {"rm": rm}, opt)
    if done == 0:
        open(LOG, "w").close()
        print("val accuracy before training:", pairwise_accuracy(rm, val, pad))
    # fixed generator -> same shuffle order every run, so a resume skips exactly the batches already seen
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED),
                        collate_fn=partial(collate_pref, pad_id=pad))

    start, step, running, running_acc = time.time() - prev_min * 60, 0, [], []
    for batch in loader:
        if step < done:          # already trained on this batch before the cut
            step += 1
            continue
        rw, rl = pair_scores(rm, batch)
        loss = -F.logsigmoid(rw - rl).mean()       # Bradley-Terry
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        running.append(loss.item())
        running_acc.append((rw > rl).float().mean().item())

        if step % args.log_every == 0 or step == len(loader) or step == args.max_steps:
            acc = pairwise_accuracy(rm, val, pad)
            rec = {"step": step, "train_loss": sum(running) / len(running),
                   "train_acc": sum(running_acc) / len(running_acc), "val_acc": acc,
                   "minutes": (time.time() - start) / 60}
            running, running_acc = [], []
            append_jsonl(LOG, rec)
            print(rec)
            save_resume("reward", step, {"rm": rm}, opt, rec["minutes"])
        if args.max_steps and step >= args.max_steps:
            break

    minutes = (time.time() - start) / 60
    rm.save(f"{CKPT}/reward")
    tok.save_pretrained(f"{CKPT}/reward")
    clear_resume("reward")

    # Final accuracy + the pairs it gets wrong (for Task 7.1)
    acc, rows = pairwise_accuracy(rm, val, pad, return_rows=True)
    wrong = sorted([r for r in rows if r["margin"] <= 0], key=lambda r: r["margin"])
    with open(f"{RESULTS}/reward_val_scores.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(__import__("json").dumps(r) + "\n")
    print(f"final val pairwise accuracy: {acc:.3f}  ({len(wrong)} wrong of {len(rows)})")
    append_jsonl(f"{RESULTS}/stage_times.jsonl", {"stage": "Reward model", "minutes": minutes,
                                                  "final_train_loss": rec["train_loss"], "val_acc": acc,
                                                  "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"})
