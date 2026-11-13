"""
Task 4 - Stage 3: PPO (the "RL" in RLHF).

The story of one PPO iteration, in plain words:
  1. ROLLOUT   - the policy writes a summary for 16 posts (sampling, not greedy).
  2. STORE     - with no gradients, remember for every generated token:
                 old log-prob, reference log-prob, old value V(s_t); and the
                 reward-model score of each whole summary.
  3. REWARD    - every token pays a small KL "fine" for drifting from SFT;
                 the last token also receives the reward-model score.
                     r_t = -beta * (log pi_old - log pi_ref) + 1[t=T] * r_phi(x, y)
  4. ADVANTAGE - R_t = sum of future rewards (gamma = 1), A_t = R_t - V_old(s_t),
                 then normalise A over all tokens in the batch.
  5. UPDATE    - 4 epochs over the same rollout with the clipped PPO loss
                 + value loss - entropy bonus.

Models in GPU memory: policy (+ value head, trained), reference (frozen SFT),
reward model (frozen). That's 3 transformers.

    python train_ppo.py
    python train_ppo.py --iterations 3     # smoke test
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn

from common import (load_tokenizer, load_causal_lm, lm_forward, gather_logprobs,
                    generate_summaries, freeze, all_frozen, append_jsonl, read_jsonl,
                    set_seed, save_resume, load_resume, clear_resume,
                    DEVICE, CKPT, RESULTS, SEED)
from data import load_sft_splits
from batching import collate_prompts
from train_reward import RewardModel

ap = argparse.ArgumentParser()
ap.add_argument("--iterations", type=int, default=200)
ap.add_argument("--prompts_per_rollout", type=int, default=16)
ap.add_argument("--chunks", type=int, default=2, help="split each rollout to save memory (grads are added)")
args = ap.parse_args()

# ---- the exact settings from the manual -------------------------------------
CLIP_EPS = 0.2
BETA_KL = 0.05
C_V = 0.1
C_H = 0.01
GAMMA = 1.0
K_EPOCHS = 4
LR = 1e-5
MAX_GRAD_NORM = 1.0


class PolicyWithValue(nn.Module):
    """The SFT language model plus a tiny linear value head V(s_t) that reads
    the same last hidden layer the LM head reads."""

    def __init__(self, path):
        super().__init__()
        self.lm = load_causal_lm(path)
        self.value_head = nn.Linear(self.lm.config.hidden_size, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.to(DEVICE)


def response_stats(lm, full_ids, full_mask, P, value_head=None):
    """For the generated part only (positions P .. end):
         logp[t]    = log pi(a_t | s_t)
         entropy[t] = H(pi(. | s_t))
         values[t]  = V(s_t)
    The hidden state at position P-1+t is the one whose logits predict a_t,
    so that's also where we read V(s_t) (manual 4.2).
    We apply the LM head only to these ~48 positions -> big memory saving."""
    h = lm_forward(lm.model, full_ids, full_mask).last_hidden_state[:, P - 1:-1]  # (B, G, H)
    logits = lm.lm_head(h).float()
    logp = gather_logprobs(logits, full_ids[:, P:])
    log_z = torch.logsumexp(logits, -1)
    entropy = log_z - (torch.softmax(logits, -1) * logits).sum(-1)
    values = value_head(h.float()).squeeze(-1).float() if value_head is not None else None
    return logp, entropy, values


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1)


def main():
    set_seed()
    tok = load_tokenizer()
    sft_train, _ = load_sft_splits(tok)
    pad = tok.pad_token_id

    policy = PolicyWithValue(f"{CKPT}/sft")
    ref = freeze(load_causal_lm(f"{CKPT}/sft"))
    rm = freeze(RewardModel.load(f"{CKPT}/reward"))
    frozen_msg = f"frozen check: reference all frozen = {all_frozen(ref)}, reward model all frozen = {all_frozen(rm)}"
    print(frozen_msg)
    assert all_frozen(ref) and all_frozen(rm)

    opt = torch.optim.AdamW(policy.parameters(), lr=LR)

    order = list(range(len(sft_train)))
    random.Random(SEED).shuffle(order)

    LOG = f"{RESULTS}/ppo_log.jsonl"
    done, prev_min = load_resume("ppo", {"policy": policy}, opt)
    # We keep the whole log in memory and rewrite the file every iteration.
    # (200 quick appends to a Google-Drive file got lost in our first run.)
    history = read_jsonl(LOG)[:done] if done and os.path.exists(LOG) else []
    checks = [frozen_msg]
    start = time.time() - prev_min * 60

    for it in range(done, args.iterations):
        # ================= 1. ROLLOUT =================
        idx = order[it * args.prompts_per_rollout:(it + 1) * args.prompts_per_rollout]
        prompts = collate_prompts([sft_train[i] for i in idx], pad)
        P = prompts["input_ids"].shape[1]
        policy.eval()
        full_ids, full_mask, gen_mask, _ = generate_summaries(policy.lm, tok, prompts, greedy=False)
        full_mask = full_mask.to(DEVICE)
        m = gen_mask.float().to(DEVICE)                     # (B, G) 1 on real summary tokens

        # ================= 2. STORE (no gradients) =================
        with torch.no_grad():
            old_logp, _, old_values = response_stats(policy.lm, full_ids, full_mask, P, policy.value_head)
            ref_logp, _, _ = response_stats(ref, full_ids, full_mask, P)
            score = rm(full_ids, full_mask)                 # (B,) read at last real token (the EOS)

            # ================= 3. PER-TOKEN REWARD =================
            kl = (old_logp - ref_logp) * m
            rewards = -BETA_KL * kl
            last = (m * torch.arange(m.shape[1], device=DEVICE)).argmax(1)  # index T of each summary
            rewards[torch.arange(m.shape[0]), last] += score

            # ================= 4. RETURNS & ADVANTAGES =================
            returns = torch.zeros_like(rewards)
            running = torch.zeros(m.shape[0], device=DEVICE)
            for t in reversed(range(m.shape[1])):
                running = rewards[:, t] + GAMMA * running * m[:, t]  # padding never carries reward
                returns[:, t] = running
            returns = returns * m
            adv = returns - old_values
            mu = masked_mean(adv, m)
            std = torch.sqrt(masked_mean((adv - mu) ** 2, m)) + 1e-8
            adv = ((adv - mu) / std) * m

        # ================= 5. K EPOCHS OF PPO UPDATES =================
        policy.train()
        n_tok = m.sum()
        B = m.shape[0]
        step = max(1, B // args.chunks)
        stats = {"pg_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        first_epoch_max_ratio_dev = 0.0
        first_epoch_clip = 0.0
        for epoch in range(K_EPOCHS):
            for s in range(0, B, step):
                sl = slice(s, s + step)
                new_logp, ent, values = response_stats(policy.lm, full_ids[sl], full_mask[sl], P, policy.value_head)
                mm = m[sl]
                ratio = torch.exp(new_logp - old_logp[sl])
                unclipped = ratio * adv[sl]
                clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[sl]
                # Sums divided by the token count of the WHOLE rollout, so two
                # halves add up to exactly the full-batch mean.
                pg = -(torch.min(unclipped, clipped) * mm).sum() / n_tok
                vf = (((values - returns[sl]) ** 2) * mm).sum() / n_tok
                h = (ent * mm).sum() / n_tok
                loss = pg + C_V * vf - C_H * h
                loss.backward()

                with torch.no_grad():
                    outside = (((ratio < 1 - CLIP_EPS) | (ratio > 1 + CLIP_EPS)).float() * mm).sum() / n_tok
                    stats["pg_loss"] += pg.item() / K_EPOCHS
                    stats["value_loss"] += vf.item() / K_EPOCHS
                    stats["entropy"] += h.item() / K_EPOCHS
                    stats["clip_frac"] += outside.item() / K_EPOCHS
                    if epoch == 0:
                        dev = ((ratio - 1).abs() * mm).max().item()
                        first_epoch_max_ratio_dev = max(first_epoch_max_ratio_dev, dev)
                        first_epoch_clip += outside.item()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # ================= 6. LOG =================
        rec = {
            "iteration": it + 1,
            "mean_score": score.mean().item(),
            "mean_kl": kl.sum(1).mean().item(),          # summed over tokens, averaged over batch
            "mean_length": m.sum(1).mean().item(),
            "clip_frac": stats["clip_frac"],
            "pg_loss": stats["pg_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "epoch0_max_abs_ratio_minus_1": first_epoch_max_ratio_dev,
            "epoch0_clip_frac": first_epoch_clip,
            "minutes": (time.time() - start) / 60,
        }
        history.append(rec)
        with open(LOG, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in history))
        if (it + 1) % 10 == 0:                     # save every 10 iterations for resume
            save_resume("ppo", it + 1, {"policy": policy}, opt, rec["minutes"])
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        if it == 0:
            checks.append(f"iteration 1: mean KL to reference = {rec['mean_kl']:.6f} (should be 0)")
        if it < 3 or it == args.iterations - 1:
            checks.append(f"iteration {it + 1}: epoch-0 max |ratio-1| = {first_epoch_max_ratio_dev:.2e}, "
                          f"epoch-0 clip fraction = {first_epoch_clip:.3f} (should be ~0 and 0)")

    minutes = (time.time() - start) / 60
    os.makedirs(f"{CKPT}/ppo", exist_ok=True)
    policy.lm.save_pretrained(f"{CKPT}/ppo")          # policy only, value head is thrown away
    tok.save_pretrained(f"{CKPT}/ppo")
    clear_resume("ppo")
    with open(f"{RESULTS}/check_ppo.txt", "a" if done else "w", encoding="utf-8") as f:
        f.write("\n".join(checks) + "\n")
    append_jsonl(f"{RESULTS}/stage_times.jsonl", {"stage": "PPO", "minutes": minutes,
                                                  "final_train_loss": rec["pg_loss"] + C_V * rec["value_loss"] - C_H * rec["entropy"],
                                                  "final_mean_score": rec["mean_score"],
                                                  "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"})
    print(f"done in {minutes:.1f} min -> checkpoints/ppo/")


if __name__ == "__main__":
    main()
