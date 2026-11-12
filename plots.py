"""
Section 4.5 - all training figures, drawn from the jsonl logs.

    python plots.py   ->  results/figures/*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import read_jsonl, RESULTS

FIG = f"{RESULTS}/figures"
os.makedirs(FIG, exist_ok=True)


def load(name):
    path = f"{RESULTS}/{name}"
    return read_jsonl(path) if os.path.exists(path) else []


def col(rows, k):
    return [r[k] for r in rows]


def smooth(x, w=10):
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


# 1. SFT
sft = load("sft_log.jsonl")
if sft:
    plt.figure(figsize=(6, 4))
    plt.plot(col(sft, "step"), col(sft, "train_loss"), "o-", label="train loss (batch)")
    plt.plot(col(sft, "step"), col(sft, "val_loss"), "s-", label="validation loss")
    plt.xlabel("step"); plt.ylabel("NLL per summary token"); plt.title("SFT")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{FIG}/1_sft_loss.png", dpi=150); plt.close()

# 2. Reward model
rw = load("reward_log.jsonl")
if rw:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(col(rw, "step"), col(rw, "train_loss"), "o-"); ax[0].set_title("Reward model: train loss")
    ax[0].axhline(np.log(2), ls="--", c="gray", label="log 2 (guessing)"); ax[0].legend()
    ax[1].plot(col(rw, "step"), np.array(col(rw, "val_acc")) * 100, "o-", c="C2")
    ax[1].axhline(50, ls="--", c="gray", label="chance"); ax[1].legend()
    ax[1].set_title("Reward model: val pairwise accuracy (%)")
    for a in ax: a.set_xlabel("step"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/2_reward_model.png", dpi=150); plt.close()

# 3. PPO
ppo = load("ppo_log.jsonl")
if ppo:
    it = col(ppo, "iteration")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for a, k, t in [(ax[0, 0], "mean_score", "mean reward-model score"),
                    (ax[0, 1], "mean_kl", "mean KL to reference"),
                    (ax[1, 0], "mean_length", "mean summary length (tokens)"),
                    (ax[1, 1], "clip_frac", "clip fraction")]:
        a.plot(it, col(ppo, k), alpha=.3)
        s = smooth(col(ppo, k))
        a.plot(it[len(it) - len(s):], s, lw=2)
        a.set_title(t); a.set_xlabel("PPO iteration"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/3_ppo_curves.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for a, k in zip(ax, ["pg_loss", "value_loss", "entropy"]):
        a.plot(it, col(ppo, k), alpha=.4); a.set_title(k); a.set_xlabel("iteration"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/3b_ppo_losses.png", dpi=150); plt.close()

# 4. DPO
dpo = load("dpo_log.jsonl")
if dpo:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(col(dpo, "step"), col(dpo, "train_loss"), "o-", label="train")
    ax[0].plot(col(dpo, "step"), col(dpo, "val_loss"), "s-", label="validation")
    ax[0].axhline(np.log(2), ls="--", c="gray", label="log 2 (start)"); ax[0].legend()
    ax[0].set_title("DPO loss")
    ax[1].plot(col(dpo, "step"), np.array(col(dpo, "val_implicit_acc")) * 100, "o-", c="C2")
    ax[1].axhline(50, ls="--", c="gray"); ax[1].set_title("DPO: val implicit reward accuracy (%)")
    for a in ax: a.set_xlabel("step"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/4_dpo.png", dpi=150); plt.close()

print("figures saved in", FIG)
