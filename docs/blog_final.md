# We Rebuilt the RLHF Recipe Behind ChatGPT on a 135M Model and a Free GPU

*SFT, a reward model, PPO and DPO, every loss written by hand in PyTorch.*

A language model can continue any piece of text, but nobody ever told it what a *good* summary looks like. In 2020, OpenAI's paper *Learning to Summarize from Human Feedback* showed how to teach a model that from human preferences. The same recipe later powered ChatGPT.

For our Generative AI course, we rebuilt that pipeline on a model small enough for a free Colab T4: **SmolLM2-135M**. The task is to read a Reddit post and write a one-line TL;DR. We used no RLHF library (no TRL, no PEFT, no Trainer). Every loss is our own PyTorch code.

## The pipeline in four stages

**1. Supervised fine-tuning (SFT).** We showed the model 20,000 posts with human-written summaries and trained it to predict the summary, one token at a time. The trick is that the loss only counts the *summary* tokens. The post is given, so we don't grade the model on it.

**2. Reward model.** We took the SFT model, removed its vocabulary head and added a single linear layer that outputs one number: a score for (post, summary). We trained it on 20,000 pairs where a human said "summary A is better than B", using the Bradley–Terry loss, `−log σ(r(A) − r(B))`. The score is read at the last real token, because in a causal model only that token has "seen" the whole post and summary.

**3. PPO.** Now the model practises. It writes summaries for 16 posts, the reward model grades them, and the model is nudged toward the tokens that scored better than expected. Two safety belts stop it from going off the rails. A **KL penalty** charges every token a small fine for drifting away from the SFT model. The **clipped objective** stops any token's probability from changing by more than ±20% in one update.

**4. DPO.** This reaches the same goal with a shortcut. DPO never trains a reward model. It directly makes the preferred summary more likely, relative to a frozen copy of the SFT model, than the rejected one. The maths shows the reward is hidden inside the policy itself, `β·log(π/π_ref)`, and the awkward normaliser Z(x) cancels out because both summaries share the same post.

## Results

| Stage | Result | Time on a T4 |
|---|---|---|
| SFT | validation loss 2.488 → **2.445** | 71 min |
| Reward model | pairwise accuracy 51% → **63.1%** (chance = 50%) | 103 min |
| PPO | mean reward-model score ≈ 0 → **1.85** | 47 min |
| DPO | loss 0.693 → 0.627, implicit reward accuracy **61.1%** | 186 min |

*(Figure 1: SFT loss, `results/figures/1_sft_loss.png`)*

*(Figure 2: reward-model loss and accuracy, `results/figures/2_reward_model.png`)*

*(Figure 3: DPO loss and implicit accuracy, `results/figures/4_dpo.png`)*

Every sanity check from the assignment passed. The KL to the reference was exactly 0 at the first PPO iteration. The PPO ratio was 1 in the first epoch of every update. The DPO loss started at exactly log 2 = 0.6931, because the policy and the reference are identical at step one.

## The most interesting part: reward hacking

PPO did what we asked: the reward-model score went up. But look at *how* it went up. By iteration 100, summaries had grown from about 35 to 44–47 tokens (the limit is 48), and the KL to the SFT model had climbed to 30–40 nats.

Our reward model is only 63% accurate, and one of the things it learned is that longer usually means better. PPO found that shortcut and exploited it. This is Goodhart's law in action: when a measure becomes a target, it stops being a good measure. A higher reward-model score is **not** the same thing as a better summary. That's exactly why the KL penalty exists. Without it (β = 0), nothing would stop the policy from drifting into whatever the judge over-rates.

## PPO vs DPO

| | PPO | DPO |
|---|---|---|
| Models on the GPU | 3 (policy + value head, frozen reference, frozen reward model) | 2 (policy, frozen reference) |
| Needs a trained reward model first | yes | no |
| Generates text during training | yes | no |
| Hyper-parameters to tune | many | basically β |

DPO's implicit reward accuracy (61%) came close to the dedicated reward model's (63%) on the same validation pairs. For a new project we would start with DPO: it's simpler, cheaper and as stable as supervised learning. PPO is worth its complexity when you need on-policy exploration or a reusable reward model.

## What actually went wrong (and what we learned)

- **A silent bfloat16 bug.** New versions of `transformers` load SmolLM2 in bfloat16. In pure bf16, an update of size 2e-6 simply rounds to zero, so our first DPO run was barely learning. Casting every model to float32 fixed it.
- **Colab disconnects.** A power cut taught us to save resumable checkpoints (model, optimizer and step) during training, not just at the end.
- **Lost logs.** Hundreds of quick appends to a Google Drive file kept only the first few lines. We switched to rewriting the whole log every iteration.

## Try it

We wrapped the three models in a small Gradio app. Paste a Reddit post and see the SFT, PPO and DPO summaries side by side, each with its reward-model score.

**Code and README:** https://github.com/Amama-Mukhtar/rlhf-tldr-summarization

*Amama Mukhtar & Zeenat Ijaz, FAST NUCES*
