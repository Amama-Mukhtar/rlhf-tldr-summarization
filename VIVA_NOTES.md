# Viva notes: explain it like you built it (because you did)

Read this once slowly, then again the night before. Every section has
**"say it in one breath"** (the 10-second answer), then the details, then
**"where in our code"** so you can open the file and point.

---

## 0. The 30-second story of the whole project

> "A pretrained model can continue text, but it doesn't know what a *good* summary is.
> So first we **show** it 20k human summaries (SFT). Then we train a **judge**, the reward model,
> on 20k pairs where a human said 'this summary is better than that one'. Then we let the
> model **practise** with PPO: it writes summaries, the judge scores them, and it learns to
> score higher, with a KL leash so it doesn't drift too far from SFT. Finally we do the same
> thing a shortcut way with DPO, which learns straight from the pairs without any judge. At
> the end we compare SFT, PPO and DPO on 300 posts none of them saw."

Same base model everywhere: **SmolLM2-135M**, fully fine-tuned, on a free T4.

---

## 1. The data (Task 1)

**Say it in one breath:** every prompt is `SUBREDDIT / TITLE / POST` + `\nTL;DR:`, every
summary starts with a space and ends with EOS, and posts >448 tokens or summaries >48 tokens are dropped.

- SFT: 20,000 train / 500 val from `openai_summarize_tldr`
- Preferences: 20,000 train / 1,000 val from `openai_summarize_comparisons` (prompt, chosen, rejected)
- Eval: 300 posts from the tldr **test** split, never trained on

### The one helper everything uses: `token_logprobs`

A causal LM's logits at position t-1 predict token t. So:

```
logits[:, :-1]   lines up with   input_ids[:, 1:]
```

and we pick out the log-prob of the actual next token. Output shape is **(B, L-1)**, and
column j is the log-prob of token j+1.

**`summary_mask`** is 1 where token j is (a) at or after the prompt and (b) a real token per
the attention mask. Then shift by one so it lines up with `token_logprobs`.

**Classic trick question:** *"Why not find padding with `input_ids != pad_token_id`?"*
→ Because in this tokenizer **pad = EOS**. Every summary ends in a real EOS that we must train
on. Comparing ids would throw it away. So we always use the attention mask.

**Where:** `common.py` → `token_logprobs`, `summary_mask`. Check output: `results/check_summary_mask.txt`.

**Position ids, a detail examiners like:** with left padding, row 2 might start at index 5.
`generate()` quietly computes position ids from the attention mask (first real token = position 0).
We do the same in *every* forward (`position_ids_from_mask`), so the model sees the same thing
during rollout and during the PPO update. Without this, the KL at PPO iteration 1 wouldn't be 0
for left-padded rows.

---

## 2. SFT (Task 2)

**Say it in one breath:** plain next-token prediction, but we only grade the summary tokens.

```
L_SFT = - (1 / #summary tokens) · Σ_{t in summary} log π(x_t | x_<t)
```

- *Why not the prompt tokens too?* We don't want it to learn to write Reddit posts; we want
  it to learn "given a post, write a summary". The post is given, it's not the job.
- Settings: 1 epoch, AdamW 5e-5, batch 16, grad clip 1.0, log every 250 steps.
- **Micro-batches:** we may split 16 into 2×8 to fit memory. We sum the loss and divide by the
  token count of the *whole* batch, so the gradient is identical to one batch of 16.
- Before SFT the base model rambles or keeps writing a Reddit post; after SFT it writes a short
  TL;DR and stops (it learned to emit EOS). See `results/sft_samples.md`.

**Where:** `train_sft.py` → `sft_loss_sum`.

---

## 3. Reward model (Task 3)

**Say it in one breath:** take the SFT transformer, throw away the vocabulary head, put a
`Linear(hidden, 1)` on top, and read it at the **last real token**. That gives one number per (post, summary).

**Why start from SFT?** It already understands posts and summaries, so the judge starts out knowing the language.

**Loss: Bradley–Terry (Eq. 40)**
```
L = - mean log σ( r(x, y_w) − r(x, y_l) )
```
Plain English: the probability the human prefers y_w is modelled as σ(score difference).
We maximise the likelihood of the human's choices. Only the **difference** matters, so adding
a constant to every score changes nothing. That's why reward scores have no absolute meaning
and can be negative.

- Starts at loss ≈ log 2 = 0.693 (random guessing), and accuracy ≈ 50%.
- Metric: pairwise accuracy = share of val pairs where r(chosen) > r(rejected).
- **Last real token**: `(arange(L) * attention_mask).argmax()`. Works for left- and right-padding.
- Why the last token? Causal attention means only the last token has "read" the whole post + summary.

**Expect** something like 60–70% for a 135M model. Clearly above chance, well below the paper's big models.

**Where:** `train_reward.py` → `RewardModel.forward`, Bradley–Terry line in the training loop.

---

## 4. PPO (Task 4), the big one

**Say it in one breath:** the policy writes summaries, the judge scores them, every token pays
a small fine for drifting from SFT, and we nudge the policy toward high-advantage tokens, but
never more than ±20% per step.

### RL vocabulary in our setting
- **State** s_t = post + tokens written so far
- **Action** a_t = next token
- **Episode** = one summary; ends at the first EOS (or 48 tokens)
- **Reward** arrives mostly at the end (the judge's score)

### Three/four models in memory
| Model | From | Trained? |
|---|---|---|
| Policy π_θ + value head V_θ | SFT + new Linear | yes |
| Reference π_ref | SFT | frozen |
| Reward model r_φ | reward ckpt | frozen |

### One iteration, step by step (point at the code!)

1. **Rollout.** 16 prompts, left-padded, *sampled* (temp 1, top_k 0, top_p 1, so pure sampling).
   Why sample and not greedy? RL needs exploration. Greedy gives the same answer every time, so
   nothing new is learned.
2. **Store, no gradients:** `log π_old(a_t|s_t)`, `log π_ref(a_t|s_t)`, `V_old(s_t)`, and `r_φ(x, y)`.
   V(s_t) is read at the same position whose logits predict a_t (position P-1+t).
3. **Per-token reward (Eq. 41):**
   ```
   r_t = −β_KL · (log π_old − log π_ref)  +  1[t = T] · r_φ(x, y)
   ```
   Every token pays a KL fine, and the last token also gets the judge's score.
4. **Returns and advantages:** γ = 1, so `R_t` = sum of all rewards from t to the end.
   `A_t = R_t − V_old(s_t)` asks "was this better or worse than expected?" Then normalise A
   over all tokens in the batch (mean 0, std 1) for stable step sizes.
5. **Update, K = 4 epochs on the same rollout:**
   ```
   ratio = π_θ / π_old = exp(new_logp − old_logp)
   L_clip = E[ min(ratio·A, clip(ratio, 0.8, 1.2)·A) ]
   L = −L_clip + 0.1·E[(V − R)²] − 0.01·E[H]
   ```
   - **min + clip:** if A > 0 we can raise that token's prob by at most 20%; if A < 0 we lower it
     by at most 20%. Beyond that the gradient is zero. This is the "trust region" in cheap form.
   - **value loss:** teaches V to predict returns, so advantages get better.
   - **entropy bonus (minus sign):** keeps the policy from collapsing to one answer too fast.
   - Every E[·] is a mean over *real generated tokens*, not padding.

### Logged every iteration
mean r_φ, mean KL (sum over summary tokens, mean over batch), mean length, clip fraction,
pg loss, value loss, entropy.

### Sanity checks (be ready to explain *why* each holds)
- **Iteration 1, KL = 0:** policy == SFT == reference, and the same tokens go through the same code.
- **Epoch 1 of every iteration, ratio = 1, clip fraction = 0:** we haven't stepped yet, so π_θ = π_old.
- **KL slightly negative early?** Normal. It's a *sample* estimate (log-ratio of the sampled
  token), not the exact KL, so one batch can come out below 0.

### Questions they love
- *Why a value head?* Without a baseline, raw returns have huge variance. `R − V` centres them.
- *Why γ = 1?* Episodes are short (≤48) and we care about the whole summary equally.
- *Why reuse a rollout for 4 epochs?* Generation is the expensive part, so we squeeze more learning
  out of it. The clip is what makes reusing it safe.
- *KL explodes, what do you check?* That the reference is frozen, and that the sign is right
  (it's a *penalty*: `−β·KL`).
- *Out of memory?* `--chunks 2` splits the 16 rollouts into two halves and **adds the gradients**
  before `optimizer.step()`. Each half divides by the token count of the *whole* rollout, so the result is exact.
- *Why apply lm_head only to the 48 response positions?* The vocab is 49k. Logits for 500
  positions × 16 rows is gigabytes. We only need the summary positions.

**Where:** `train_ppo.py`, where the numbered comments 1–6 match the steps above.

---

## 5. DPO (Task 5)

**Say it in one breath:** same goal as PPO, but we skip the judge. We directly make the chosen
summary more likely, *relative to the reference*, than the rejected one.

```
L_DPO = −E log σ( β·[log π(y_w|x)/π_ref(y_w|x)] − β·[log π(y_l|x)/π_ref(y_l|x)] )
```

- `log π(y|x)` = **sum** of token log-probs over summary tokens only (`sequence_logprob`).
- β = 0.1, lr 2e-6 (tiny!), batch 8 pairs, 1 epoch.
- **Implicit reward** = `β · log(π/π_ref)`. "Implicit reward accuracy" = share of val pairs where
  the thing inside σ is > 0.
- **First step loss = log 2:** π = π_ref, so both log-ratios are 0, σ(0) = 0.5, and −log 0.5 = 0.693.

### Where did the reward go? (Task 7.5, very likely question)
1. The best policy for "max reward − β·KL" has a closed form: `π* = π_ref · exp(r/β) / Z(x)`.
2. Flip it around: `r = β log(π*/π_ref) + β log Z(x)`. **The policy *is* the reward model.**
3. Put that into Bradley–Terry, which only needs `r(y_w) − r(y_l)`. Same prompt x, same Z(x),
   so it **cancels**. Z(x) would be a sum over every possible summary, which is impossible to
   compute, and we never need it.

**Where:** `train_dpo.py` → `margins()`.

---

## 6. Evaluation (Task 6 + Section 3)

On 300 unseen posts, greedy, ≤48 tokens:

| Metric | What it means |
|---|---|
| Mean r_φ | how much our judge likes it |
| Win rate vs SFT | % of posts where its r_φ beats SFT's r_φ on the same post |
| KL to SFT | Σ_t (log π − log π_sft) on its *own* greedy summary: how far it moved |
| Length | tokens |
| ROUGE-L | word-overlap with the human reference (longest common subsequence F1) |

**Human eval:** 30 posts, three summaries shuffled as A/B/C (no names), each member picks the best
on their own, then count picks per model + agreement.

**Expected pattern and how to explain it:** PPO raises r_φ while ROUGE-L can *fall* and length
changes. That's not a bug. PPO optimises the judge, not the human reference. ROUGE only rewards
copying the reference's words.

---

## 7. Comparing PPO vs DPO (Task 7.6)

| | PPO | DPO |
|---|---|---|
| Models on GPU | 3 (policy+value, ref, RM), plus the RM has to be trained first | 2 (policy, ref) |
| Sampling during training | yes (slow) | no |
| Hyper-parameters | many | basically β |
| Stability | finicky | like supervised learning |
| Exploration / reusable judge | yes | no |

**Our pick:** DPO for a new project (cheaper, simpler, stable); PPO when we need on-policy
exploration or a reward that isn't a preference pair.

---

## 8. Rapid-fire Q&A (practise out loud)

- **What is RLHF?** Fine-tune a model with RL using a reward learned from human preferences.
- **Why not just do more SFT?** SFT copies one reference; it can't use "A is better than B", and it
  can't tell good from bad, only "like the reference".
- **Why preferences instead of scores 1–10?** Comparing two is easier and more consistent for humans.
- **What is σ?** Sigmoid, 1/(1+e^−z). Turns a score difference into a probability.
- **Why is π_ref frozen?** It's the anchor for the KL. If it moved, the leash would move with the dog.
- **What does β_KL = 0 do?** Reward hacking: KL blows up, weird or long outputs, high r_φ, worse summaries.
- **What is clip fraction?** Share of tokens whose ratio left [0.8, 1.2]. Small means small, safe updates.
- **Advantage normalisation?** Mean 0 / std 1 over all tokens, so the step size doesn't depend on reward scale.
- **Why does the reward model use the SFT checkpoint and not the base?** It already understands summaries.
- **Left vs right padding?** Right for training (prompt starts at 0, easy masks); left for generation
  so every prompt's last token sits at the same column, where generation continues.
- **Why does the summary need EOS?** So the model learns to *stop*, and so the reward model reads a fixed "end" token.
- **Why forbid trl/peft?** The point is that we implement and understand the losses ourselves.
- **What's Goodhart's law here?** "When a measure becomes a target it stops being a good measure."
  r_φ is the measure, and PPO makes it the target.
- **Why is reward-model accuracy only ~65%?** Small model, only 20k pairs, noisy labels
  (humans themselves agree only ~70–75%), and many pairs are near-ties.

---

## 9. Who did what (fill in, both should know everything anyway)

- Member 1: [...]
- Member 2: [...]

Make sure **both** of you have commits in the repo (manual requirement).

---

## 10. OUR actual run: numbers and story (kept updated by Claude)

### Numbers to remember
| Stage | Result | Time (T4) |
|---|---|---|
| Data | SFT 20000/500, Pref 20000/1000, Eval 300. Prompt mean 317.4 / max 448 tokens, summary mean 33.8 / max 48 (incl. EOS) | – |
| SFT | val loss 2.488 (step 250) → 2.468 → 2.458 → 2.451 → **2.445** (step 1250). Final train loss 2.51 | 71 min |
| Reward model | val pairwise acc 51.2% before → 60.4% (250) → peak 65.3% (1750) → **63.1%** final. Final train loss 0.59 | 103 min |
| PPO | mean r_φ −0.02 (it 1) → ~1.4–2.0 (it 100), first run ended at **1.85** | 47 min (first run) |
| DPO | loss 0.693 → 0.627, val implicit reward acc **61.1%** (stays ~60% the whole run) | 186 min |

### Correctness checks, all passed
- Summary mask: all 4 rows `match=True`, decoded text ends with `<|endoftext|>`.
- Frozen: reference + reward model `requires_grad=False` → `True`.
- PPO iteration 1: KL = 0.000000.
- PPO epoch 1 of every iteration: max |ratio−1| ≈ 3e-05 (float error), clip fraction 0.000.
- DPO first step loss = 0.6931 = log 2.
- Reproducible: the PPO re-run gave exactly the same iterations 1–4 (−0.0247, 0.2848, −0.5401, 0.0999), because the seed is fixed.

### What we observed (use these in Task 7)
- **Reward hacking in PPO:** by iteration ~100 the length went 35 → 44–47 tokens (max 48), KL went 0 → 31–40 nats, and entropy rose 2.7 → 5.2. The reward model likes longer summaries, so PPO learned "write longer", not "write better". This is Goodhart's law: a higher r_φ is not the same as a better summary.
- **RM vs DPO accuracy (7.5):** RM 63.1% vs DPO implicit 61.1% on the same 1000 pairs. DPO is a bit lower: tiny lr (2e-6), and its implicit reward β·log(π/π_ref) starts at exactly 0.
- **Validation loss < train loss in SFT:** train loss is one batch at that moment; val is the average over 500 pairs.
- **GPU memory (7.6):** PPO needs 3 transformers (policy+value head, reference, reward model) plus the RM had to be trained first. DPO needs 2 (policy, reference).

### Engineering problems we hit (good viva stories, they show we really ran it)
1. **bfloat16 bug:** new transformers loads SmolLM2 in bf16 (from its config). The reward head was float32 → dtype error. Worse, in pure bf16 the tiny updates of lr 1e-5 / 2e-6 round to zero, so DPO barely learned. Fix: `.float()` on every model in `common.py` (starter files unchanged). SFT had already been trained in bf16; we kept it because its loss clearly went down.
2. **Colab disconnects (power/wifi):** checkpoints were only saved at the end. We added resume points (model + optimizer + step), written with a write-then-rename so a cut never leaves a broken file. A fixed DataLoader generator means a resume skips exactly the batches already seen.
3. **Lost PPO log:** 200 quick appends to a Google Drive file kept only 3 lines. Fix: keep the log in memory and rewrite the whole file every iteration.
4. **Resume saves were too heavy for PPO:** each save is ~1.6 GB (model + Adam state) to Drive. Every 10 iterations slowed PPO from 0.22 to 2–3 min/iteration and filled Drive. Fix: `--resume_every 0` for the final PPO run.
5. **Colab free GPU quota:** after many hours one account fell back to CPU (10 min per PPO iteration instead of 13 s), so we split the stages over two accounts.
