# Task 7 – Analysis

> Fill every `[...]` with your own numbers after the runs. The reasoning parts are
> already written — read them, make sure you agree, and rephrase in your own words.

## 7.1 How accurate is the reward model, and what does it get wrong?

Validation pairwise accuracy: **[..]%** on 1,000 pairs (chance = 50%).

`results/reward_val_scores.jsonl` holds every validation pair with both scores. Sort by
`margin` and read the most negative ones. Three examples it gets wrong:

1. [chosen] vs [rejected] — margin [..]. *Why:* [e.g. both summaries say the same thing in different words; the human preference is close to a coin flip]
2. [...] *Why:* [e.g. the rejected one is longer and more detailed — the model has learned "longer = better"]
3. [...] *Why:* [e.g. the rejected one has a factual mistake (wrong gender/age) that a 135M model can't spot]

Typical patterns: near-duplicate pairs, length bias, factual errors that need real
understanding of the post, and noisy labels (human labellers also disagree ~25–30% of the time in the paper).

## 7.2 PPO training curves

![PPO](figures/3_ppo_curves.png)

- Mean r_φ went from [..] to [..] — [rose / rose then flattened].
- Mean KL grew from 0 to [..] nats, slowly, because of the β_KL penalty.
- Mean length went from [..] to [..] tokens.
- **Is a higher r_φ the same as a better summary?** No, not necessarily. In our human
  eval PPO was picked [..]/30 and [..]/30 times vs SFT [..] and [..]. ROUGE-L
  moved from [..] to [..]. The policy optimises the reward model, which is an imperfect
  proxy (it has [..]% accuracy), so part of the reward gain is the policy exploiting the
  judge's quirks (e.g. length) rather than becoming more useful. That's Goodhart's law —
  "reward hacking" / over-optimisation, exactly what the paper shows in its Fig. 5.

## 7.3 What does β_KL do? What if β_KL = 0?

Eq. 41 maximises `E[r_φ(x,y)] − β·KL(π_θ || π_ref)`. In our per-token reward every token
pays `−β(log π_old − log π_ref)`, so moving away from SFT costs reward.

β_KL is a leash. It keeps the policy close to the SFT model, where (a) the text is still
fluent and (b) the reward model is still accurate, because it was trained on summaries that
look like SFT's.

With β_KL = 0 the only signal left is r_φ. I'd expect the score to rise faster and higher, the KL to
grow without limit, and the policy to find whatever the reward model over-rates: very long
or very short outputs, repeated phrases, odd tokens, the same generic template for every post.
Entropy would collapse (mode collapse), and human-judged quality would drop even while r_φ goes
up. That's classic reward hacking.

## 7.4 Clip fraction

Epoch 1 of every iteration: clip fraction exactly 0 (ratio = 1, policy = old policy).
Averaged over the 4 epochs it was around [..] (max [..]).

So only a small share of tokens had their probability changed by more than ±20%, which means
each update is small and controlled, and that's the point of the clip. If it were large (>0.3) the
learning rate or K would be too big. If it stayed ~0 forever the policy would barely be learning.

## 7.5 Where did DPO's reward go?

- Eq. 55: the optimal policy for the KL-constrained objective is
  `π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)`.
- Eq. 56–57: solve it for r: `r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)`.
  So **the reward is hidden inside the policy itself**. Any policy defines an implicit reward
  `β log(π_θ/π_ref)`.
- Eq. 58: plug that into Bradley–Terry. It only uses the *difference* `r(x,y_w) − r(x,y_l)`.
  Both summaries share the same prompt x, so `β log Z(x)` appears twice and cancels. That's why
  we never need Z(x), which would be a sum over every possible summary and can't be computed.

Accuracy on the same 1,000 validation pairs: DPO implicit reward [..]% vs reward model [..]%.
[Comment: e.g. "close / lower because DPO's lr is tiny (2e-6) and β·log-ratio starts at 0", or
"similar, because both are basically the same classifier trained on the same pairs".]

## 7.6 Cost and choice

| | PPO | DPO |
|---|---|---|
| Models in GPU memory | 3 transformers (policy + value head, frozen reference, frozen reward model), and the reward model had to be trained first | 2 (policy, frozen reference) |
| Time | [..] min (+ [..] min for the reward model) | [..] min |
| Generation during training | yes (slow, 48 sampled tokens × 16 every iteration) | no |
| Hyper-parameters | many (ε, β, c_V, c_H, K, γ) | one (β) |

**Our choice for a new project:** DPO, because it's simpler, cheaper, more stable and has
fewer moving parts, and it reached [similar] quality here. PPO is worth it when you need
on-policy exploration, a reusable reward model (e.g. to score many new outputs, or for
best-of-n), or a reward that isn't a pairwise preference (tests passing, a math checker).
