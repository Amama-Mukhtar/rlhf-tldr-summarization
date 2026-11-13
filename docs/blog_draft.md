# I Trained a 135M Model to Summarize Reddit With Human Feedback (SFT, Reward Model, PPO and DPO, All From Scratch)

*Medium draft, 800–1500 words. Replace every [..] with your numbers. Needs at least 2 figures.*

## Why a model needs to be "aligned" at all
A language model can continue any text, but "write a good TL;DR" isn't something it knows how to do.
In 2020 OpenAI showed you can teach it this from human *preferences*. The same recipe later powered ChatGPT.
We rebuilt that pipeline on a model small enough for a free Colab GPU: SmolLM2-135M.

## The pipeline in four stages
(Figure: a simple diagram, SFT → Reward model → PPO, and SFT → DPO)
1. **SFT:** 20k Reddit posts with human summaries. Loss only on the summary tokens.
2. **Reward model:** the SFT body plus a single linear layer that outputs a score. Trained on 20k
   "A is better than B" pairs with the Bradley–Terry loss. Accuracy: [..]% (chance is 50%).
3. **PPO:** the model writes, the reward model grades, and a KL penalty keeps it close to SFT. 200 iterations.
4. **DPO:** the same goal with no reward model at all, learned directly from the pairs.

## What happened during PPO
(Figure: results/figures/3_ppo_curves.png)
The reward went from [..] to [..], KL crept up to [..], and summaries got [longer/shorter].

## Results
(Table 3 here)
(Table 4, human eval, here)

## Where the methods differ: real examples
(2–3 examples from samples.md)

## What we learned
- A higher reward-model score ≠ a better summary (Goodhart).
- β_KL is the leash that stops reward hacking.
- DPO hides the reward inside the policy, so Z(x) cancels.
- PPO needed 3 models on the GPU and [..] min; DPO needed 2 and [..] min.

Code: [GitHub link] · Checkpoints: [link]
