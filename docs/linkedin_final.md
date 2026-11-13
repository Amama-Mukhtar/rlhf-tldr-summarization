We rebuilt the RLHF recipe behind ChatGPT from scratch, on a 135M-parameter model and a free GPU. 🧠

For our Generative AI course at FAST NUCES, Zeenat Ijaz and I trained SmolLM2-135M to summarize Reddit posts in four stages, with every loss written by hand in PyTorch (no TRL, no shortcuts):

1️⃣ Supervised fine-tuning on 20k human summaries
2️⃣ A reward model trained on 20k human preference pairs (63% accuracy, chance is 50%)
3️⃣ PPO with a KL penalty (reward score 0 → 1.85)
4️⃣ DPO: the same goal with no reward model at all (61% implicit accuracy)

Biggest lesson: a higher reward score isn't always a better summary. Our PPO model learned that the judge likes longer summaries and started writing longer ones. That's reward hacking, live on a T4.

📝 Blog: [Medium link]
💻 Code: https://github.com/Amama-Mukhtar/rlhf-tldr-summarization

#GenerativeAI #RLHF #MachineLearning #LLM #PPO #DPO #FASTNUCES
