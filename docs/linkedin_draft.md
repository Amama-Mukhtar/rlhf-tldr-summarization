We just rebuilt the RLHF pipeline behind ChatGPT from scratch, on a 135M-parameter model and a free GPU. 🧠

For our Generative AI course at FAST NUCES, [partner] and I trained SmolLM2-135M to summarize Reddit posts in four stages, every loss written by hand in PyTorch (no TRL, no shortcuts):
1️⃣ Supervised fine-tuning on human summaries
2️⃣ A reward model trained on 20k human preference pairs ([..]% accuracy)
3️⃣ PPO with a KL penalty
4️⃣ DPO, the same goal with no reward model at all

Biggest lesson: a higher reward score isn't always a better summary. Reward models can be gamed.

📝 Blog: [Medium link]
💻 Code + demo: [GitHub link]

#GenerativeAI #RLHF #MachineLearning #LLM #PPO #DPO
