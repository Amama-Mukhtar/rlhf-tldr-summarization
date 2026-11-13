"""
Task 8.1 - front end. Paste a Reddit post, see the SFT, PPO and DPO summaries
side by side, each scored by our reward model.

    pip install gradio
    python app/app.py                       # uses ./checkpoints/{sft,ppo,dpo,reward}
    CKPT_DIR=/content/drive/MyDrive/ckpt python app/app.py
"""
import os
import sys

import gradio as gr
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import load_tokenizer, load_causal_lm, generate_summaries, freeze  # noqa: E402
from data import clean_post  # noqa: E402
from batching import collate_prompts  # noqa: E402
from train_reward import RewardModel  # noqa: E402

CKPT = os.environ.get("CKPT_DIR", os.path.join(ROOT, "checkpoints"))
tok = load_tokenizer()
print("loading checkpoints from", CKPT)
policies = {name: freeze(load_causal_lm(f"{CKPT}/{name}")) for name in ["sft", "ppo", "dpo"]}
rm = freeze(RewardModel.load(f"{CKPT}/reward"))


def summarise(subreddit, title, body):
    if not body.strip():
        return ["(paste a post first)", ""] * 3
    raw = f"SUBREDDIT: r/{subreddit.strip() or 'AskReddit'}\nTITLE: {title.strip()}\nPOST: {body.strip()}"
    prompt = clean_post(raw)
    ids = tok(prompt, add_special_tokens=False)["input_ids"][-448:]  # keep the end (it holds "TL;DR:")
    batch = collate_prompts([{"prompt_ids": ids}], tok.pad_token_id)
    outputs = []
    for name, model in policies.items():
        full_ids, full_mask, _, texts = generate_summaries(model, tok, batch, greedy=True)
        with torch.no_grad():
            score = rm(full_ids, full_mask.to(full_ids.device)).item()
        outputs += [texts[0].strip(), f"{score:+.3f}"]
    return outputs


EXAMPLE = ["relationships", "My (24F) boyfriend (26M) keeps forgetting plans we make",
           "We've been together two years. Lately he forgets almost every plan we make - dinner, "
           "movies, even my sister's birthday last week. When I bring it up he says he's stressed "
           "at work and I'm overreacting. I don't want to nag but I feel like I'm not a priority. "
           "How do I talk to him about this without starting a fight?"]

with gr.Blocks(title="RLHF TL;DR - SFT vs PPO vs DPO") as demo:
    gr.Markdown("## Learning to Summarize from Human Feedback\n"
                "SmolLM2-135M trained by us in four stages. Paste a Reddit post and compare the three policies. "
                "The number under each summary is our reward model's score (higher = it thinks humans would like it more).")
    with gr.Row():
        sub = gr.Textbox(label="Subreddit", value=EXAMPLE[0], scale=1)
        title = gr.Textbox(label="Title", value=EXAMPLE[1], scale=3)
    body = gr.Textbox(label="Post", value=EXAMPLE[2], lines=8)
    btn = gr.Button("Summarise", variant="primary")
    outs = []
    with gr.Row():
        for name in ["SFT", "PPO", "DPO"]:
            with gr.Column():
                gr.Markdown(f"### {name}")
                outs.append(gr.Textbox(label="Summary", lines=4))
                outs.append(gr.Textbox(label="Reward-model score"))
    btn.click(summarise, [sub, title, body], outs)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("SHARE", "0") == "1")
