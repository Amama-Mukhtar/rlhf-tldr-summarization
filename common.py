"""
common.py - the small helpers every stage leans on.

If you only remember one file for the viva, remember this one. Everything
else (SFT, reward model, PPO, DPO, evaluation) is built from these pieces:

    token_logprobs   -> log pi(x_t | x_<t) for every position, shape (B, L-1)
    summary_mask     -> 1 exactly on the summary tokens, also shape (B, L-1)
    position_ids     -> correct positions even when a row is left-padded
    generate_summaries -> greedy or sampled summaries, cut at the first EOS
"""
import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

# The starter files import each other as "from config import ...", so the
# starter folder has to be on the path. We never edit those files.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from config import SEED, MAX_SUMMARY_TOKENS  # noqa: E402
from models import load_tokenizer  # noqa: E402,F401
from models import load_causal_lm as _starter_load_causal_lm  # noqa: E402


def load_causal_lm(name_or_path=None, device=None):
    """Starter loader + .float().

    New transformers versions load SmolLM2 in bfloat16 by default (its config
    says so). Training in pure bf16 is bad here: with lr 1e-5 / 2e-6 most
    weight updates are smaller than bf16 can represent and simply vanish, and
    the T4 has no fast bf16 anyway. So every model is kept in float32."""
    model = _starter_load_causal_lm(name_or_path, device) if name_or_path else _starter_load_causal_lm(device=device)
    return model.float()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def position_ids_from_mask(attention_mask):
    """Position of each token counting only real tokens.

    Right padding: this is just 0, 1, 2, ... (same as the default).
    Left padding:  the first real token gets position 0, not position 5.
    generate() already does this internally, so we do it everywhere else too,
    otherwise a left-padded prompt would look different in training vs. rollout.
    """
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)


def lm_forward(model, input_ids, attention_mask, **kw):
    """One forward pass of a causal LM with the right position ids."""
    return model(input_ids=input_ids,
                 attention_mask=attention_mask,
                 position_ids=position_ids_from_mask(attention_mask),
                 **kw)


def gather_logprobs(logits, targets):
    """log softmax(logits)[target] without building the full log-softmax
    tensor twice. logits: (B, T, V), targets: (B, T) -> (B, T)."""
    logits = logits.float()
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


def token_logprobs(model, input_ids, attention_mask):
    """Task 1.3. Returns log pi(x_t | x_<t) for every t >= 1.

    The logits at position t-1 predict token t, so we drop the last logit
    and the first token, and line them up. Output shape: (B, L-1).
    Column j of the output is the log-prob of token j+1.
    """
    logits = lm_forward(model, input_ids, attention_mask).logits
    return gather_logprobs(logits[:, :-1], input_ids[:, 1:])


def summary_mask(attention_mask, prompt_lens):
    """1 exactly on the summary tokens, aligned with token_logprobs (B, L-1).

    For right-padded prompt+summary rows, token j is a summary token when
    j >= prompt_len and it is not padding. We use the attention mask (never
    input_ids != pad_id) because pad == EOS in this tokenizer, and the EOS at
    the end of every summary is a real token we must keep.
    """
    B, L = attention_mask.shape
    j = torch.arange(L, device=attention_mask.device).unsqueeze(0)
    is_summary = (j >= prompt_lens.to(attention_mask.device).unsqueeze(1)) & (attention_mask == 1)
    return is_summary[:, 1:].float()  # shift by one to match token_logprobs


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def sequence_logprob(model, input_ids, attention_mask, prompt_lens):
    """log pi(y | x) = sum of token log-probs over the summary tokens only.
    Used by DPO and by the evaluation KL."""
    lp = token_logprobs(model, input_ids, attention_mask)
    return (lp * summary_mask(attention_mask, prompt_lens)).sum(-1)


def cut_at_eos(gen_ids, eos_id):
    """gen_ids: (B, G) tokens produced after the prompt.
    Returns a (B, G) mask that is 1 up to and INCLUDING the first EOS and 0
    after it. If a row never produced EOS (hit 48 tokens), all of it counts.
    """
    is_eos = (gen_ids == eos_id).long()
    eos_before = is_eos.cumsum(-1) - is_eos  # how many EOS came strictly before me
    return (eos_before == 0).long()


@torch.no_grad()
def generate_summaries(model, tok, prompt_batch, greedy=True, max_new_tokens=MAX_SUMMARY_TOKENS):
    """Generate from a LEFT-padded prompt batch (collate_prompts).

    Returns:
      full_ids   (B, P+G) prompt + generated tokens
      full_mask  (B, P+G) 1 on real prompt tokens and real summary tokens
      gen_mask   (B, G)   1 on the summary tokens (up to and incl. first EOS)
      texts      list[str] decoded summaries (EOS removed)
    """
    ids = prompt_batch["input_ids"].to(model.device)
    mask = prompt_batch["attention_mask"].to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                  eos_token_id=tok.eos_token_id)
    if greedy:
        kwargs.update(do_sample=False)
    else:  # exactly the PPO rollout settings from the manual
        kwargs.update(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
    out = model.generate(input_ids=ids, attention_mask=mask, **kwargs)

    P = ids.shape[1]
    gen = out[:, P:]
    gen_mask = cut_at_eos(gen, tok.eos_token_id)
    full_mask = torch.cat([mask, gen_mask], dim=1)
    texts = []
    for row, m in zip(gen, gen_mask):
        toks = row[m.bool()].tolist()
        if toks and toks[-1] == tok.eos_token_id:
            toks = toks[:-1]
        texts.append(tok.decode(toks))
    return out, full_mask, gen_mask, texts


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def all_frozen(model):
    return all(not p.requires_grad for p in model.parameters())


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Resume support (Colab can die on a power cut / wifi drop).
# Every few steps we save model + optimizer + step to Drive. If the script is
# started again it continues from there instead of from step 0.
# ---------------------------------------------------------------------------
def _resume_path(name):
    return os.path.join(CKPT, f"_resume_{name}.pt")


def save_resume(name, step, modules, opt, minutes):
    tmp = _resume_path(name) + ".tmp"
    torch.save({"step": step, "minutes": minutes, "opt": opt.state_dict(),
                "modules": {k: m.state_dict() for k, m in modules.items()}}, tmp)
    os.replace(tmp, _resume_path(name))       # write-then-rename, so a cut never leaves a broken file


def load_resume(name, modules, opt):
    """Returns (step_done, minutes_so_far). (0, 0) means a fresh start."""
    path = _resume_path(name)
    if not os.path.exists(path):
        return 0, 0.0
    state = torch.load(path, map_location=DEVICE)
    for k, m in modules.items():
        m.load_state_dict(state["modules"][k])
    opt.load_state_dict(state["opt"])
    print(f"RESUMING {name} from step {state['step']} ({state['minutes']:.1f} min already done)")
    return state["step"], state["minutes"]


def clear_resume(name):
    if os.path.exists(_resume_path(name)):
        os.remove(_resume_path(name))
