"""
Task 5 - Stage 4: DPO (Direct Preference Optimisation).

Same goal as PPO, but no reward model, no sampling, no value head.
We push up the (policy / reference) ratio of the chosen summary and push down
the ratio of the rejected one:

    L_DPO = - mean log sigmoid( beta * [ log pi(y_w|x)/pi_ref(y_w|x)
                                        - log pi(y_l|x)/pi_ref(y_l|x) ] )

beta * log(pi/pi_ref) is the "implicit reward". If it is bigger for the
chosen summary than the rejected one, the policy "agrees" with the human.

    python train_dpo.py
"""
import argparse
import time
from functools import partial

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (load_tokenizer, load_causal_lm, sequence_logprob, freeze, all_frozen,
                    append_jsonl, set_seed, save_resume, load_resume, clear_resume,
                    DEVICE, CKPT, RESULTS, SEED)
from data import load_preference_splits
from batching import collate_pref

ap = argparse.ArgumentParser()
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=2e-6)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--micro_batch", type=int, default=4)
ap.add_argument("--log_every", type=int, default=250)
ap.add_argument("--max_steps", type=int, default=None)
args = ap.parse_args()

set_seed()
tok = load_tokenizer()
train, val = load_preference_splits(tok)
pad = tok.pad_token_id

policy = load_causal_lm(f"{CKPT}/sft")
ref = freeze(load_causal_lm(f"{CKPT}/sft"))
frozen_msg = f"frozen check: reference all frozen = {all_frozen(ref)}"
print(frozen_msg)
assert all_frozen(ref)


def margins(batch, with_grad=True):
    """The thing inside sigmoid(.) for each pair (shape (B,))."""
    d = lambda k: batch[k].to(DEVICE)
    pl = batch["prompt_lens"]
    with torch.no_grad():
        ref_w = sequence_logprob(ref, d("chosen_ids"), d("chosen_mask"), pl)
        ref_l = sequence_logprob(ref, d("rejected_ids"), d("rejected_mask"), pl)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        pi_w = sequence_logprob(policy, d("chosen_ids"), d("chosen_mask"), pl)
        pi_l = sequence_logprob(policy, d("rejected_ids"), d("rejected_mask"), pl)
    return args.beta * ((pi_w - ref_w) - (pi_l - ref_l))


@torch.no_grad()
def validate():
    policy.eval()
    losses, correct = [], 0
    for b in DataLoader(val, batch_size=16, collate_fn=partial(collate_pref, pad_id=pad)):
        z = margins(b, with_grad=False)
        losses.append(-F.logsigmoid(z).sum().item())
        correct += (z > 0).sum().item()
    policy.train()
    return sum(losses) / len(val), correct / len(val)


LOG = f"{RESULTS}/dpo_log.jsonl"
opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
done, prev_min = load_resume("dpo", {"policy": policy}, opt)
if done == 0:
    open(LOG, "w").close()
# fixed generator -> same shuffle order every run, so a resume skips exactly the batches already seen
loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED),
                    collate_fn=partial(collate_pref, pad_id=pad))
policy.train()
checks = [frozen_msg]
start, step, running = time.time() - prev_min * 60, 0, []

for batch in loader:
    if step < done:              # already trained on this batch before the cut
        step += 1
        continue
    B = batch["prompt_lens"].shape[0]
    total = 0.0
    for s in range(0, B, args.micro_batch):
        piece = {k: v[s:s + args.micro_batch] for k, v in batch.items()}
        z = margins(piece)
        loss = -F.logsigmoid(z).sum() / B          # mean over the whole batch
        loss.backward()
        total += loss.item()
    if step == 0 and done == 0:
        msg = f"DPO first step loss = {total:.4f} (should be log 2 = {math.log(2):.4f})"
        print(msg)
        checks.append(msg)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    running.append(total)

    if step % args.log_every == 0 or step == len(loader) or step == args.max_steps:
        vl, va = validate()
        rec = {"step": step, "train_loss": sum(running) / len(running), "val_loss": vl,
               "val_implicit_acc": va, "minutes": (time.time() - start) / 60}
        running = []
        append_jsonl(LOG, rec)
        print(rec)
        save_resume("dpo", step, {"policy": policy}, opt, rec["minutes"])
    if args.max_steps and step >= args.max_steps:
        break

minutes = (time.time() - start) / 60
policy.save_pretrained(f"{CKPT}/dpo")
tok.save_pretrained(f"{CKPT}/dpo")
clear_resume("dpo")
if len(checks) > 1:   # only a run that started at step 0 saw the first-step loss
    with open(f"{RESULTS}/check_dpo.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(checks) + "\n")
append_jsonl(f"{RESULTS}/stage_times.jsonl", {"stage": "DPO", "minutes": minutes,
                                              "final_train_loss": rec["train_loss"],
                                              "val_implicit_acc": rec["val_implicit_acc"],
                                              "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"})
print(f"done in {minutes:.1f} min -> checkpoints/dpo/")
