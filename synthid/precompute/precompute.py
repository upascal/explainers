"""Precompute the next-token tree the explainer runs on.

For each prompt we walk the model's own top-8 continuations to depth 3, storing
a distribution at every node (1 + 8 + 64 + 512 = 585 per prompt). Raw logits are
stored, not probabilities, so the page can apply temperature client-side.

    python precompute/precompute.py --out site/data.json
"""

import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ordered fixed -> open. The final display order is set from measured first-step
# entropy (see the `entropy` field in the output), not from this listing.
PROMPTS = [
    ("fox", "The quick brown fox"),
    ("australia", "The capital of Australia is"),
    ("jordan", "Michael Jordan is a"),
    ("president", "The President indicated that"),
    ("economy", "Economic growth is"),
    ("humanity", "Humanity is destined to"),
    # Extras kept for Section 3 / swapping. Not in the main dropdown.
    ("olympiad", "The 1936 Summer Olympics, officially known as the Games of the"),
    ("moon", "The first person to walk on the Moon was"),
]

TOP_K = 8
DEPTH = 3


def entropy_bits(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(-(logp.exp() * logp).sum() / math.log(2))


@torch.no_grad()
def batched_logits(model, seqs, device, batch_size=32):
    """Final-position logits for a list of equal-length token id lists."""
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(input_ids=ids).logits[:, -1, :]
        out.append(logits.float().cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="site/data.json")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    prompts_out = []
    for pid, text in PROMPTS:
        t0 = time.time()
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        # Nodes are keyed by the path of chosen candidate slots, "" for the root,
        # then "3", "3-0", "3-0-5" ... Each node stores the top-8 at that point.
        nodes = {}
        # level: list of (path, full token id sequence)
        level = [("", list(prompt_ids))]
        for depth in range(DEPTH + 1):
            logits = batched_logits(
                model, [s for _, s in level], args.device, args.batch_size
            )
            nxt = []
            for (path, seq), row in zip(level, logits):
                top = torch.topk(row, TOP_K)
                ids = [int(i) for i in top.indices]
                vals = [round(float(v), 4) for v in top.values]
                probs = torch.softmax(row, dim=-1)
                nodes[path] = {
                    "seq": seq,
                    "cand": [
                        {"id": i, "s": tok.decode([i]), "logit": v}
                        for i, v in zip(ids, vals)
                    ],
                    # what the page is hiding: full-vocab entropy, and how much
                    # probability mass the visible top-8 actually covers
                    "entropy": round(entropy_bits(row), 3),
                    "pmass8": round(float(probs[top.indices].sum()), 4),
                }
                if depth < DEPTH:
                    for slot, i in enumerate(ids):
                        child = f"{path}-{slot}" if path else str(slot)
                        nxt.append((child, seq + [i]))
            level = nxt

        prompts_out.append(
            {
                "id": pid,
                "text": text,
                "tokens": [
                    {"id": int(i), "s": tok.decode([i])} for i in prompt_ids
                ],
                "entropy": nodes[""]["entropy"],
                "nodes": nodes,
            }
        )
        print(
            f"{pid:12s} H1={nodes['']['entropy']:.3f} bits  "
            f"top8 mass={nodes['']['pmass8']:.3f}  "
            f"{len(nodes)} nodes  {time.time() - t0:.0f}s"
        )
        print("   top-8:", " | ".join(f"{c['s']!r} {c['logit']}" for c in nodes[""]["cand"]))

    out = {
        "model": args.model,
        "top_k": TOP_K,
        "depth": DEPTH,
        "vocab_size": int(model.config.vocab_size),
        "prompts": prompts_out,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
