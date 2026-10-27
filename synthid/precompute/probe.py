"""Quick first-step probe for candidate prompts: top-12 with probabilities."""
import sys, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("mps").eval()
prompts = [l.rstrip("\n") for l in open(sys.argv[1]) if l.strip()]
with torch.no_grad():
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to("mps")
        logits = model(input_ids=ids).logits[0, -1].float().cpu()
        pr = torch.softmax(logits, -1)
        H = float(-(pr * torch.log2(pr.clamp_min(1e-30))).sum())
        top = torch.topk(pr, 12)
        print(f"\n{p!r}  H={H:.2f}")
        print("  " + " | ".join(f"{tok.decode([int(i)])!r} {float(v):.3f}" for v, i in zip(top.values, top.indices)))
        for probe in [" she", " Munich", " Sydney", " Buzz", " VIII", " X", " XII", " Saturn"]:
            pid = tok(probe, add_special_tokens=False).input_ids
            if len(pid) == 1:
                rank = int((pr > pr[pid[0]]).sum()) + 1
                if rank <= 60:
                    print(f"    probe {probe!r}: p={float(pr[pid[0]]):.4f} rank={rank}")
