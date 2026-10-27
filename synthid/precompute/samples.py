"""Build site/samples.json: pre-tokenised passages for the detector in section 3.

    python precompute/samples.py --selftest        # print the coin-table fingerprint (compare with node)
    python precompute/samples.py                   # ~10 min on MPS, needs google/gemma-2-2b

Four registers (oratory / prose / law / code), three kinds each:
  human        public-domain text from precompute/texts/, just tokenised
  gemma-plain  gemma-2-2b continuing a prompt in that register, ordinary sampling at T=1
  gemma-wm     the same prompt, sampled through the page's tournament under DEMO_KEY

The hash, coin table and tournament below are a bit-exact port of src/app.js (accum, TABLE,
rowFor, coin, runBracket, tokenScore), so the page's tokenScore() scores these samples the
way the Python here does. transformers' own SynthID processor is *not* used: it seeds its
table differently and the page could not read its watermark.
"""
import argparse, json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXTS = ROOT / "precompute" / "texts"
MODEL = "google/gemma-2-2b"
H, M, N_NEW, MIN_NEW = 4, 4, 280, 200
_rng = random.Random(20260907)
DEMO_KEY = [_rng.randrange(10000) for _ in range(30)]   # one fixed key per layer, shipped with the page

# ---------- port of the page's arithmetic (src/app.js "maths") ----------
M32, MASK64, MULT, TABLE_SIZE = 0xFFFFFFFF, (1 << 64) - 1, 6364136223846793005, 65536


def imul(a, b):
    return (a * b) & M32


def build_table(seed=0x9E3779B9):
    a, t = seed, bytearray(TABLE_SIZE)
    for i in range(TABLE_SIZE):
        a = (a + 0x6D2B79F5) & M32                            # a = a + 0x6D2B79F5 | 0
        x = imul(a ^ (a >> 15), 1 | a)                        # Math.imul(a ^ a >>> 15, 1 | a)
        x = ((x + imul(x ^ (x >> 7), 61 | x)) & M32) ^ x      # x + Math.imul(x ^ x >>> 7, 61 | x) ^ x
        t[i] = (x ^ (x >> 14)) & 1                            # ((x ^ x >>> 14) >>> 0) & 1
    return t


TABLE = build_table()


def accum(h, d):
    return ((h + d) * MULT + 1) & MASK64


def coin(keys, window, cand, layer):
    h = 1
    for t in window:
        h = accum(h, t)
    h = accum(h, cand)
    h = accum(h, keys[layer % len(keys)])
    return TABLE[h % TABLE_SIZE]


def token_score(ids, idx, keys, m=M):
    w = ids[max(0, idx - H):idx]
    return sum(coin(keys, w, ids[idx], l) for l in range(m)) / m


def scored_positions(ids, start):
    """Positions >= start whose H-token context has not occurred earlier in the text.
    Repeated-context masking, as in the paper and transformers' SynthID detector: a repeated
    window yields the same coins again, so it is not fresh evidence (and the generator below
    does not watermark there either)."""
    seen, out = set(), []
    for i in range(1, len(ids)):
        w = tuple(ids[max(0, i - H):i])
        if w not in seen and i >= start:
            out.append(i)
        seen.add(w)
    return out


def mean_score(ids, start, keys, m=M):
    pos = scored_positions(ids, start)
    return sum(token_score(ids, i, keys, m) for i in pos) / len(pos), len(pos)


def tournament(probs, window, keys, m, rng):
    """Port of runBracket: 2^m independent draws from the full distribution, paired off,
    the token that read a 1 beats the one that read a 0, ties broken at random."""
    import torch
    cur = torch.multinomial(probs, 2 ** m, replacement=True).tolist()
    for layer in range(m):
        nxt = []
        for a, b in zip(cur[0::2], cur[1::2]):
            ca, cb = coin(keys, window, a, layer), coin(keys, window, b, layer)
            nxt.append(a if ca > cb else b if cb > ca else (a if rng.random() < 0.5 else b))
        cur = nxt
    return cur[0]


# ---------- generation ----------
PROMPTS = {
    "oratory": "My fellow citizens, we gather tonight at a moment of",
    "prose": "The morning the letter arrived, Eleanor had already decided to",
    "law": "Section 1. The Commission shall have the authority to",
    "code": 'def merge_intervals(intervals):\n    """Merge overlapping [start, end] pairs."""\n',
}
HUMAN = [
    ("gettysburg", "oratory", "Gettysburg Address", "Lincoln, 1863 (public domain)"),
    ("pride_prejudice", "prose", "Pride and Prejudice, opening", "Austen, 1813 (public domain)"),
    ("constitution_art1", "law", "US Constitution, Art. I §8", "1787 (public domain)"),
    ("python_func", "code", "Python, three small utilities", "written for this page"),
]


@__import__("torch").no_grad()
def generate(model, tok, prompt, wm, keys, seed, device):
    import torch
    torch.manual_seed(seed); rng = random.Random(seed)
    ids = tok(prompt)["input_ids"]                      # leading <bos>, like data.json
    n_prompt = len(ids)
    seen = {tuple(ids[max(0, i - H):i]) for i in range(1, len(ids))}
    out = model(input_ids=torch.tensor([ids], device=device), use_cache=True)
    for _ in range(N_NEW):
        probs = torch.softmax(out.logits[0, -1].float().cpu(), -1)
        w = tuple(ids[-H:]); fresh = w not in seen; seen.add(w)
        t = tournament(probs, ids[-H:], keys, M, rng) if wm and fresh else int(torch.multinomial(probs, 1))
        if t == tok.eos_token_id:
            break
        ids.append(t)
        out = model(input_ids=torch.tensor([[t]], device=device), past_key_values=out.past_key_values, use_cache=True)
    return ids, n_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(ROOT / "site" / "samples.json"))
    args = ap.parse_args()

    if args.selftest:
        print("".join(map(str, TABLE[:48])), sum(TABLE))
        h = 1
        for x in [2, 651, 4320, 8426]:
            h = accum(h, x)
        h = accum(accum(h, 25341), 4242)
        print("row", h % TABLE_SIZE, "coin", TABLE[h % TABLE_SIZE])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(args.device).eval()

    def pack(sid, label, kind, register, source, ids, n_prompt):
        return dict(id=sid, label=label, kind=kind, register=register, source=source, prompt_ntok=n_prompt,
                    ids=ids, toks=[tok.decode([i]) for i in ids], text=tok.decode(ids[1:]))

    samples = []
    for fname, register, label, source in HUMAN:
        ids = tok((TEXTS / f"{fname}.txt").read_text().strip())["input_ids"]
        samples.append(pack(fname, label, "human", register, source, ids, 1))
        print(f"{fname}: {len(ids)} tokens", file=sys.stderr)

    for register, prompt in PROMPTS.items():
        for wm in (False, True):
            seed = 1
            while True:
                ids, n_prompt = generate(model, tok, prompt, wm, DEMO_KEY, seed, args.device)
                if len(ids) - n_prompt >= MIN_NEW:
                    break
                print(f"  {register} {'wm' if wm else 'plain'} seed {seed}: only {len(ids) - n_prompt} tokens, retrying", file=sys.stderr)
                seed += 1
            kind = "gemma-wm" if wm else "gemma-plain"
            sid = f"{register}-{'wm' if wm else 'plain'}"
            label = f"Gemma, {'watermarked' if wm else 'unwatermarked'} ({register})"
            source = f"gemma-2-2b, {'tournament m=4 under the demo key' if wm else 'plain sampling'}, T=1, seed {seed}"
            samples.append(pack(sid, label, kind, register, source, ids, n_prompt))
            print(f"{sid}: {len(ids) - n_prompt} tokens", file=sys.stderr)

    # self-check: the page will compute exactly these numbers under the demo key
    ok = True
    for s in samples:
        mu, n = mean_score(s["ids"], s["prompt_ntok"], DEMO_KEY)
        s["demo_mean"] = round(mu, 4); s["n_scored"] = n
        lo = 0.52 if s["register"] == "code" else 0.55
        good = mu > lo if s["kind"] == "gemma-wm" else 0.44 < mu < 0.56
        ok &= good
        print(f"{s['id']:22s} {s['kind']:12s} n={n:4d} mean={mu:.4f} {'ok' if good else 'UNEXPECTED'}", file=sys.stderr)
    if not ok:
        print("self-check failed; inspect the means above before shipping", file=sys.stderr)

    out = dict(model=MODEL, m=M, H=H, demo_key=DEMO_KEY, samples=samples)
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {args.out}", file=sys.stderr)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
