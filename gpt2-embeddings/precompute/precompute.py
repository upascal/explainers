"""Precompute the real-GPT-2 data the explainer shows: src/gpt2data.js.

    python precompute/precompute.py        # ~1 min on CPU; needs the `gpt2` checkpoint (Hugging Face)

Everything the toy model does is computed live in the browser. This file only covers what a
browser can't carry: GPT-2 small's own merge list (so the page runs its real tokenizer), a few
of its embedding rows, nearest neighbours in its 50,257-row table, how far each row sits from
the average row, its position table, its next-token guesses for the toy corpus, and how one
word's vector changes through its 12 layers.
"""
import base64, json, pathlib, re
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
# the local Hugging Face cache if it has GPT-2 small, else a download (~550 MB)
try:
    SNAP = pathlib.Path(snapshot_download("gpt2", local_files_only=True))
except Exception:
    SNAP = pathlib.Path(snapshot_download("gpt2", allow_patterns=["*.json", "merges.txt", "model.safetensors"]))
OUT = ROOT / "src" / "gpt2data.js"
EOT = 50256

tok = GPT2TokenizerFast.from_pretrained(SNAP)
vocab = json.load(open(SNAP / "vocab.json"))
inv = {v: k for k, v in vocab.items()}
W = load_file(SNAP / "model.safetensors")
wte, wpe = W["wte.weight"].astype(np.float64), W["wpe.weight"].astype(np.float64)

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

# ---- the merge list, as pairs of token ids (a token's id is 256 + the rank of the merge that made it)
merges = [l.rstrip("\n").split(" ") for l in open(SNAP / "merges.txt", encoding="utf-8")][1:]
merges = [m for m in merges if len(m) == 2]
pairs = np.array([[vocab[a], vocab[b]] for a, b in merges], dtype=np.uint16)
assert all(vocab[a + b] == 256 + r for r, (a, b) in enumerate(merges))

# ---- the toy corpus, shared with ../word2vec
corpus_src = (ROOT.parent / "word2vec" / "src" / "model.js").read_text().split("`")[1]
SENTS = [l.strip() for l in corpus_src.strip().split("\n")]
toy_ids = sorted({i for s in SENTS for i in tok(s).input_ids} | {EOT})

# ---- some rows, int8 with one scale per row
EXTRA = [" Coffee", "coffee", " espresso", " Paris", " king", " queen", " 1987", " the", " SolidGoldMagikarp", " bank"]
row_ids = toy_ids + [i for w in EXTRA for i in tok(w).input_ids if i not in toy_ids]
row_ids = list(dict.fromkeys(row_ids))
R = wte[row_ids]
scale = np.abs(R).max(1) / 127
rows_q = np.round(R / scale[:, None]).astype(np.int8)

# ---- nearest neighbours (cosine, whole table) for word-like tokens
U = wte / np.linalg.norm(wte, axis=1, keepdims=True)
qids = [i for i in range(256, 50256) if re.fullmatch(r"Ġ?[A-Za-z]{2,}", inv[i]) and i < 16000]
K = 10
nn_ids, nn_cos = np.zeros((len(qids), K), np.uint16), np.zeros((len(qids), K), np.uint8)
for s in range(0, len(qids), 2048):
    q = qids[s:s + 2048]
    sim = U[q] @ U.T
    sim[np.arange(len(q)), q] = -9
    top = np.argpartition(-sim, K, axis=1)[:, :K]
    for r, i in enumerate(q):
        t = top[r][np.argsort(-sim[r, top[r]])]
        nn_ids[s + r] = t
        nn_cos[s + r] = np.clip(np.round(sim[r, t] * 255), 0, 255)

# ---- every row's distance from the average row
mu = wte.mean(0)
dmu = np.linalg.norm(wte - mu, axis=1)
dlo, dhi = float(dmu.min()), float(dmu.max())
dmu_q = np.round((dmu - dlo) / (dhi - dlo) * 255).astype(np.uint8)
rng = np.random.default_rng(0)
a, b = rng.integers(0, 50257, 50000), rng.integers(0, 50257, 50000)
Uc = (wte - mu) / np.linalg.norm(wte - mu, axis=1, keepdims=True)
stats = {
    "meanCos": float((U[a] * U[b]).sum(1).mean()),
    "meanCosCentered": float((Uc[a] * Uc[b]).sum(1).mean()),
    "meanNorm": float(np.linalg.norm(wte, axis=1).mean()),
    "muNorm": float(np.linalg.norm(mu)),
    "std": float(wte.std()),
    "nearMean": [int(i) for i in np.argsort(dmu)[:12]],
}

# ---- positions: cosine similarity between position rows (every 8th), and the top 3 directions over position
PS = 8
P = wpe / np.linalg.norm(wpe, axis=1, keepdims=True)
Sp = P[::PS] @ P[::PS].T
pos_q = np.round((Sp + 1) / 2 * 255).astype(np.uint8)
pos_norm = np.linalg.norm(wpe, axis=1)
# every dimension at every 8th position, ranked by how much it changes with position (position 0 aside),
# for the chart that adds dimensions one group at a time; and how many change much at all
pstd = wpe[1:].std(0)
pos_order = [int(d) for d in np.argsort(-pstd)]
pos_moving = int((pstd > 0.1).sum())
pos_typical = int(np.argsort(pstd)[len(pstd) // 2])   # the median one: how flat most of them are
# reading the position back: how close a weighted sum of the top k dimensions gets to the position
# number, fitted on the even positions and tested on the odd ones (median error, in positions); for
# the trained table and for random rows drawn the way GPT-2 starts them
READ_STOPS = [1, 2, 4, 8, 16, 32, pos_moving, 768]
def readout(T, k):
    Z = T[:, pos_order[:k]]; Z = (Z - Z[0::2].mean(0)) / (Z[0::2].std(0) + 1e-12)
    A = np.c_[Z, np.ones(len(Z))]; y = np.arange(len(Z), dtype=np.float64); I = np.eye(A.shape[1]); I[-1, -1] = 0
    lam = 1e-6 if k <= pos_moving else 10.0   # a little ridge once there are more dimensions than the fit can pin down
    w = np.linalg.solve(A[0::2].T @ A[0::2] + lam * I, A[0::2].T @ y[0::2])
    return round(float(np.median(np.abs(A[1::2] @ w - y[1::2]))), 1)
rand_pos = np.random.default_rng(0).normal(0, 0.01, wpe.shape)
pos_read = {"stops": READ_STOPS, "end": [readout(wpe, k) for k in READ_STOPS], "start": [readout(rand_pos, k) for k in READ_STOPS]}
pos_all = wpe[::PS][:, pos_order]
pos_all_scale = float(np.abs(pos_all).max() / 127)
pos_all_q = np.round(pos_all / pos_all_scale).astype(np.int8)

# ---- GPT-2's own next-token guesses for every prefix of every corpus sentence
model = GPT2LMHeadModel.from_pretrained(SNAP, output_hidden_states=True).eval()
nexts = []
with torch.no_grad():
    for s in SENTS:
        ids = [EOT] + tok(s).input_ids
        p = torch.softmax(model(torch.tensor([ids])).logits[0], -1)
        row = []
        for t in range(len(ids)):
            top = torch.topk(p[t], 8)
            row.append([[int(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)])
        nexts.append(row)

# ---- one word, two senses: how alike its vectors are at every layer (0 = the embedding layer).
# Each layer's vectors share a large common offset, which makes any two look alike, so the cosine is
# taken after subtracting that layer's average vector over some ordinary sentences. Layer 12 is the
# last block's raw output, before GPT-2's final LayerNorm.
REF = """The weather was cold and grey all week.
He opened the door and walked into the kitchen.
Scientists have discovered a new species of frog in the rainforest.
The committee will meet again next Tuesday to discuss the budget.
She played the piano every evening after dinner.
Prices rose sharply after the announcement.
The children ran across the field toward the old barn.
Our team lost the final game by two points.
Please send me the report before the end of the day.
The museum is closed on Mondays and public holidays.
A strong wind knocked down several trees overnight.
They argued for hours about where to go on holiday.""".split("\n")
# minimal pairs: only the word before differs, so at layer 0 (same token, same position) the two
# vectors are identical, and any difference later comes from the layers reading the context
SENSES = [
    ("plot", "the garden plot", "the vegetable plot", "the secret plot"),
    ("cell", "the prison cell", "the jail cell", "the blood cell"),
    ("pitch", "the football pitch", "the cricket pitch", "the high pitch"),
    ("bat", "the vampire bat", "the fruit bat", "the baseball bat"),
    ("spring", "the early spring", "the warm spring", "the metal spring"),
]
last = {}
model.transformer.h[11].register_forward_hook(lambda mod, i, o: last.__setitem__("h", o[0] if isinstance(o, tuple) else o))
def states(sent):
    ids = [EOT] + tok(sent).input_ids
    hs = model(torch.tensor([ids])).hidden_states
    return ids, [h[0].numpy().astype(np.float64) for h in hs[:12]] + [last["h"][0].numpy().astype(np.float64)]
with torch.no_grad():
    ref = [states(s)[1] for s in REF]
    lmean = [np.concatenate([r[L][1:] for r in ref]).mean(0) for L in range(13)]
    def vec(sent, word):
        ids, hs = states(sent); k = ids.index(tok(" " + word).input_ids[0])
        return k, [h[k] - lmean[L] for L, h in enumerate(hs)]
    ccos = lambda x, y: round(float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)), 4)
    senses = []
    for word, a1, a2, b1 in SENSES:
        (ka, A), (ka2, A2), (kb, B) = vec(a1, word), vec(a2, word), vec(b1, word)
        senses.append({"word": word, "same": [a1, a2], "diff": b1, "tok": tok(" " + word).input_ids[0],
                       "cosSame": [ccos(x, y) for x, y in zip(A, A2)], "cosDiff": [ccos(x, y) for x, y in zip(A, B)]})

data = {
    "nVocab": 50257, "dModel": 768, "nCtx": 1024,
    "params": int(sum(v.size for k, v in W.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias"))),
    "merges": b64(pairs),
    "rows": {"ids": row_ids, "scale": [round(float(x), 7) for x in scale], "q": b64(rows_q)},
    "nn": {"q": qids, "k": K, "ids": b64(nn_ids), "cos": b64(nn_cos)},
    "dmu": {"lo": dlo, "hi": dhi, "q": b64(dmu_q)},
    "stats": stats,
    "pos": {"step": PS, "n": int(Sp.shape[0]), "cos": b64(pos_q), "order": pos_order, "all": b64(pos_all_q), "allScale": pos_all_scale, "moving": pos_moving, "typical": pos_typical, "read": pos_read},
    
    "next": nexts,
    "senses": senses,
}
js = "// generated by precompute/precompute.py from GPT-2 small (openai-community/gpt2); do not edit\nconst GPT2 = " + json.dumps(data, separators=(",", ":")) + ";\n"
OUT.write_text(js)
print(f"wrote {OUT.relative_to(ROOT)} ({len(js) / 1e6:.2f} MB): {len(qids)} neighbour lists, {len(row_ids)} rows, params {data['params']:,}")
print("stats", {k: v for k, v in stats.items() if k != "nearMean"}, [inv[i] for i in stats["nearMean"]])
print("readout", pos_read)
print("senses", [(s["word"], s["cosSame"][12], s["cosDiff"][12]) for s in senses])
