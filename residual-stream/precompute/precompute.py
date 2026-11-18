"""Precompute everything the residual-stream explainer shows: src/resdata.js.

    python precompute/precompute.py        # under a minute on CPU; needs the `gpt2` checkpoint (Hugging Face)

All of it comes from GPT-2 small. Weights are used as GPT-2 applies them to row vectors (x @ W).
Each head's LayerNorm gain is folded into its W_Q, W_K and W_V; the normalization itself (centre,
divide by the length) is applied to the input where an input is used.
"""
import base64, json, pathlib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
try:
    SNAP = snapshot_download("gpt2", local_files_only=True)
except Exception:
    SNAP = snapshot_download("gpt2", allow_patterns=["*.json", "merges.txt", "model.safetensors"])
OUT = ROOT / "src" / "resdata.js"
EOT = 50256
tok = GPT2TokenizerFast.from_pretrained(SNAP)
model = GPT2LMHeadModel.from_pretrained(SNAP, attn_implementation="eager").eval()
W = {k: v.astype(np.float64) for k, v in load_file(pathlib.Path(SNAP) / "model.safetensors").items()}
L, H, D, DH = 12, 12, 768, 64
E = W["wte.weight"]                                   # also the unembedding (tied)
rng = np.random.default_rng(0)
r3 = lambda a: [round(float(x), 3) for x in np.ravel(a)]


def head(l, h):
    g, A = W[f"h.{l}.ln_1.weight"][:, None], W[f"h.{l}.attn.c_attn.weight"]
    s = slice(h * DH, (h + 1) * DH)
    return (g * A[:, :D])[:, s], (g * A[:, D:2 * D])[:, s], (g * A[:, 2 * D:])[:, s], W[f"h.{l}.attn.c_proj.weight"][s]


HEADS = [(l, h) for l in range(L) for h in range(H)]
P = {lh: head(*lh) for lh in HEADS}
OV = {lh: P[lh][2] @ P[lh][3] for lh in HEADS}
QK = {lh: P[lh][0] @ P[lh][1].T for lh in HEADS}
orth = lambda M: np.linalg.qr(M)[0]
cos = lambda a, b: np.linalg.svd(orth(a).T @ orth(b), compute_uv=False)

# ---- 3: each head's singular values, and the overlap between what it reads and what it writes
sv, overlap = [], []
for lh in HEADS:
    Wq, Wk, Wv, Wo = P[lh]
    sv.append(np.linalg.svd(OV[lh], compute_uv=False)[:DH])
    overlap.append(cos(Wv, Wo.T))                     # read: col(W_V); write: row(W_O)
Ec = E - E.mean(-1, keepdims=True)
U = E - E.mean(0)   # a score relative to the average token's score: only differences reach the softmax
rand_overlap = np.mean([cos(rng.standard_normal((D, DH)), rng.standard_normal((D, DH))) for _ in range(200)], 0)

# ---- 4: copying. The OV circuit as a map from tokens to tokens. A head past layer 0 doesn't read a
# token's raw embedding but what layer 0 made of it, so the input is the "effective embedding",
# E + MLP0(E), normalized as the head's LayerNorm would; the output is centred, as the final
# LayerNorm does, and scored against every token's embedding (the unembedding, tied).
def ln(x): x = x - x.mean(-1, keepdims=True); return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5)
blk0 = model.transformer.h[0]
def eff(ids):
    e = torch.tensor(E[ids], dtype=torch.float32)
    with torch.no_grad(): m = blk0.mlp(blk0.ln_2(e)).double().numpy()
    return E[ids] + m
EXAMPLES = [" artist", " mask", " coffee", " dog", " Paris", " seven", " happy", " running", " she", " the"]
EX = [tok.encode(w)[0] for w in EXAMPLES]
assert all(len(tok.encode(w)) == 1 for w in EXAMPLES)
vocab = tok.get_vocab()
DICT = set(pathlib.Path("/usr/share/dict/words").read_text().split())
COMMON = [i for t, i in sorted(vocab.items(), key=lambda kv: kv[1]) if t.startswith("Ġ") and t[1:] in DICT and t[1:].isalpha() and i < 20000][:500]
Xex, Xc = ln(eff(EX)), ln(eff(COMMON))
copy_score, ov_top, ov_self, med_rank = [], [], [], []
for lh in HEADS:
    Wv, Wo = P[lh][2], P[lh][3]
    ev = np.linalg.eigvals(Wo @ Ec[COMMON].T @ Xc @ Wv)        # nonzero eigenvalues of Xc·W_OV·Ecᵀ (common tokens in, the same tokens out)
    copy_score.append(float(np.real(ev.sum()) / np.abs(ev).sum()))
    sc = (Xex @ OV[lh]) @ Ec.T
    ov_top.append(np.argsort(-sc, -1)[:, :5].tolist())
    ov_self.append([int((sc[i] > sc[i, EX[i]]).sum()) for i in range(len(EX))])   # where the token itself ranks (0 = top)
    sc = (Xc @ OV[lh]) @ Ec.T
    med_rank.append(int(np.median([(sc[i] > sc[i, t]).sum() for i, t in enumerate(COMMON)])))
rand_copy = []
for _ in range(50):
    A_, B_ = rng.standard_normal((D, DH)), rng.standard_normal((DH, D))
    ev = np.linalg.eigvals(B_ @ Ec[COMMON].T @ Xc @ A_); rand_copy.append(float(np.real(ev.sum()) / np.abs(ev).sum()))

# ---- 5: composition (Elhage et al.): how much of an earlier head's OV output lands where a later head reads
fro = np.linalg.norm
def comp(M, ov): return float(fro(ov @ M) / (fro(ov) * fro(M)))
qcomp, kcomp, vcomp = np.zeros((144, 144)), np.zeros((144, 144)), np.zeros((144, 144))
for i, a in enumerate(HEADS):
    for j, b in enumerate(HEADS):
        if b[0] <= a[0]: continue
        kcomp[i, j] = comp(QK[b].T, OV[a])   # a writes at the source token, which b reads with its keys
        qcomp[i, j] = comp(QK[b], OV[a])     # a writes at the destination, which b reads with its queries
        vcomp[i, j] = comp(OV[b], OV[a])     # a writes at the source, which b reads with its values
rand_comp = []
for _ in range(50):
    a = rng.standard_normal((D, DH)) @ rng.standard_normal((DH, D)); b = rng.standard_normal((D, DH)) @ rng.standard_normal((DH, D))
    rand_comp.append(comp(b, a))

# ---- 1 and 6: a forward pass, split into the parts that add up to the final vector
def run(ids):
    acts = {}
    hooks = []
    for l, blk in enumerate(model.transformer.h):
        hooks.append(blk.attn.register_forward_hook(lambda m, i, o, l=l: acts.__setitem__(("attn", l), o[0][0].double().numpy())))
        hooks.append(blk.mlp.register_forward_hook(lambda m, i, o, l=l: acts.__setitem__(("mlp", l), o[0].double().numpy())))
        hooks.append(blk.ln_1.register_forward_hook(lambda m, i, o, l=l: acts.__setitem__(("ln1", l), o[0].double().numpy())))
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(lambda m, i, l=l: acts.__setitem__(("z", l), i[0][0].double().numpy().reshape(-1, H, DH))))
        hooks.append(blk.register_forward_pre_hook(lambda m, i, l=l: acts.__setitem__(("resid", l), i[0][0].double().numpy())))
    with torch.no_grad():
        o = model(torch.tensor([ids]), output_attentions=True, output_hidden_states=True)
    for hk in hooks: hk.remove()
    A = np.stack([a[0].double().numpy() for a in o.attentions])
    final = o.hidden_states[-1][0].double().numpy()       # after ln_f
    return acts, A, o.logits[0].double().numpy(), final

def parts_of(ids):
    acts, A, logits, _ = run(ids)
    T = len(ids)
    emb = W["wte.weight"][ids] + W["wpe.weight"][:T]
    parts = [emb] + [p for l in range(L) for p in (acts[("attn", l)], acts[("mlp", l)])]
    resid = sum(parts)
    sig = np.sqrt(((resid - resid.mean(-1, keepdims=True)) ** 2).mean(-1) + 1e-5)   # the final LayerNorm's scale, frozen
    g, b = W["ln_f.weight"], W["ln_f.bias"]
    nxt = ids[1:] + [ids[-1]]
    # each part's contribution to the logit of the token that really came next; plus a constant (ln_f's bias)
    contrib = np.array([[((p[t] - p[t].mean()) * g / sig[t]) @ U[nxt[t]] for p in parts] for t in range(T)])
    const = np.array([b @ U[nxt[t]] for t in range(T)])
    rel = logits[np.arange(T), nxt] - logits.mean(-1)
    assert np.allclose(contrib.sum(1) + const, rel, atol=1e-3), "the parts don't add up to the score"
    norms = np.array([[np.linalg.norm(sum(parts[:k + 1])[t]) for k in range(len(parts))] for t in range(T)])
    return acts, A, contrib, const, norms, sig, logits

NAT = "My cat likes to sleep on the warm laptop, and my dog likes to play in the garden."
words = sorted(t for t, i in vocab.items() if t.startswith("Ġ") and t[1:] in DICT and t[1:].islower() and 4 <= len(t) - 1 <= 7 and i < 12000)
R = [vocab[w] for w in np.random.default_rng(5).choice(words, 20, replace=False)]   # the same 20 words as the attention page
SPLIT = [(5, 5), (7, 2), (9, 9), (10, 7)]
texts = {"nat": [EOT] + tok.encode(NAT), "rep": [EOT] + R + R}
out_texts = {}
for k, ids in texts.items():
    acts, A, contrib, const, norms, sig, logits = parts_of(ids)
    T = len(ids)
    p = torch.softmax(torch.tensor(logits), -1).numpy()
    d = {"ids": ids, "contrib": r3(contrib), "const": r3(const), "norm": r3(norms),
         "pNext": r3([p[t, ids[t + 1]] for t in range(T - 1)])}
    # every head's direct push on the logit of the token that really came next, at every position
    z = {l: acts[("z", l)] for l in range(L)}
    nxt = ids[1:] + [ids[-1]]
    g = W["ln_f.weight"]
    dla = np.zeros((T, 144))
    for l in range(L):
        for h in range(H):
            out = z[l][:, h] @ W[f"h.{l}.attn.c_proj.weight"][h * DH:(h + 1) * DH]
            dla[:, l * H + h] = [((out[t] - out[t].mean()) * g / sig[t]) @ U[nxt[t]] for t in range(T)]
    d["heads"] = r3(dla)
    if k == "rep":
        # 6: a few heads' output at each position, split by the source token it came from
        d["split"] = {}
        for l, h in SPLIT:
            lnw, lnb = W[f"h.{l}.ln_1.weight"], W[f"h.{l}.ln_1.bias"]
            x = (acts[("ln1", l)] - lnb) / lnw                      # normalized, before LayerNorm's gain (folded into W_V) and bias
            cs = slice(2 * D + h * DH, 2 * D + (h + 1) * DH)
            Wv, Wo = P[(l, h)][2], P[(l, h)][3]
            bv = lnb @ W[f"h.{l}.attn.c_attn.weight"][:, cs] + W[f"h.{l}.attn.c_attn.bias"][cs]
            push = lambda v, q: ((v - v.mean()) * g / sig[q]) @ U[nxt[q]]
            src = np.zeros((T, T)); bias = np.zeros(T)
            for q in range(T):
                for s_ in range(q + 1): src[q, s_] = push(A[l, h, q, s_] * (x[s_] @ Wv) @ Wo, q)
                bias[q] = push(bv @ Wo, q)   # the value bias: the weights add up to 1, so it arrives whole, wherever the head looks
            assert np.allclose(src.sum(1) + bias, dla[:, l * H + h], atol=2e-3), (l, h)
            d["split"][f"{l}.{h}"] = {"src": r3(src), "bias": r3(bias), "att": r3(A[l, h])}
    out_texts[k] = d

Wv55, Wo55 = P[(5, 5)][2], P[(5, 5)][3]
dial = {"wv": r3(Wv55[:4]), "wo": r3(Wo55[:, :4]), "ov": r3((Wv55 @ Wo55)[:4, :4])}
iu = lambda M: [r3(M[i]) for i in range(144)]
data = {
    "layers": L, "heads": H, "dModel": D, "dHead": DH,
    "sv": [r3(s) for s in sv], "overlap": [r3(o) for o in overlap], "randOverlap": r3(rand_overlap),
    "copy": r3(copy_score), "medRank": med_rank, "dial": dial, "randCopy": r3([np.mean(rand_copy), np.std(rand_copy)]),
    "examples": EX, "ovTop": ov_top, "ovSelf": ov_self,
    "kcomp": iu(kcomp), "qcomp": iu(qcomp), "vcomp": iu(vcomp), "randComp": r3([np.mean(rand_comp), np.std(rand_comp)]),
    "texts": out_texts, "repN": len(R),
}
# the text of every token the page shows (a byte that isn't text on its own shows as its hex)
shown = set(EX) | {t for rows in ov_top for r in rows for t in r} | {t for ids in texts.values() for t in ids}
def text_of(t):
    if t == EOT: return "<|endoftext|>"
    b = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(t)])
    return b if "\ufffd" not in b else "<" + tok.convert_ids_to_tokens(t).encode().hex() + ">"
data["str"] = {int(t): text_of(int(t)) for t in shown}
OUT.write_text("// generated by precompute/precompute.py from GPT-2 small (openai-community/gpt2); do not edit\nconst RES = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")

# ---- what the prose quotes
name = lambda i: f"{i // H}.{i % H}"
ov = np.array([o[0] for o in overlap])
print("overlap top cosine: heads mean %.2f, range %.2f-%.2f; random %.2f; heads above random's 99th pct:" % (ov.mean(), ov.min(), ov.max(), rand_overlap[0]),
      int((ov > np.percentile([cos(rng.standard_normal((D, DH)), rng.standard_normal((D, DH)))[0] for _ in range(300)], 99)).sum()))
print("most / least overlap:", [name(i) for i in np.argsort(-ov)[:5]], [name(i) for i in np.argsort(ov)[:5]])
cs = np.array(copy_score)
print("copy score: random %.2f ± %.2f; heads > 0.5: %d, < -0.5: %d; top:" % (np.mean(rand_copy), np.std(rand_copy), (cs > 0.5).sum(), (cs < -0.5).sum()), [(name(i), round(cs[i], 2)) for i in np.argsort(-cs)[:6]], "bottom:", [(name(i), round(cs[i], 2)) for i in np.argsort(cs)[:4]])
mr = np.array(med_rank)
print("median self-rank: copies (rank<10):", [name(i) for i in np.where(mr < 10)[0]], " suppresses (rank>40000):", [(name(i), mr[i]) for i in np.where(mr > 40000)[0]])
for nm in ["5.5", "4.11", "7.2", "7.10", "9.9", "10.7"]:
    i = HEADS.index(tuple(map(int, nm.split("."))))
    print(f"  {nm}: copy {cs[i]:+.2f}, median rank {mr[i]}; OV top tokens for", [(EXAMPLES[e], [tok.decode([t]) for t in ov_top[i][e][:3]], ov_self[i][e]) for e in (0, 2, 4)])
k = kcomp[HEADS.index((4, 11))]
print("K-comp from 4.11 top:", [(name(j), round(k[j], 3)) for j in np.argsort(-k)[:7]], "random %.3f ± %.3f" % (np.mean(rand_comp), np.std(rand_comp)))
for M, nm in [(kcomp, "K"), (qcomp, "Q"), (vcomp, "V")]:
    i, j = np.unravel_index(np.argmax(M), M.shape); print(f"max {nm}-comp: {name(i)} -> {name(j)} {M[i, j]:.3f}")
rep = out_texts["rep"]; T = len(texts["rep"]); q = 1 + 20 + 2
for nm, sp in rep["split"].items():
    src = np.array(sp["src"]).reshape(T, T)
    print(f"rep, 2nd ' mask', head {nm}: top source", tok.decode([texts['rep'][int(np.argmax(src[q]))]]), round(src[q].max(), 2), "bias", sp["bias"][q], "total", round(src[q].sum() + sp["bias"][q], 2))
hd = np.array(rep["heads"]).reshape(T, 144)
m2 = hd[22:2 * 20 + 1].mean(0)
print("direct push, 2nd copy mean, top heads:", [(name(i), round(m2[i], 2)) for i in np.argsort(-m2)[:6]], "bottom:", [(name(i), round(m2[i], 2)) for i in np.argsort(m2)[:3]])
c = np.array(rep["contrib"]).reshape(T, 25)
labels = ["embed"] + [f"{k}{l}" for l in range(12) for k in ("attn", "mlp")]
print("rep, 2nd ' mask', top parts toward ' artist':", [(labels[i], round(c[q, i], 2)) for i in np.argsort(-c[q])[:6]], "bottom", [(labels[i], round(c[q, i], 2)) for i in np.argsort(c[q])[:3]])
nn = np.array(out_texts["nat"]["norm"]).reshape(len(texts["nat"]), 25)
print("nat norms after embed / layer 6 / layer 12 (median over positions 1+):", np.median(nn[1:, 0]).round(1), np.median(nn[1:, 12]).round(1), np.median(nn[1:, 24]).round(1), "pos0 final", nn[0, 24].round(0))
