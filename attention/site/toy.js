// What this page adds to the toy model (../gpt2-embeddings/src/model.js, concatenated ahead of
// this file): one position's way through the layer, step by step, with the attention weights
// left as the model has them, set by hand, or switched off.

// Every intermediate for position t of sequence `seq` (an input sequence with its targets, as in
// V.seqs). `w`: attention weights over positions 0..t to use instead of the model's own; `off`:
// leave attention out entirely (the layer adds nothing from it).
function headAt(m, seq, t, o = {}) {
  const { D, H } = CFG, sc = 1 / Math.sqrt(D);
  const E = view(m, 'E'), P = view(m, 'P'), Wq = view(m, 'Wq'), Wk = view(m, 'Wk'), Wv = view(m, 'Wv'), Wo = view(m, 'Wo');
  const W1 = view(m, 'W1'), b1 = view(m, 'b1'), W2 = view(m, 'W2'), b2 = view(m, 'b2');
  const x = []; for (let u = 0; u <= t; u++) { const r = new Float64Array(D); for (let d = 0; d < D; d++) r[d] = E[seq[u] * D + d] + P[u * D + d]; x.push(r); }
  const q = mv(x[t], Wq, D), k = x.map(r => mv(r, Wk, D)), v = x.map(r => mv(r, Wv, D));
  const raw = k.map(kk => dot(q, kk)), score = raw.map(s => s * sc);
  const mx = Math.max(...score), ex = score.map(s => Math.exp(s - mx)), Z = ex.reduce((a, b) => a + b, 0);
  const own = ex.map(e => e / Z), a = o.w || own;
  const z = new Float64Array(D); a.forEach((w, u) => { for (let d = 0; d < D; d++) z[d] += w * v[u][d]; });
  const out = o.off ? new Float64Array(D) : mv(z, Wo, D);
  const h1 = x[t].map((xx, d) => xx + out[d]);
  const pre = mv(h1, W1, H).map((y, j) => y + b1[j]), act = pre.map(gelu), mlp = mv(act, W2, D).map((y, d) => y + b2[d]);
  const h2 = h1.map((y, d) => y + mlp[d]);
  const logits = new Float64Array(V.n); for (let j = 0; j < V.n; j++) { let s = 0; for (let d = 0; d < D; d++) s += h2[d] * E[j * D + d]; logits[j] = s; }
  const lm = Math.max(...logits), le = Array.from(logits, l => Math.exp(l - lm)), LZ = le.reduce((x1, y) => x1 + y, 0), p = le.map(e => e / LZ);
  return { x, q, k, v, raw, score, own, a, z, out, h1, mlp, h2, p };
}

const seqOf = s => V.seqs[CORPUS.indexOf(s)];

if (typeof module !== 'undefined') module.exports = { headAt, seqOf };
