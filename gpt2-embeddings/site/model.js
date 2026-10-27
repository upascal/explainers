// ---------- GPT-2's tokenizer ----------
// Byte-level BPE, run from GPT-2's own merge list (gpt2data.js, concatenated ahead of this file).
// A token's id is 256 + the rank of the merge that made it; ids 0–255 are single bytes, in
// GPT-2's byte order, and 50256 is <|endoftext|>.
const b64bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const EOT = 50256;
const BPE = (() => {
  // GPT-2 maps each byte to a printable character so the merge list can be plain text; its ids
  // 0–255 follow that mapping's order: the printable bytes first, then the rest
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const printable = new Set(bs);
  for (let b = 0; b < 256; b++) if (!printable.has(b)) bs.push(b);
  const idOfByte = new Uint16Array(256); bs.forEach((b, i) => { idOfByte[b] = i; });
  const pairs = new Uint16Array(b64bytes(GPT2.merges).buffer);
  const n = pairs.length / 2, rank = new Map(), bytes = new Array(256 + n + 1);
  bs.forEach((b, i) => { bytes[i] = Uint8Array.of(b); });
  for (let r = 0; r < n; r++) {
    const a = pairs[2 * r], b = pairs[2 * r + 1];
    rank.set(a * 65536 + b, r);
    const x = bytes[a], y = bytes[b], z = new Uint8Array(x.length + y.length); z.set(x); z.set(y, x.length); bytes[256 + r] = z;
  }
  bytes[EOT] = new TextEncoder().encode('<|endoftext|>');
  // GPT-2 first splits text into words, numbers, punctuation runs and spaces (a space stays
  // with the word after it), and never merges across those pieces
  const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const enc = new TextEncoder(), dec = new TextDecoder('utf-8', { fatal: true });
  // one piece, merged step by step: the lowest-ranked (earliest learned) adjacent pair each time
  function mergeSteps(piece) {
    let ids = Array.from(enc.encode(piece), b => idOfByte[b]);
    const steps = [{ ids: ids.slice(), rank: null, at: -1 }];
    for (;;) {
      let best = Infinity, at = -1;
      for (let i = 0; i + 1 < ids.length; i++) { const r = rank.get(ids[i] * 65536 + ids[i + 1]); if (r !== undefined && r < best) { best = r; at = i; } }
      if (at < 0) return steps;
      // every occurrence of that pair merges in the same step, left to right
      const a = ids[at], b = ids[at + 1], out = [];
      for (let i = 0; i < ids.length; i++) { if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) { out.push(256 + best); i++; } else out.push(ids[i]); }
      ids = out; steps.push({ ids: ids.slice(), rank: best, at });
    }
  }
  const cache = new Map();
  function encodePiece(p) { if (!cache.has(p)) cache.set(p, mergeSteps(p).pop().ids); return cache.get(p); }
  // <|endoftext|> written out is the special token itself, as in GPT-2's tokenizer
  function encode(text) { return text.split('<|endoftext|>').flatMap((part, i) => (i ? [EOT] : []).concat((part.match(SPLIT) || []).flatMap(encodePiece))); }
  // a token as text; a piece of a multi-byte character can't be shown as text, so it shows as its bytes
  function str(id) { try { return dec.decode(bytes[id]); } catch (e) { return null; } }
  const hex = id => Array.from(bytes[id], b => b.toString(16).toUpperCase().padStart(2, '0')).join(' ');
  return { encode, mergeSteps, str, hex, bytes: id => bytes[id], pieces: t => t.match(SPLIT) || [], split: SPLIT, idOfByte, nMerges: n };
})();

// ---------- the toy model ----------
// A one-layer, one-head GPT with GPT-2's structure, shrunk: a token table (also used at the top,
// as in GPT-2), a position table, one attention layer, one MLP. No LayerNorm. Trained by plain
// stochastic gradient descent, one sentence at a time. Deterministic given the two seeds.

// the same 30 sentences as ../word2vec, as GPT-2 tokenizes them
const CORPUS = `
i drink hot coffee every morning
i drink hot tea every morning
pour the coffee into a cup
pour the tea into my cup
a cup of hot coffee with milk
a cup of hot tea with sugar
i drink my coffee with sugar
i drink my tea with milk
the coffee is hot
the tea is hot
pour milk into the coffee
pour sugar into the tea
a hot drink every morning
i like coffee
i like tea
the dog has soft fur
the cat has soft fur
i feed my dog every morning
i feed my cat every morning
i walk the dog
the dog likes to play
the cat likes to sleep
the dog can bark
the cat can purr
my pet dog is soft
my pet cat is soft
a dog is a good pet
a cat is a good pet
i play with my dog
the cat and the dog sleep
`.trim().split('\n').map(l => l.trim());

const CFG = {
  D: 6,            // dimensions per row (768 in GPT-2 small)
  H: 24,           // MLP width, 4 × D as in GPT-2
  K: 0,            // set below: the longest sequence, which is also the number of positions
  ALPHA0: 0.05,    // starting learning rate; it decays linearly to 0
  EPOCHS: 120,     // passes over the corpus
  STD: 0.02,       // initialisation: N(0, 0.02), positions N(0, 0.01), as in GPT-2's code
};

function rng(seed) {   // mulberry32
  let a = seed >>> 0;
  return () => {
    a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gauss(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }

// The vocabulary: every GPT-2 token in the corpus, plus <|endoftext|>, most common first.
// A sentence is one training sequence: <|endoftext|>, its tokens, <|endoftext|>.
const V = (() => {
  const toks = CORPUS.map(s => BPE.encode(s)), counts = new Map();
  toks.forEach(t => [EOT, ...t, EOT].forEach(id => counts.set(id, (counts.get(id) || 0) + 1)));
  counts.set(EOT, counts.get(EOT) / 2);   // one <|endoftext|> between each pair of sentences, as in GPT-2's training text
  const ids = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || BPE.str(a).localeCompare(BPE.str(b)));
  const index = new Map(ids.map((g, i) => [g, i]));
  const seqs = toks.map(t => [EOT, ...t, EOT].map(g => index.get(g)));
  const total = toks.reduce((s, t) => s + t.length, 0);
  return { ids, index, count: ids.map(g => counts.get(g)), str: ids.map(g => BPE.str(g)), seqs, total, n: ids.length };
})();
CFG.K = Math.max(...V.seqs.map(s => s.length - 1));
const tokIx = s => V.str.indexOf(s);   // the toy index of a token, by its text

// all parameters in one flat array, so a model is cheap to copy and store
const LAYOUT = (() => {
  const { D, H, K } = CFG, L = {}; let o = 0;
  for (const [k, n] of [['E', V.n * D], ['P', K * D], ['Wq', D * D], ['Wk', D * D], ['Wv', D * D], ['Wo', D * D], ['W1', D * H], ['b1', H], ['W2', H * D], ['b2', D]]) { L[k] = [o, n]; o += n; }
  L.size = o; return L;
})();
const view = (m, k) => m.subarray(LAYOUT[k][0], LAYOUT[k][0] + LAYOUT[k][1]);
const rowOf = (m, k, i) => { const o = LAYOUT[k][0] + i * CFG.D; return m.subarray(o, o + CFG.D); };

function initModel(seed) {
  const r = rng(seed), m = new Float64Array(LAYOUT.size), D = CFG.D;
  const fill = (k, s) => { const v = view(m, k); for (let i = 0; i < v.length; i++) v[i] = gauss(r) * s; };
  fill('E', CFG.STD); fill('P', CFG.STD / 2); fill('Wq', CFG.STD); fill('Wk', CFG.STD); fill('Wv', CFG.STD); fill('W1', CFG.STD);
  fill('Wo', CFG.STD / Math.sqrt(2)); fill('W2', CFG.STD / Math.sqrt(2));   // the two layers that write into the residual stream, scaled down as in GPT-2
  return m;
}

const sigmoid = x => 1 / (1 + Math.exp(-x));
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const GC = Math.sqrt(2 / Math.PI);
const gelu = x => 0.5 * x * (1 + Math.tanh(GC * (x + 0.044715 * x * x * x)));
const dgelu = x => { const th = Math.tanh(GC * (x + 0.044715 * x * x * x)); return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * GC * (1 + 3 * 0.044715 * x * x); };
// x (length n) times W (n × m, row-major)
function mv(x, W, m) { const o = new Float64Array(m); for (let i = 0; i < x.length; i++) { const xi = x[i]; if (xi) for (let j = 0; j < m; j++) o[j] += xi * W[i * m + j]; } return o; }
// g (length m) times Wᵀ
function mtv(g, W, n, m) { const o = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < m; j++) s += W[i * m + j] * g[j]; o[i] = s; } return o; }

// One sequence through the model. Position t sees tokens 0..t and predicts token t + 1.
function forward(m, seq) {
  const { D, H } = CFG, n = seq.length - 1, inp = seq.slice(0, n), tg = seq.slice(1);
  const E = view(m, 'E'), P = view(m, 'P'), Wq = view(m, 'Wq'), Wk = view(m, 'Wk'), Wv = view(m, 'Wv'), Wo = view(m, 'Wo');
  const W1 = view(m, 'W1'), b1 = view(m, 'b1'), W2 = view(m, 'W2'), b2 = view(m, 'b2'), sc = 1 / Math.sqrt(D);
  const h0 = inp.map((w, t) => { const h = new Float64Array(D); for (let d = 0; d < D; d++) h[d] = E[w * D + d] + P[t * D + d]; return h; });
  const q = h0.map(h => mv(h, Wq, D)), k = h0.map(h => mv(h, Wk, D)), v = h0.map(h => mv(h, Wv, D));
  const a = q.map((qt, t) => { const s = []; for (let u = 0; u <= t; u++) s.push(dot(qt, k[u]) * sc); const mx = Math.max(...s), e = s.map(z => Math.exp(z - mx)), Z = e.reduce((x, y) => x + y, 0); return e.map(z => z / Z); });
  const z = a.map(at => { const o = new Float64Array(D); at.forEach((w, u) => { for (let d = 0; d < D; d++) o[d] += w * v[u][d]; }); return o; });
  const h1 = h0.map((h, t) => { const o = mv(z[t], Wo, D); return h.map((x, d) => x + o[d]); });
  const pre = h1.map(h => mv(h, W1, H).map((x, j) => x + b1[j])), act = pre.map(p => p.map(gelu));
  const h2 = h1.map((h, t) => { const y = mv(act[t], W2, D); return h.map((x, d) => x + y[d] + b2[d]); });
  // the top: score the last vector against every row of the same token table, then softmax
  const logits = h2.map(h => { const l = new Float64Array(V.n); for (let j = 0; j < V.n; j++) { let s = 0; for (let d = 0; d < D; d++) s += h[d] * E[j * D + d]; l[j] = s; } return l; });
  const p = logits.map(l => { const mx = Math.max(...l), e = l.map(x => Math.exp(x - mx)), Z = e.reduce((x, y) => x + y, 0); return e.map(x => x / Z); });
  let loss = 0; tg.forEach((y, t) => { loss -= Math.log(Math.max(p[t][y], 1e-300)); });
  return { n, inp, tg, h0, q, k, v, a, z, h1, pre, act, h2, logits, p, loss };
}

// The gradient of the loss (summed over the sequence's positions, or of position `only`'s loss
// alone). Besides the full gradient it keeps what the page shows: dh0[t], the gradient that
// reaches the input vector at position t.
function backward(m, f, only = -1) {
  const { D, H } = CFG, n = f.n, g = new Float64Array(LAYOUT.size), sc = 1 / Math.sqrt(D);
  const E = view(m, 'E'), Wq = view(m, 'Wq'), Wk = view(m, 'Wk'), Wv = view(m, 'Wv'), Wo = view(m, 'Wo'), W1 = view(m, 'W1'), W2 = view(m, 'W2');
  const gE = view(g, 'E'), gP = view(g, 'P'), gWq = view(g, 'Wq'), gWk = view(g, 'Wk'), gWv = view(g, 'Wv'), gWo = view(g, 'Wo');
  const gW1 = view(g, 'W1'), gb1 = view(g, 'b1'), gW2 = view(g, 'W2'), gb2 = view(g, 'b2');
  // the top: every row j gets (p_j − label_j) × h2, and h2 gets the rows weighted the same way
  const dh2 = f.p.map((pt, t) => {
    const o = new Float64Array(D);
    for (let j = 0; j < V.n; j++) { const c = only >= 0 && t !== only ? 0 : pt[j] - (j === f.tg[t] ? 1 : 0); for (let d = 0; d < D; d++) { gE[j * D + d] += c * f.h2[t][d]; o[d] += c * E[j * D + d]; } }
    return o;
  });
  const dh1 = dh2.map(r => r.slice());
  for (let t = 0; t < n; t++) {
    for (let d = 0; d < D; d++) gb2[d] += dh2[t][d];
    for (let j = 0; j < H; j++) for (let d = 0; d < D; d++) gW2[j * D + d] += f.act[t][j] * dh2[t][d];
    const dact = mtv(dh2[t], W2, H, D), dpre = dact.map((x, j) => x * dgelu(f.pre[t][j]));
    for (let j = 0; j < H; j++) { gb1[j] += dpre[j]; for (let d = 0; d < D; d++) gW1[d * H + j] += f.h1[t][d] * dpre[j]; }
    const dx = mtv(dpre, W1, D, H); for (let d = 0; d < D; d++) dh1[t][d] += dx[d];
  }
  const dh0 = dh1.map(r => r.slice()), dz = dh1.map(r => mtv(r, Wo, D, D));
  for (let t = 0; t < n; t++) for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) gWo[i * D + j] += f.z[t][i] * dh1[t][j];
  const dq = f.q.map(() => new Float64Array(D)), dk = f.q.map(() => new Float64Array(D)), dv = f.q.map(() => new Float64Array(D));
  for (let t = 0; t < n; t++) {
    const at = f.a[t], da = at.map((_, u) => dot(dz[t], f.v[u])), s = at.reduce((x, w, u) => x + w * da[u], 0);
    at.forEach((w, u) => { const ds = w * (da[u] - s) * sc; for (let d = 0; d < D; d++) { dv[u][d] += w * dz[t][d]; dq[t][d] += ds * f.k[u][d]; dk[u][d] += ds * f.q[t][d]; } });
  }
  for (let t = 0; t < n; t++) {
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) { gWq[i * D + j] += f.h0[t][i] * dq[t][j]; gWk[i * D + j] += f.h0[t][i] * dk[t][j]; gWv[i * D + j] += f.h0[t][i] * dv[t][j]; }
    const a1 = mtv(dq[t], Wq, D, D), a2 = mtv(dk[t], Wk, D, D), a3 = mtv(dv[t], Wv, D, D);
    for (let d = 0; d < D; d++) dh0[t][d] += a1[d] + a2[d] + a3[d];
  }
  // the bottom: the input vector at position t is token row + position row, so both get dh0[t]
  for (let t = 0; t < n; t++) for (let d = 0; d < D; d++) { gE[f.inp[t] * D + d] += dh0[t][d]; gP[t * D + d] += dh0[t][d]; }
  return { g, dh0 };
}

// The training run: EPOCHS passes, each over the 30 sentences in a shuffled order (one fixed
// seed), one update per sentence. The learning rate falls linearly from ALPHA0 to 0.
function makeStream(seed) {
  const r = rng(seed), order = [];
  for (let e = 0; e < CFG.EPOCHS; e++) {
    const o = V.seqs.map((_, i) => i);
    for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; }
    order.push(...o);
  }
  return { total: order.length, sent: i => order[i], epoch: i => Math.floor(i / V.seqs.length) };
}
const alphaAt = (step, total) => CFG.ALPHA0 * (1 - step / total);
function trainStep(m, seq, alpha) {
  const f = forward(m, seq), { g } = backward(m, f);
  for (let i = 0; i < m.length; i++) m[i] -= alpha * g[i];
  return f.loss / f.n;   // loss per token, for the chart
}

function cosine(a, b) { const na = Math.hypot(...a), nb = Math.hypot(...b); return na && nb ? dot(a, b) / (na * nb) : 0; }
function neighbors(rows, w, n = 5) {
  return rows.map((r, i) => ({ i, c: cosine(rows[w], r) })).filter(x => x.i !== w).sort((a, b) => b.c - a.c).slice(0, n);
}
// top-2 principal components of the rows of M, by power iteration on the covariance
function pca2(M, prev) {
  const n = M.length, d = M[0].length;
  const mean = Array.from({ length: d }, (_, j) => M.reduce((s, r) => s + r[j], 0) / n);
  const X = M.map(r => Array.from(r, (x, j) => x - mean[j]));
  const C = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0) / n));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let v = prev ? prev[c].slice() : Array.from({ length: d }, (_, j) => j === c ? 1 : 0.3);
    for (let it = 0; it < 60; it++) {
      let w = C.map(row => dot(row, v));
      comps.forEach(u => { const p = dot(w, u); w = w.map((x, j) => x - p * u[j]); });
      const nw = Math.hypot(...w) || 1; v = w.map(x => x / nw);
    }
    if (prev && dot(v, prev[c]) < 0) v = v.map(x => -x);   // keep the axes from flipping between frames
    comps.push(v);
  }
  return { mean, comps, project: r => comps.map(u => dot(Array.from(r, (x, j) => x - mean[j]), u)) };
}

if (typeof module !== 'undefined') module.exports = { BPE, CORPUS, CFG, V, LAYOUT, view, rowOf, rng, initModel, forward, backward, makeStream, alphaAt, trainStep, neighbors, cosine, pca2, dot, sigmoid, tokIx, EOT };
