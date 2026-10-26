// ---------- the toy model ----------
// Skip-gram with negative sampling, following the update order of Mikolov's word2vec.c
// (the center word's input vector predicts each context word's output vector; the paper's
// framing). Everything here is deterministic given the two seeds, so reset is honest.

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
`.trim().split('\n').map(l => l.trim().split(/\s+/));

const CFG = {
  D: 5,            // dimensions (300 in the paper)
  WINDOW: 5,       // max window; each center word draws b in 1..WINDOW
  K: 5,            // negative samples per pair
  ALPHA0: 0.025,   // starting learning rate (skip-gram default)
  EPOCHS: 200,     // passes over the corpus (the toy corpus is tiny; the code's default is 5)
  SAMPLE: 0.01,    // subsampling threshold t (the code's default is 1e-3; raised so it bites on a 156-word corpus)
  POW: 0.75,       // noise distribution exponent
};

// mulberry32: small seeded PRNG, so a seed reproduces every draw
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function buildVocab(corpus) {
  const counts = new Map();
  corpus.flat().forEach(w => counts.set(w, (counts.get(w) || 0) + 1));
  const words = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b));
  const index = new Map(words.map((w, i) => [w, i]));
  const count = words.map(w => counts.get(w));
  const total = count.reduce((a, b) => a + b, 0);
  // unigram^0.75 noise distribution, as a cumulative table
  const pw = count.map(c => Math.pow(c, CFG.POW)); const z = pw.reduce((a, b) => a + b, 0);
  const noise = pw.map(p => p / z), raw = count.map(c => c / total);
  const cum = []; noise.reduce((a, p, i) => (cum[i] = a + p), 0);
  // word2vec.c keep probability: (sqrt(f / (t·N)) + 1) · (t·N) / f
  const tn = CFG.SAMPLE * total;
  const keep = count.map(c => Math.min(1, (Math.sqrt(c / tn) + 1) * tn / c));
  const sents = corpus.map(s => s.map(w => index.get(w)));
  return { words, index, count, total, raw, noise, cum, keep, sents };
}

const V = buildVocab(CORPUS);

function drawNoise(r) {
  const x = r(); let lo = 0, hi = V.cum.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (V.cum[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}

const sigmoid = x => 1 / (1 + Math.exp(-x));
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

// The training stream: every (center, context, negatives) pair in order, with the window
// shrink, subsampling and noise draws decided by one seeded generator. Built in full up front
// (about 66,000 pairs) into typed arrays, so any pair can be revisited and the total is known.
function makeStream(seed) {
  const r = rng(seed), K = CFG.K;
  const cap = CFG.EPOCHS * V.total * 2 * CFG.WINDOW;
  const center = new Int16Array(cap), ctx = new Int16Array(cap), negs = new Int16Array(cap * K);
  const epochA = new Int16Array(cap), sentA = new Int16Array(cap), posA = new Int8Array(cap), ctxPosA = new Int8Array(cap);
  const bA = new Int8Array(cap), ciA = new Int32Array(cap), wordsA = new Int32Array(cap);
  const centers = [];      // first pass only, for the window walker: {sent, pos, b, kept, first, n}
  let n = 0, wordsDone = 0, nc = 0;
  for (let epoch = 0; epoch < CFG.EPOCHS; epoch++) {
    V.sents.forEach((s, sent) => {
      const kept = s.map(w => r() < V.keep[w]);
      const live = s.map((w, i) => i).filter(i => kept[i]);
      live.forEach((pos, li) => {
        const b = 1 + Math.floor(r() * CFG.WINDOW);
        const first = n;
        for (let j = li - b; j <= li + b; j++) {
          if (j < 0 || j >= live.length || j === li) continue;
          center[n] = s[pos]; ctx[n] = s[live[j]]; epochA[n] = epoch; sentA[n] = sent; posA[n] = pos; ctxPosA[n] = live[j];
          bA[n] = b; ciA[n] = nc; wordsA[n] = wordsDone;
          for (let k = 0; k < K; k++) negs[n * K + k] = drawNoise(r);
          n++;
        }
        if (epoch === 0) centers.push({ sent, pos, b, kept, first, n: n - first });
        nc++;
      });
      wordsDone += s.length;
    });
  }
  return {
    total: n, centers,
    get(i) {
      if (i < 0 || i >= n) return null;
      const ng = Array.from(negs.subarray(i * K, i * K + K));
      return { i, epoch: epochA[i], sent: sentA[i], pos: posA[i], ctxPos: ctxPosA[i], b: bA[i], center: center[i], ctx: ctx[i],
        negs: ng.map(w => ({ w, skip: w === ctx[i] })), wordsDone: wordsA[i] };
    },
  };
}
const TRAIN_WORDS = CFG.EPOCHS * V.total;
// linear decay over all training words, floored at 1e-4 of the start, as in word2vec.c
function alphaAt(wordsDone) { return Math.max(CFG.ALPHA0 * 1e-4, CFG.ALPHA0 * (1 - wordsDone / (TRAIN_WORDS + 1))); }

function initModel(seed) {
  const r = rng(seed);
  const Win = V.words.map(() => Array.from({ length: CFG.D }, () => (r() - 0.5) / CFG.D));
  const Wout = V.words.map(() => new Array(CFG.D).fill(0));
  return { Win, Wout };
}
function cloneModel(m) { return { Win: m.Win.map(r => r.slice()), Wout: m.Wout.map(r => r.slice()) }; }

// One full pair update. Returns the loss before the update.
function trainPair(m, p, alpha) {
  const v = m.Win[p.center]; const pend = new Array(CFG.D).fill(0); let loss = 0;
  const targets = [{ w: p.ctx, label: 1 }].concat(p.negs.filter(n => !n.skip).map(n => ({ w: n.w, label: 0 })));
  for (const t of targets) {
    const vo = m.Wout[t.w]; const f = dot(v, vo); const s = sigmoid(f);
    loss -= Math.log(t.label ? Math.max(s, 1e-12) : Math.max(1 - s, 1e-12));
    const g = (t.label - s) * alpha;
    for (let d = 0; d < CFG.D; d++) pend[d] += g * vo[d];
    for (let d = 0; d < CFG.D; d++) vo[d] += g * v[d];
  }
  for (let d = 0; d < CFG.D; d++) v[d] += pend[d];
  return loss;
}

function cosine(a, b) { const na = Math.hypot(...a), nb = Math.hypot(...b); return na && nb ? dot(a, b) / (na * nb) : 0; }
function neighbors(M, w, n = 5) {
  return M.map((row, i) => ({ i, c: cosine(M[w], row) })).filter(x => x.i !== w).sort((a, b) => b.c - a.c).slice(0, n);
}

// top-2 principal components of the rows of M, by power iteration on the covariance
function pca2(M, prev) {
  const n = M.length, d = M[0].length;
  const mean = Array.from({ length: d }, (_, j) => M.reduce((s, r) => s + r[j], 0) / n);
  const X = M.map(r => r.map((x, j) => x - mean[j]));
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
  return { mean, comps, project: r => comps.map(u => dot(r.map((x, j) => x - mean[j]), u)) };
}

if (typeof module !== 'undefined') module.exports = { CORPUS, CFG, V, rng, makeStream, alphaAt, initModel, cloneModel, trainPair, neighbors, cosine, pca2, sigmoid, dot, TRAIN_WORDS };
