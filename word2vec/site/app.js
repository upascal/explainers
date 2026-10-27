(function () {
// model.js is concatenated ahead of this file by build.py (CORPUS, CFG, V, makeStream, trainPair, …)
const $ = id => document.getElementById(id);
const D = CFG.D, W = V.words;
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const commas = v => String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
// a number with a real minus sign, and no sign on a value that rounds to zero
function num(x, n = 3) { const s = Math.abs(x).toFixed(n); return (x < 0 && +s !== 0 ? '−' : '') + s; }
const zeros = () => new Array(D).fill(0);
const norm = a => Math.hypot(...a);

// transport icons, drawn the same way as on the SynthID page
const I_RESET = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><rect x="1" y="2" width="2" height="12" fill="currentColor"/><polygon points="11,2 11,14 4,8" fill="currentColor"/><polygon points="18,2 18,14 11,8" fill="currentColor"/></svg>';
const I_PLAY = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><polygon points="4,2 4,14 13,8" fill="currentColor"/></svg>';
const I_PAUSE = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" fill="currentColor"/><rect x="9.5" y="2" width="3.5" height="12" fill="currentColor"/></svg>';
const I_STEP = '<svg viewBox="0 0 18 16" width="12" height="10" aria-hidden="true"><polygon points="2,2 2,14 11,8" fill="currentColor"/><rect x="12" y="2" width="2.5" height="12" fill="currentColor"/></svg>';
const I_END = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><polygon points="2,2 2,14 9,8" fill="currentColor"/><polygon points="9,2 9,14 16,8" fill="currentColor"/><rect x="17" y="2" width="2" height="12" fill="currentColor"/></svg>';

// ---------- shared model state ----------
// One initialisation (2) and one training stream for the whole page. Section 8 owns the
// training timeline: G.model is the model after G.step pairs. The full run is computed once
// up front, with a checkpoint every CK pairs, so any point of training can be revisited.
// Sections 6–7 take their own copy of the model at one chosen pair (P6 below).
const G = {
  seed: 1,                 // initialisation seed (re-rolled in 2)
  streamSeed: 2013,        // seed for window shrink, subsampling and negative draws
  stream: null, init: null, model: null, mid: null, final: null,
  step: 0,                 // pairs trained so far on the timeline
  ck: [], lossSum: null, lossN: null, bucket: 1,
  pca: null, rad: null,
};
const NB = 150;            // loss-curve buckets
const CK = 500;            // pairs between checkpoints

function flat(m) { return Float64Array.from(m.Win.flat().concat(m.Wout.flat())); }
function unflat(f) {
  const n = W.length, row = (o, i) => Array.from(f.subarray(o + i * D, o + i * D + D));
  return { Win: W.map((_, i) => row(0, i)), Wout: W.map((_, i) => row(n * D, i)) };
}
function resetTraining() {
  G.init = initModel(G.seed);
  G.bucket = Math.ceil(G.stream.total / NB); G.lossSum = new Float64Array(NB + 1); G.lossN = new Uint32Array(NB + 1);
  const m = cloneModel(G.init), N = G.stream.total; G.ck = []; G.lossAll = new Float32Array(N);
  // input vectors at log-spaced steps, for charts whose time axis is logarithmic (early training in detail)
  const snapAt = [...new Set([0, ...Array.from({ length: 90 }, (_, k) => Math.round(Math.pow(N, (k + 1) / 90)))])].filter(x => x < N);
  G.wsnaps = []; let si = 0;
  for (let i = 0; i < N; i++) {
    if (i % CK === 0) G.ck.push(flat(m));
    if (i === snapAt[si]) { G.wsnaps.push({ step: i, Win: m.Win.map(r => r.slice()) }); si++; }
    if (i === Math.floor(G.stream.total / 2)) G.mid = cloneModel(m);
    const p = G.stream.get(i), l = trainPair(m, p, alphaAt(p.wordsDone));
    const b = Math.floor(i / G.bucket); G.lossSum[b] += l; G.lossN[b]++; G.lossAll[i] = l;
  }
  G.wsnaps.push({ step: N, Win: m.Win.map(r => r.slice()) });
  G.final = m; G.model = cloneModel(G.init); G.step = 0; G.pca = null; G.rad = null;
}
// the model after `step` pairs, from the nearest checkpoint
function modelAt(step) {
  const c = Math.min(Math.floor(step / CK), G.ck.length - 1), m = unflat(G.ck[c]);
  for (let i = c * CK; i < step; i++) { const p = G.stream.get(i); trainPair(m, p, alphaAt(p.wordsDone)); }
  return m;
}
function seek(step) { G.step = Math.max(0, Math.min(G.stream.total, step)); G.model = modelAt(G.step); }
function advance(n) {
  const end = Math.min(G.stream.total, G.step + n);
  for (; G.step < end; G.step++) { const p = G.stream.get(G.step); trainPair(G.model, p, alphaAt(p.wordsDone)); }
}
const timelinePair = () => G.stream.get(Math.min(G.step, G.stream.total - 1));

// ---------- one pair, stepped through (6–7) ----------
// The reader picks a real pair (center, context) and a moment in training; P6 holds that
// pair from the stream, the model as it stood just before it, and how far the update has got.
const P6 = { center: 'coffee', ctx: 'cup', idx: 0, pair: null, PL: null, phase: 0, R: null, snap: null, stand: null };
const MOMENTS = { start: 0, middle: 0.5, trained: 0.95 };   // where in training 6–7 look for the pair, by the shared stage

// The phases of one pair update. 0–5 belong to 6 (the positive), the rest to 7.
const POS_END = 5;
function phaseList(p) {
  const L = ['start', 'score', 'sigma', 'g', 'out', 'pend'].map(k => ({ k }));
  p.negs.forEach((n, j) => { L.push({ k: 'draw', j }); if (!n.skip) L.push({ k: 'neg', j }); });
  L.push({ k: 'apply' });
  return L;
}
const END = () => P6.PL.length - 1;

// Every number shown in 6 and 7, recomputed from the model as it stood before the pair.
// Updates are applied in word2vec.c's order: each target's v′ moves at once, while the
// change to v collects in “Δv so far”, which is added at the very end.
function replay(upto = P6.phase) {
  const p = P6.pair, PL = P6.PL, m = cloneModel(P6.snap), alpha = alphaAt(p.wordsDone);
  const v = m.Win[p.center], pend = zeros(); let loss = 0;
  const T = [{ w: p.ctx, label: 1, skip: false, drawAt: 0, outAt: 4, pendAt: 5 }];
  p.negs.forEach((n, j) => {
    const di = PL.findIndex(q => q.k === 'draw' && q.j === j);
    T.push({ w: n.w, label: 0, skip: n.skip, j, drawAt: di, outAt: n.skip ? Infinity : di + 1, pendAt: n.skip ? Infinity : di + 1 });
  });
  for (const t of T) {
    if (t.skip) continue;
    const vo = m.Wout[t.w]; t.vo0 = vo.slice(); t.f = dot(v, vo); t.s = sigmoid(t.f); t.g = (t.label - t.s) * alpha;
    t.dvo = v.map(x => t.g * x); t.contrib = t.vo0.map(x => t.g * x);
    loss -= Math.log(Math.max(t.label ? t.s : 1 - t.s, 1e-12));
    if (upto >= t.outAt) for (let d = 0; d < D; d++) vo[d] += t.dvo[d];
    if (upto >= t.pendAt) for (let d = 0; d < D; d++) pend[d] += t.contrib[d];
    t.vo1 = vo.slice();
  }
  const v0 = v.slice(), applied = upto >= PL.length - 1;
  if (applied) for (let d = 0; d < D; d++) v[d] += pend[d];
  return { m, T, pend, v0, v1: v.slice(), alpha, loss, applied };
}
// context words that really occur with a center word, alphabetical; and a default among them,
// the most frequent one that isn't a small word like "the"
const SMALL = new Set(['the', 'a', 'i', 'my', 'is', 'of', 'into', 'with', 'to', 'and', 'can', 'has', 'every']);
function contexts6(c, best) {
  const ci = V.index.get(c), n = new Map();
  for (let i = 0; i < G.stream.total; i++) { const p = G.stream.get(i); if (p.center === ci) n.set(W[p.ctx], (n.get(W[p.ctx]) || 0) + 1); }
  if (best) return [...n].filter(([w]) => !SMALL.has(w)).sort((a, b) => b[1] - a[1])[0]?.[0] || [...n.keys()][0];
  return [...n.keys()].sort();
}
// find the pair in the stream at the chosen moment and load the model just before it
function loadPair6() {
  const ci = V.index.get(P6.center), xi = V.index.get(P6.ctx), ni = P6.stand ? V.index.get(P6.stand) : -1, N = G.stream.total, from = Math.floor(MOMENTS[S4.stage] * N);
  // a positive pair occurs as itself; a negative one as a negative draw for some positive pair with that center
  const hit = p => p.center === ci && (ni >= 0 ? p.negs.some(n => n.w === ni && !n.skip) : p.ctx === xi);
  let idx = -1;
  for (let k = 0; k < N && idx < 0; k++) { const i = (from + k) % N, p = G.stream.get(i); if (hit(p)) idx = i; }
  if (idx < 0) { const k = (from + N) % N; idx = [...Array(N).keys()].map(j => (k + j) % N).find(i => G.stream.get(i).center === ci); }
  if (ni >= 0) P6.ctx = W[G.stream.get(idx).ctx];   // section 7 shows the positive pair it was drawn for
  P6.idx = idx; P6.pair = G.stream.get(idx); P6.PL = phaseList(P6.pair); P6.snap = modelAt(idx); setPhase(POS_END);
}
function setPhase(ph) { P6.phase = ph; P6.R = replay(); }

// ---------- the shared play bar ----------
// reset · play/pause · step · skip to end, the same in every section. The keyboard drives the
// bar you last used, or else the one nearest the middle of the screen.
const BARS = []; let ACTIVE = null;
function PlayBar(host, o) {
  const bar = { o, host, playing: false, timer: null };
  host.classList.add('seg', 'transport', 'playbar');
  host.innerHTML = `<button data-a="reset" title="reset (shift + ←)">${I_RESET}</button><button data-a="play" title="play / pause (space)">${I_PLAY}</button><button data-a="step" title="step (→)">${I_STEP}</button><button data-a="end" title="skip to end (shift + →)">${I_END}</button>`;
  const ms = () => typeof o.ms === 'function' ? o.ms() : o.ms;
  const tick = () => {
    if (!bar.playing) return;
    if (!o.canStep()) { bar.stop(); return; }
    o.step(); refreshBars();
    if (!o.canStep()) { bar.stop(); return; }
    bar.timer = setTimeout(tick, ms());
  };
  bar.stop = () => { bar.playing = false; clearTimeout(bar.timer); bar.refresh(); };
  bar.act = a => {
    ACTIVE = bar;
    BARS.forEach(b => { if (b !== bar && b.playing) b.stop(); });
    if (a === 'play') {
      if (bar.playing) bar.stop();
      else { if (!o.canStep() && o.rewind) o.rewind(); if (o.canStep()) { bar.playing = true; tick(); } }   // play at the end starts over
    }
    else { bar.stop(); if (a === 'step' && o.canStep()) o.step(); else if (a === 'end' && o.canStep()) o.end(); else if (a === 'reset' && (!o.canReset || o.canReset())) o.reset(); }
    refreshBars();
  };
  bar.refresh = () => {
    const can = o.canStep(), b = k => host.querySelector(`[data-a="${k}"]`);
    b('step').disabled = !can || bar.playing; b('end').disabled = !can; b('play').disabled = !can && !bar.playing && !o.rewind;
    b('reset').disabled = o.canReset ? !o.canReset() : false;
    b('play').innerHTML = bar.playing ? I_PAUSE : I_PLAY;
    b('play').classList.toggle('on', bar.playing);
  };
  host.addEventListener('click', e => { const b = e.target.closest('button'); if (b && !b.disabled) bar.act(b.dataset.a); });
  BARS.push(bar); return bar;
}
function refreshBars() { BARS.forEach(b => b.refresh()); }
function pickBar() {
  const vis = BARS.filter(b => { const r = b.host.closest('.module').getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; });
  if (ACTIVE && vis.includes(ACTIVE)) return ACTIVE;
  const mid = innerHeight / 2, dist = b => { const r = b.host.closest('.module').getBoundingClientRect(); return r.top <= mid && r.bottom >= mid ? 0 : Math.min(Math.abs(r.top - mid), Math.abs(r.bottom - mid)); };
  return vis.sort((a, b) => dist(a) - dist(b))[0] || null;
}
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input:not([type=checkbox]), select, textarea')) return;
  if (e.target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
  let a = null;
  if (e.key === ' ') a = 'play';
  else if (e.key === 'ArrowRight') a = e.shiftKey ? 'end' : 'step';
  else if (e.key === 'ArrowLeft' && e.shiftKey) a = 'reset';
  const bar = a && pickBar(); if (!bar) return;
  e.preventDefault(); bar.act(a);
});

// ---------- small render helpers ----------
// the two vector symbols, in their colors: violet v (input), green v′ (output)
const SV = '<span class="cv">v</span>', SVO = '<span class="cvo">v′</span>', SG = '<span class="cg">g</span>';   // and g, the scaling factor, in blue
const wchip = (w, cls = '') => `<span class="chip ${cls}" data-w="${esc(w)}">${esc(w)}</span>`;
function vecCells(a, cls = '', n = 3) { return a.map((x, d) => `<td class="n ${cls}" style="--d:${d}">${num(x, n)}</td>`).join(''); }
const blankCells = () => Array.from({ length: D }, () => '<td class="n dim">·</td>').join('');
const dimHead = () => Array.from({ length: D }, (_, d) => `<th class="n">d${d + 1}</th>`).join('');
let svgUid = 0;

// Two vectors always lie in one flat plane, so they can be drawn at their true lengths and
// true angle however many dimensions they have. `ghost` is an earlier position of b; an
// update Δv′ = g·v keeps v′ in the plane of v and the old v′, so all three are exact here.
function pairPlane(a, b, o = {}) {
  const ghost = o.ghost || null, id = 'pp' + (svgUid++), bc = o.bCls || 'b';
  const W0 = 300, H0 = 172, ox = W0 / 2, oy = H0 - 22, Rpx = 120;
  const ref = ghost && norm(ghost) > 1e-12 ? ghost : b;
  let e1, e2;
  const na = norm(a);
  if (na > 1e-12) e1 = a.map(x => x / na); else if (norm(ref) > 1e-12) e1 = ref.map(x => x / norm(ref));
  if (e1) {
    const pr = dot(ref, e1), perp = ref.map((x, i) => x - pr * e1[i]), np = norm(perp);
    e2 = np > 1e-9 * Math.max(1, norm(ref)) ? perp.map(x => x / np) : null;
    if (!e2) { const r2 = (norm(b) > 1e-12 ? b : ref).map((x, i) => x - dot(b, e1) * e1[i]); const n2 = norm(r2); e2 = n2 > 1e-9 ? r2.map(x => x / n2) : null; }
  }
  const xy = u => !e1 ? [0, 0] : [dot(u, e1), e2 ? dot(u, e2) : 0];
  const pts = [a, b].concat(ghost ? [ghost] : []).map(xy);
  const R = Math.max(1e-9, ...pts.map(([x, y]) => Math.hypot(x, y)));
  const nb = norm(b), cos = na > 1e-12 && nb > 1e-12 ? dot(a, b) / (na * nb) : null;
  // one vector all zeros: it has no direction, so tilt the other off the baseline (lying flat
  // along it would read as an angle of 0°) and mark the origin "angle undefined" instead
  const tilt = cos === null ? 0.3 : 0, ct = Math.cos(tilt), st = Math.sin(tilt);
  const k = Rpx / R, P = ([x, y]) => [ox + (x * ct - y * st) * k, oy - (x * st + y * ct) * k];
  const labs = [];
  const arrow = (u, cls, label, dash) => {
    const [x, y] = P(xy(u)), len = Math.hypot(x - ox, y - oy);
    // a zero vector's label goes below the dot, clear of the other (tilted) arrow
    if (len < 2) return `<circle cx="${ox}" cy="${oy}" r="4" class="pp-${cls}" fill="currentColor"/>${label ? `<text x="${ox + 8}" y="${cos === null ? oy + 16 : oy - 10}" class="pp-lab pp-${cls}">${label} ${norm(u) < 1e-12 ? '= 0' : '(tiny)'}</text>` : ''}`;
    let tx = x + (x - ox) / len * 12, ty = y + (y - oy) / len * 12;
    if (label && labs.some(([lx, ly]) => Math.hypot(lx - tx, ly - ty) < 16)) ty -= 15;   // two tips close together: lift this label clear
    let anc = 'middle'; if (tx > W0 - 34) { anc = 'end'; tx = W0 - 2; ty = Math.min(ty, y - 10); }   // keep labels at the right edge inside the box
    if (label) labs.push([tx, ty]);
    return `<line x1="${ox}" y1="${oy}" x2="${x}" y2="${y}" class="pp-${cls}" stroke="currentColor" stroke-width="${dash ? 1.5 : 2.5}" ${dash ? 'stroke-dasharray="4 3"' : ''} marker-end="url(#${id}-${cls}${dash ? 'g' : ''})"/>` +
      (label ? `<text x="${tx}" y="${ty}" text-anchor="${anc}" dominant-baseline="middle" class="pp-lab pp-${cls}">${label}</text>` : '');
  };
  const mk = (cls, g) => `<marker id="${id}-${cls}${g ? 'g' : ''}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="${g ? 5 : 6}" markerHeight="${g ? 5 : 6}" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="pp-${cls}" fill="currentColor"/></marker>`;
  const ang = cos === null ? null : Math.acos(Math.max(-1, Math.min(1, cos)));
  let arc = '';
  if (ang !== null && Math.min(na, nb) * k > 14 && ang > 0.12) {
    const r = 26, [bx, by] = xy(b), t = Math.atan2(by, bx);
    const x2 = ox + r * Math.cos(t), y2 = oy - r * Math.sin(t);
    arc = `<path d="M${ox + r},${oy} A${r},${r} 0 0 0 ${x2},${y2}" fill="none" stroke="var(--muted)"/>` +
      `<text x="${ox + (r + 12) * Math.cos(t / 2)}" y="${oy - (r + 12) * Math.sin(t / 2)}" class="pp-ang" text-anchor="middle" dominant-baseline="middle">${Math.round(ang * 180 / Math.PI)}°</text>`;
  }
  const svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block;max-width:${W0}px">
    <defs>${mk('a')}${mk(bc)}${mk(bc, true)}</defs>
    ${cos === null ? `<g class="pp-${na <= 1e-12 ? 'a' : bc}"><circle cx="${ox}" cy="${oy}" r="24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3"/>
    <text x="${ox}" y="${oy - 32}" text-anchor="middle" fill="currentColor" class="pp-undef">angle undefined</text></g>`
    : `<line x1="10" y1="${oy}" x2="${W0 - 10}" y2="${oy}" stroke="var(--grid)"/>
    <path d="M${ox - Rpx},${oy} A${Rpx},${Rpx} 0 0 1 ${ox + Rpx},${oy}" fill="none" stroke="var(--grid)" stroke-dasharray="2 4"/>`}
    ${arc}${ghost ? arrow(ghost, bc, '', true) : ''}${arrow(b, bc, o.bLabel || 'v′')}${arrow(a, 'a', o.aLabel || 'v')}
  </svg>`;
  const deg = Math.round(ang * 180 / Math.PI);
  const cap = o.cos && cos !== null ? `<span class="note">angle ${deg}° · cosine similarity <b>${cos.toFixed(2)}</b></span>` : cos === null
    ? `<span class="note">${na <= 1e-12 ? (o.aLabel || 'v') : (o.bLabel || 'v′')} is all zeros. With no length it has no direction, so the angle is undefined. Every product is × 0, so the dot product is 0.</span>`
    : `<span class="note">|${o.aLabel || 'v'}| = ${num(na)} · |${o.bLabel || 'v′'}| = ${num(nb, nb < 0.01 ? 4 : 3)} · angle ${Math.round(ang * 180 / Math.PI)}°<br>${num(na)} × ${num(nb, nb < 0.01 ? 4 : 3)} × cos ${Math.round(ang * 180 / Math.PI)}° = <b>${num(dot(a, b), 4)}</b></span>`;
  return `<div class="pplane">${o.title ? `<div class="pp-title">${o.title}</div>` : ''}${svg}${cap}${ghost ? '<div class="note">dashed: v′ before this update</div>' : ''}</div>`;
}

// ---------- sticky status bar ----------
function renderStatebar() {
  // the pair selected in 3, 4 or 5, and where it stands
  const real = isReal(S4.center, S4.ctx), x = dot(S4.v, S4.u);
  $('statebar-in').innerHTML = `<div class="sb-set">
    <span class="sb-pair"><a href="#s3">selected pair</a> ${wchip(S4.center, 'v')} ${wchip(S4.ctx, real ? 'vo' : 'neg')} <span class="badge ${real ? 'pos' : 'neg'}">${real ? 'positive' : 'negative'}</span></span>
    <span class="sb-vals"><a href="#s4">training <b>${{ start: 'start', middle: 'middle', trained: 'end' }[S4.stage]}</b></a><a href="#s5">score <b>${num(x, 2)}</b></a><a href="#s2" title="initialisation seed — re-roll it in section 2">seed <b>${G.seed}</b></a></span>
  </div>`;
}

// the header only makes sense once pairs exist: show it from section 3's widget on
(function headerWhenPairs() {
  let ticking = false;
  const check = () => { ticking = false; $('statebar').classList.toggle('shown', $('m3').getBoundingClientRect().top < innerHeight * 0.75); };
  addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(check); } }, { passive: true });
  addEventListener('resize', check); setTimeout(check, 0);
})();

// ---------- 1 the problem ----------
// A fill-in-the-blank puzzle from four corpus sentences: the reader matches words by their
// company, the way word2vec does. Each word goes in one blank; coffee and tea fit either cup.
const M1_WORDS = ['coffee', 'tea', 'purr', 'fur'];
// `no` holds a hand-written reply for each wrong word: playful, and never naming the answer
const M1_SENTS = [
  { pre: 'the dog has soft', post: '', ok: ['fur'], yes: 'Yes!',
    no: { purr: 'A soft purr is lovely, but it isn’t something a dog has.', coffee: 'A dog with soft coffee? Try again.', tea: 'A dog with soft tea? Try again.' } },
  { pre: 'the cat can', post: '', ok: ['purr'], yes: 'Yes!',
    no: { fur: 'Unfortunately, “fur” is not something a cat can do.', coffee: 'Unfortunately, “coffee” is not something a cat can do.', tea: 'Unfortunately, “tea” is not something a cat can do.' } },
  { pre: 'a cup of hot', post: 'with milk', ok: ['coffee', 'tea'], yes: 'Yes!',
    no: { fur: 'Sorry, I don’t think that would taste good.', purr: 'Try again, this time pick a word that makes sense!' } },
  { pre: 'pour the', post: 'into my cup', ok: ['coffee', 'tea'], yes: 'Yes!',
    no: { fur: 'While technically possible, there is a more delicious option.', purr: 'You can’t pour a purr. Try again.' } },
];
const m1 = { fill: M1_SENTS.map(() => null), sel: null, last: null, dragged: false };

function m1Message(i, w) {
  const s = M1_SENTS[i];
  if (!s.ok.includes(w)) return s.no[w];
  const o = s.ok.find(x => x !== w);
  return o ? `${s.yes} “${o}” would work here too.` : s.yes;
}

function render1() {
  const used = new Set(m1.fill);
  $('m1-bank').innerHTML = M1_WORDS.map(w => `<button class="m1w${m1.sel === w ? ' on' : ''}${used.has(w) ? ' used' : ''}" data-m1w="${w}" aria-pressed="${m1.sel === w}">${w}</button>`).join('');
  $('m1-sents').innerHTML = M1_SENTS.map((s, i) => {
    const w = m1.fill[i], cls = w == null ? '' : s.ok.includes(w) ? ' ok' : ' bad';
    const blank = `<button class="m1blank${cls}" data-m1b="${i}" aria-label="${w == null ? 'empty blank' : 'blank filled with ' + w}">${w ?? '&nbsp;'}</button>`;
    return `<div class="sline m1line">${[s.pre, blank, s.post].filter(Boolean).join(' ')}</div>`;
  }).join('');
  const right = m1.fill.filter((w, i) => w != null && M1_SENTS[i].ok.includes(w)).length;
  let fb = m1.last != null && m1.fill[m1.last] != null ? m1Message(m1.last, m1.fill[m1.last]) : '';
  if (right === M1_SENTS.length) fb += ' <b>Every word is in a sentence it fits.</b>';
  $('m1-fb').innerHTML = fb;
}

// each word sits in one blank, so placing it again moves it
function m1Place(i, w) { m1.fill = m1.fill.map(x => x === w ? null : x); m1.fill[i] = w; m1.last = i; m1.sel = null; render1(); }

// tap a word, then a blank; tapping a filled blank with no word selected empties it
$('m1').addEventListener('click', e => {
  if (m1.dragged) return;
  const wb = e.target.closest('[data-m1w]'), bb = e.target.closest('[data-m1b]');
  if (wb) { m1.sel = m1.sel === wb.dataset.m1w ? null : wb.dataset.m1w; render1(); }
  else if (bb) {
    const i = +bb.dataset.m1b;
    if (m1.sel) m1Place(i, m1.sel);
    else if (m1.fill[i] != null) { m1.fill[i] = null; m1.last = null; render1(); }
  }
});
$('m1-reset').addEventListener('click', () => { m1.fill.fill(null); m1.sel = m1.last = null; render1(); });

// drag a word onto a blank (pointer events, so touch works too); a short press is a tap
(function m1Drag() {
  let drag = null;
  const blankAt = (x, y) => document.elementFromPoint(x, y)?.closest('[data-m1b]');
  const hover = el => document.querySelectorAll('.m1blank.over').forEach(b => b !== el && b.classList.remove('over'));
  $('m1-bank').addEventListener('pointerdown', e => {
    const wb = e.target.closest('[data-m1w]'); if (!wb || e.button) return;
    e.preventDefault();              // no text selection while dragging
    drag = { w: wb.dataset.m1w, x: e.clientX, y: e.clientY, ghost: null };
  });
  document.addEventListener('pointermove', e => {
    if (!drag) return;
    if (!drag.ghost) {
      if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 6) return;
      drag.ghost = Object.assign(document.createElement('div'), { className: 'm1w m1ghost', textContent: drag.w });
      document.body.appendChild(drag.ghost);
    }
    drag.ghost.style.left = e.clientX + 'px'; drag.ghost.style.top = e.clientY + 'px';
    const b = blankAt(e.clientX, e.clientY); hover(b); b?.classList.add('over');
  });
  const end = e => {
    if (!drag) return;
    const d = drag; drag = null;
    if (!d.ghost) return;            // never moved: the click handler treats it as a tap
    d.ghost.remove(); hover(null);
    const b = e.type === 'pointerup' && blankAt(e.clientX, e.clientY);
    if (b) m1Place(+b.dataset.m1b, d.w);
    // ignore the click that may follow a drag, so it doesn't also select or clear anything
    m1.dragged = true; setTimeout(() => { m1.dragged = false; }, 0);
  };
  document.addEventListener('pointerup', end);
  document.addEventListener('pointercancel', end);
})();

// ---------- 2 vocabulary and the two tables ----------
// section 2 highlights the selected pair's center word; clicking a word selects it page-wide
function render2() {
  const sel = V.index.get(S4.center), maxc = Math.max(...V.count);
  $('m2-corpus').innerHTML = V.sents.map(s => `<div class="sline">${s.map(w => `<span class="w${w === sel ? ' on' : ''}" data-w="${W[w]}">${W[w]}</span>`).join(' ')}</div>`).join('');
  $('m2-vocab').innerHTML = `<table class="data vocab"><tbody>${W.map((w, i) => `<tr class="${i === sel ? 'on' : ''}" data-w="${w}"><td class="n rk">${i + 1}</td><td class="mono">${w}</td><td class="n">${V.count[i]}</td><td class="bar"><i style="width:${100 * V.count[i] / maxc}%"></i></td></tr>`).join('')}</tbody></table>`;
  $('m2-stats').textContent = `${W.length} words in the vocabulary · ${V.total} words of running text · ${V.sents.length} sentences`;
  $('m2-seed').textContent = `seed ${G.seed}`;
}
function selWord2(w) {
  if (!w || !V.index.has(w)) return;
  selectPair(w, contexts6(w).includes(S4.ctx) ? S4.ctx : contexts6(w, true), '2');   // center of the page's selected pair
}
['m2-corpus', 'm2-vocab'].forEach(id => $(id).addEventListener('click', e => { const el = e.target.closest('[data-w]'); if (el) selWord2(el.dataset.w); }));
// retrain the whole page's model, after the seed or the learning rate changes
function retrain(prev) { stopAll(); G.prev = prev; resetTraining(); loadPair6(); S8A.key = ''; restart4(); renderAll(); }
$('reroll').addEventListener('click', () => { const prev = snapshotRun(); G.seed = 1 + Math.floor(Math.random() * 9999); retrain(prev); });

// ---------- 2 an example input vector, at the beginning, middle and end of training ----------
// Four fixed words, plus a fifth slot: a random word at first, then the selected pair's center
// word whenever that is not one of the four.
const FIXED2V = ['coffee', 'tea', 'dog', 'cat'];
const S2V = { word: 'coffee', stage: 'start', extra: (o => o[Math.floor(Math.random() * o.length)])(W.filter(w => !FIXED2V.includes(w))) };
function follow2v() {                      // called when the selected pair changes: show its center word
  if (!FIXED2V.includes(S4.center)) S2V.extra = S4.center;   // a word that isn't one of the four takes the fifth slot
  S2V.word = S4.center;
  render2(); render2v();
}
function render2v() {
  const M = { start: G.init, middle: G.mid, end: G.final }[S2V.stage], v = M.Win[V.index.get(S2V.word)];
  const top = Math.max(...G.final.Win.flat().map(Math.abs));          // one scale for all three stages
  // a column vector, as in section 4's table: one dimension per line, label on the right
  const cell = (x, d) => `<span class="m2v-cell"><span class="d">d${d + 1}</span><span class="v" style="background:${x >= 0 ? 'rgba(42,120,214,' : 'rgba(235,104,52,'}${(0.08 + 0.42 * Math.abs(x) / top).toFixed(3)})">${num(x, 2)}</span></span>`;
  $('m2v-vec').innerHTML = `<div class="m2v-row">
    <span class="m2v-word">${esc(S2V.word)}<span class="sub">input vector</span></span>
    <span class="m2v-arrow" aria-hidden="true">→</span>
    <span class="m2v-br">${v.map(cell).join('')}</span></div>`;
  $('m2v-note').textContent = {
    start: `Five random numbers between −0.1 and 0.1. Nothing about “${S2V.word}” is in them yet.`,
    middle: 'Halfway through training the numbers have already grown and moved. Switch between the words to compare.',
    end: 'Training nudges the numbers up and down to bring similar words closer together. Notice how the numbers for “coffee” become more similar to “tea” than to “dog”.',
  }[S2V.stage];
  const ex = $('m2v-extra'); ex.dataset.v = S2V.extra; ex.textContent = S2V.extra;
  document.querySelectorAll('#m2v-word button').forEach(b => b.classList.toggle('on', b.dataset.v === S2V.word));
  document.querySelectorAll('#m2v-state button').forEach(b => b.classList.toggle('on', b.dataset.v === S2V.stage));
}
$('m2v-word').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S2V.word = b.dataset.v; render2v(); } });
$('m2v-state').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S2V.stage = b.dataset.v; render2v(); } });

// ---------- formulas: in words or in symbols ----------
// One switch for the whole page; the choice is remembered in this browser when it can be.
function setFx(m) {
  document.body.classList.toggle('fx-sym', m === 's');
  document.querySelectorAll('.fxsw button').forEach(b => b.classList.toggle('on', b.dataset.fx === m));
  try { localStorage.setItem('w2v-fx', m); } catch (e) {}
}
document.querySelectorAll('.fxsw').forEach(sw => sw.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setFx(b.dataset.fx); }));
setFx((() => { try { return localStorage.getItem('w2v-fx') || 'w'; } catch (e) { return 'w'; } })());

// ---------- callouts with a hidden answer ----------
document.querySelectorAll('.reveal-btn').forEach(b => b.addEventListener('click', () => {
  const body = b.nextElementSibling, open = body.hidden;
  body.hidden = !open; b.setAttribute('aria-expanded', open); b.textContent = open ? 'Hide answer' : 'Reveal answer';
}));

// ---------- 3 the window ----------
// A sandbox, separate from training: pick a sentence, drag the window, see the pairs it makes.
// (Training shrinks the window at random and drops frequent words; the text explains that.)
const S3 = { sent: 0, pos: 3, b: 2, shown: new Set() };

// ---------- the selected pair ----------
// One (center, context) pair is selected for the whole page: clicked in 3, or picked in 4 or 5.
// It is real if the two words ever sit within a full window (±5) of each other in the corpus.
const REAL = new Set();
V.sents.forEach(s => s.forEach((c, i) => s.forEach((x, j) => { if (i !== j && Math.abs(i - j) <= CFG.WINDOW) REAL.add(c + ',' + x); })));
const isReal = (c, x) => REAL.has(V.index.get(c) + ',' + V.index.get(x));
function selectPair(c, x, from) {
  S4.center = c; S4.ctx = x;
  if (from !== '3') sync3(); else pairs3();
  load4(); if (S4.math) play4(); else render4(); render5(); renderStatebar(); follow2v(); sync6(); render8a(); refreshBars();
}
// 6–7 step through the selected pair; a negative pair never gets a positive update, so then
// they show the center word with its most common real neighbour instead (and say so)
function sync6() {
  const ok = contexts6(S4.center).includes(S4.ctx);
  P6.center = S4.center; P6.ctx = ok ? S4.ctx : contexts6(S4.center, true); P6.stand = ok ? null : S4.ctx;
  stopAll(); loadPair6(); render6(); render7();
}
// show the selected pair in 3: a real pair at a place where it falls inside the window
// (widening the window only if it must), a fake one with its center word at the center
function sync3() {
  const c = V.index.get(S4.center), x = V.index.get(S4.ctx), cur = V.sents[S3.sent];
  if (cur[S3.pos] === c && (isReal(S4.center, S4.ctx) ? cur.some((w, j) => w === x && j !== S3.pos && Math.abs(j - S3.pos) <= S3.b) : true)) { render3(); return; }
  let best = null;
  V.sents.forEach((s, k) => s.forEach((w, i) => { if (w !== c) return; s.forEach((y, j) => { if (y === x && j !== i) { const d = Math.abs(i - j); if (!best || d < best.d) best = { k, i, d }; } }); }));
  if (best && best.d <= CFG.WINDOW) { S3.sent = best.k; S3.pos = best.i; S3.b = Math.max(S3.b, best.d); }
  else { const k = V.sents.findIndex(s => s.includes(c)); S3.sent = k; S3.pos = V.sents[k].indexOf(c); }
  render3();
}
// after the window moves in 3, keep the selected context if it is still inside, else take the nearest word
function follow3() {
  const s = V.sents[S3.sent], c = S3.pos, ks = s.map((w, k) => k).filter(k => k !== c && Math.abs(k - c) <= S3.b);
  const keep = W[s[c]] === S4.center && ks.some(k => W[s[k]] === S4.ctx);
  const k = keep ? null : ks.sort((a, b) => Math.abs(a - c) - Math.abs(b - c) || b - a)[0];
  selectPair(W[s[c]], keep ? S4.ctx : W[s[k]], '3');
}
function render3() {
  const s = V.sents[S3.sent], c = S3.pos, b = S3.b;
  $('m3-track').innerHTML = s.map((w, i) => `<span class="chip ${i === c ? 'v' : Math.abs(i - c) <= b ? 'vo' : ''}" data-i="${i}">${W[w]}</span>`).join('')
    + `<span class="m3band" id="m3-band"><span class="m3lbl">window ±${b}</span><span class="m3grip" aria-hidden="true"><svg viewBox="0 0 16 6" width="16" height="6"><g fill="currentColor"><circle cx="2" cy="1.2" r="1.1"/><circle cx="8" cy="1.2" r="1.1"/><circle cx="14" cy="1.2" r="1.1"/><circle cx="2" cy="4.8" r="1.1"/><circle cx="8" cy="4.8" r="1.1"/><circle cx="14" cy="4.8" r="1.1"/></g></svg></span></span>`;
  place3();
  pairs3();
  $('m3-sent-pick').value = S3.sent;
  $('m3-bv').textContent = b; $('m3-minus').disabled = b <= 1; $('m3-plus').disabled = b >= CFG.WINDOW;
}
// size the band to cover the window's chips; only classes and the band move during a drag
function place3() {
  const chips = [...$('m3-track').querySelectorAll('.chip')], n = chips.length;
  const lo = chips[Math.max(0, S3.pos - S3.b)], hi = chips[Math.min(n - 1, S3.pos + S3.b)], band = $('m3-band');
  band.style.left = (lo.offsetLeft - 5) + 'px';
  band.style.width = (hi.offsetLeft + hi.offsetWidth - lo.offsetLeft + 10) + 'px';
  chips.forEach((ch, i) => ch.className = 'chip ' + (i === S3.pos ? 'v' : Math.abs(i - S3.pos) <= S3.b ? 'vo' : ''));
  // on a narrow screen the sentence can overflow: keep the window in view
  const sc = $('m3-track').parentElement, L = lo.offsetLeft - 8, R = hi.offsetLeft + hi.offsetWidth + 8;
  if (R > sc.scrollLeft + sc.clientWidth) sc.scrollLeft = R - sc.clientWidth;
  if (L < sc.scrollLeft) sc.scrollLeft = L;
}
function setCenter3(i) {
  if (i === S3.pos) return;
  S3.pos = i; place3(); follow3();
}
// list the window's pairs; the context chip of any pair not shown a moment ago bubbles in
function pairs3() {
  const s = V.sents[S3.sent], c = S3.pos, key = k => `${S3.sent}:${c}:${k}`;
  const ks = s.map((w, k) => k).filter(k => k !== c && Math.abs(k - c) <= S3.b);
  let n = 0;
  $('m3-pairs').innerHTML = ks.map(k => {
    // a new pair first opens up its space, then its context word pops in
    const on = W[s[c]] === S4.center && W[s[k]] === S4.ctx ? ' on' : '', data = `data-c="${W[s[c]]}" data-x="${W[s[k]]}"`;
    if (S3.shown.has(key(k))) return `<span class="pairchip${on}" ${data}>${wchip(W[s[c]], 'v')}${wchip(W[s[k]], 'vo')}</span>`;
    const d = 90 * n++;
    return `<span class="pairchip grow${on}" ${data} style="animation-delay:${d}ms">${wchip(W[s[c]], 'v')}<span class="chip vo pop" style="animation-delay:${d + 160}ms">${esc(W[s[k]])}</span></span>`;
  }).join('');
  const shown = new Set(ks.map(key));
  // three fake pairs: words drawn by noise frequency (as in 7) that never sit near this center
  // word in the corpus; seeded by the center's place so the same spot shows the same ones.
  // A selected fake pair for this center always takes the first place.
  const cw = W[s[c]], r = rng(1000 * S3.sent + c + 7), fakes = [];
  if (S4.center === cw && !isReal(cw, S4.ctx) && S4.ctx !== cw) fakes.push(V.index.get(S4.ctx));
  for (let tries = 0; fakes.length < 3 && tries < 300; tries++) { const w = drawNoise(r); if (w !== s[c] && !isReal(cw, W[w]) && !fakes.includes(w)) fakes.push(w); }
  // the same entrance as the positive pairs, following on after them
  $('m3-fakes').innerHTML = fakes.map(w => {
    const k = `${S3.sent}:${c}:n${w}`, on = S4.center === cw && S4.ctx === W[w] ? ' on' : '', data = `data-c="${cw}" data-x="${W[w]}"`;
    shown.add(k);
    if (S3.shown.has(k)) return `<span class="pairchip fake${on}" ${data}>${wchip(cw, 'v')}${wchip(W[w], 'neg')}</span>`;
    const d = 90 * n++;
    return `<span class="pairchip fake grow${on}" ${data} style="animation-delay:${d}ms">${wchip(cw, 'v')}<span class="chip neg pop" style="animation-delay:${d + 160}ms">${esc(W[w])}</span></span>`;
  }).join('');
  S3.shown = shown;
}
function setSent3(k) { S3.sent = k; S3.pos = Math.min(S3.pos, V.sents[S3.sent].length - 1); render3(); follow3(); }
$('m3-sent-pick').innerHTML = V.sents.map((s, k) => `<option value="${k}">${s.map(w => W[w]).join(' ')}</option>`).join('');
$('m3-sent-pick').addEventListener('change', e => setSent3(+e.target.value));
$('m3-minus').addEventListener('click', () => { S3.b = Math.max(1, S3.b - 1); render3(); follow3(); });
$('m3-plus').addEventListener('click', () => { S3.b = Math.min(CFG.WINDOW, S3.b + 1); render3(); follow3(); });
['m3-pairs', 'm3-fakes'].forEach(id => $(id).addEventListener('click', e => { const x = e.target.closest('[data-x]'); if (x) selectPair(x.dataset.c, x.dataset.x, '3'); }));
window.addEventListener('resize', () => { place3(); if (typeof render8 === 'function') render8(); });
(function drag3() {
  // press and drag moves the window; a press that doesn't move picks the word as center
  const track = $('m3-track'); let g = null;
  const mid = el => el.offsetLeft + el.offsetWidth / 2;
  const nearest = x => { let best = 0, d = Infinity; track.querySelectorAll('.chip').forEach((ch, i) => { const e = Math.abs(mid(ch) - x); if (e < d) { d = e; best = i; } }); return best; };
  const xOf = e => e.clientX - track.getBoundingClientRect().left;
  track.addEventListener('pointerdown', e => {
    const chip = e.target.closest('.chip'), x = xOf(e);
    g = { x0: x, off: x - mid(track.querySelectorAll('.chip')[S3.pos]), chip: chip ? +chip.dataset.i : null, moved: false };
    track.setPointerCapture(e.pointerId); track.classList.add('dragging'); e.preventDefault();
  });
  track.addEventListener('pointermove', e => {
    if (!g) return;
    const x = xOf(e);
    if (!g.moved && Math.abs(x - g.x0) < 5) return;
    g.moved = true; setCenter3(nearest(x - g.off));
  });
  const end = e => { if (g && !g.moved && g.chip != null && e.type === 'pointerup') setCenter3(g.chip); g = null; track.classList.remove('dragging'); };
  track.addEventListener('pointerup', end); track.addEventListener('pointercancel', end);
})();

// ---------- 4 the dot product ----------
// any center word's v against any context word's v′, at the start, middle or end of training
const S4 = { center: 'coffee', ctx: 'hot', stage: 'start', edited: false, math: false, step: 0, v: null, u: null };
function load4() {
  const M = { start: G.init, middle: G.mid, trained: G.final }[S4.stage];
  S4.v = M.Win[V.index.get(S4.center)].slice(); S4.u = M.Wout[V.index.get(S4.ctx)].slice(); S4.edited = false;
  ['m4', 'm5'].forEach(k => {               // 5 shows the same pair, with its own copy of the controls
    const cs = $(k + '-center'), xs = $(k + '-ctx');
    if (!cs.options.length) W.slice().sort().forEach(w => { cs.add(new Option(w, w)); xs.add(new Option(w, w)); });
    cs.value = S4.center; xs.value = S4.ctx;
    $(k + '-stage').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === S4.stage));
  });
}
function restart4() { load4(); render4(); render5(); }   // after a re-roll: new initial and trained models
function score4() { return dot(S4.v, S4.u); }
function render4() {
  const prod = S4.v.map((x, d) => x * S4.u[d]), run = prod.reduce((a, b) => a + b, 0), real = isReal(S4.center, S4.ctx);
  // "show math" opens each vector into its column (as in section 2) and works through one
  // dimension at a time: the two numbers light up, their product appears, then the sum
  // steps 1–5 multiply one dimension each; steps 6–10 add the products one by one; then done
  const st = S4.math ? S4.step : 2 * D + 1, cur = st - 1, mult = st >= 1 && st <= D, k = st - D;   // k: products added so far
  // the products column spells out the row being worked on: v × v′ = product
  const col = (a, cls, isProd) => `<span class="m4br ${cls}${isProd && S4.math && st <= D ? ' working' : ''}">${a.map((x, d) => {
    const shown = !isProd || d < st, on = S4.math && mult && d === cur ? ' on' : '', summed = isProd && k >= 1 && k <= D && d < k ? (d === k - 1 ? ' on' : ' sumd') : '';
    const txt = shown ? num(x, 2) : '·';
    const fresh = isProd && mult && d === cur ? ' cur' : '';   // the product just made
    return `<span class="c${on}${summed}${fresh}">${txt}</span>`;
  }).join('')}</span>`;
  const term = (l, v, f, cls = '') => `<div class="term ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="f">${f}</div></div>`;
  const cc = wchip(S4.center, 'v'), xc = wchip(S4.ctx, real ? 'vo' : 'neg'), done = st > 2 * D, adding = k >= 1 && k <= D;
  const sofar = adding ? prod.slice(0, k).reduce((a, b) => a + b, 0) : run;
  $('m4-eq').innerHTML = (S4.math
    ? term(cc, col(S4.v, 'cv'), '', 'sym open') + '<div class="op"><span class="dotch">·</span></div>' + term(xc, col(S4.u, 'cvo'), '', 'sym open')
      + `<div class="m4scoregrp"><div class="op"><span class="eqch">=</span> <span class="sumw">sum</span></div>${term('products', `<span class="m4paren">${col(prod, 'cp', true)}</span>`, '', 'sym open')}</div>`   // breaks before the =, never after
    : term('center', SV, cc, 'sym') + '<div class="op"><span class="dotch">·</span></div>' + term('context', SVO, xc, 'sym'))
    // "= score" travels as one piece, so on a narrow screen it wraps onto its own line
    + `<div class="m4scoregrp"><div class="op">=</div>${term(adding ? 'sum so far' : 'score', done || adding ? num(adding ? sofar : run, 3) : '?', adding ? `${k} of ${D} products` : '', 'score' + (adding ? ' adding' : ''))}</div>`
    + `<button class="m4math" id="m4-math">${S4.math ? 'Hide math' : 'Show math'}</button>`;
  $('m4-ctl').hidden = !S4.math; $('m4-eq').classList.toggle('mathon', S4.math);
  // on the row being worked on, a × and an = sit between its three numbers
  $('m4-eq').classList.toggle('working', S4.math && !done);
  $('m4-eq').classList.toggle('mult', S4.math && mult); $('m4-eq').classList.toggle('adding', S4.math && adding);
  if (S4.math && mult) {
    const box = $('m4-eq').getBoundingClientRect(), cell = k => $('m4-eq').querySelectorAll(`.m4br.${k} .c`)[cur].getBoundingClientRect();
    const a = cell('cv'), b = cell('cvo'), c = cell('cp'), y = a.top + a.height / 2 - box.top;
    const mark = (x, t) => `<span class="m4op" style="left:${x - box.left}px;top:${y}px">${t}</span>`;
    // each badge sits right over the faded sign it stands for
    const over = q => { const r = $('m4-eq').querySelector(q).getBoundingClientRect(); return r.left + r.width / 2; };
    // one band across the whole row, from the v number to its product
    const band = `<span class="m4band" style="left:${a.left - 14 - box.left}px;top:${a.top - 5 - box.top}px;width:${c.right - a.left + 28}px;height:${a.height + 10}px"></span>`;
    $('m4-eq').insertAdjacentHTML('beforeend', band + mark(over('.dotch'), '×') + mark((b.right + c.left) / 2, '='));   // = halfway between the v′ number and its product
  }
  // while adding, one band around the products added so far, growing a row each step,
  // and a + between the product just added and the one above it
  if (S4.math && adding) {
    const box = $('m4-eq').getBoundingClientRect(), cs = $('m4-eq').querySelectorAll('.m4br.cp .c'), f = cs[0].getBoundingClientRect(), l = cs[k - 1].getBoundingClientRect();
    // room on the left inside the band for the +, which sits on the newest (bottom) row
    let h = `<span class="m4band" style="left:${f.left - 30 - box.left}px;top:${f.top - 5 - box.top}px;width:${f.width + 38}px;height:${l.bottom - f.top + 10}px"></span>`;
    if (k >= 2) h += `<span class="m4op plus" style="left:${f.left - 15 - box.left}px;top:${l.top + l.height / 2 - box.top}px">+</span>`;
    $('m4-eq').insertAdjacentHTML('beforeend', h);
  }
  $('m4-sum').innerHTML = cmp4(run);
  const kind = `<span class="badge ${real ? 'pos' : 'neg'}">${real ? 'positive pair' : 'negative pair'}</span>`;
  $('m4-kind').innerHTML = kind; $('m5-kind').innerHTML = kind;   // 5 shows the same pair
}
// the math runs on the page's usual play bar: 2 × 5 steps (multiply each dimension, then add each product)
const bar4 = PlayBar($('m4-bar'), {
  ms: () => S4.step < D ? 700 : 600, canStep: () => S4.step <= 2 * D, canReset: () => S4.step > 0, rewind: () => { S4.step = 0; },
  step: () => { S4.step++; render4(); }, end: () => { S4.step = 2 * D + 1; render4(); }, reset: () => { S4.step = 0; render4(); },
});
function play4() { bar4.stop(); S4.step = 0; render4(); refreshBars(); bar4.act('play'); }
$('m4-eq').addEventListener('click', e => {
  if (e.target.closest('#m4-math')) { S4.math = !S4.math; if (S4.math) play4(); else { bar4.stop(); render4(); } }
});
function pick4() { stopAll(); load4(); if (S4.math) play4(); else render4(); render5(); renderStatebar(); render8a(); sync6(); refreshBars(); }   // the stage is shared: 6–7 follow it too
['m4', 'm5'].forEach(k => {
  $(k + '-center').addEventListener('change', e => { stopAll(); selectPair(e.target.value, S4.ctx, k); });
  $(k + '-ctx').addEventListener('change', e => { stopAll(); selectPair(S4.center, e.target.value, k); });
  $(k + '-stage').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S4.stage = b.dataset.v; pick4(); } });
});

// beside the sum: the same center word's lowest score with any word, for comparison
function cmp4(run) {
  if (S4.edited) return '';
  if (Math.abs(run) < 1e-12) return '<b>Score: 0.</b> Every score is 0 at the start, because v′ is all zeros.';
  const M = { start: G.init, middle: G.mid, trained: G.final }[S4.stage], v = M.Win[V.index.get(S4.center)];
  let lo = 0; M.Wout.forEach((u, i) => { if (dot(v, u) < dot(v, M.Wout[lo])) lo = i; });
  const me = `<b>Score</b> for ${wchip(S4.center, 'v')} · ${wchip(S4.ctx, isReal(S4.center, S4.ctx) ? 'vo' : 'neg')}: <b>${num(run, 2)}</b>`;
  if (W[lo] === S4.ctx) return `${me}, the lowest score of any word with “${esc(S4.center)}”.`;
  return `${me}. Compare ${wchip(S4.center, 'v')} · ${wchip(W[lo], 'neg')}: ${num(dot(v, M.Wout[lo]), 2)}, the lowest score of any word with “${esc(S4.center)}”.`;
}

// ---------- 5 the sigmoid ----------
const XM = 8;
function render5() {
  const x = score4(), s = sigmoid(x), lab = S4.edited || isReal(S4.center, S4.ctx) ? 1 : 0;   // the selected pair's score, and its right answer
  const W0 = 700, H0 = 250, l = 62, r = 20, t = 14, b = 48;
  const X = v => l + (Math.max(-XM, Math.min(XM, v)) + XM) / (2 * XM) * (W0 - l - r), Y = v => t + (1 - v) * (H0 - t - b);
  let path = ''; for (let i = 0; i <= 200; i++) { const v = -XM + 2 * XM * i / 200; path += (i ? 'L' : 'M') + X(v).toFixed(1) + ',' + Y(sigmoid(v)).toFixed(1); }
  const ticks = [-8, -6, -4, -2, 0, 2, 4, 6, 8].map(v => `<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(0) + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${Y(0) + 17}" text-anchor="middle" font-size="11">${num(v, 0)}</text>`).join('');
  const yt = [0, 0.5, 1].map(v => `<line x1="${l}" y1="${Y(v)}" x2="${W0 - r}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${l - 8}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="11">${v}</text>`).join('');
  const off = Math.abs(x) > XM;
  $('m5-chart').innerHTML = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block">${yt}${ticks}
    <line x1="${X(0)}" y1="${t}" x2="${X(0)}" y2="${Y(0)}" stroke="var(--grid)"/>
    <path d="${path}" fill="none" stroke="var(--ink)" stroke-width="2"/>
    <line x1="${X(x)}" y1="${Y(s)}" x2="${X(x)}" y2="${Y(lab)}" stroke="var(--orange)" stroke-width="2"/>
    <line x1="${l}" y1="${Y(s)}" x2="${X(x)}" y2="${Y(s)}" stroke="var(--ink-2)" stroke-dasharray="3 3"/>
    <line x1="${X(x)}" y1="${Y(s)}" x2="${X(x)}" y2="${Y(0)}" stroke="var(--ink-2)" stroke-dasharray="3 3"/>
    <rect x="${X(x) - 27}" y="${Y(0) + 5}" width="54" height="18" rx="9" fill="var(--ink)"/>
    <text x="${X(x)}" y="${Y(0) + 15}" text-anchor="middle" dominant-baseline="middle" font-size="11" font-weight="600" fill="white">${num(x, 3)}</text>
    <circle cx="${X(x)}" cy="${Y(s)}" r="7" fill="var(--ink)" stroke="white" stroke-width="2"/>
    <circle cx="${X(x)}" cy="${Y(lab)}" r="6" fill="${lab ? '#1d5fae' : '#8a3a1a'}" stroke="white" stroke-width="2"/>
    <text x="${X(x) + (x > 5 ? 11 : -11)}" y="${Y(lab) + (lab ? 4 : -2)}" font-size="11" font-weight="600" fill="${lab ? '#1d5fae' : '#8a3a1a'}" text-anchor="${x > 5 ? 'start' : 'end'}">label = ${lab}</text>
    <text x="${X(x) + (x > 5 ? -12 : 12)}" y="${Y(lab) + (Y(s) - Y(lab)) / 2}" font-size="11" font-weight="600" fill="var(--orange)" text-anchor="${x > 5 ? 'end' : 'start'}" dominant-baseline="middle">error = ${num(lab - s, 3)}</text>
    <text x="${l + (W0 - l - r) / 2}" y="${H0 - 6}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--ink-2)">score = v · v′${off ? ' — off the chart' : ''}</text>
    <text transform="translate(16,${t + (Y(0) - t) / 2}) rotate(-90)" text-anchor="middle" font-size="12" font-weight="600" fill="var(--ink-2)">probability = σ(score)</text>
  </svg>`;
  // the pair and its score, then one equation read left to right: label − σ(score) = error
  const term = (cls, l, v, f) => `<div class="term ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="f">${f}</div></div>`;
  $('m5-chain').innerHTML = `<div class="m5eq">${term(lab ? 'lab1' : 'lab0', 'label', lab, lab ? 'positive pair' : 'negative pair')}<div class="op">−</div>
      ${term('prob', 'σ(score)', s.toFixed(3), `σ(${num(x, 3)})`)}<div class="op">=</div>
      ${term('err', 'error', num(lab - s, 3), '&nbsp;')}</div>`;
}


// ---------- a vector equation with "show math" (used in 6) ----------
// get() returns { left: [items], result: { label, chip, vals, cls } }, items being
// { t: 'vec', label, chip, vals, cls }, { t: 'scal', label, val, f, cls } or { t: 'op', s }.
// Collapsed, the vectors on the left show as symbols and the result as its column; "show math"
// opens them into columns and works out the result one row at a time on a play bar.
function VecEq(host, get, dec = 2, cls = '') {   // dec: decimal places shown in the columns; cls: extra class
  const me = { math: false, step: 0 };
  host.innerHTML = `<div class="m4wrap"><div class="m5eq m4eq vq ${cls}"></div><div class="m4ctl" hidden><span></span></div></div>`;
  const eq = host.querySelector('.m4eq'), ctl = host.querySelector('.m4ctl');
  const col = (vals, cls, rowOf) => `<span class="m4br ${cls}">${vals.map((x, d) => `<span class="c${rowOf(d)}">${x == null ? '·' : num(x, dec)}</span>`).join('')}</span>`;
  const term = (l, v, f, cls = '') => `<div class="term ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="f">${f}</div></div>`;
  me.render = () => {
    const g = get(); if (!g) { eq.innerHTML = ''; ctl.hidden = true; return; }
    const st = me.math ? me.step : D + 1, cur = st - 1, work = me.math && st >= 1 && st <= D;
    const rowOf = d => work && d === cur ? ' on' : '';
    const res = g.result, resVals = res.vals.map((x, d) => d < st ? x : null);
    let h = g.left.map(it => it.t === 'op' ? `<div class="op">${it.s}</div>` : it.t === 'paren' ? `<div class="paren">${it.s}</div>`
      : it.t === 'scal' ? term(it.label, `<span class="${it.cls || ''}">${it.val}</span>`, it.f || '', 'scal')
      : me.math ? term(it.chip, col(it.vals, it.cls, rowOf), '', 'sym open') : term(it.label, it.sym, it.chip, 'sym')).join('');
    h += '<div class="op">=</div>' + term(me.math ? res.chip : res.label, col(resVals, res.cls + ' res', d => (work && d === cur ? ' on cur' : '')), me.math ? '' : res.chip, 'sym open');
    h += `<button class="m4math">${me.math ? 'Hide math' : 'Show math'}</button>`;
    eq.innerHTML = h; ctl.hidden = !me.math;
    eq.classList.toggle('mathon', me.math); eq.classList.toggle('working', work); eq.classList.toggle('mult', work);
    if (work) {   // one band across the row being worked out
      const box = eq.getBoundingClientRect(), cells = [...eq.querySelectorAll('.m4br')].map(b => b.querySelectorAll('.c')[cur].getBoundingClientRect());
      const a = cells[0], z = cells[cells.length - 1];
      const firstScal = eq.querySelector(':scope > .term.scal > .v'), sr = firstScal && firstScal.getBoundingClientRect();
      const left = sr && sr.left < a.left ? sr.left - 6 : a.left;
      const y = a.top + a.height / 2 - box.top, cx = el => { const r = el.getBoundingClientRect(); return r.left + r.width / 2 - box.left; };
      let h2 = `<span class="m4band" style="left:${left - 14 - box.left}px;top:${a.top - 5 - box.top}px;width:${z.right - left + 28}px;height:${a.height + 10}px"></span>`;
      // the whole row's sum on the band: each sign over its faded one, and the scalar as a small pill
      eq.querySelectorAll(':scope > .op').forEach(o => { h2 += `<span class="m4op" style="left:${cx(o)}px;top:${y}px">${o.textContent.trim()}</span>`; });
      eq.querySelectorAll(':scope > .term.scal').forEach(t => { const v = t.querySelector(':scope > .v'); h2 += `<span class="m4pill" style="left:${cx(v)}px;top:${y}px">${v.textContent}</span>`; });
      eq.insertAdjacentHTML('beforeend', h2);
    }
  };
  me.bar = PlayBar(ctl.querySelector('span'), {
    ms: 800, canStep: () => me.step <= D, canReset: () => me.step > 0, rewind: () => { me.step = 0; },
    step: () => { me.step++; me.render(); }, end: () => { me.step = D + 1; me.render(); }, reset: () => { me.step = 0; me.render(); },
  });
  me.play = () => { me.bar.stop(); me.step = 0; me.render(); refreshBars(); me.bar.act('play'); };
  eq.addEventListener('click', e => { if (e.target.closest('.m4math')) { me.math = !me.math; if (me.math) me.play(); else { me.bar.stop(); me.render(); } } });
  return me;
}

// ---------- 6 one positive update ----------
function render6() {
  $('m6-neg').hidden = true; $('m6-body').hidden = false;
  const p = P6.pair, R = P6.R, t = t6(); if (!t) return;
  const lab = t.label, where = `pair ${commas(P6.idx + 1)} of ${commas(G.stream.total)}, pass ${p.epoch + 1} of ${CFG.EPOCHS}`;
  $('m6-head').innerHTML = `<p class="note" style="margin:0 0 10px">${lab ? where
    : `Here ${wchip(W[t.w], 'neg')} is drawn as a negative sample for the positive pair ${wchip(W[p.center], 'v')} · ${wchip(W[p.ctx], 'vo')} (${where}). Its right answer is 0, so the same update pushes the vectors apart.`}</p>`;
  // the scaling factor: error × learning rate = g
  const sc = (cls, l, v, f) => `<div class="term ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="f">${f}</div></div>`;
  $('m6-g').innerHTML = sc('err', 'error', num(lab - t.s, 3), `${lab} − σ(${num(t.f, 2)}), as in section 5`) + '<div class="op">×</div>'
    + sc('', 'learning rate α', R.alpha.toFixed(4), 'at this point in training') + '<div class="op">=</div>'
    + sc('gval', `scaling factor ${SG}`, num(t.g, 5), t.g > 0 ? 'positive: pull together' : 'negative: push apart');
  EQ6U.render(); EQ6P.render();
  $('m6-zero').hidden = norm(t.vo0) !== 0;
  // pickers
  const cs = $('m6-center'), xs = $('m6-ctx');
  if (!cs.options.length) W.slice().sort().forEach(w => cs.add(new Option(w, w)));
  cs.value = P6.center;
  const opts = contexts6(P6.center).concat(P6.stand ? [P6.stand] : []).sort();   // keep a negative selection in the list
  xs.innerHTML = opts.map(w => `<option value="${w}">${w}</option>`).join(''); xs.value = P6.stand || P6.ctx;
  $('m6-kind').innerHTML = `<span class="badge ${P6.stand ? 'neg' : 'pos'}">${P6.stand ? 'negative pair · label = 0' : 'positive pair · label = 1'}</span>`;   // as in 4 and 5
  $('m6-stage').querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.v === S4.stage));
}
// the pair section 6 works on: the positive pair, or the selected negative sample within it
function t6() {
  if (!P6.R) return null;
  if (!P6.stand) return P6.R.T[0];
  const ni = V.index.get(P6.stand); return P6.R.T.find(t => t.label === 0 && !t.skip && t.w === ni) || null;
}
// the two updates, each with its own "show math"
const EQ6U = VecEq($('m6-upd'), () => {
  const t = t6(); if (!t) return null; const c = W[P6.pair.center], x = W[t.w], xc = t.label ? 'vo' : 'neg';
  // v′ + (g × v): start from v′ and add the scaled v; the parentheses say the multiplication comes first
  return { left: [{ t: 'vec', label: 'context', sym: SVO, chip: wchip(x, xc), vals: t.vo0, cls: 'cvo' }, { t: 'op', s: '+' }, { t: 'paren', s: '(' },
    { t: 'scal', label: SG, val: num(t.g, 4), cls: 'cg' }, { t: 'op', s: '×' }, { t: 'vec', label: 'center', sym: SV, chip: wchip(c, 'v'), vals: P6.R.v0, cls: 'cv' }, { t: 'paren', s: ')' }],
    result: { label: `new ${SVO}`, chip: wchip(x, xc), vals: t.vo0.map((y, d) => y + t.dvo[d]), cls: 'cvo' } };
}, 4);
const EQ6P = VecEq($('m6-pend'), () => {
  const t = t6(); if (!t) return null; const x = W[t.w], c = W[P6.pair.center];
  return { left: [{ t: 'scal', label: SG, val: num(t.g, 4), cls: 'cg' }, { t: 'op', s: '×' }, { t: 'vec', label: 'context', sym: SVO, chip: wchip(x, t.label ? 'vo' : 'neg'), vals: t.vo0, cls: 'cvo' }],
    result: { label: `Δ${SV} so far`, chip: wchip(c, 'v'), vals: t.contrib, cls: 'cp' } };
}, 4);
function pick6() { stopAll(); loadPair6(); render6(); render7(); refreshBars(); }
// the pickers here change the page's selected pair, like those in 3–5
$('m6-center').addEventListener('change', e => { const c = e.target.value; selectPair(c, contexts6(c).includes(S4.ctx) ? S4.ctx : contexts6(c, true), '6'); });
$('m6-ctx').addEventListener('change', e => selectPair(S4.center, e.target.value, '6'));
$('m7-center').addEventListener('change', e => { const c = e.target.value; selectPair(c, contexts6(c).includes(S4.ctx) ? S4.ctx : contexts6(c, true), '7'); });
$('m7-ctx').addEventListener('change', e => selectPair(S4.center, e.target.value, '7'));
$('m6-stage').addEventListener('click', e => { const x = e.target.closest('button'); if (x) { S4.stage = x.dataset.v; pick4(); } });

// ---------- 7 negative samples ----------
function render7() {
  // pickers, as in 6: they change the page's selected pair
  const cs = $('m7-center'), xs = $('m7-ctx');
  if (!cs.options.length) W.slice().sort().forEach(w => cs.add(new Option(w, w)));
  cs.value = P6.center;
  xs.innerHTML = contexts6(P6.center).concat(P6.stand ? [P6.stand] : []).sort().map(w => `<option value="${w}">${w}</option>`).join(''); xs.value = P6.stand || P6.ctx;
  $('m7-kind').innerHTML = `<span class="badge ${P6.stand ? 'neg' : 'pos'}">${P6.stand ? 'negative pair' : 'positive pair'}</span>`;
  const neg = P6.stand ? `${wchip(P6.center, 'v')} · ${wchip(P6.stand, 'neg')} is one of the negative samples drawn for the positive pair ${wchip(P6.center, 'v')} · ${wchip(P6.ctx, 'vo')}. Here is that whole set of positive and negative pairs.` : '';
  $('m7-neg').innerHTML = neg; $('m7-neg').hidden = !neg; $('m7-body').hidden = false;
  const p = P6.pair, R = P6.R, ph = P6.phase, cur = P6.PL[ph];
  // frequency vs frequency^0.75
  const drawnUpTo = R.T.filter(t => t.label === 0 && ph >= t.drawAt);
  const hits = new Map(); drawnUpTo.forEach(t => hits.set(t.w, (hits.get(t.w) || 0) + 1));
  const W0 = 700, H0 = 210, l = 34, bt = 62, top = 16, n = W.length, bw = (W0 - l - 24) / n, mx = Math.max(...V.raw, ...V.noise);
  const Y = v => top + (1 - v / mx) * (H0 - bt - top);
  let svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block">`;
  [0, 0.05, 0.1].filter(v => v <= mx).forEach(v => { svg += `<line x1="${l}" x2="${W0}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${l - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${Math.round(v * 100)}%</text>`; });
  W.forEach((w, i) => {
    const x = l + i * bw, h = hits.get(i) || 0, isCtx = i === p.ctx;
    svg += `<g><title>${w}: ${(100 * V.raw[i]).toFixed(1)}% of words, ${(100 * V.noise[i]).toFixed(1)}% chance of being picked as a negative sample</title>
      <rect x="${x + 1}" y="${Y(V.raw[i])}" width="${bw / 2 - 1}" height="${Y(0) - Y(V.raw[i])}" fill="var(--axis)"/>
      <rect x="${x + bw / 2}" y="${Y(V.noise[i])}" width="${bw / 2 - 1}" height="${Y(0) - Y(V.noise[i])}" fill="${h ? 'var(--orange)' : '#f6c3ad'}"/>
      ${h ? `<text x="${x + bw * 0.75}" y="${Y(V.noise[i]) - 4}" text-anchor="middle" font-size="10" fill="var(--orange)">${h > 1 ? '×' + h : '●'}</text>` : ''}
      <text transform="translate(${x + bw / 2 + 3},${Y(0) + 6}) rotate(60)" font-size="10" font-family="var(--mono)" fill="${isCtx ? 'var(--aqua)' : 'var(--ink-2)'}" font-weight="${isCtx ? 700 : 400}">${w}</text></g>`;
  });
  $('m7-chart').innerHTML = svg + `<line x1="${l}" x2="${W0}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/></svg>`;
  // the draws
  const negs = R.T.slice(1);
  $('m7-draws').innerHTML = negs.map(t => ph >= t.drawAt
    ? (t.skip ? `<span class="chip skip" title="same word as the context: skipped">${W[t.w]}</span>` : `<span class="chip neg">${W[t.w]}</span>`)
    : '<span class="dempty"></span>').join('');
  EQ7S.render();
  EQ7.render();
  $('m7-applied').textContent = R.applied ? 'All the parts are in, so the update is applied to the input vector, once.' : 'Not applied yet: it waits until every negative sample has added its part.';
  let msg;
  if (ph < POS_END) msg = 'The positive update from section 6 comes first. Step to complete it.';
  else if (!cur || cur.k === 'pend') msg = `Next: draw ${CFG.K} noise words from the orange distribution.`;
  else if (cur.k === 'draw') { const t = R.T[cur.j + 1]; msg = t.skip ? `Drew “${W[t.w]}”, the context word itself. word2vec skips such draws.` : `Drew “${W[t.w]}”. It is treated as a negative pair: label = 0.`; }
  else if (cur.k === 'neg') { const t = R.T[cur.j + 1]; msg = `Negative pair with “${W[t.w]}”: σ = ${t.s.toFixed(4)}, so g = (0 − σ) × α = ${num(t.g, 6)} is negative. Its v′ is pushed away from v, and g·v′ is added to Δv so far.`; }
  else msg = `All ${CFG.K} draws done. Δv so far is added to v, once. That finishes this pair; training moves on to the next one.`;
  $('m7-note').textContent = msg;
}
// the parts added up: what each part is (g × v′ of its word), and "show math" for the numbers
const EQ7S = VecEq($('m7-sum'), () => {
  const R = P6.R; if (!R) return null; const ph = P6.phase, c = W[P6.pair.center], sel = P6.stand ? V.index.get(P6.stand) : -1;
  const parts = R.T.filter(t => !t.skip && ph >= t.pendAt), left = [];
  parts.forEach((t, i) => {
    if (i) left.push({ t: 'op', s: '+' });
    left.push({ t: 'vec', label: t.label ? 'positive' : 'negative', sym: `${SG}<span class="gx">·</span>${SVO}`, chip: wchip(W[t.w], t.label ? 'vo' : 'neg'),
      vals: t.contrib, cls: 'cp' + (!t.label && t.w === sel ? ' selpart' : '') });
  });
  return { left, result: { label: `Δ${SV} so far`, chip: wchip(c, 'v'), vals: R.pend, cls: 'cv' } };
}, 4, 'compact');
// v + Δv so far = new v, with "show math"
const EQ7 = VecEq($('m7-apply'), () => {
  const R = P6.R; if (!R) return null; const c = W[P6.pair.center];
  return { left: [{ t: 'vec', label: 'center', sym: SV, chip: wchip(c, 'v'), vals: R.v0, cls: 'cv' }, { t: 'op', s: '+' },
    { t: 'vec', label: `Δ${SV} so far`, sym: `Δ${SV}`, chip: wchip(c, 'v'), vals: R.pend, cls: 'cp' }],
    result: { label: `new ${SV}`, chip: wchip(c, 'v'), vals: R.v0.map((x, d) => x + R.pend[d]), cls: 'cv' } };
}, 4);
const bar7 = PlayBar($('m7-bar'), {
  ms: 900, canStep: () => P6.phase < END(), canReset: () => P6.phase > POS_END,
  step: () => { setPhase(P6.phase < POS_END ? POS_END : P6.phase + 1); render6(); render7(); },
  end: () => { setPhase(END()); render6(); render7(); },
  reset: () => { setPhase(POS_END); render6(); render7(); },
});

// ---------- 8 two words drawing together ----------
// The selected pair's center word against any other word, at the start, middle or end of
// training (shared with 4–5), plus the angle between their input vectors over the whole run.
const S8A = { other: null, series: null, key: '' };
const angle = (x, y) => Math.acos(Math.max(-1, Math.min(1, cosine(Array.from(x), Array.from(y))))) * 180 / Math.PI;
function other8a() {   // default: the word most like the center word at the end of training
  if (!S8A.other || S8A.other === S4.center) S8A.other = W[neighbors(G.final.Win, V.index.get(S4.center), 1)[0].i];
  return S8A.other;
}
function angleSeries(a, b) {   // [step, angle] at every log-spaced snapshot
  const key = G.seed + ':' + CFG.ALPHA0 + ':' + a + ',' + b; if (S8A.key === key) return S8A.series;
  S8A.series = G.wsnaps.map(w => [w.step, angle(w.Win[a], w.Win[b])]);
  S8A.key = key; return S8A.series;
}
function render8a() {
  const aw = S4.center, bw = other8a(), a = V.index.get(aw), b = V.index.get(bw), N = G.stream.total;
  const M = { start: G.init, middle: G.mid, trained: G.final }[S4.stage], at = { start: 0, middle: Math.floor(N / 2), trained: N }[S4.stage];
  $('m8a-plane').innerHTML = pairPlane(M.Win[a], M.Win[b], { aLabel: aw, bLabel: bw, bCls: 'a2', cos: true });
  const ser = angleSeries(a, b), cur = angle(M.Win[a], M.Win[b]);
  const cw = 340, ch = 190, cl = 40, cr = 10, ct = 12, cb = 30, X = v => cl + logx(v) * (cw - cl - cr), Y = v => ct + (1 - v / 180) * (ch - ct - cb);   // log time axis
  const grid = [0, 45, 90, 135, 180].map(v => `<line x1="${cl}" x2="${cw - cr}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${cl - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${v}°</text>`).join('');
  const ticks = LOG_TICKS().map(([x, l]) => `<text x="${X(x)}" y="${ch - 14}" text-anchor="${x === N ? 'end' : 'middle'}" font-size="9">${l}</text>`).join('')
    + `<text x="${X(N / 2)}" y="${ch - 26}" text-anchor="middle" font-size="9" fill="var(--muted)">middle</text>`;
  $('m8a-chart').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${grid}${ticks}
    <path d="${ser.map(([x, v], i) => (i ? 'L' : 'M') + X(x).toFixed(1) + ',' + Y(v).toFixed(1)).join('')}" fill="none" stroke="var(--violet)" stroke-width="2"/>
    <line x1="${X(at)}" x2="${X(at)}" y1="${ct}" y2="${ch - cb}" stroke="var(--ink-2)" stroke-dasharray="3 3"/>
    <circle cx="${X(at)}" cy="${Y(cur)}" r="5" fill="var(--violet)" stroke="white" stroke-width="2"/>
    <text x="${X(at) + (at > N * 0.8 ? -8 : 8)}" y="${Y(cur) - 10}" text-anchor="${at > N * 0.8 ? 'end' : 'start'}" font-size="11" font-weight="600" fill="var(--violet)">${Math.round(cur)}°</text>
    <text x="${cl + (cw - cl - cr) / 2}" y="${ch - 2}" text-anchor="middle" font-size="10" fill="var(--ink-2)">pairs trained (log scale)</text></svg>`;
  const cs = $('m8a-center'), os = $('m8a-other');
  if (!cs.options.length) W.slice().sort().forEach(w => { cs.add(new Option(w, w)); os.add(new Option(w, w)); });
  cs.value = aw; os.value = bw;
  $('m8a-stage').querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.v === S4.stage));
}
$('m8a-center').addEventListener('change', e => { const c = e.target.value; selectPair(c, contexts6(c).includes(S4.ctx) ? S4.ctx : contexts6(c, true), '8'); render8a(); });
$('m8a-other').addEventListener('change', e => { S8A.other = e.target.value; render8a(); });
$('m8a-stage').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S4.stage = b.dataset.v; pick4(); render8a(); } });

// ---------- 8 training at scale ----------
const S8 = { speed: 500, sel: 'coffee', showOut: false, zoom: 1, pan: [0, 0] };   // zoom and pan of the map, in PCA units   // speed: pairs trained per frame (a full run plays in about 5 s)
function render8() {
  const M = G.model, sel = V.index.get(S8.sel);
  const pc = pca2(M.Win, G.pca && G.pca.comps); G.pca = pc;
  const ptsIn = M.Win.map(pc.project), ptsOut = S8.showOut ? M.Wout.map(pc.project) : [];
  const all = ptsIn.concat(ptsOut), want = 1.12 * Math.max(1e-6, ...all.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y))));
  G.rad = !G.rad || want > G.rad ? want : G.rad * 0.92 + want * 0.08;   // grow at once, shrink gently
  // the map's shape follows its width: wide on a big screen, squarer as it narrows
  const box = $('m8-scatter').clientWidth || 700, W0 = 700, H0 = box < 420 ? 700 : box < 560 ? 560 : 440;
  const cx = W0 / 2, cy = H0 / 2, k = Math.min(W0, H0) / 2 / G.rad * S8.zoom;
  const P = ([x, y]) => [cx + (x - S8.pan[0]) * k, cy - (y - S8.pan[1]) * k];
  S8.k = k; S8.W0 = W0; S8.selPt = ptsIn[sel];   // for dragging and zooming
  const nb = neighbors(M.Win, sel, 5), nbs = new Set(nb.map(x => x.i));
  const [ox, oy] = P([0, 0]);   // the axes follow the pan
  // labels and dots are sized in screen pixels (the drawing is scaled to fit its box), then zoom magnifies them
  const z = (W0 / box) * S8.zoom;
  let svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block;--z:${z}" id="scatter">
    <line x1="0" x2="${W0}" y1="${oy}" y2="${oy}" stroke="var(--grid)"/><line x1="${ox}" x2="${ox}" y1="0" y2="${H0}" stroke="var(--grid)"/>`;
  nb.forEach(({ i }) => { const [x1, y1] = P(ptsIn[sel]), [x2, y2] = P(ptsIn[i]); svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--violet)" stroke-width="${z}" opacity="0.35"/>`; });
  ptsOut.forEach((q, i) => { const [x, y] = P(q); svg += `<g class="pt out"><circle cx="${x}" cy="${y}" r="${3 * z}"/><text x="${x + 5 * z}" y="${y + 3 * z}">${W[i]}′</text></g>`; });
  ptsIn.forEach((q, i) => { const [x, y] = P(q); svg += `<g class="pt${i === sel ? ' sel' : nbs.has(i) ? ' nb' : ''}" data-w="${W[i]}"><circle cx="${x}" cy="${y}" r="${(i === sel ? 6 : 4) * z}"/><text x="${x + 7 * z}" y="${y + 4 * z}">${W[i]}</text></g>`; });
  $('m8-scatter').innerHTML = svg + '</svg>';
  $('m8-scatter').classList.toggle('zoomed', S8.zoom > 1);   // only a zoomed map takes over touch drags
  // neighbours
  $('m8-nn').innerHTML = `<table class="data"><tbody>${nb.map(({ i, c }) => `<tr data-w="${W[i]}"><td class="mono">${W[i]}</td><td class="n">${c.toFixed(2)}</td><td class="bar"><i style="width:${Math.max(0, c) * 100}%"></i></td></tr>`).join('')}</tbody></table>`;
  const sel8 = $('m8-word'); if (!sel8.options.length) W.slice().sort().forEach(w => sel8.add(new Option(w, w)));
  sel8.value = S8.sel;
  // the timeline
  const sc = $('m8-scrub'); sc.max = G.stream.total; sc.value = G.step;
  const at = `${commas(G.step)} of ${commas(G.stream.total)} pairs trained · pass ${G.step >= G.stream.total ? CFG.EPOCHS : timelinePair().epoch + 1} of ${CFG.EPOCHS}`;
  $('m8-pos').textContent = at;
}
// the loss and learning rate over the whole run (section 9's first piece)
// time on a log scale: x in 0..1 for a step, so the first few percent of training get room
const logx = step => Math.log10(1 + step) / Math.log10(1 + G.stream.total);
const LOG_TICKS = () => [[100, '100'], [1000, '1,000'], [10000, '10,000'], [G.stream.total, 'end']];
// the loss in equal-width bins along training (a linear time axis), as [x 0..1, mean loss, last step in bin]
function lossBinsLinear() {
  const N = G.stream.total, out = [];
  for (let b = 0; b * G.bucket < N; b++) if (G.lossN[b]) out.push([(b + 0.5) * G.bucket / N, G.lossSum[b] / G.lossN[b], Math.min(N, (b + 1) * G.bucket)]);
  return out;
}
// the run before the latest retrain, drawn faintly for comparison
function snapshotRun() { return G.lossAll ? { bins: lossBinsLinear(), alpha: CFG.ALPHA0, seed: G.seed } : null; }
function render8l() {
  $('m8-seed').value = G.seed;
  $('m8-lr').querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.v === CFG.ALPHA0));
  const cw = 340, ch = 160, cl = 46, cb = 34, ct = 10, N = G.stream.total, X = f => cl + f * (cw - cl - 6);
  const L = lossBinsLinear(), lmax = Math.max(4.5, ...L.map(x => x[1])), LY = v => ct + (1 - v / lmax) * (ch - ct - cb);
  const lp = A => A.map(([x, v], i) => (i ? 'L' : 'M') + X(x).toFixed(1) + ',' + LY(v).toFixed(1)).join('');
  const axis = (fy, ticks, f) => ticks.map(v => `<line x1="${cl}" x2="${cw - 6}" y1="${fy(v)}" y2="${fy(v)}" stroke="var(--grid)"/><text x="${cl - 5}" y="${fy(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${f(v)}</text>`).join('');
  const xticks = [[0, '0'], [N / 2, 'middle'], [N, 'end']].map(([v, l]) => `<line x1="${X(v / N)}" x2="${X(v / N)}" y1="${ch - cb}" y2="${ch - cb + 4}" stroke="var(--axis)"/><text x="${X(v / N)}" y="${ch - cb + 14}" text-anchor="${v === N ? 'end' : v ? 'middle' : 'start'}" font-size="9">${l}</text>`).join('')
    + `<text x="${cl + (cw - cl - 6) / 2}" y="${ch - 3}" text-anchor="middle" font-size="10">pairs trained</text>`;
  const now = X(G.step / N), prev = G.prev;
  const ghost = prev ? `<path d="${lp(prev.bins)}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>` : '';
  $('m8-loss').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${axis(LY, [0, 2, 4], v => v)}${xticks}${ghost}
    <path d="${lp(L.filter(([, , e]) => e <= G.step))}" fill="none" stroke="var(--blue)" stroke-width="2"/>
    <line x1="${now}" x2="${now}" y1="${ct}" y2="${ch - cb}" stroke="var(--ink-2)" stroke-dasharray="3 3"/></svg>`;
  // learning rate on the same log time axis, on a fixed scale up to the largest choice
  const AMAX = 0.1, AY = v => ct + (1 - v / AMAX) * (ch - ct - cb);
  const aline = a0 => `M${X(0)},${AY(a0)}L${X(1)},${AY(0)}`;   // linear decay: a straight line on a linear axis
  $('m8-alpha').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${axis(AY, [0, 0.025, 0.05, 0.075, 0.1], v => v ? String(v) : '0')}${xticks}
    ${prev && prev.alpha !== CFG.ALPHA0 ? `<path d="${aline(prev.alpha)}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>` : ''}
    <path d="${aline(CFG.ALPHA0)}" fill="none" stroke="var(--blue)" stroke-width="2"/>
    <circle cx="${now}" cy="${AY(CFG.ALPHA0 * (1 - G.step / N))}" r="4" fill="var(--blue)" stroke="white" stroke-width="2"/></svg>`;
  $('m8-ghost').innerHTML = prev ? `<span class="ghostkey"></span> previous run: seed ${prev.seed}, α ${prev.alpha}` : '';
}
function renderTimeline() { renderStatebar(); render8l(); render8(); refreshBars(); }
// Train: watch the run from the start (the model itself is already computed)
$('m8-train').addEventListener('click', () => { bar8.stop(); seek(0); renderTimeline(); bar8.act('play'); });
$('m8-seed').addEventListener('change', e => { const prev = snapshotRun(); G.seed = Math.max(1, Math.min(9999, Math.round(+e.target.value) || 1)); retrain(prev); });
$('m8-seedrand').addEventListener('click', () => { const prev = snapshotRun(); G.seed = 1 + Math.floor(Math.random() * 9999); retrain(prev); });
$('m8-lr').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { const prev = snapshotRun(); CFG.ALPHA0 = +b.dataset.v; retrain(prev); } });
$('m8-scrub').addEventListener('input', e => { stopAll(); seek(+e.target.value); renderTimeline(); });

$('m8-scatter').addEventListener('click', e => { if (S8.dragged) return; const g = e.target.closest('[data-w]'); if (g) { S8.sel = g.dataset.w; render8(); } });
// zoom with the buttons; drag the map to pan (a press that doesn't move still selects a word)
document.querySelector('.m8zoom').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.z === 'in' && S8.zoom === 1 && S8.selPt) S8.pan = S8.selPt.slice();   // first zoom-in centres on the selected word
  if (b.dataset.z === 'fit') { S8.zoom = 1; S8.pan = [0, 0]; } else S8.zoom = Math.max(1, Math.min(12, S8.zoom * (b.dataset.z === 'in' ? 1.6 : 1 / 1.6)));
  if (S8.zoom === 1) S8.pan = [0, 0];
  render8();
});
(function pan8() {
  const el = $('m8-scatter'); let d = null;
  el.addEventListener('pointerdown', e => { if (e.button || S8.zoom === 1) return; d = { x: e.clientX, y: e.clientY, pan: S8.pan.slice() }; S8.dragged = false; });   // pan only when zoomed in
  window.addEventListener('pointermove', e => {
    if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y; if (!S8.dragged && Math.hypot(dx, dy) < 5) return;
    S8.dragged = true; el.setPointerCapture?.(e.pointerId);
    const px = S8.W0 / el.clientWidth / S8.k;   // screen pixels → PCA units
    S8.pan = [d.pan[0] - dx * px, d.pan[1] + dy * px]; render8();
  });
  window.addEventListener('pointerup', () => { d = null; setTimeout(() => { S8.dragged = false; }, 0); });
})();
$('m8-nn').addEventListener('click', e => { const g = e.target.closest('[data-w]'); if (g) { S8.sel = g.dataset.w; render8(); } });
$('m8-word').addEventListener('change', e => { S8.sel = e.target.value; render8(); });
$('m8-out').addEventListener('change', e => { S8.showOut = e.target.checked; G.rad = null; render8(); });

const bar8 = PlayBar($('m8-bar'), {
  ms: 40, canStep: () => G.step < G.stream.total, canReset: () => G.step > 0, rewind: () => seek(0),
  step: () => { advance(S8.speed); renderTimeline(); }, end: () => { seek(G.stream.total); renderTimeline(); },
  reset: () => { seek(0); renderTimeline(); },
});

// ---------- 9 CBOW ----------
function box(x, y, w, txt, cls) { return `<g class="dg ${cls}"><rect x="${x - w / 2}" y="${y - 14}" width="${w}" height="28" rx="6"/><text x="${x}" y="${y + 1}" text-anchor="middle" dominant-baseline="middle">${txt}</text></g>`; }
function arr(x1, y1, x2, y2, id) { return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--ink-2)" stroke-width="1.3" marker-end="url(#${id})"/>`; }
function render9() {
  const ctx = ['drink', 'hot', 'every', 'morning'], mk = '<marker id="dg-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--ink-2)"/></marker>';
  const xs = [45, 125, 215, 295];
  let a = `<svg viewBox="0 0 340 250" width="100%" style="display:block"><defs>${mk}</defs><text x="170" y="16" text-anchor="middle" class="dg-t">skip-gram</text>`;
  ctx.forEach((w, i) => { a += box(xs[i], 60, 74, w, 'vo') + arr(170, 196, xs[i], 76, 'dg-ar'); });
  a += box(170, 210, 84, 'coffee', 'v') + `<text x="170" y="244" text-anchor="middle" class="dg-n">4 pairs → 4 updates of v</text></svg>`;
  let b = `<svg viewBox="0 0 340 250" width="100%" style="display:block"><defs>${mk}</defs><text x="170" y="16" text-anchor="middle" class="dg-t">CBOW</text>`;
  ctx.forEach((w, i) => { b += box(xs[i], 60, 74, w, 'v') + arr(xs[i], 76, 170, 124, 'dg-ar'); });
  b += `<g class="dg avg"><circle cx="170" cy="138" r="15"/><text x="170" y="139" text-anchor="middle" dominant-baseline="middle">avg</text></g>` + arr(170, 154, 170, 194, 'dg-ar');
  b += box(170, 210, 84, 'coffee', 'vo') + `<text x="170" y="244" text-anchor="middle" class="dg-n">1 update, shared by all 4 context words</text></svg>`;
  $('m9-body').innerHTML = `<div class="cols2">${a}${b}</div>`;
}

// ---------- 10 the bridge to LLMs ----------
function render10() {
  const cells = (y, x, tone) => tone.map((c, i) => `<rect x="${x + i * 18}" y="${y}" width="16" height="16" rx="2" fill="${c}"/>`).join('');
  const same = ['#cfcae9', '#8a7fd0', '#e4e1f5', '#4a3aa7', '#b3abe0'];
  const riv = ['#bfe9d6', '#1baf7a', '#e0f5ec', '#73cfa8', '#9fdfc2'], mon = ['#73cfa8', '#e0f5ec', '#0e7a53', '#bfe9d6', '#1baf7a'];
  const mk = '<marker id="br-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--ink-2)"/></marker>';
  $('m10-body').innerHTML = `<div class="tscroll"><svg viewBox="0 0 700 200" width="100%" style="display:block;min-width:560px"><defs>${mk}</defs>
    <text x="10" y="44" class="dg-s">we sat on the river <tspan font-weight="700">bank</tspan></text>
    <text x="10" y="164" class="dg-s">she paid it into the <tspan font-weight="700">bank</tspan></text>
    ${arr(196, 40, 236, 90, 'br-ar')}${arr(196, 160, 236, 112, 'br-ar')}
    <text x="290" y="72" text-anchor="middle" class="dg-n">one row of the table</text>${cells(92, 245, same)}
    ${arr(342, 100, 392, 100, 'br-ar')}
    <g class="dg"><rect x="396" y="58" width="104" height="84" rx="8"/><text x="448" y="94" text-anchor="middle" class="dg-n">transformer</text><text x="448" y="110" text-anchor="middle" class="dg-n">layers</text></g>
    ${arr(504, 88, 548, 48, 'br-ar')}${arr(504, 112, 548, 152, 'br-ar')}
    ${cells(32, 552, riv)}<text x="552" y="66" class="dg-n">“bank” by a river</text>
    ${cells(144, 552, mon)}<text x="552" y="178" class="dg-n">“bank” for money</text>
  </svg></div><p class="note" style="margin:8px 0 0">Schematic. The colours stand in for numbers: the same row comes out of the table both times, and the layers that follow pull it two ways.</p>`;
}

// ---------- everything ----------
function stopAll() { BARS.forEach(b => b.playing && b.stop()); }
function renderAll() {
  renderStatebar(); render2(); render2v(); render3(); render6(); render7(); render8a(); render8l(); render8(); refreshBars();
}
G.stream = makeStream(G.streamSeed);
resetTraining(); load4(); sync3(); P6.center = S4.center; P6.ctx = S4.ctx; loadPair6();
render1(); render4(); render5(); render9(); render10(); renderAll();
})();
