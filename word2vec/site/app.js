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
  stream: null, init: null, model: null, final: null,
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
  const m = cloneModel(G.init); G.ck = [];
  for (let i = 0; i < G.stream.total; i++) {
    if (i % CK === 0) G.ck.push(flat(m));
    const p = G.stream.get(i), l = trainPair(m, p, alphaAt(p.wordsDone));
    const b = Math.floor(i / G.bucket); G.lossSum[b] += l; G.lossN[b]++;
  }
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
const P6 = { center: 'coffee', ctx: 'cup', moment: 'early', idx: 0, pair: null, PL: null, phase: 0, R: null, snap: null };
const MOMENTS = { early: 0, middle: 0.5, late: 0.9 };

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
// change to v collects in a pending row that is added at the very end.
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
  const ci = V.index.get(P6.center), xi = V.index.get(P6.ctx), N = G.stream.total, from = Math.floor(MOMENTS[P6.moment] * N);
  let idx = -1;
  for (let k = 0; k < N && idx < 0; k++) { const i = (from + k) % N, p = G.stream.get(i); if (p.center === ci && p.ctx === xi) idx = i; }
  P6.idx = idx; P6.pair = G.stream.get(idx); P6.PL = phaseList(P6.pair); P6.snap = modelAt(idx); setPhase(0);
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
  // where section 8's training timeline is
  const p = timelinePair(), sb = $('statebar-in'), s = V.sents[p.sent];
  const sent = s.map((w, i) => wchip(W[w], i === p.pos ? 'v' : i === p.ctxPos ? 'vo' : '')).join('');
  sb.innerHTML = `
    <div class="sb-set"><span class="sb-vals">
      <a href="#s8" title="training timeline — scrub it in 8">pair <b>${commas(Math.min(G.step + 1, G.stream.total))}</b> / ${commas(G.stream.total)}</a>
      <span>pass <b>${p.epoch + 1}</b> / ${CFG.EPOCHS}</span>
      <a href="#s6" title="learning rate α — decays linearly over training (6)">α <b>${alphaAt(p.wordsDone).toFixed(4)}</b></a>
      <a href="#s2" title="initialisation seed — re-roll it in 2">seed <b>${G.seed}</b></a>
    </span></div>
    <span class="sb-hr"></span>
    <div class="sb-row"><span class="sb-pair">center ${wchip(W[p.center], 'v')} context ${wchip(W[p.ctx], 'vo')}</span><span class="sb-sent">${sent}</span></div>`;
}

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
const S2 = { sel: 'coffee' };
function render2() {
  const sel = V.index.get(S2.sel), maxc = Math.max(...V.count);
  $('m2-corpus').innerHTML = V.sents.map(s => `<div class="sline">${s.map(w => `<span class="w${w === sel ? ' on' : ''}" data-w="${W[w]}">${W[w]}</span>`).join(' ')}</div>`).join('');
  $('m2-vocab').innerHTML = `<table class="data vocab"><tbody>${W.map((w, i) => `<tr class="${i === sel ? 'on' : ''}" data-w="${w}"><td class="n rk">${i + 1}</td><td class="mono">${w}</td><td class="n">${V.count[i]}</td><td class="bar"><i style="width:${100 * V.count[i] / maxc}%"></i></td></tr>`).join('')}</tbody></table>`;
  $('m2-stats').textContent = `${W.length} words in the vocabulary · ${V.total} words of running text · ${V.sents.length} sentences`;
  $('m2-seed').textContent = `seed ${G.seed}`;
}
function selWord2(w) { if (w && V.index.has(w)) { S2.sel = w; render2(); } }
['m2-corpus', 'm2-vocab'].forEach(id => $(id).addEventListener('click', e => { const el = e.target.closest('[data-w]'); if (el) selWord2(el.dataset.w); }));
$('reroll').addEventListener('click', () => { stopAll(); G.seed = 1 + Math.floor(Math.random() * 9999); resetTraining(); loadPair6(); restart4(); renderAll(); });

// ---------- 2 an example input vector, before and after training ----------
// "trained" runs the whole stream on a private copy (what 8 does), so this works whatever
// state the shared model is in.
const S2V = { word: 'coffee', trained: false, cache: null };
function trainedWin() {
  if (S2V.cache && S2V.cache.seed === G.seed) return S2V.cache.Win;
  const m = cloneModel(G.init);
  for (let i = 0; i < G.stream.total; i++) { const p = G.stream.get(i); trainPair(m, p, alphaAt(p.wordsDone)); }
  S2V.cache = { seed: G.seed, Win: m.Win };
  return m.Win;
}
function render2v() {
  const i = V.index.get(S2V.word), T = trainedWin(), v = (S2V.trained ? T : G.init.Win)[i];
  const top = Math.max(...T.flat().map(Math.abs));          // one scale for both views
  const cell = (x, d) => `<span class="m2v-cell"><span class="v" style="background:${x >= 0 ? 'rgba(42,120,214,' : 'rgba(235,104,52,'}${(0.08 + 0.42 * Math.abs(x) / top).toFixed(3)})">${num(x, 2)}</span><span class="d">d${d + 1}</span></span>`;
  $('m2v-vec').innerHTML = `<div class="m2v-row">
    <span class="m2v-word">${esc(S2V.word)}<span class="sub">input vector</span></span>
    <span class="m2v-arrow" aria-hidden="true">→</span>
    <span class="m2v-br">${v.map(cell).join('')}</span></div>`;
  $('m2v-note').textContent = S2V.trained
    ? 'Training nudges the numbers up and down to bring similar words closer together. Notice how the numbers for “coffee” are more similar to “tea” than to “dog”.'
    : `Five random numbers between −0.1 and 0.1. Nothing about “${S2V.word}” is in them yet.`;
  document.querySelectorAll('#m2v-word button').forEach(b => b.classList.toggle('on', b.dataset.v === S2V.word));
  document.querySelectorAll('#m2v-state button').forEach(b => b.classList.toggle('on', (b.dataset.v === 'trained') === S2V.trained));
}
$('m2v-word').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S2V.word = b.dataset.v; render2v(); } });
$('m2v-state').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S2V.trained = b.dataset.v === 'trained'; render2v(); } });

// ---------- callouts with a hidden answer ----------
document.querySelectorAll('.reveal-btn').forEach(b => b.addEventListener('click', () => {
  const body = b.nextElementSibling, open = body.hidden;
  body.hidden = !open; b.setAttribute('aria-expanded', open); b.textContent = open ? 'Hide answer' : 'Reveal answer';
}));

// ---------- 3 the window ----------
// A sandbox, separate from training: pick a sentence, drag the window, see the pairs it makes.
// (Training shrinks the window at random and drops frequent words; the text explains that.)
const S3 = { sent: 0, pos: 2, b: 2, shown: new Set() };
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
  S3.pos = i; place3(); pairs3();
}
// list the window's pairs; the context chip of any pair not shown a moment ago bubbles in
function pairs3() {
  const s = V.sents[S3.sent], c = S3.pos, key = k => `${S3.sent}:${c}:${k}`;
  const ks = s.map((w, k) => k).filter(k => k !== c && Math.abs(k - c) <= S3.b);
  let n = 0;
  $('m3-pairs').innerHTML = ks.map(k => {
    // a new pair first opens up its space, then its context word pops in
    if (S3.shown.has(key(k))) return `<span class="pairchip">${wchip(W[s[c]], 'v')}${wchip(W[s[k]], 'vo')}</span>`;
    const d = 90 * n++;
    return `<span class="pairchip grow" style="animation-delay:${d}ms">${wchip(W[s[c]], 'v')}<span class="chip vo pop" style="animation-delay:${d + 160}ms">${esc(W[s[k]])}</span></span>`;
  }).join('');
  S3.shown = new Set(ks.map(key));
}
function setSent3(k) { S3.sent = k; S3.pos = Math.min(S3.pos, V.sents[S3.sent].length - 1); render3(); }
$('m3-sent-pick').innerHTML = V.sents.map((s, k) => `<option value="${k}">${s.map(w => W[w]).join(' ')}</option>`).join('');
$('m3-sent-pick').addEventListener('change', e => setSent3(+e.target.value));
$('m3-minus').addEventListener('click', () => { S3.b = Math.max(1, S3.b - 1); render3(); });
$('m3-plus').addEventListener('click', () => { S3.b = Math.min(CFG.WINDOW, S3.b + 1); render3(); });
window.addEventListener('resize', () => place3());
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
// any center word's v against any context word's v′, at the start of training or after it
const S4 = { center: 'coffee', ctx: 'cup', stage: 'start', edited: false, v: null, u: null, ph: 0 };
function load4() {
  const M = S4.stage === 'start' ? G.init : G.final;
  S4.v = M.Win[V.index.get(S4.center)].slice(); S4.u = M.Wout[V.index.get(S4.ctx)].slice(); S4.ph = 0; S4.edited = false;
  const cs = $('m4-center'), xs = $('m4-ctx');
  if (!cs.options.length) W.slice().sort().forEach(w => { cs.add(new Option(w, w)); xs.add(new Option(w, w)); });
  cs.value = S4.center; xs.value = S4.ctx;
  $('m4-stage').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === S4.stage));
}
function restart4() { load4(); render4(); render5(); }   // after a re-roll: new initial and trained models
function score4() { return dot(S4.v, S4.u); }
function render4() {
  const ph = S4.ph, prod = S4.v.map((x, d) => x * S4.u[d]);
  let run = 0; const parts = [];
  for (let d = 0; d < Math.min(ph, D); d++) { run += prod[d]; parts.push(num(prod[d], 4)); }
  const inp = (a, row) => a.map((x, d) => `<td class="n ${ph - 1 === d ? 'merge' : ''}"><input type="number" step="0.05" data-r="${row}" data-d="${d}" value="${+x.toFixed(4)}"></td>`).join('');
  const prodCells = prod.map((x, d) => d < ph ? `<td class="n prod ${ph - 1 === d ? 'pop' : ''}">${num(x, 4)}</td>` : '<td class="n dim">·</td>').join('');
  const done = ph > D;
  $('m4-table').innerHTML = `<table class="data vec"><thead><tr><th></th>${dimHead()}<th class="n">sum</th></tr></thead><tbody>
    <tr class="r-v"><th>center v ${S4.edited ? '<span class="note">(edited)</span>' : wchip(S4.center, 'v')}</th>${inp(S4.v, 'v')}<td></td></tr>
    <tr class="r-vo"><th>context v′ ${S4.edited ? '<span class="note">(edited)</span>' : wchip(S4.ctx, 'vo')}</th>${inp(S4.u, 'u')}<td></td></tr>
    <tr class="r-p"><th>product</th>${prodCells}<td class="n score ${done ? 'pop' : ''}">${ph ? num(run, 4) : '·'}</td></tr></tbody></table>`;
  $('m4-sum').innerHTML = ph === 0 ? 'Press play to multiply the two rows, one dimension at a time.'
    : `score = ${parts.join(' + ').replace(/\+ −/g, '− ')}${ph < D ? ' + …' : ''} = <b>${num(run, 4)}</b>${done ? (Math.abs(run) < 1e-12 ? ' — exactly zero, because v′ starts as all zeros' : run > 0 ? ' — positive: the vectors point the same way' : ' — negative: the vectors point apart') : ''}`;
}
$('m4-table').addEventListener('change', e => {
  const el = e.target.closest('input'); if (!el) return;
  const x = parseFloat(el.value); if (!isFinite(x)) return;
  (el.dataset.r === 'v' ? S4.v : S4.u)[+el.dataset.d] = x;
  S4.edited = true; S4.ph = D + 1;
  render4(); render5(); refreshBars();
});
function pick4() { stopAll(); load4(); render4(); render5(); refreshBars(); }
$('m4-center').addEventListener('change', e => { S4.center = e.target.value; pick4(); });
$('m4-ctx').addEventListener('change', e => { S4.ctx = e.target.value; pick4(); });
$('m4-stage').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S4.stage = b.dataset.v; pick4(); } });
const bar4 = PlayBar($('m4-bar'), {
  ms: 650, canStep: () => S4.ph <= D, canReset: () => S4.ph > 0,
  step: () => { S4.ph++; render4(); render5(); }, end: () => { S4.ph = D + 1; render4(); render5(); }, reset: () => { S4.ph = 0; render4(); render5(); },
});

// ---------- 5 the sigmoid ----------
const S5 = { manual: null };
const XM = 8;
function render5() {
  const x = S5.manual === null ? score4() : S5.manual, s = sigmoid(x);
  const W0 = 700, H0 = 230, l = 44, r = 20, t = 14, b = 34;
  const X = v => l + (Math.max(-XM, Math.min(XM, v)) + XM) / (2 * XM) * (W0 - l - r), Y = v => t + (1 - v) * (H0 - t - b);
  let path = ''; for (let i = 0; i <= 200; i++) { const v = -XM + 2 * XM * i / 200; path += (i ? 'L' : 'M') + X(v).toFixed(1) + ',' + Y(sigmoid(v)).toFixed(1); }
  const ticks = [-8, -6, -4, -2, 0, 2, 4, 6, 8].map(v => `<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(0) + 4}" stroke="var(--axis)"/><text x="${X(v)}" y="${Y(0) + 17}" text-anchor="middle" font-size="11">${num(v, 0)}</text>`).join('');
  const yt = [0, 0.5, 1].map(v => `<line x1="${l}" y1="${Y(v)}" x2="${W0 - r}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${l - 8}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="11">${v}</text>`).join('');
  const off = Math.abs(x) > XM;
  $('m5-chart').innerHTML = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block">${yt}${ticks}
    <line x1="${X(0)}" y1="${t}" x2="${X(0)}" y2="${Y(0)}" stroke="var(--grid)"/>
    <path d="${path}" fill="none" stroke="var(--ink)" stroke-width="2"/>
    <line x1="${X(x)}" y1="${Y(s)}" x2="${X(x)}" y2="${Y(1)}" stroke="var(--blue)" stroke-width="2" opacity="0.35"/>
    <line x1="${l}" y1="${Y(s)}" x2="${X(x)}" y2="${Y(s)}" stroke="var(--blue)" stroke-dasharray="3 3"/>
    <circle cx="${X(x)}" cy="${Y(s)}" r="7" fill="var(--blue)" stroke="white" stroke-width="2"/>
    <text x="${X(x) + (x > 5 ? -12 : 12)}" y="${Y(1) + (Y(s) - Y(1)) / 2}" font-size="11" fill="var(--blue)" text-anchor="${x > 5 ? 'end' : 'start'}" dominant-baseline="middle">error ${num(1 - s, 3)}</text>
    <text x="${W0 - r}" y="${H0 - 4}" text-anchor="end" font-size="11">score${off ? ' (off the chart)' : ''}</text>
  </svg>`;
  $('m5-stats').innerHTML = `<div class="s"><div class="l">score</div><div class="v">${num(x, 3)}</div></div>
    <div class="s"><div class="l"><span class="lc">σ</span>(score)</div><div class="v">${s.toFixed(3)}</div></div>
    <div class="s"><div class="l">error 1 − <span class="lc">σ</span></div><div class="v">${(1 - s).toFixed(3)}</div></div>`;
  $('m5-src').textContent = S5.manual === null ? 'The dot follows the score from 4.' : 'Score set by hand. Reset to follow 4 again.';
  $('m5-x').value = Math.max(-XM, Math.min(XM, x)); $('m5-xv').textContent = num(x, 2);
}
$('m5-x').addEventListener('input', e => { S5.manual = +e.target.value; render5(); refreshBars(); });
const bar5 = PlayBar($('m5-bar'), {
  ms: 350, canStep: () => (S5.manual === null ? score4() : S5.manual) < XM, canReset: () => S5.manual !== null,
  step: () => { const x = S5.manual === null ? score4() : S5.manual; S5.manual = Math.min(XM, Math.floor(x * 2 + 1e-9) / 2 + 0.5); render5(); },
  end: () => { S5.manual = XM; render5(); }, reset: () => { S5.manual = null; render5(); },
});

// ---------- 6 one positive update ----------
function render6() {
  const p = P6.pair, R = P6.R, ph = Math.min(P6.phase, POS_END), t = R.T[0], fresh = k => P6.phase === k ? ' new' : '';
  const show = (k, cells) => ph >= k ? cells : blankCells();
  const prod = R.v0.map((x, d) => x * t.vo0[d]);
  $('m6-head').innerHTML = `<div class="pairhead">center ${wchip(W[p.center], 'v')} context ${wchip(W[p.ctx], 'vo')} <span class="badge pos">real pair · label 1</span><span class="note">pair ${commas(P6.idx + 1)} of ${commas(G.stream.total)}, pass ${p.epoch + 1} · α = ${R.alpha.toFixed(5)}</span></div>`;
  $('m6-table').innerHTML = `<table class="data vec"><thead><tr><th></th>${dimHead()}<th class="n"></th></tr></thead><tbody>
      <tr class="r-v"><th>center v</th>${vecCells(R.v0)}<td></td></tr>
      <tr class="r-vo"><th>context v′</th>${vecCells(t.vo0)}<td></td></tr>
      <tr class="r-p"><th>product</th>${show(1, vecCells(prod, 'prod' + fresh(1), 4))}<td class="n score${fresh(1)}">${ph >= 1 ? 'score ' + num(t.f, 4) : ''}</td></tr>
      <tr class="r-s"><th>σ, g</th><td colspan="${D + 1}" class="scal${fresh(2)}">${ph >= 2 ? `σ(${num(t.f, 4)}) = <b>${t.s.toFixed(4)}</b>` : ''}${ph >= 3 ? `<span class="${fresh(3)}"> &nbsp;·&nbsp; g = (1 − ${t.s.toFixed(4)}) × ${R.alpha.toFixed(5)} = <b>${num(t.g, 6)}</b></span>` : ''}</td></tr>
      <tr class="r-d"><th>Δv′ = g · v</th>${show(4, vecCells(t.dvo, 'chg' + fresh(4), 5))}<td></td></tr>
      <tr class="r-vo"><th>new v′</th>${show(4, vecCells(ph >= 4 ? t.vo1 : t.vo0, 'chg' + fresh(4), 4))}<td class="n note">${ph >= 4 ? 'applied now' : ''}</td></tr>
      <tr class="r-pend"><th>pending Δv += g · v′</th>${show(5, vecCells(t.contrib, 'chg' + fresh(5), 5))}<td class="n note">${ph >= 5 ? 'held for 7' : ''}</td></tr>
    </tbody></table>`;
  const msg = [
    'Press play to score this pair and update it.',
    `The score is the dot product of v and v′, ${num(t.f, 4)}.`,
    `σ turns it into the model’s probability that the pair is real: ${t.s.toFixed(4)}. The pair is real, so the error is 1 − σ = ${(1 - t.s).toFixed(4)}.`,
    `g scales the error by the learning rate. It is positive, so both vectors will move toward each other.`,
    `v′ moves by g times v: a small step in v’s direction. It is changed straight away.`,
    P6.phase > POS_END ? 'The positive update is done. 7 is now working through the negative samples for this pair.' : `v’s change, g times the old v′, is only written into the pending row. 7 adds negative samples before it is applied.`,
  ];
  $('m6-note').textContent = msg[ph] + (norm(t.vo0) === 0 && ph >= 3 ? ' (v′ is still all zeros at this point, so v’s pending change is zero too.)' : '');
  // pickers
  const cs = $('m6-center'), xs = $('m6-ctx');
  if (!cs.options.length) W.slice().sort().forEach(w => cs.add(new Option(w, w)));
  cs.value = P6.center;
  xs.innerHTML = contexts6(P6.center).map(w => `<option>${w}</option>`).join(''); xs.value = P6.ctx;
  $('m6-moment').querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.v === P6.moment));
}
const bar6 = PlayBar($('m6-bar'), {
  ms: 900, canStep: () => P6.phase < POS_END, canReset: () => P6.phase > 0,
  step: () => { setPhase(P6.phase + 1); render6(); render7(); }, end: () => { setPhase(POS_END); render6(); render7(); },
  reset: () => { setPhase(0); render6(); render7(); },
});
function pick6() { stopAll(); loadPair6(); render6(); render7(); refreshBars(); }
$('m6-center').addEventListener('change', e => { P6.center = e.target.value; P6.ctx = contexts6(P6.center, true); pick6(); });
$('m6-ctx').addEventListener('change', e => { P6.ctx = e.target.value; pick6(); });
$('m6-moment').addEventListener('click', e => { const x = e.target.closest('button'); if (x) { P6.moment = x.dataset.v; pick6(); } });

// ---------- 7 negative samples ----------
function render7() {
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
    svg += `<g><title>${w}: ${(100 * V.raw[i]).toFixed(1)}% of words, ${(100 * V.noise[i]).toFixed(1)}% of noise draws</title>
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
  // the running table
  const row = t => {
    const lab = t.label ? `<span class="badge pos">real</span>` : `<span class="badge neg">noise</span>`;
    if (t.label === 0 && ph < t.drawAt) return '';
    if (t.skip) return `<tr class="skipped"><th>${wchip(W[t.w], 'neg')}${lab}</th><td colspan="${3 + D}" class="note">same word as the context, so this draw is skipped</td></tr>`;
    const doneT = ph >= t.pendAt, isNew = cur && ((cur.k === 'neg' && cur.j === t.j) || (cur.k === 'pend' && t.label));
    return `<tr class="${doneT ? '' : 'wait'}${isNew ? ' newrow' : ''}"><th>${wchip(W[t.w], t.label ? 'vo' : 'neg')}${lab}</th>
      <td class="n">${doneT ? num(t.f, 4) : '·'}</td><td class="n">${doneT ? t.s.toFixed(4) : '·'}</td><td class="n">${doneT ? num(t.g, 6) : '·'}</td>
      ${doneT ? vecCells(t.contrib, 'chg' + (isNew ? ' new' : ''), 5) : blankCells()}</tr>`;
  };
  const applied = R.applied;
  $('m7-table').innerHTML = `<table class="data vec upd"><thead><tr><th>target</th><th class="n">score</th><th class="n">σ</th><th class="n">g</th>${Array.from({ length: D }, (_, d) => `<th class="n">g·v′ d${d + 1}</th>`).join('')}</tr></thead><tbody>
    ${R.T.map(row).join('')}
    <tr class="r-pend tot"><th colspan="4">pending Δv (sum of the rows above)</th>${vecCells(R.pend, 'chg' + (cur && (cur.k === 'neg' || cur.k === 'pend') ? ' new' : ''), 5)}</tr>
    <tr class="r-v"><th colspan="4">center v ${wchip(W[p.center], 'v')} before</th>${vecCells(R.v0, '', 4)}</tr>
    <tr class="r-v ${applied ? '' : 'wait'}"><th colspan="4">v + pending Δv${applied ? '' : ' (not yet applied)'}</th>${applied ? vecCells(R.v1, 'chg new', 4) : blankCells()}</tr>
  </tbody></table>`;
  let msg;
  if (ph < POS_END) msg = 'The positive update from 6 comes first. Step to complete it.';
  else if (!cur || cur.k === 'pend') msg = `Next: draw ${CFG.K} noise words from the orange distribution.`;
  else if (cur.k === 'draw') { const t = R.T[cur.j + 1]; msg = t.skip ? `Drew “${W[t.w]}”, the context word itself. word2vec skips such draws.` : `Drew “${W[t.w]}”. It is treated as a fake pair: label 0.`; }
  else if (cur.k === 'neg') { const t = R.T[cur.j + 1]; msg = `Fake pair with “${W[t.w]}”: σ = ${t.s.toFixed(4)}, so g = (0 − σ) × α = ${num(t.g, 6)} is negative. Its v′ is pushed away from v, and g·v′ joins the pending row.`; }
  else msg = `All ${CFG.K} draws done. The pending row is added to v once. That finishes this pair; training moves on to the next one.`;
  $('m7-note').textContent = msg;
}
const bar7 = PlayBar($('m7-bar'), {
  ms: 900, canStep: () => P6.phase < END(), canReset: () => P6.phase > POS_END,
  step: () => { setPhase(P6.phase < POS_END ? POS_END : P6.phase + 1); render6(); render7(); },
  end: () => { setPhase(END()); render6(); render7(); },
  reset: () => { setPhase(POS_END); render6(); render7(); },
});

// ---------- 8 training at scale ----------
const S8 = { speed: 100, sel: 'coffee', showOut: false, pair: ['coffee', 'tea'] };
function render8() {
  const M = G.model, sel = V.index.get(S8.sel);
  const pc = pca2(M.Win, G.pca && G.pca.comps); G.pca = pc;
  const ptsIn = M.Win.map(pc.project), ptsOut = S8.showOut ? M.Wout.map(pc.project) : [];
  const all = ptsIn.concat(ptsOut), want = 1.12 * Math.max(1e-6, ...all.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y))));
  G.rad = !G.rad || want > G.rad ? want : G.rad * 0.92 + want * 0.08;   // grow at once, shrink gently
  const W0 = 700, H0 = 440, cx = W0 / 2, cy = H0 / 2, k = Math.min(W0, H0) / 2 / G.rad;
  const P = ([x, y]) => [cx + x * k, cy - y * k];
  const nb = neighbors(M.Win, sel, 5), nbs = new Set(nb.map(x => x.i));
  let svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block" id="scatter">
    <line x1="0" x2="${W0}" y1="${cy}" y2="${cy}" stroke="var(--grid)"/><line x1="${cx}" x2="${cx}" y1="0" y2="${H0}" stroke="var(--grid)"/>`;
  nb.forEach(({ i }) => { const [x1, y1] = P(ptsIn[sel]), [x2, y2] = P(ptsIn[i]); svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--violet)" opacity="0.35"/>`; });
  ptsOut.forEach((q, i) => { const [x, y] = P(q); svg += `<g class="pt out"><circle cx="${x}" cy="${y}" r="3"/><text x="${x + 5}" y="${y + 3}">${W[i]}′</text></g>`; });
  ptsIn.forEach((q, i) => { const [x, y] = P(q); svg += `<g class="pt${i === sel ? ' sel' : nbs.has(i) ? ' nb' : ''}" data-w="${W[i]}"><circle cx="${x}" cy="${y}" r="${i === sel ? 6 : 4}"/><text x="${x + 7}" y="${y + 4}">${W[i]}</text></g>`; });
  $('m8-scatter').innerHTML = svg + '</svg>';
  // loss and learning rate
  const nbk = Math.ceil(G.stream.total / G.bucket), L = [];
  for (let b = 0; b < nbk; b++) if (G.lossN[b]) L.push([b, G.lossSum[b] / G.lossN[b]]);
  const done = L.filter(([b]) => b * G.bucket < G.step);   // the full run is known; blue up to where the timeline is
  const cw = 340, ch = 150, cl = 46, cb = 22, ct = 10;
  const lmax = Math.max(4.5, ...L.map(x => x[1]));
  const LX = b => cl + b / nbk * (cw - cl - 6), LY = v => ct + (1 - v / lmax) * (ch - ct - cb);
  const lp = A => A.map(([b, v], i) => (i ? 'L' : 'M') + LX(b + 0.5).toFixed(1) + ',' + LY(v).toFixed(1)).join('');
  const frac = G.step / G.stream.total;
  const axis = (fy, ticks, f) => ticks.map(v => `<line x1="${cl}" x2="${cw - 6}" y1="${fy(v)}" y2="${fy(v)}" stroke="var(--grid)"/><text x="${cl - 5}" y="${fy(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${f(v)}</text>`).join('');
  $('m8-loss').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${axis(LY, [0, 2, 4], v => v)}
    <path d="${lp(L)}" fill="none" stroke="var(--blue-100)" stroke-width="2"/><path d="${lp(done)}" fill="none" stroke="var(--blue)" stroke-width="2"/>
    <text x="${cw - 6}" y="${ch - 6}" text-anchor="end" font-size="10">training →</text></svg>`;
  const AY = v => ct + (1 - v / CFG.ALPHA0) * (ch - ct - cb), AX = f => cl + f * (cw - cl - 6);
  $('m8-alpha').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${axis(AY, [0, CFG.ALPHA0 / 2, CFG.ALPHA0], v => v.toFixed(4).replace(/0+$/, '').replace(/\.$/, ''))}
    <line x1="${AX(0)}" y1="${AY(CFG.ALPHA0)}" x2="${AX(1)}" y2="${AY(0)}" stroke="var(--axis)" stroke-width="2"/>
    <line x1="${AX(0)}" y1="${AY(CFG.ALPHA0)}" x2="${AX(frac)}" y2="${AY(CFG.ALPHA0 * (1 - frac))}" stroke="var(--blue)" stroke-width="2"/>
    <circle cx="${AX(frac)}" cy="${AY(CFG.ALPHA0 * (1 - frac))}" r="4" fill="var(--blue)"/>
    <text x="${cw - 6}" y="${ch - 6}" text-anchor="end" font-size="10">training →</text></svg>`;
  // neighbours
  $('m8-nn').innerHTML = `<table class="data"><tbody>${nb.map(({ i, c }) => `<tr data-w="${W[i]}"><td class="mono">${W[i]}</td><td class="n">${c.toFixed(2)}</td><td class="bar"><i style="width:${Math.max(0, c) * 100}%"></i></td></tr>`).join('')}</tbody></table>`;
  const cs = (a, b) => cosine(M.Win[V.index.get(a)], M.Win[V.index.get(b)]).toFixed(2);
  $('m8-stats').innerHTML = `<div class="s"><div class="l">pass</div><div class="v">${G.step >= G.stream.total ? CFG.EPOCHS : timelinePair().epoch + 1}<small> / ${CFG.EPOCHS}</small></div></div>
    <div class="s"><div class="l">cos(coffee, tea)</div><div class="v">${cs('coffee', 'tea')}</div></div>
    <div class="s"><div class="l">cos(dog, cat)</div><div class="v">${cs('dog', 'cat')}</div></div>
    <div class="s"><div class="l">cos(coffee, dog)</div><div class="v">${cs('coffee', 'dog')}</div></div>`;
  const sel8 = $('m8-word'); if (!sel8.options.length) W.slice().sort().forEach(w => sel8.add(new Option(w, w)));
  sel8.value = S8.sel;
  // two input vectors in their own plane: true lengths, true angle
  const [a, b] = S8.pair;
  $('m8-angle').innerHTML = pairPlane(M.Win[V.index.get(a)], M.Win[V.index.get(b)], { aLabel: a, bLabel: b, bCls: 'a2', cos: true });
  $('m8-pairs').querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.v === S8.pair.join(',')));
  // the timeline
  const sc = $('m8-scrub'); sc.max = G.stream.total; sc.value = G.step;
  $('m8-pos').textContent = `${commas(G.step)} of ${commas(G.stream.total)} pairs trained`;
}
function renderTimeline() { renderStatebar(); render8(); refreshBars(); }
$('m8-scrub').addEventListener('input', e => { stopAll(); seek(+e.target.value); renderTimeline(); });
$('m8-pairs').addEventListener('click', e => { const x = e.target.closest('button'); if (x) { S8.pair = x.dataset.v.split(','); render8(); } });
$('m8-scatter').addEventListener('click', e => { const g = e.target.closest('[data-w]'); if (g) { S8.sel = g.dataset.w; render8(); } });
$('m8-nn').addEventListener('click', e => { const g = e.target.closest('[data-w]'); if (g) { S8.sel = g.dataset.w; render8(); } });
$('m8-word').addEventListener('change', e => { S8.sel = e.target.value; render8(); });
$('m8-out').addEventListener('change', e => { S8.showOut = e.target.checked; G.rad = null; render8(); });
$('m8-speed').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
  S8.speed = +b.dataset.v; $('m8-speed').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
}));
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
  renderStatebar(); render2(); render2v(); render3(); render6(); render7(); render8(); refreshBars();
}
G.stream = makeStream(G.streamSeed);
resetTraining(); loadPair6(); load4();
render1(); render4(); render5(); render9(); render10(); renderAll();
})();
