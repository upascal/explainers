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
// One model for the whole page. 2 initialises it, 6–7 train it one pair at a time, 8 trains
// it in bulk, and every section reads from it.
const G = {
  seed: 1,                 // initialisation seed (re-rolled in 2)
  streamSeed: 2013,        // seed for window shrink, subsampling and negative draws
  stream: null, init: null, model: null, snap: null,
  step: 0,                 // index of the pair in progress
  pair: null, PL: null,    // that pair and its list of phases
  phase: 0,                // how far into the pair (see phaseList)
  R: null,                 // the pair's numbers at the current phase (replay())
  lossSum: null, lossN: null, bucket: 1,
  pca: null, rad: null,
};
const NB = 150;            // loss-curve buckets

// The phases of one pair update. 0–5 belong to 6 (the positive), the rest to 7.
const POS_END = 5;
function phaseList(p) {
  const L = ['start', 'score', 'sigma', 'g', 'out', 'pend'].map(k => ({ k }));
  p.negs.forEach((n, j) => { L.push({ k: 'draw', j }); if (!n.skip) L.push({ k: 'neg', j }); });
  L.push({ k: 'apply' });
  return L;
}
const END = () => G.PL.length - 1;

// Every number shown in 6 and 7, recomputed from the model as it stood before the pair.
// Updates are applied in word2vec.c's order: each target's v′ moves at once, while the
// change to v collects in a pending row that is added at the very end.
function replay(upto = G.phase) {
  const p = G.pair, PL = G.PL, m = cloneModel(G.snap), alpha = alphaAt(p.wordsDone);
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

function loadPair() {
  G.pair = G.stream.get(G.step);
  G.PL = G.pair ? phaseList(G.pair) : null; G.phase = 0;
  G.R = G.pair ? replay(0) : null;
}
function logLoss(i, l) { const b = Math.floor(i / G.bucket); G.lossSum[b] += l; G.lossN[b]++; }
function resetTraining() {
  G.init = initModel(G.seed); G.model = cloneModel(G.init); G.snap = cloneModel(G.init); G.step = 0;
  G.bucket = Math.ceil(G.stream.total / NB); G.lossSum = new Float64Array(NB + 1); G.lossN = new Uint32Array(NB + 1);
  G.pca = null; G.rad = null; loadPair();
}
function setPhase(ph) { G.phase = ph; G.R = replay(); G.model = G.R.m; }
function commit() {                      // finish the pair in progress and move to the next
  const r = replay(END()); G.model = r.m; logLoss(G.step, r.loss);
  G.step++; G.snap = cloneModel(G.model); loadPair();
}
function trainN(n) {
  if (!G.pair) return;
  if (G.phase > 0) { commit(); n--; }
  const m = G.model;
  for (; n > 0 && G.step < G.stream.total; n--, G.step++) {
    const p = G.stream.get(G.step); logLoss(G.step, trainPair(m, p, alphaAt(p.wordsDone)));
  }
  G.snap = cloneModel(m); loadPair();
}

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
    if (a === 'play') { if (bar.playing) bar.stop(); else if (o.canStep()) { bar.playing = true; tick(); } }
    else { bar.stop(); if (a === 'step' && o.canStep()) o.step(); else if (a === 'end' && o.canStep()) o.end(); else if (a === 'reset' && (!o.canReset || o.canReset())) o.reset(); }
    refreshBars();
  };
  bar.refresh = () => {
    const can = o.canStep(), b = k => host.querySelector(`[data-a="${k}"]`);
    b('step').disabled = !can || bar.playing; b('end').disabled = !can; b('play').disabled = !can && !bar.playing;
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
  const k = Rpx / R, P = ([x, y]) => [ox + x * k, oy - y * k];
  const labs = [];
  const arrow = (u, cls, label, dash) => {
    const [x, y] = P(xy(u)), len = Math.hypot(x - ox, y - oy);
    if (len < 2) return `<circle cx="${ox}" cy="${oy}" r="4" class="pp-${cls}" fill="currentColor"/>${label ? `<text x="${ox + 8}" y="${oy - 10}" class="pp-lab pp-${cls}">${label} ${norm(u) < 1e-12 ? '= 0' : '(tiny)'}</text>` : ''}`;
    let tx = x + (x - ox) / len * 12, ty = y + (y - oy) / len * 12;
    if (label && labs.some(([lx, ly]) => Math.hypot(lx - tx, ly - ty) < 16)) ty -= 15;   // two tips close together: lift this label clear
    let anc = 'middle'; if (tx > W0 - 34) { anc = 'end'; tx = W0 - 2; ty = Math.min(ty, y - 10); }   // keep labels at the right edge inside the box
    if (label) labs.push([tx, ty]);
    return `<line x1="${ox}" y1="${oy}" x2="${x}" y2="${y}" class="pp-${cls}" stroke="currentColor" stroke-width="${dash ? 1.5 : 2.5}" ${dash ? 'stroke-dasharray="4 3"' : ''} marker-end="url(#${id}-${cls}${dash ? 'g' : ''})"/>` +
      (label ? `<text x="${tx}" y="${ty}" text-anchor="${anc}" dominant-baseline="middle" class="pp-lab pp-${cls}">${label}</text>` : '');
  };
  const mk = (cls, g) => `<marker id="${id}-${cls}${g ? 'g' : ''}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="${g ? 5 : 6}" markerHeight="${g ? 5 : 6}" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="pp-${cls}" fill="currentColor"/></marker>`;
  const nb = norm(b), cos = na > 1e-12 && nb > 1e-12 ? dot(a, b) / (na * nb) : null;
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
    <line x1="10" y1="${oy}" x2="${W0 - 10}" y2="${oy}" stroke="var(--grid)"/>
    <path d="M${ox - Rpx},${oy} A${Rpx},${Rpx} 0 0 1 ${ox + Rpx},${oy}" fill="none" stroke="var(--grid)" stroke-dasharray="2 4"/>
    ${arc}${ghost ? arrow(ghost, bc, '', true) : ''}${arrow(b, bc, o.bLabel || 'v′')}${arrow(a, 'a', o.aLabel || 'v')}
  </svg>`;
  const cap = cos === null
    ? `<span class="note">${na <= 1e-12 ? (o.aLabel || 'v') : (o.bLabel || 'v′')} is all zeros, so there is no angle and the dot product is 0.</span>`
    : `<span class="note">|${o.aLabel || 'v'}| = ${num(na)} · |${o.bLabel || 'v′'}| = ${num(nb, nb < 0.01 ? 4 : 3)} · angle ${Math.round(ang * 180 / Math.PI)}°<br>${num(na)} × ${num(nb, nb < 0.01 ? 4 : 3)} × cos ${Math.round(ang * 180 / Math.PI)}° = <b>${num(dot(a, b), 4)}</b></span>`;
  return `<div class="pplane">${o.title ? `<div class="pp-title">${o.title}</div>` : ''}${svg}${cap}${ghost ? '<div class="note">dashed: v′ before this update</div>' : ''}</div>`;
}

// ---------- sticky status bar ----------
function renderStatebar() {
  const p = G.pair, sb = $('statebar-in');
  if (!p) { sb.innerHTML = `<div class="sb-set"><span class="sb-vals"><span>training finished · <b>${commas(G.stream.total)}</b> pairs</span><a href="#s2">seed <b>${G.seed}</b></a></span></div>`; return; }
  const s = V.sents[p.sent];
  const sent = s.map((w, i) => wchip(W[w], i === p.pos ? 'v' : i === p.ctxPos ? 'vo' : '')).join('');
  sb.innerHTML = `
    <div class="sb-set"><span class="sb-vals">
      <span>pair <b>${commas(G.step + 1)}</b> / ${commas(G.stream.total)}</span>
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
  $('m2-seed').textContent = G.step === 0 && G.phase === 0 ? `seed ${G.seed}, untrained` : `seed ${G.seed}, after ${commas(G.step)} pair updates`;
}
function selWord2(w) { if (w && V.index.has(w)) { S2.sel = w; render2(); } }
['m2-corpus', 'm2-vocab'].forEach(id => $(id).addEventListener('click', e => { const el = e.target.closest('[data-w]'); if (el) selWord2(el.dataset.w); }));
$('reroll').addEventListener('click', () => { stopAll(); G.seed = 1 + Math.floor(Math.random() * 9999); resetTraining(); restart4(); renderAll(); });

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
const S3 = { ci: 0 };
function render3() {
  const C = G.stream.centers, c = C[S3.ci], s = V.sents[c.sent];
  const live = s.map((w, i) => i).filter(i => c.kept[i]), li = live.indexOf(c.pos);
  const inWin = new Set(live.filter((_, j) => j !== li && Math.abs(j - li) <= c.b));
  const lo = live[Math.max(0, li - c.b)], hi = live[Math.min(live.length - 1, li + c.b)];
  let html = '';
  s.forEach((w, i) => {
    if (i === lo) html += '<span class="winbox">';
    const cls = i === c.pos ? 'v' : !c.kept[i] ? 'drop' : inWin.has(i) ? 'vo' : '';
    const t = !c.kept[i] ? `dropped by subsampling (kept with probability ${V.keep[w].toFixed(2)})` : `kept with probability ${V.keep[w].toFixed(2)}`;
    html += `<span class="chip ${cls}" title="${t}">${W[w]}</span>`;
    if (i === hi) html += `<span class="winlbl">window ${c.b} (max ${CFG.WINDOW})</span></span>`;
  });
  $('m3-sent').innerHTML = html;
  const mine = Array.from({ length: c.n }, (_, k) => G.stream.get(c.first + k));
  $('m3-pairs').innerHTML = mine.length ? mine.map(p => `<span class="pairchip">${wchip(W[p.center], 'v')}${wchip(W[p.ctx], 'vo')}</span>`).join('') : '<span class="note">no pairs: every other word in this sentence was dropped</span>';
  const done = c.first + c.n;
  $('m3-note').textContent = `center word ${S3.ci + 1} of ${C.length} in the first pass · sentence ${c.sent + 1} · ${commas(done)} pairs so far · ${commas(G.stream.total)} pairs over all ${CFG.EPOCHS} passes`;
  $('win-v').textContent = CFG.WINDOW;
}
const bar3 = PlayBar($('m3-bar'), {
  ms: 700, canStep: () => S3.ci < G.stream.centers.length - 1, canReset: () => S3.ci > 0,
  step: () => { S3.ci++; render3(); }, end: () => { S3.ci = G.stream.centers.length - 1; render3(); }, reset: () => { S3.ci = 0; render3(); },
});
$('win').addEventListener('input', e => {
  stopAll(); CFG.WINDOW = +e.target.value; G.stream = makeStream(G.streamSeed);
  S3.ci = Math.min(S3.ci, G.stream.centers.length - 1); resetTraining(); restart4(); renderAll();
});

// ---------- 4 the dot product ----------
const S4 = { preset: 'start', v: null, u: null, ph: 0, cw: '', ow: '' };
function preset4(k) {
  S4.preset = k; S4.ph = 0;
  if (k === 'start') { const p = G.stream.get(0); S4.v = G.init.Win[p.center].slice(); S4.u = G.init.Wout[p.ctx].slice(); S4.cw = W[p.center]; S4.ow = W[p.ctx]; }
  if (k === 'similar') { S4.v = [0.8, -0.3, 0.5, 0.1, -0.6]; S4.u = [0.7, -0.2, 0.6, 0.0, -0.5]; S4.cw = S4.ow = ''; }
  if (k === 'unrelated') { S4.v = [0.8, -0.3, 0.5, 0.1, -0.6]; S4.u = [0.3, 0.6, -0.2, 0.4, 0.1]; S4.cw = S4.ow = ''; }
  $('m4-presets').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === k));
}
function restart4() { if (S4.preset === 'start') { preset4('start'); render4(); render5(); } }   // its vectors come from the initial model
function score4() { return dot(S4.v, S4.u); }
function render4() {
  const ph = S4.ph, prod = S4.v.map((x, d) => x * S4.u[d]);
  let run = 0; const parts = [];
  for (let d = 0; d < Math.min(ph, D); d++) { run += prod[d]; parts.push(num(prod[d], 4)); }
  const inp = (a, row) => a.map((x, d) => `<td class="n ${ph - 1 === d ? 'merge' : ''}"><input type="number" step="0.05" data-r="${row}" data-d="${d}" value="${+x.toFixed(4)}"></td>`).join('');
  const prodCells = prod.map((x, d) => d < ph ? `<td class="n prod ${ph - 1 === d ? 'pop' : ''}">${num(x, 4)}</td>` : '<td class="n dim">·</td>').join('');
  const done = ph > D;
  $('m4-table').innerHTML = `<table class="data vec"><thead><tr><th></th>${dimHead()}<th class="n">sum</th></tr></thead><tbody>
    <tr class="r-v"><th>center v${S4.cw ? ` <span class="mono">${esc(S4.cw)}</span>` : ''}</th>${inp(S4.v, 'v')}<td></td></tr>
    <tr class="r-vo"><th>context v′${S4.ow ? ` <span class="mono">${esc(S4.ow)}</span>` : ''}</th>${inp(S4.u, 'u')}<td></td></tr>
    <tr class="r-p"><th>product</th>${prodCells}<td class="n score ${done ? 'pop' : ''}">${ph ? num(run, 4) : '·'}</td></tr></tbody></table>`;
  $('m4-sum').innerHTML = ph === 0 ? 'Press play to multiply the two rows, one dimension at a time.'
    : `score = ${parts.join(' + ').replace(/\+ −/g, '− ')}${ph < D ? ' + …' : ''} = <b>${num(run, 4)}</b>${done ? (Math.abs(run) < 1e-12 ? ' — exactly zero, because v′ starts as all zeros' : run > 0 ? ' — positive: the vectors point the same way' : ' — negative: the vectors point apart') : ''}`;
  $('m4-plane').innerHTML = pairPlane(S4.v, S4.u, { aLabel: 'v', bLabel: 'v′' });
}
$('m4-table').addEventListener('change', e => {
  const el = e.target.closest('input'); if (!el) return;
  const x = parseFloat(el.value); if (!isFinite(x)) return;
  (el.dataset.r === 'v' ? S4.v : S4.u)[+el.dataset.d] = x;
  S4.preset = 'custom'; S4.ph = D + 1; S4.cw = S4.ow = '';
  $('m4-presets').querySelectorAll('button').forEach(b => b.classList.remove('on'));
  render4(); render5(); refreshBars();
});
$('m4-presets').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { preset4(b.dataset.v); render4(); render5(); refreshBars(); }));
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
  if (!G.pair) { $('m6-table').innerHTML = '<p class="note">Training has finished. Reset it in 8 to step through pairs again.</p>'; ['m6-head', 'm6-plane', 'm6-note'].forEach(id => $(id).innerHTML = ''); return; }
  const p = G.pair, R = G.R, ph = Math.min(G.phase, POS_END), t = R.T[0], fresh = k => G.phase === k ? ' new' : '';
  const show = (k, cells) => ph >= k ? cells : blankCells();
  const prod = R.v0.map((x, d) => x * t.vo0[d]);
  $('m6-head').innerHTML = `<div class="pairhead">center ${wchip(W[p.center], 'v')} context ${wchip(W[p.ctx], 'vo')} <span class="badge pos">real pair · label 1</span><span class="note">pair ${commas(G.step + 1)} · α = ${R.alpha.toFixed(5)}</span></div>`;
  $('m6-table').innerHTML = `<table class="data vec"><thead><tr><th></th>${dimHead()}<th class="n"></th></tr></thead><tbody>
      <tr class="r-v"><th>center v</th>${vecCells(R.v0)}<td></td></tr>
      <tr class="r-vo"><th>context v′</th>${vecCells(t.vo0)}<td></td></tr>
      <tr class="r-p"><th>product</th>${show(1, vecCells(prod, 'prod' + fresh(1), 4))}<td class="n score${fresh(1)}">${ph >= 1 ? 'score ' + num(t.f, 4) : ''}</td></tr>
      <tr class="r-s"><th>σ, g</th><td colspan="${D + 1}" class="scal${fresh(2)}">${ph >= 2 ? `σ(${num(t.f, 4)}) = <b>${t.s.toFixed(4)}</b>` : ''}${ph >= 3 ? `<span class="${fresh(3)}"> &nbsp;·&nbsp; g = (1 − ${t.s.toFixed(4)}) × ${R.alpha.toFixed(5)} = <b>${num(t.g, 6)}</b></span>` : ''}</td></tr>
      <tr class="r-d"><th>Δv′ = g · v</th>${show(4, vecCells(t.dvo, 'chg' + fresh(4), 5))}<td></td></tr>
      <tr class="r-vo"><th>new v′</th>${show(4, vecCells(ph >= 4 ? t.vo1 : t.vo0, 'chg' + fresh(4), 4))}<td class="n note">${ph >= 4 ? 'applied now' : ''}</td></tr>
      <tr class="r-pend"><th>pending Δv += g · v′</th>${show(5, vecCells(t.contrib, 'chg' + fresh(5), 5))}<td class="n note">${ph >= 5 ? 'held for 7' : ''}</td></tr>
    </tbody></table>`;
  $('m6-plane').innerHTML = pairPlane(R.v0, ph >= 4 ? t.vo1 : t.vo0, { aLabel: 'v', bLabel: 'v′', ghost: ph >= 4 ? t.vo0 : null });
  const msg = [
    'Press play to score this pair and update it.',
    `The score is the dot product of v and v′, ${num(t.f, 4)}.`,
    `σ turns it into the model’s probability that the pair is real: ${t.s.toFixed(4)}. The pair is real, so the error is 1 − σ = ${(1 - t.s).toFixed(4)}.`,
    `g scales the error by the learning rate. It is positive, so both vectors will move toward each other.`,
    `v′ moves by g times v: a small step in v’s direction. It is changed straight away.`,
    G.phase > POS_END ? 'The positive update is done. 7 is now working through the negative samples for this pair.' : `v’s change, g times the old v′, is only written into the pending row. 7 adds negative samples before it is applied.`,
  ];
  $('m6-note').textContent = msg[ph] + (G.step === 0 && ph >= 3 ? ' (On the very first pair v′ is all zeros, so v’s pending change is zero too.)' : '');
}
const bar6 = PlayBar($('m6-bar'), {
  ms: 900, canStep: () => !!G.pair && (G.phase < POS_END || G.phase === END()), canReset: () => !!G.pair && G.phase > 0,
  step: () => { if (G.phase === END()) commit(); else setPhase(G.phase + 1); renderAll(); },
  end: () => { if (G.phase === END()) commit(); setPhase(POS_END); renderAll(); },
  reset: () => { setPhase(0); renderAll(); },
});

// ---------- 7 negative samples ----------
function render7() {
  if (!G.pair) { $('m7-table').innerHTML = '<p class="note">Training has finished. Reset it in 8 to step through pairs again.</p>'; ['m7-draws', 'm7-plane', 'm7-note'].forEach(id => $(id).innerHTML = ''); return; }
  const p = G.pair, R = G.R, ph = G.phase, cur = G.PL[ph];
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
  // the pair plane: the latest negative, or v before and after once the pending row is applied
  let plane;
  if (cur && cur.k === 'apply') plane = pairPlane(R.v0, R.v1, { aLabel: 'v', bLabel: 'new v', bCls: 'a2', title: 'center v, before and after the pending row is added' });
  else {
    const last = negs.filter(t => !t.skip && ph >= t.outAt).pop();
    plane = last ? pairPlane(R.v0, last.vo1, { aLabel: 'v', bLabel: `v′ ${W[last.w]}`, ghost: last.vo0, title: `center v and noise word “${W[last.w]}”` })
      : `<div class="pplane"><div class="note">Each negative sample’s v′ will be drawn here as it is pushed away from v.</div></div>`;
  }
  $('m7-plane').innerHTML = plane;
  let msg;
  if (ph < POS_END) msg = 'The positive update from 6 comes first. Step to complete it.';
  else if (!cur || cur.k === 'pend') msg = `Next: draw ${CFG.K} noise words from the orange distribution.`;
  else if (cur.k === 'draw') { const t = R.T[cur.j + 1]; msg = t.skip ? `Drew “${W[t.w]}”, the context word itself. word2vec skips such draws.` : `Drew “${W[t.w]}”. It is treated as a fake pair: label 0.`; }
  else if (cur.k === 'neg') { const t = R.T[cur.j + 1]; msg = `Fake pair with “${W[t.w]}”: σ = ${t.s.toFixed(4)}, so g = (0 − σ) × α = ${num(t.g, 6)} is negative. Its v′ is pushed away from v, and g·v′ joins the pending row.`; }
  else msg = `All ${CFG.K} draws done. The pending row is added to v once. The next step moves on to pair ${commas(G.step + 2)}.`;
  $('m7-note').textContent = msg;
}
const bar7 = PlayBar($('m7-bar'), {
  ms: 900, canStep: () => !!G.pair, canReset: () => !!G.pair && G.phase > POS_END,
  step: () => { if (G.phase === END()) commit(); else setPhase(G.phase < POS_END ? POS_END : G.phase + 1); renderAll(); },
  end: () => { if (G.phase === END()) commit(); setPhase(END()); renderAll(); },
  reset: () => { setPhase(POS_END); renderAll(); },
});

// ---------- 8 training at scale ----------
const S8 = { speed: 100, sel: 'coffee', showOut: false };
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
  const cw = 340, ch = 150, cl = 46, cb = 22, ct = 10;
  const lmax = Math.max(4.5, ...L.map(x => x[1]));
  const LX = b => cl + b / nbk * (cw - cl - 6), LY = v => ct + (1 - v / lmax) * (ch - ct - cb);
  const lpath = L.map(([b, v], i) => (i ? 'L' : 'M') + LX(b + 0.5).toFixed(1) + ',' + LY(v).toFixed(1)).join('');
  const frac = G.step / G.stream.total;
  const axis = (fy, ticks, f) => ticks.map(v => `<line x1="${cl}" x2="${cw - 6}" y1="${fy(v)}" y2="${fy(v)}" stroke="var(--grid)"/><text x="${cl - 5}" y="${fy(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${f(v)}</text>`).join('');
  $('m8-loss').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${axis(LY, [0, 2, 4], v => v)}
    <path d="${lpath}" fill="none" stroke="var(--blue)" stroke-width="2"/>
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
  $('m8-stats').innerHTML = `<div class="s"><div class="l">pass</div><div class="v">${G.pair ? G.pair.epoch + 1 : CFG.EPOCHS}<small> / ${CFG.EPOCHS}</small></div></div>
    <div class="s"><div class="l">cos(coffee, tea)</div><div class="v">${cs('coffee', 'tea')}</div></div>
    <div class="s"><div class="l">cos(dog, cat)</div><div class="v">${cs('dog', 'cat')}</div></div>
    <div class="s"><div class="l">cos(coffee, dog)</div><div class="v">${cs('coffee', 'dog')}</div></div>`;
  const sel8 = $('m8-word'); if (!sel8.options.length) W.slice().sort().forEach(w => sel8.add(new Option(w, w)));
  sel8.value = S8.sel;
}
$('m8-scatter').addEventListener('click', e => { const g = e.target.closest('[data-w]'); if (g) { S8.sel = g.dataset.w; render8(); } });
$('m8-nn').addEventListener('click', e => { const g = e.target.closest('[data-w]'); if (g) { S8.sel = g.dataset.w; render8(); } });
$('m8-word').addEventListener('change', e => { S8.sel = e.target.value; render8(); });
$('m8-out').addEventListener('change', e => { S8.showOut = e.target.checked; G.rad = null; render8(); });
$('m8-speed').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
  S8.speed = +b.dataset.v; $('m8-speed').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
}));
const bar8 = PlayBar($('m8-bar'), {
  ms: 16, canStep: () => !!G.pair, canReset: () => G.step > 0 || G.phase > 0,
  step: () => { trainN(S8.speed); renderAll(); }, end: () => { trainN(Infinity); renderAll(); },
  reset: () => { resetTraining(); renderAll(); },
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
resetTraining(); preset4('start');
render1(); render4(); render5(); render9(); render10(); renderAll();
})();
