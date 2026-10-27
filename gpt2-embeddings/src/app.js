(function () {
// gpt2data.js and model.js are concatenated ahead of this file by build.py (GPT2, BPE, V, CFG, forward, …)
const $ = id => document.getElementById(id);
const D = CFG.D;
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const commas = v => String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
// a number with a real minus sign, and no sign on a value that rounds to zero
function num(x, n = 3) { const s = Math.abs(x).toFixed(n); return (x < 0 && +s !== 0 ? '−' : '') + s; }
const norm = a => Math.hypot(...a);
const sum = a => a.reduce((x, y) => x + y, 0);
const arr = a => Array.from(a);
const b64 = (s, T) => new T(b64bytes(s).buffer);

// transport icons, drawn the same way as on the SynthID and word2vec pages
const I_RESET = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><rect x="1" y="2" width="2" height="12" fill="currentColor"/><polygon points="11,2 11,14 4,8" fill="currentColor"/><polygon points="18,2 18,14 11,8" fill="currentColor"/></svg>';
const I_PLAY = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><polygon points="4,2 4,14 13,8" fill="currentColor"/></svg>';
const I_PAUSE = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" fill="currentColor"/><rect x="9.5" y="2" width="3.5" height="12" fill="currentColor"/></svg>';
const I_STEP = '<svg viewBox="0 0 18 16" width="12" height="10" aria-hidden="true"><polygon points="2,2 2,14 11,8" fill="currentColor"/><rect x="12" y="2" width="2.5" height="12" fill="currentColor"/></svg>';
const I_END = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><polygon points="2,2 2,14 9,8" fill="currentColor"/><polygon points="9,2 9,14 16,8" fill="currentColor"/><rect x="17" y="2" width="2" height="12" fill="currentColor"/></svg>';

// ---------- showing tokens ----------
// a space shows as a faint dot, so " coffee" and "coffee" look different; a piece of a
// multi-byte character, which isn't text on its own, shows as its bytes
const EOT_HTML = '<span class="eot">&lt;|endoftext|&gt;</span>';
const vis = s => esc(s).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="sp">↵</span>').replace(/\t/g, '<span class="sp">⇥</span>');
function gstr(id) { if (id === EOT) return EOT_HTML; const s = BPE.str(id); return s === null ? `<span class="hex">${BPE.hex(id)}</span>` : vis(s); }
const tstr = i => gstr(V.ids[i]);                       // a toy token, by its index in the toy vocabulary
const wchip = (i, cls = '') => `<span class="chip ${cls}" data-t="${i}">${tstr(i)}</span>`;
const plain = i => V.ids[i] === EOT ? '<|endoftext|>' : V.str[i];   // for <option> labels and titles
const optLabel = i => plain(i).replace(/^ /, '·');
const SV = '<i class="cv">E</i>';   // an input embedding: a row of E, the token embedding table

// ---------- shared model state ----------
// One initialization (re-rolled in 3 or 10) and one training stream for the whole page. The whole
// run is computed up front and every step's model kept, so any point of training can be revisited.
const G = { seed: 1, streamSeed: 2019, stream: null, snaps: null, loss: null, step: 0, prev: null, pca: null, rad: null };
function resetTraining() {
  const N = G.stream.total, m = initModel(G.seed);
  G.snaps = [m.slice()]; G.loss = new Float32Array(N);
  for (let i = 0; i < N; i++) { G.loss[i] = trainStep(m, V.seqs[G.stream.sent(i)], alphaAt(i, N)); G.snaps.push(m.slice()); }
  G.step = 0; G.pca = null; G.rad = null; U.key = ''; S9.key = '';
}
const rows = (m, k = 'E') => Array.from({ length: k === 'E' ? V.n : CFG.K }, (_, i) => arr(rowOf(m, k, i)));
// a step's place in training: step s (1-based) is the s-th update, in pass ceil(s / 30)
const passOfStep = s => Math.ceil(s / V.seqs.length);

// ---------- the selected prediction ----------
// One prediction is selected for the whole page: in sentence `sent`, the vector at position `pos`
// predicts the token at pos + 1. Sections 6–9 show it as the model stood just before that
// sentence's update in the first, middle or last pass.
const SEL = { sent: 4, pos: 4, pass: 1 };   // pass: which of the 120 times this sentence is trained
const seqSel = () => V.seqs[SEL.sent];
const inTok = () => seqSel()[SEL.pos], tgTok = () => seqSel()[SEL.pos + 1];
// the step (0-based: how many updates came before) at which `sent` is trained in pass `pass`
function stepOf(sent, pass) {
  const n = V.seqs.length, ep = pass - 1;
  for (let i = ep * n; i < ep * n + n; i++) if (G.stream.sent(i) === sent) return i;
}
// the forward pass and gradients for the selected sentence at the selected stage, cached
const U = { key: '' };
function upd() {
  const key = [G.seed, CFG.ALPHA0, SEL.sent, SEL.pass].join(':');
  if (U.key !== key) {
    const step = stepOf(SEL.sent, SEL.pass), m = G.snaps[step], f = forward(m, V.seqs[SEL.sent]);
    Object.assign(U, { key, step, m, f, alpha: alphaAt(step, G.stream.total), full: backward(m, f), one: {} });
  }
  if (!U.one[SEL.pos]) U.one[SEL.pos] = backward(U.m, U.f, SEL.pos);
  return U;
}
// The header holds the selected prediction's controls for sections 5–9: the sentence, the context
// (drag it, or click a token) and the pass. Built once; renderStatebar() only updates it.
$('statebar-in').innerHTML = `<div class="sb-set">
    <a class="sb-lab" href="#s5">selected prediction</a>
    <label class="m3pick sb-pick"><select id="sb-sent" aria-label="sentence"></select></label>
    <span class="passctl"><span class="lab">pass</span><input type="range" min="1" max="1" step="1" aria-label="which pass over the corpus"><span class="note pv"></span></span>
  </div>
  <div class="m3scroll sb-scroll"><div class="m3track m5track" id="sb-track"></div></div>
  <div class="sb-sentview" id="sb-sentview" hidden></div>`;
function selectPred(sent, pos) {
  SEL.sent = sent; SEL.pos = Math.max(0, Math.min(V.seqs[sent].length - 2, pos));
  S8.row = tgTok(); follow3v(); S9.a = inTok() === V.index.get(EOT) ? S9.a : inTok();
  stopAll(); renderSel();
}
function renderSel() { render3(); render3v(); render4(); render5(); renderMix(); render6(); render7(); render8(); render9(); renderStatebar(); refreshBars(); }

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
function stopAll() { BARS.forEach(b => b.playing && b.stop()); }
function pickBar() {
  const vis = BARS.filter(b => { const r = b.host.closest('.module').getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && b.host.offsetParent; });
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

// ---------- formulas: in words or in symbols ----------
function setFx(m) {
  document.body.classList.toggle('fx-sym', m === 's');
  document.querySelectorAll('.fxsw button').forEach(b => b.classList.toggle('on', b.dataset.fx === m));
  try { localStorage.setItem('gpt2-fx', m); } catch (e) {}
}
document.querySelectorAll('.fxsw').forEach(sw => sw.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setFx(b.dataset.fx); }));
setFx((() => { try { return localStorage.getItem('gpt2-fx') || 'w'; } catch (e) { return 'w'; } })());

// ---------- callouts with a hidden answer ----------
document.querySelectorAll('.reveal-btn').forEach(b => b.addEventListener('click', () => {
  const body = b.nextElementSibling, open = body.hidden;
  body.hidden = !open; b.setAttribute('aria-expanded', open); b.textContent = open ? 'Hide answer' : 'Reveal answer';
}));

// ---------- where in training: a pass (6–9) or a step (3, 4, 10, 11) ----------
// Sections 6–9 follow the selected sentence's own update, so they pick which of its 120 passes;
// sections 3, 4 and 10 show the tables as a whole, so they share the training step with 11.
document.querySelectorAll('.passctl input').forEach(inp => {
  inp.max = CFG.EPOCHS;
  inp.addEventListener('input', () => { SEL.pass = +inp.value; stopAll(); renderMix(); render6(); render7(); render8(); renderStatebar(); refreshBars(); });
});
function syncPass() {
  const step = stepOf(SEL.sent, SEL.pass) + 1;
  document.querySelectorAll('.passctl').forEach(c => { c.querySelector('input').value = SEL.pass; c.querySelector('.pv').textContent = `${SEL.pass} of ${CFG.EPOCHS} · step ${commas(step)} of ${commas(G.stream.total)}`; });
}
document.querySelectorAll('.stepctl input').forEach(inp => inp.addEventListener('input', () => { stopAll(); G.step = +inp.value; renderTimeline(); }));
function syncStep() {
  const s = G.step, txt = s ? `${commas(s)} of ${commas(G.stream.total)} · pass ${passOfStep(s)}` : `0 of ${commas(G.stream.total)} · before training`;
  document.querySelectorAll('.stepctl').forEach(c => { const inp = c.querySelector('input'); inp.max = G.stream.total; inp.value = s; c.querySelector('.sv').textContent = txt; });
}
// everything that shows the tables at the shared step
function renderStep() { render3vToy(); render4(); render9(); syncStep(); }

// ---------- a vector equation with "show math" ----------
// get() returns { left: [items], result: { label, chip, vals, cls } }, items being
// { t: 'vec', label, sym, chip, vals, cls }, { t: 'scal', label, val, f, cls }, { t: 'op', s } or { t: 'paren', s }.
// Collapsed, the vectors on the left show as symbols and the result as its column; "show math"
// opens them into columns and works out the result one row at a time on a play bar.
function VecEq(host, get, dec = 2, cls = '') {
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
  // "show math" opens the finished working; the play bar steps through it from the start
  me.done = () => { me.bar.stop(); me.step = D + 1; me.render(); refreshBars(); };
  me.refresh = () => { if (me.math) me.done(); else me.render(); };
  eq.addEventListener('click', e => { if (e.target.closest('.m4math')) { me.math = !me.math; if (me.math) me.done(); else { me.bar.stop(); me.render(); } } });
  return me;
}

// ---------- a dot product with "show math" (used in 6) ----------
// get() returns { a: { label, sym, chip, vals, cls }, b: { … }, scoreLabel }. "Show math" opens both
// vectors into columns and works through them: multiply each dimension, then add the products.
function DotEq(host, get, dec = 2, pdec = dec) {   // dec: decimal places for the two vectors; pdec: for the products
  const me = { math: false, step: 0 };
  host.innerHTML = `<div class="m4wrap"><div class="m5eq m4eq"></div><div class="m4ctl" hidden><span></span></div></div>`;
  const eq = host.querySelector('.m4eq'), ctl = host.querySelector('.m4ctl');
  me.render = () => {
    const g = get(), A = g.a.vals, B = g.b.vals, prod = A.map((x, d) => x * B[d]), run = sum(prod);
    // steps 1–D multiply one dimension each; steps D+1–2D add the products one by one; then done
    const st = me.math ? me.step : 2 * D + 1, cur = st - 1, mult = st >= 1 && st <= D, k = st - D;
    const col = (a, cls, isProd) => `<span class="m4br ${cls}">${a.map((x, d) => {
      const shown = !isProd || d < st, on = me.math && mult && d === cur ? ' on' : '', summed = isProd && k >= 1 && k <= D && d < k ? (d === k - 1 ? ' on' : ' sumd') : '';
      const fresh = isProd && mult && d === cur ? ' cur' : '';
      return `<span class="c${on}${summed}${fresh}">${shown ? num(x, isProd ? pdec : dec) : '·'}</span>`;
    }).join('')}</span>`;
    const term = (l, v, f, cls = '') => `<div class="term ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="f">${f}</div></div>`;
    const done = st > 2 * D, adding = k >= 1 && k <= D, sofar = adding ? sum(prod.slice(0, k)) : run;
    eq.innerHTML = (me.math
      ? term(g.a.chip, col(A, g.a.cls), '', 'sym open') + '<div class="op"><span class="dotch">·</span></div>' + term(g.b.chip, col(B, g.b.cls), '', 'sym open')
        + `<div class="m4scoregrp"><div class="op"><span class="eqch">=</span> <span class="sumw">sum</span></div>${term('products', `<span class="m4paren">${col(prod, 'cp', true)}</span>`, '', 'sym open')}</div>`
      : term(g.a.label, g.a.sym, g.a.chip, 'sym') + '<div class="op"><span class="dotch">·</span></div>' + term(g.b.label, g.b.sym, g.b.chip, 'sym'))
      + `<div class="m4scoregrp"><div class="op">=</div>${term(adding ? 'sum so far' : g.scoreLabel || 'score', done || adding ? num(sofar, pdec) : '?', adding ? `${k} of ${D} products` : '', 'score' + (adding ? ' adding' : ''))}</div>`
      + `<button class="m4math">${me.math ? 'Hide math' : 'Show math'}</button>`;
    ctl.hidden = !me.math; eq.classList.toggle('mathon', me.math);
    eq.classList.toggle('working', me.math && !done); eq.classList.toggle('mult', me.math && mult); eq.classList.toggle('adding', me.math && adding);
    const brs = eq.querySelectorAll('.m4br'), box = eq.getBoundingClientRect();
    if (me.math && mult) {
      const cell = i => brs[i].querySelectorAll('.c')[cur].getBoundingClientRect(), a = cell(0), b = cell(1), c = cell(2), y = a.top + a.height / 2 - box.top;
      const mark = (x, t) => `<span class="m4op" style="left:${x - box.left}px;top:${y}px">${t}</span>`;
      const over = q => { const r = eq.querySelector(q).getBoundingClientRect(); return r.left + r.width / 2; };
      const band = `<span class="m4band" style="left:${a.left - 14 - box.left}px;top:${a.top - 5 - box.top}px;width:${c.right - a.left + 28}px;height:${a.height + 10}px"></span>`;
      eq.insertAdjacentHTML('beforeend', band + mark(over('.dotch'), '×') + mark((b.right + c.left) / 2, '='));
    }
    if (me.math && adding) {   // one band around the products added so far, a + on the newest
      const cs = brs[2].querySelectorAll('.c'), f = cs[0].getBoundingClientRect(), l = cs[k - 1].getBoundingClientRect();
      let h = `<span class="m4band" style="left:${f.left - 30 - box.left}px;top:${f.top - 5 - box.top}px;width:${f.width + 38}px;height:${l.bottom - f.top + 10}px"></span>`;
      if (k >= 2) h += `<span class="m4op plus" style="left:${f.left - 15 - box.left}px;top:${l.top + l.height / 2 - box.top}px">+</span>`;
      eq.insertAdjacentHTML('beforeend', h);
    }
  };
  me.bar = PlayBar(ctl.querySelector('span'), {
    ms: () => me.step < D ? 700 : 600, canStep: () => me.step <= 2 * D, canReset: () => me.step > 0, rewind: () => { me.step = 0; },
    step: () => { me.step++; me.render(); }, end: () => { me.step = 2 * D + 1; me.render(); }, reset: () => { me.step = 0; me.render(); },
  });
  // "show math" opens the finished working; the play bar steps through it from the start
  me.done = () => { me.bar.stop(); me.step = 2 * D + 1; me.render(); refreshBars(); };
  me.refresh = () => { if (me.math) me.done(); else me.render(); };
  eq.addEventListener('click', e => { if (e.target.closest('.m4math')) { me.math = !me.math; if (me.math) me.done(); else { me.bar.stop(); me.render(); } } });
  return me;
}

// Two vectors always lie in one flat plane, so they can be drawn at their true lengths and
// true angle however many dimensions they have.
let svgUid = 0;
function pairPlane(a, b, o = {}) {
  const id = 'pp' + (svgUid++), bc = o.bCls || 'b';
  const W0 = 300, H0 = 172, ox = W0 / 2, oy = H0 - 22, Rpx = 120;
  const na = norm(a), nb = norm(b);
  const e1 = na > 1e-12 ? a.map(x => x / na) : b.map(x => x / (nb || 1));
  const pr = dot(b, e1), perp = b.map((x, i) => x - pr * e1[i]), np = norm(perp), e2 = np > 1e-12 ? perp.map(x => x / np) : null;
  const xy = u => [dot(u, e1), e2 ? dot(u, e2) : 0];
  const R = Math.max(1e-9, ...[a, b].map(xy).map(([x, y]) => Math.hypot(x, y))), k = Rpx / R, P = ([x, y]) => [ox + x * k, oy - y * k];
  const cos = na > 1e-12 && nb > 1e-12 ? dot(a, b) / (na * nb) : null, ang = cos === null ? null : Math.acos(Math.max(-1, Math.min(1, cos)));
  const labs = [];
  const arrow = (u, cls, label) => {
    const [x, y] = P(xy(u)), len = Math.hypot(x - ox, y - oy);
    let tx = x + (x - ox) / (len || 1) * 12, ty = y + (y - oy) / (len || 1) * 12;
    if (labs.some(([lx, ly]) => Math.hypot(lx - tx, ly - ty) < 16)) ty -= 15;
    let anc = 'middle'; if (tx > W0 - 34) { anc = 'end'; tx = W0 - 2; ty = Math.min(ty, y - 10); } if (tx < 34) { anc = 'start'; tx = 2; ty = Math.min(ty, y - 10); }
    labs.push([tx, ty]);
    return `<line x1="${ox}" y1="${oy}" x2="${x}" y2="${y}" class="pp-${cls}" stroke="currentColor" stroke-width="2.5" marker-end="url(#${id}-${cls})"/>
      <text x="${tx}" y="${ty}" text-anchor="${anc}" dominant-baseline="middle" class="pp-lab pp-${cls}">${esc(label)}</text>`;
  };
  const mk = cls => `<marker id="${id}-${cls}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="pp-${cls}" fill="currentColor"/></marker>`;
  let arc = '';
  if (ang !== null && Math.min(na, nb) * k > 14 && ang > 0.12) {
    const r = 26, [bx, by] = xy(b), t = Math.atan2(by, bx), x2 = ox + r * Math.cos(t), y2 = oy - r * Math.sin(t);
    arc = `<path d="M${ox + r},${oy} A${r},${r} 0 0 0 ${x2},${y2}" fill="none" stroke="var(--muted)"/>` +
      `<text x="${ox + (r + 12) * Math.cos(t / 2)}" y="${oy - (r + 12) * Math.sin(t / 2)}" class="pp-ang" text-anchor="middle" dominant-baseline="middle">${Math.round(ang * 180 / Math.PI)}°</text>`;
  }
  const svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block;max-width:${W0}px"><defs>${mk('a')}${mk(bc)}</defs>
    <line x1="10" y1="${oy}" x2="${W0 - 10}" y2="${oy}" stroke="var(--grid)"/>
    <path d="M${ox - Rpx},${oy} A${Rpx},${Rpx} 0 0 1 ${ox + Rpx},${oy}" fill="none" stroke="var(--grid)" stroke-dasharray="2 4"/>
    ${arc}${arrow(b, bc, o.bLabel)}${arrow(a, 'a', o.aLabel)}</svg>`;
  return `<div class="pplane">${svg}<span class="note">angle ${Math.round(ang * 180 / Math.PI)}° · cosine similarity <b>${cos.toFixed(2)}</b></span></div>`;
}

// ---------- sticky status bar ----------
// The header changes with where you are: the context track (5–6), with 7's stack (7–8), the whole
// sentence (9, whose update uses every prediction at once; click a token to pick 9's row), then hides.
function renderStatebar() {
  const sent = S6.sbMode === 'sent';
  document.querySelector('#statebar .sb-scroll').hidden = sent; $('sb-sentview').hidden = !sent;
  if (sent) $('sb-sentview').innerHTML = `<span class="lab">every prediction in this sentence · row</span>${seqSel().map(t => `<span class="chip${t === S8.row ? ' v' : ''}" data-t="${t}" title="pick this row">${tstr(t)}</span>`).join('')}`;
  else { TRACKSB.render(); bindStack($('sb-track').querySelector('.m6vp')); }
  syncPass();
}
// the header only makes sense once predictions exist: show it from section 5's widget on
(function headerWhenPreds() {
  let ticking = false;
  const check = () => {
    ticking = false; const H = $('statebar').offsetHeight, top = id => $(id).getBoundingClientRect().top;
    // shown once 5's widget is scrolled past, until the end of 9's (past that nothing uses the selected
    // prediction), and hidden while 7's own track is on screen, so there is never a second track in view
    const r7 = $('m6-track').getBoundingClientRect(), track7 = r7.bottom > H && r7.top < innerHeight;
    $('statebar').classList.toggle('shown', $('m5').getBoundingClientRect().bottom < H && !track7 && $('m8').getBoundingClientRect().bottom > H);
    const mode = top('s9') < H ? 'sent' : $('m6-track').getBoundingClientRect().bottom < H ? 'stack' : 'track';
    if (mode !== S6.sbMode) { S6.sbMode = mode; renderStatebar(); }
  };
  addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(check); } }, { passive: true });
  addEventListener('resize', check); setTimeout(check, 0);
})();

// ---------- 1 GPT-2's tokens for any text ----------
const M1_EX = [
  ['a sentence', 'The cat sat on the mat and drank a cup of coffee.'],
  ['capitals', 'coffee Coffee COFFEE, and coffee mid-sentence.'],
  ['rare words', 'Tokenization of embeddings is unbelievably useful.'],
  ['numbers', 'In 1987, 1,234,567 people paid $3.50 each.'],
  ['code', 'def square(x):\n    return x ** 2'],
  ['other scripts', 'Café ☕ naïve 東京 Москва 🙂'],
];
const S1 = { sel: -1 };
function render1() {
  const text = $('m1-text').value, ids = BPE.encode(text);
  const n = [...text].length;
  $('m1-nchar').innerHTML = `<b>${commas(n)}</b> character${n === 1 ? '' : 's'}`; $('m1-ntok').innerHTML = `<b>${commas(ids.length)}</b> token${ids.length === 1 ? '' : 's'}`;
  $('m1-toks').innerHTML = ids.map((id, i) => `<span class="m1tok t${i % 5}${i === S1.sel ? ' on' : ''}" data-i="${i}"><span class="s">${gstr(id)}</span><span class="id">${id}</span></span>${BPE.bytes(id).includes(10) ? '<span class="m1br"></span>' : ''}`).join('');
  const id = ids[S1.sel];
  $('m1-detail').innerHTML = id === undefined ? 'Click a token to see its ID and bytes. A dot · marks a space.'
    : `Token <b>${id}</b> ${id === EOT ? 'is the special end-of-text token.' : id < 256 ? `is a single byte, <span class="mono">${BPE.hex(id)}</span>.` : `is made of ${BPE.bytes(id).length} byte${BPE.bytes(id).length > 1 ? 's' : ''}, <span class="mono">${BPE.hex(id)}</span>, by merge number ${commas(id - 256 + 1)} of 50,000.`}${BPE.str(id) === null ? ' It is part of a character that takes several bytes, so it can’t be shown as text on its own.' : ''}`;
  document.querySelectorAll('#m1-ex button').forEach(b => b.classList.toggle('on', M1_EX[+b.dataset.i][1] === text));
}
$('m1-ex').innerHTML = M1_EX.map(([l], i) => `<button data-i="${i}">${l}</button>`).join('');
$('m1-ex').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { $('m1-text').value = M1_EX[+b.dataset.i][1]; S1.sel = -1; render1(); } });
$('m1-text').value = M1_EX[0][1];
$('m1-text').addEventListener('input', () => { S1.sel = -1; render1(); });
$('m1-toks').addEventListener('click', e => { const t = e.target.closest('[data-i]'); if (t) { S1.sel = +t.dataset.i === S1.sel ? -1 : +t.dataset.i; render1(); } });

// ---------- 2 byte-pair encoding on the corpus ----------
// The corpus split into GPT-2's pieces (words with their space), each piece starting as single
// characters. Every step merges the most common neighbouring pair, counted across the corpus.
const S2 = (() => {
  const freq = new Map();
  CORPUS.forEach(s => BPE.pieces(s).forEach(p => freq.set(p, (freq.get(p) || 0) + 1)));
  const pieces = [...freq.keys()], chars = new Set(pieces.join(''));
  let cur = pieces.map(p => [...p]);
  const states = [cur], merges = [], counts = [];
  for (;;) {
    const c = new Map();
    cur.forEach((t, i) => { for (let j = 0; j + 1 < t.length; j++) { const k = t[j] + '\u0000' + t[j + 1]; c.set(k, (c.get(k) || 0) + freq.get(pieces[i])); } });
    // most common first; ties go to the pair that comes first alphabetically
    const top = [...c].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
    counts.push(top);
    if (!top.length) break;
    const [a, b] = top[0][0].split('\u0000');
    merges.push({ a, b, n: top[0][1] });
    cur = cur.map(t => { const o = []; for (let j = 0; j < t.length; j++) { if (j + 1 < t.length && t[j] === a && t[j + 1] === b) { o.push(a + b); j++; } else o.push(t[j]); } return o; });
    states.push(cur);
  }
  return { s: 0, freq, pieces, index: new Map(pieces.map((p, i) => [p, i])), nChars: chars.size, states, merges, counts };
})();
// The loop, as a flowchart that lights up the step the widget is on, one box at a time. Steps:
// 0 = split into characters; then for each merge: count the pairs, check whether to stop ("no"),
// merge; at the end: count (no pairs left), check ("yes"), return the merge list.
function flow2(lit, way) {
  const on = k => lit === k ? ' on' : '', go = w => way === w ? ' on' : '';
  const box = (k, x, y, w, h, l1, l2) => `<g class="fb${on(k)}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text x="${x + w / 2}" y="${y + h / 2 + (l2 ? -9 : 0)}">${l1}</text>${l2 ? `<text class="s" x="${x + w / 2}" y="${y + h / 2 + 11}">${l2}</text>` : ''}</g>`;
  return `<svg viewBox="0 0 680 450" width="100%" role="img" aria-label="the byte-pair encoding loop"><defs>${['f2a', 'f2b'].map((id, n) => `<marker id="${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="${n ? 'var(--violet)' : 'var(--ink-2)'}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>`).join('')}</defs>
    ${box('split', 190, 10, 300, 56, 'Split text into words', 'then each word into characters')}
    <line class="fa" x1="340" y1="66" x2="340" y2="88"/>
    ${box('count', 190, 90, 300, 44, 'Count every neighbouring pair')}
    <line class="fa" x1="340" y1="134" x2="340" y2="154"/>
    <g class="fb${on('check')}"><polygon points="340,156 480,216 340,276 200,216"/><text x="340" y="206">50,000 merges done,</text><text x="340" y="228">or no pairs left?</text></g>
    <line class="fa${go('no')}" x1="340" y1="276" x2="340" y2="306"/><text class="fl${go('no')}" x="352" y="294">no</text>
    ${box('merge', 190, 308, 300, 44, 'Merge the most common pair')}
    <path class="fa" d="M190 330 L150 330 L150 112 L188 112"/>
    <path class="fa${go('yes')}" d="M480 216 L560 216 L560 368"/><text class="fl${go('yes')}" x="520" y="206">yes</text>
    ${box('ret', 440, 370, 220, 56, 'Return the merge list', 'in order')}
  </svg>`;
}
const STEPS2 = () => 3 * S2.merges.length + 4;   // split; count, check, merge per merge; then count, check, return
function render2() {
  const s = S2.s, K = S2.merges.length, r = s ? (s - 1) % 3 : -1, k0 = s ? Math.floor((s - 1) / 3) : 0;
  const phase = s === 0 ? 'split' : r === 0 ? 'count' : r === 1 ? 'check' : k0 < K ? 'merge' : 'ret';
  const k = phase === 'merge' ? k0 + 1 : k0;   // merges made so far
  const st = S2.states[k], last = phase === 'merge' ? S2.merges[k0] : null, fresh = last && last.a + last.b;
  const counted = phase === 'count' || phase === 'check', win = counted && S2.counts[k0].length ? S2.counts[k0][0][0].split('\u0000') : null;
  $('m2-flow').innerHTML = flow2(phase, phase === 'check' ? (win ? 'no' : 'yes') : null);
  // the corpus: once counted, the winning pair is marked wherever it sits; after a merge, the new token pops in
  const piece = toks => { let h = ''; for (let i = 0; i < toks.length; i++) {
    const w = win && i + 1 < toks.length && toks[i] === win[0] && toks[i + 1] === win[1];
    if (w) { h += `<span class="m2t win">${vis(toks[i])}</span><span class="m2t win">${vis(toks[i + 1])}</span>`; i++; }
    else h += `<span class="m2t${toks[i] === fresh ? ' pop' : ''}">${vis(toks[i])}</span>`;
  } return h; };
  $('m2-corpus').innerHTML = CORPUS.map(c => `<div class="sline m2line">${BPE.pieces(c).map(p => `<span class="m2piece">${piece(st[S2.index.get(p)])}</span>`).join('')}</div>`).join('');
  // the counts: shown once counted; after a merge, the counts that chose it
  const top = s === 0 ? [] : S2.counts[k0].slice(0, 10), mx = top.length ? top[0][1] : 1;
  $('m2-pairs').innerHTML = s === 0 ? '<p class="note" style="padding:8px 10px;margin:0">Not counted yet.</p>'
    : top.length ? `<table class="data vocab"><tbody>${top.map(([p, n], i) => { const [a, b] = p.split('\u0000'); const w = !i && counted ? ' win' : ''; return `<tr class="${i ? '' : 'on'}"><td class="mono"><span class="m2t${w}">${vis(a)}</span> + <span class="m2t${w}">${vis(b)}</span></td><td class="n">${n}</td><td class="bar"><i style="width:${100 * n / mx}%"></i></td></tr>`; }).join('')}</tbody></table>`
    : '<p class="note" style="padding:8px 10px;margin:0">No pairs left: every word is a single token.</p>';
  $('m2-merges').innerHTML = k ? S2.merges.slice(0, k).map((m, i) => `<span class="m2m${i === k - 1 && phase === 'merge' ? ' last' : ''}"><span class="m2t">${vis(m.a)}</span>+<span class="m2t">${vis(m.b)}</span></span>`).reverse().join('') : '<span class="note">none yet</span>';
  const tokens = st.reduce((n, t, i) => n + t.length * S2.freq.get(S2.pieces[i]), 0);
  $('m2-stats').innerHTML = `merges <b>${k}</b> · vocabulary <b>${S2.nChars + k}</b> (${S2.nChars} characters + ${k} merges) · the corpus is <b>${tokens}</b> tokens long`;
  $('m2-note').innerHTML = {
    split: `The corpus starts as ${S2.nChars} different characters, spaces included. Step to count every pair of neighbours.`,
    count: win ? `Counted. The most common pair is “${vis(win[0])}” + “${vis(win[1])}”, side by side ${S2.counts[k0][0][1]} times.` : 'Counted: there are no pairs left. Every word is a single token.',
    check: win ? `${k} merge${k === 1 ? '' : 's'} so far, far from 50,000, and pairs are left: no, not done. So merge.` : 'No pairs left: yes, done.',
    merge: last && `Merged “${vis(last.a)}” + “${vis(last.b)}” into one token, everywhere it occurs. Then back to counting.`,
    ret: `Done after ${k} merges: every word in this small corpus is now a single token, so no pairs are left to merge. A large corpus has far too many different words for that. There, merging stops at a chosen number instead: GPT-2 stopped after 50,000, so rarer words are still split into pieces.`,
  }[phase];
}
PlayBar($('m2-bar'), {
  ms: 280, canStep: () => S2.s < STEPS2() - 1, canReset: () => S2.s > 0, rewind: () => { S2.s = 0; },
  step: () => { S2.s++; render2(); }, end: () => { S2.s = STEPS2() - 1; render2(); }, reset: () => { S2.s = 0; render2(); },
});

// ---------- 2 GPT-2's merges replayed on one word ----------
const M2G_EX = [' embeddings', ' unbelievably', ' coffee', ' SolidGoldMagikarp', ' naïve'];
const S2G = { steps: [], k: 0, word: '' };
function load2g() {
  const t = $('m2g-word').value, p = BPE.pieces(t);
  S2G.word = p[0] || ''; S2G.more = p.length > 1; S2G.steps = S2G.word ? BPE.mergeSteps(S2G.word) : []; S2G.k = S2G.steps.length - 1;
  document.querySelectorAll('#m2g-ex button').forEach(b => b.classList.toggle('on', b.dataset.w === t));
}
function render2g() {
  const S = S2G.steps;
  if (!S.length) { $('m2g-steps').innerHTML = ''; $('m2g-stats').textContent = 'Type a word.'; return; }
  $('m2g-steps').innerHTML = S.slice(0, S2G.k + 1).map((s, i) => {
    const chips = s.ids.map((id, j) => `<span class="chip${i && j === s.at ? ' v pop' : ''}">${gstr(id)}</span>`).join('');
    const what = i ? `merge <b>${commas(s.rank + 1)}</b> → ID ${256 + s.rank}` : `${s.ids.length} single bytes`;
    return `<div class="m2gs${i === S2G.k ? ' cur' : ''}"><span class="m2gn">${what}</span><span class="chips">${chips}</span></div>`;
  }).join('');
  const fin = S[S.length - 1].ids;
  $('m2g-stats').innerHTML = (S2G.k === S.length - 1 ? `No merge applies any more: <b>${fin.length}</b> token${fin.length > 1 ? 's' : ''}, ID${fin.length > 1 ? 's' : ''} ${fin.join(', ')}.` : `step ${S2G.k} of ${S.length - 1}`)
    + (S2G.more ? ` Only the first word, “${vis(S2G.word)}”, is shown.` : '');
}
$('m2g-ex').innerHTML = M2G_EX.map(w => `<button data-w="${esc(w)}">${esc(w.trim())}</button>`).join('');
$('m2g-ex').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { $('m2g-word').value = b.dataset.w; load2g(); S2G.k = 0; render2g(); bar2g.act('play'); } });
$('m2g-word').addEventListener('input', () => { bar2g.stop(); load2g(); render2g(); refreshBars(); });
const bar2g = PlayBar($('m2g-bar'), {
  ms: 650, canStep: () => S2G.k < S2G.steps.length - 1, canReset: () => S2G.k > 0, rewind: () => { S2G.k = 0; },
  step: () => { S2G.k++; render2g(); }, end: () => { S2G.k = S2G.steps.length - 1; render2g(); }, reset: () => { S2G.k = 0; render2g(); },
});

// ---------- 3 the corpus as GPT-2 tokens, and the vocabulary ----------
function render3() {
  const sel = inTok(), maxc = Math.max(...V.count);
  $('m3-corpus').innerHTML = V.seqs.map((s, k) => `<div class="sline m3line">${s.slice(1, -1).map((t, j) => `<span class="m3t${t === sel ? ' on' : ''}${k === SEL.sent && j + 1 === SEL.pos ? ' here' : ''}" data-s="${k}" data-p="${j + 1}">${tstr(t)}</span>`).join('')}</div>`).join('');
  $('m3-vocab').innerHTML = `<table class="data vocab"><thead><tr><th></th><th>token</th><th class="n">GPT-2 ID</th><th class="n">count</th><th></th></tr></thead><tbody>${V.ids.map((g, i) => `<tr class="${i === sel ? 'on' : ''}" data-t="${i}"><td class="n rk">${i + 1}</td><td class="mono">${tstr(i)}</td><td class="n mono">${g}</td><td class="n">${V.count[i]}</td><td class="bar"><i style="width:${100 * V.count[i] / maxc}%"></i></td></tr>`).join('')}</tbody></table>`;
  $('m3-stats').textContent = `${V.n} tokens in the vocabulary · ${V.total} tokens of text · ${V.seqs.length} sentences, each ended by <|endoftext|>`;
  $('m3-seed').textContent = `seed ${G.seed}`;
}
// a token's first place as an input: the prediction it makes there
function firstUse(t) { for (let k = 0; k < V.seqs.length; k++) { const j = V.seqs[k].indexOf(t); if (j >= 0 && j < V.seqs[k].length - 1) return [k, j]; } return [0, 0]; }
$('m3-corpus').addEventListener('click', e => { const el = e.target.closest('[data-s]'); if (el) selectPred(+el.dataset.s, +el.dataset.p); });
$('m3-vocab').addEventListener('click', e => { const el = e.target.closest('[data-t]'); if (el) selectPred(...firstUse(+el.dataset.t)); });
// retrain the whole page's model, after the seed or the learning rate changes
function retrain(prev) { stopAll(); G.prev = prev; resetTraining(); renderAll(); }
$('reroll').addEventListener('click', () => { const prev = snapshotRun(); G.seed = 1 + Math.floor(Math.random() * 9999); retrain(prev); });

// ---------- 3 one token's row: the toy's 6 numbers, and GPT-2's 768 ----------
// Four fixed tokens, plus a fifth slot: a random one at first, then the selected prediction's
// input token whenever that is not one of the four.
const FIXED3 = [' coffee', ' tea', ' dog', ' cat'].map(tokIx);
const S3V = { tok: FIXED3[0], real: 'end', extra: (o => o[Math.floor(Math.random() * o.length)])(V.ids.map((_, i) => i).filter(i => !FIXED3.includes(i) && V.ids[i] !== EOT)) };
function follow3v() { if (!FIXED3.includes(inTok())) S3V.extra = inTok(); S3V.tok = inTok(); }
const REAL = (() => {   // GPT-2's own rows for the toy vocabulary and a few more, int8 with one scale per row
  const q = b64(GPT2.rows.q, Int8Array), m = new Map();
  GPT2.rows.ids.forEach((id, r) => m.set(id, Float32Array.from(q.subarray(r * 768, r * 768 + 768), x => x * GPT2.rows.scale[r])));
  return m;
})();
function heatColor(x, top) { const a = Math.min(1, Math.abs(x) / top); return x >= 0 ? `rgba(42,120,214,${(0.06 + 0.9 * a).toFixed(3)})` : `rgba(235,104,52,${(0.06 + 0.9 * a).toFixed(3)})`; }
function render3v() { render3vToy(); render3vReal(); syncStep(); }
function render3vToy() {
  const m = G.snaps[G.step], v = arr(rowOf(m, 'E', S3V.tok)), N = G.stream.total;
  const top = Math.max(...arr(view(G.snaps[G.snaps.length - 1], 'E')).map(Math.abs));   // one scale for all three stages
  const cell = (x, d) => `<span class="m2v-cell"><span class="d">d${d + 1}</span><span class="v" style="background:${x >= 0 ? 'rgba(42,120,214,' : 'rgba(235,104,52,'}${(0.08 + 0.42 * Math.abs(x) / top).toFixed(3)})">${num(x, 2)}</span></span>`;
  $('m3v-vec').innerHTML = `<div class="m2v-row"><span class="m2v-word"><span>${tstr(S3V.tok)}</span><span class="sub">ID ${V.ids[S3V.tok]}</span></span><span class="m2v-arrow" aria-hidden="true">→</span><span class="m2v-br">${v.map(cell).join('')}</span></div>`;
  $('m3v-note').innerHTML = G.step === 0 ? `Six random numbers, most within ±0.04. Nothing about “${plain(S3V.tok).trim()}” is in them yet.`
    : G.step < N ? `After ${commas(G.step)} of ${commas(N)} steps, the numbers have grown and moved. Drag on, or switch between the tokens to compare.`
    : 'Training nudges the numbers until tokens used alike have alike embeddings. Compare “ coffee” with “ tea”, then with “ dog”.';
}
function render3vReal() {
  // GPT-2's row: 768 numbers as a grid of squares, 32 to a line. OpenAI released only the trained
  // table, so "start" is a fresh draw from GPT-2's starting distribution, N(0, 0.02), seeded by the
  // page's initialization seed and the token's ID, so re-rolling draws it again; "middle" has nothing
  // to show. Start and end share one color scale, the trained row's, stated in the key under the
  // grid: the start looks faint because its numbers are about 7 times smaller.
  const id = V.ids[S3V.tok], R = REAL.get(id), C = 32, s = 9;
  const r0 = rng(G.seed * 65537 + id), Rs = S3V.real === 'start' ? Float32Array.from({ length: 768 }, () => gauss(r0) * CFG.STD) : R;
  const rtop = Math.max(...R.map(Math.abs)) * 0.6;
  let svg = `<svg viewBox="0 0 ${C * s} ${768 / C * s}" width="100%" style="display:block;max-width:${C * s}px" aria-label="768 numbers">`;
  for (let i = 0; i < 768; i++) svg += `<rect x="${(i % C) * s}" y="${Math.floor(i / C) * s}" width="${s - 1}" height="${s - 1}" rx="1.5" data-i="${i}" fill="${Rs ? heatColor(Rs[i], rtop) : '#f3f2ee'}"/>`;
  S3V.vals = Rs;   // for the hover readout
  // a vertical color bar beside the grid: positive at the top, 0 in the middle, negative at the bottom
  const lab = x => num(x, rtop < 0.1 ? 3 : 2), key = Rs ? `<div class="m3vkey" aria-label="color scale"><span>+${lab(rtop)}</span><i style="background:linear-gradient(180deg, ${heatColor(rtop, rtop)}, white, ${heatColor(-rtop, rtop)})"></i><span>−${lab(rtop)}</span></div>` : '';
  $('m3v-real').innerHTML = `<div class="m3vheat">${svg}</svg>${key}<span class="m3vtip" hidden></span></div><p class="note" style="margin:8px 0 0">${{
    start: `A random start for ${tstr(S3V.tok)}’s embedding (row ${id}): 768 numbers drawn with a standard deviation of 0.02, the way GPT-2’s code starts every embedding. GPT-2’s actual starting numbers were never published.`,
    end: `GPT-2’s embedding of ${tstr(S3V.tok)} (row ${id}), after training, from its released weights. Blue is positive, orange negative.`,
  }[S3V.real]}${S3V.real === 'start' ? '' : ' OpenAI released only the finished table, so there is nothing to show from partway through training.'}</p>`;
  $('m3v-word').innerHTML = FIXED3.concat([S3V.extra]).map(i => `<button data-v="${i}" class="${i === S3V.tok ? 'on' : ''}">${tstr(i)}</button>`).join('');
  document.querySelectorAll('#m3v-gstate button').forEach(b => b.classList.toggle('on', b.dataset.v === S3V.real));
}
// hovering a square shows its number; a mouse only, since a finger covers the square anyway
$('m3v-real').addEventListener('pointermove', e => {
  const tip = $('m3v-real').querySelector('.m3vtip'), sq = e.target.closest('[data-i]');
  if (!tip) return;
  if (e.pointerType !== 'mouse' || !sq || !S3V.vals) { tip.hidden = true; return; }
  const i = +sq.dataset.i, box = tip.parentElement.getBoundingClientRect(), r = sq.getBoundingClientRect();
  tip.innerHTML = `d${i + 1} <b>${num(S3V.vals[i], 3)}</b>`; tip.hidden = false;
  tip.style.left = (r.left + r.width / 2 - box.left) + 'px'; tip.style.top = (r.top - box.top) + 'px';
});
$('m3v-real').addEventListener('pointerleave', () => { const tip = $('m3v-real').querySelector('.m3vtip'); if (tip) tip.hidden = true; });
$('m3v-word').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S3V.tok = +b.dataset.v; render3v(); } });
$('m3v-gstate').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S3V.real = b.dataset.v; render3vReal(); } });

// ---------- 4 token row + position row = input vector ----------
const sentOpts = id => { $(id).innerHTML = V.seqs.map((s, k) => `<option value="${k}">${esc(CORPUS[k])}</option>`).join(''); };
sentOpts('m4-sent'); sentOpts('m5-sent'); sentOpts('sb-sent');
$('m4-sent').addEventListener('change', e => selectPred(+e.target.value, SEL.pos));
$('m5-sent').addEventListener('change', e => selectPred(+e.target.value, SEL.pos));
$('sb-sent').addEventListener('change', e => selectPred(+e.target.value, SEL.pos));
// a lookup, so one fixed moment per stage (start, halfway, end of training), whatever the sentence:
// changing the sentence changes which rows are looked up, not the tables themselves
const EQ4 = VecEq($('m4-eq'), () => {
  const m = G.snaps[G.step], t = inTok(), tok = arr(rowOf(m, 'E', t)), pos = arr(rowOf(m, 'P', SEL.pos));
  return { left: [{ t: 'vec', label: 'input embedding', sym: SV, chip: wchip(t, 'v'), vals: tok, cls: 'cv' }, { t: 'op', s: '+' },
    { t: 'vec', label: 'position embedding', sym: '<i class="cpos">P</i>', chip: `<span class="chip pos">position ${SEL.pos}</span>`, vals: pos, cls: 'cpos' }],
    result: { label: 'input vector', chip: `<span class="chip">${tstr(t)} at ${SEL.pos}</span>`, vals: tok.map((x, d) => x + pos[d]), cls: 'cp' } };
}, 3);
function render4() {
  $('m4-sent').value = SEL.sent; syncStep();
  $('m4-track').innerHTML = seqSel().slice(0, -1).map((t, i) => `<span class="m4tok${i === SEL.pos ? ' on' : ''}" data-p="${i}"><span class="chip ${i === SEL.pos ? 'v' : ''}">${tstr(t)}</span><span class="pn">${i}</span></span>`).join('');
  EQ4.refresh();
}
$('m4-track').addEventListener('click', e => { const el = e.target.closest('[data-p]'); if (el) selectPred(SEL.sent, +el.dataset.p); });

// ---------- 4 GPT-2's position rows ----------
// the heatmap at the start or end of training. GPT-2's starting rows weren't published, so "start" is a
// fresh draw from its starting distribution, N(0, 0.01), seeded by the page's initialization seed
const S4G = { stage: 'end', key: '', rand: null, dims: null, k: 2 };   // k: which stop of the dimensions slider
// the second-to-last stop is every dimension that changes much with position; the last adds the flat rest
const K4G = [1, 2, 4, 8, 16, 32, GPT2.pos.moving, 768];
// all 1,024 random rows for this seed; the heatmap and the dimensions chart both use every 8th, as
// for the trained table, with the dimensions in the trained table's order
function rand4g() {
  if (S4G.key === G.seed) return;
  const n = 1024, d = 768, r = rng(G.seed * 7919 + 3), X = new Float64Array(n * d);
  for (let i = 0; i < X.length; i++) X[i] = gauss(r) * 0.01;
  const m = GPT2.pos.n, step = GPT2.pos.step, U = Array.from({ length: m }, (_, k) => { const v = X.subarray(k * step * d, k * step * d + d), l = norm(v); return Float64Array.from(v, x => x / l); });
  S4G.rand = new Float32Array(m * m); for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) S4G.rand[i * m + j] = dot(U[i], U[j]);
  S4G.dims = new Float32Array(m * d); for (let k = 0; k < m; k++) GPT2.pos.order.forEach((dd, c) => { S4G.dims[k * d + c] = X[k * step * d + dd]; });
  S4G.key = G.seed;
}
const TRAINED4G = (() => { const q = b64(GPT2.pos.all, Int8Array); return Float32Array.from(q, x => x * GPT2.pos.allScale); })();
function cos4g() {
  if (S4G.stage === 'end') { const q = b64(GPT2.pos.cos, Uint8Array); return Float32Array.from(q, x => x / 255 * 2 - 1); }
  rand4g(); return S4G.rand;
}
function renderHeat4g() {
  const n = GPT2.pos.n, q = cos4g(), cv = document.createElement('canvas');
  cv.width = n; cv.height = n; const cx = cv.getContext('2d'), img = cx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const c = q[i], a = Math.min(1, Math.abs(c));
    const [r, g, b] = c >= 0 ? [42, 120, 214] : [235, 104, 52];   // blue alike, orange opposite, on white
    img.data[4 * i] = 255 + (r - 255) * a; img.data[4 * i + 1] = 255 + (g - 255) * a; img.data[4 * i + 2] = 255 + (b - 255) * a; img.data[4 * i + 3] = 255;
  }
  cx.putImageData(img, 0, 0);
  const ticks = [0, 256, 512, 768, 1023];
  $('m4g-heat').innerHTML = `<div class="m4gwrap"><div class="m4gy">${ticks.map(t => `<span style="top:${t / 1024 * 100}%">${t}</span>`).join('')}</div><div class="m4gbox"></div><div class="m4gx">${ticks.map(t => `<span style="left:${t / 1024 * 100}%">${t}</span>`).join('')}</div></div>`;
  cv.className = 'm4gcv'; $('m4g-heat').querySelector('.m4gbox').appendChild(cv);
  const read = () => $('m4g-read').innerHTML = S4G.stage === 'end'
    ? 'This shows GPT-2’s learned position embeddings by comparing every pair of them. Each square is two positions: blue where their embeddings point the same way, white where they are unrelated, orange where opposite. Hover to read a square.'
    : 'Before training: random embeddings, drawn the way GPT-2’s code starts them. Every position is unrelated to every other, so all but the diagonal, each position compared with itself, is nearly white. GPT-2’s actual starting embeddings were never published. Hover to read a square.';
  read();
  cv.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect(), i = Math.max(0, Math.min(n - 1, Math.floor((e.clientX - r.left) / r.width * n))), j = Math.max(0, Math.min(n - 1, Math.floor((e.clientY - r.top) / r.height * n)));
    $('m4g-read').innerHTML = `positions <b>${j * GPT2.pos.step}</b> and <b>${i * GPT2.pos.step}</b>: cosine similarity <b>${num(q[j * n + i], 2)}</b>`;
  });
  cv.addEventListener('pointerleave', read);
  document.querySelectorAll('#m4g-stage button').forEach(b => b.classList.toggle('on', b.dataset.v === S4G.stage));
}
$('m4g-stage').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S4G.stage = b.dataset.v; render4g(); } });
function render4g() { renderHeat4g(); renderDims4g(); }
// the k dimensions that change most with position, traced across positions, plus one typical
// dimension in grey for reference. Start and end share one scale, so the random start is as flat as it is.
function renderDims4g() {
  const end = S4G.stage === 'end', k = K4G[S4G.k], m = GPT2.pos.n, d = 768, step = GPT2.pos.step, mov = GPT2.pos.moving;
  if (!end) rand4g();
  const V4 = end ? TRAINED4G : S4G.dims, val = (p, c) => V4[p * d + c], typ = GPT2.pos.order.indexOf(GPT2.pos.typical);
  const mx = 1.5, W0 = 320, l = 30, r = 0, t = 0, b = 28, H0 = W0 - l - r + t + b;   // the same frame as the heatmap beside it
  const X = p => l + p * step / 1023 * (W0 - l - r), Y = v => t + (1 - (Math.max(-mx, Math.min(mx, v)) / mx + 1) / 2) * (H0 - t - b);
  const few = ['var(--violet)', 'var(--aqua)', 'var(--gold)', 'var(--blue)'];
  const line = (c, col, w, op = 1) => { let s = ''; for (let p = 0; p < m; p++) s += (p ? 'L' : 'M') + X(p).toFixed(1) + ',' + Y(val(p, c)).toFixed(1); return `<path d="${s}" fill="none" stroke="${col}" stroke-width="${w}" opacity="${op}"/>`; };
  let svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block;max-width:${W0}px"><rect x="${l}" y="${t}" width="${W0 - l - r}" height="${H0 - t - b}" fill="white" stroke="var(--grid)"/>`;
  svg += [-1, 0, 1].map(v => `<line x1="${l}" x2="${W0 - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${l - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${num(v, 0)}</text>`).join('');
  // hovering a position: the shown waves that change fastest there, the ones that best tell it from its
  // neighbours, stand out; the slope is measured across the neighbouring samples (8 positions apart)
  const hi = end && S4G.hover != null ? S4G.hover : null, shown = Math.min(k, mov);
  const steep = hi == null ? [] : Array.from({ length: shown }, (_, c) => c).filter(c => c !== typ)
    .map(c => [c, Math.abs(val(Math.min(m - 1, hi + 1), c) - val(Math.max(0, hi - 1), c))]).sort((x, y) => y[1] - x[1]).slice(0, 3).map(([c]) => c);
  const fade = hi == null ? 1 : 0.25;
  // the moving ones first, then the flat ones on top, so they show as a grey band along 0
  for (let c = 0; c < shown; c++) if (c !== typ && !steep.includes(c)) svg += k <= 4 ? line(c, few[c], 1.6, fade) : line(c, 'var(--violet)', 1, 0.5 * fade);
  for (let c = mov; c < k; c++) if (c !== typ) svg += line(c, '#a9a69d', 0.7, 0.35 * fade);
  svg += line(typ, 'var(--ink-2)', 1.6, fade);
  const hcol = (c, r) => k <= 4 ? few[c] : few[r];
  if (hi != null) {
    svg += `<line x1="${X(hi)}" x2="${X(hi)}" y1="${t}" y2="${H0 - b}" stroke="var(--ink)" stroke-dasharray="3 3"/>`;
    steep.forEach((c, r) => { svg += line(c, hcol(c, r), 2.2); });
    steep.forEach((c, r) => { svg += `<circle cx="${X(hi)}" cy="${Y(val(hi, c))}" r="4" fill="${hcol(c, r)}" stroke="white" stroke-width="1.5"/>`; });
  }
  svg += [0, 256, 512, 768, 1023].map(p => `<text x="${l + p / 1023 * (W0 - l - r)}" y="${H0 - 15}" text-anchor="${p ? p === 1023 ? 'end' : 'middle' : 'start'}" font-size="10">${p}</text>`).join('') + `<text x="${l + (W0 - l - r) / 2}" y="${H0 - 2}" text-anchor="middle" font-size="10">position</text></svg>`;
  $('m4g-pc').innerHTML = svg;
  const dn = c => 'd' + (GPT2.pos.order[c] + 1);
  $('m4g-pchead').textContent = `${k === d ? 'All 768' : `${k} of the 768`} dimensions, by position`;
  const leg = k <= 4 ? Array.from({ length: k }, (_, c) => `<span><i style="background:${few[c]}"></i>${dn(c)}</span>`).join('')
    : `<span><i style="background:var(--violet)"></i>change with position</span>${k > mov ? '<span><i style="background:#b6b4ad"></i>don’t</span>' : ''}`;
  // reading the position back from these dimensions (precomputed: a weighted sum fitted on four fifths of
  // the positions and tested on the rest, five times round; the average error, in positions)
  const R4 = GPT2.pos.read, err = R4[end ? 'end' : 'start'][S4G.k], these = k === 1 ? 'this dimension' : `these ${k === d ? 768 : k} dimensions`;
  const e63 = R4.end[R4.stops.indexOf(mov)], fmt = x => x >= 10 ? Math.round(x) : x;
  $('m4g-readout').innerHTML = !end ? `Reading the position back: from random embeddings, a weighted sum of ${these} is off by about <b>${Math.round(err)}</b> positions on average, no better than always guessing the middle.`
    : err > 200 ? `Reading the position back: a weighted sum of ${these} is off by about <b>${Math.round(err)}</b> positions on average, no better than guessing. A slow wave passes through the same height at many positions.`
    : k > mov ? `Reading the position back: a weighted sum of all 768 finds the position to within about <b>${fmt(err)}</b> positions on average, the same as the ${mov} that move (${fmt(e63)}). The flat dimensions add nothing.`
    : `Reading the position back: a weighted sum of ${these} finds the position to within about <b>${fmt(err)}</b> positions on average.`;
  if (hi != null) $('m4g-readout').innerHTML = `At position <b>${hi * step}</b>, ${steep.map((c, r) => `<b style="color:${hcol(c, r)}">${dn(c)}</b>`).join(steep.length > 2 ? ', ' : ' and ').replace(/, ([^,]*)$/, ' and $1')} change${steep.length === 1 ? 's' : ''} fastest, so ${steep.length === 1 ? 'it best tells' : 'they best tell'} it apart from its neighbours. Flatter waves here only say roughly where it is.`;
  $('m4g-kval').textContent = k === d ? 'all 768' : k === mov ? `all ${mov} that move` : `the top ${k}`;
  $('m4g-pcnote').innerHTML = `<div class="legend" style="margin-top:0">${leg}<span><i style="background:var(--ink-2)"></i>${dn(typ)}, a typical one</span></div>` + (end
    ? `Each line follows one dimension across all 1,024 positions, ranked by how much it changes. Only ${mov} of the 768 change much, in smooth waves${k === mov ? ': these are all of them' : ''}${k > mov ? '; the rest stay in a flat band near 0, like the typical one' : ''}.`
    : 'The same dimensions in the random embeddings: tiny, and no pattern from one position to the next.');
}
$('m4g-k').addEventListener('input', e => { S4G.k = +e.target.value; renderDims4g(); });
// hover (or touch) the chart to pick a position; the frame is 320 wide with the plot from x = 30
$('m4g-pc').addEventListener('pointermove', e => {
  if (S4G.stage !== 'end') return;
  const r = $('m4g-pc').querySelector('svg').getBoundingClientRect(), x = (e.clientX - r.left) / r.width * 320;
  const i = x < 30 ? null : Math.max(0, Math.min(GPT2.pos.n - 1, Math.round((x - 30) / 290 * 1023 / GPT2.pos.step)));
  if (i !== S4G.hover) { S4G.hover = i; renderDims4g(); }
});
$('m4g-pc').addEventListener('pointerleave', () => { if (S4G.hover != null) { S4G.hover = null; renderDims4g(); } });

// ---------- 5 each position predicts the next ----------
// The sentence and its context box, used in 5 and again in 6: `k` is the module's id prefix. Both
// copies show the page's selected prediction, and dragging either one changes it.
// stack(): in 7, the next position is a vertical stack of every candidate token, crossing the sentence
// like an intersection; the positions after it are blank, since the model hasn't got there yet
function ContextTrack(k, stack = null) {
  const track = $(k + '-track'), me = {};
  me.render = () => {
    if ($(k + '-sent')) $(k + '-sent').value = SEL.sent;
    const st = me.st = stack ? stack() : null;   // null: an ordinary track this time
    track.innerHTML = seqSel().map((t, i) => st && i > SEL.pos ? (i === SEL.pos + 1 ? st : `<span class="chip ph" data-i="${i}">&nbsp;&nbsp;&nbsp;</span>`) : `<span class="chip" data-i="${i}">${tstr(t)}</span>`).join('')
      + `<span class="m3band"><span class="m3lbl">context</span><span class="m3grip" aria-hidden="true"><svg viewBox="0 0 16 6" width="16" height="6"><g fill="currentColor"><circle cx="2" cy="1.2" r="1.1"/><circle cx="8" cy="1.2" r="1.1"/><circle cx="14" cy="1.2" r="1.1"/><circle cx="2" cy="4.8" r="1.1"/><circle cx="8" cy="4.8" r="1.1"/><circle cx="14" cy="4.8" r="1.1"/></g></svg></span></span>${st ? '' : '<span class="m5next">next</span>'}`;
    me.place();
  };
  me.place = () => {
    const chips = [...track.querySelectorAll('[data-i]')]; if (!chips.length) return;
    const band = track.querySelector('.m3band'), lo = chips[0], hi = chips[SEL.pos], nx = chips[SEL.pos + 1];
    band.style.left = (lo.offsetLeft - 5) + 'px'; band.style.width = (hi.offsetLeft + hi.offsetWidth - lo.offsetLeft + 10) + 'px';
    const stack = me.st;
    if (stack) { band.style.top = (hi.offsetTop - 8) + 'px'; band.style.height = (hi.offsetHeight + 16) + 'px'; band.style.bottom = 'auto'; }
    else { band.style.top = band.style.height = band.style.bottom = ''; track.querySelector('.m5next').style.left = (nx.offsetLeft + nx.offsetWidth / 2) + 'px'; }
    chips.forEach((ch, i) => { if (!ch.classList.contains('m6x')) ch.className = 'chip ' + (i <= SEL.pos ? 'v' : stack ? 'ph' : i === SEL.pos + 1 ? 'tg' : ''); });
    const sc = track.parentElement, R = nx.offsetLeft + nx.offsetWidth + 8;
    if (R > sc.scrollLeft + sc.clientWidth) sc.scrollLeft = R - sc.clientWidth;
  };
  // drag the context's right edge along the sentence; a press that doesn't move picks the token
  let g = null;
  const nearest = x => { let best = 0, d = Infinity; track.querySelectorAll('[data-i]').forEach((ch, i) => { const e = Math.abs(ch.offsetLeft + ch.offsetWidth / 2 - x); if (e < d) { d = e; best = i; } }); return best; };
  const xOf = e => e.clientX - track.getBoundingClientRect().left;
  const setPos = i => { const p = Math.max(0, Math.min(seqSel().length - 2, i)); if (p !== SEL.pos) selectPred(SEL.sent, p); };
  track.addEventListener('pointerdown', e => { if (e.target.closest('.m6x')) return; const chip = e.target.closest('[data-i]'); g = { x0: xOf(e), chip: chip ? +chip.dataset.i : null, moved: false }; track.setPointerCapture(e.pointerId); track.classList.add('dragging'); e.preventDefault(); });
  track.addEventListener('pointermove', e => { if (!g) return; const x = xOf(e); if (!g.moved && Math.abs(x - g.x0) < 5) return; g.moved = true; setPos(nearest(x)); });
  const end = e => { if (g && !g.moved && g.chip != null && e.type === 'pointerup') setPos(g.chip); g = null; track.classList.remove('dragging'); };
  track.addEventListener('pointerup', end); track.addEventListener('pointercancel', end);
  return me;
}
const TRACK5 = ContextTrack('m5'), TRACK6 = ContextTrack('m6', () => stack6()), TRACKSB = ContextTrack('sb', () => S6.sbMode === 'stack' ? stack6() : null);   // the header swaps in 7's stack once you're past it
function render5() {
  const s = seqSel(), n = s.length - 1;
  TRACK5.render();
  $('m5-preds').innerHTML = Array.from({ length: n }, (_, p) => `<span class="pairchip m5p${p === SEL.pos ? ' on' : ''}" data-p="${p}">${p > 0 ? '<span class="m5more">…</span>' : ''}${wchip(s[p], 'v')}<span class="m5ar">→</span>${wchip(s[p + 1], 'tg')}</span>`).join('');
}
$('m5-preds').addEventListener('click', e => { const el = e.target.closest('[data-p]'); if (el) selectPred(SEL.sent, +el.dataset.p); });
addEventListener('resize', () => { TRACK5.place(); TRACK6.place(); TRACKSB.place(); render10(); render11g(); });

// ---------- 6 mixing in the context ----------
// The toy's one layer, for the selected prediction: how much the vector at that position draws on
// each token of its context, and h as the exact sum input vector + attention's mix + MLP's change.
const EQM = VecEq($('mix-eq'), () => {
  const u = upd(), f = u.f, t = SEL.pos, x = arr(f.h0[t]), att = arr(f.h1[t]).map((v, d) => v - x[d]), mlp = arr(f.h2[t]).map((v, d) => v - f.h1[t][d]);
  return { left: [{ t: 'vec', label: 'input vector', sym: '<i class="cv">x</i>', chip: `<span class="chip v">${tstr(inTok())} (${t})</span>`, vals: x, cls: 'cv' }, { t: 'op', s: '+' },
    { t: 'vec', label: 'attention’s mix', sym: '<span class="mixsym">attn</span>', chip: `<span class="chip">from 0 to ${t}</span>`, vals: att, cls: 'cp' }, { t: 'op', s: '+' },
    { t: 'vec', label: 'MLP’s change', sym: '<span class="mixsym">mlp</span>', chip: '<span class="chip">this position</span>', vals: mlp, cls: 'cp' }],
    result: { label: 'h', chip: hChip(), vals: arr(f.h2[t]), cls: 'ch' } };
}, 3);
function renderMix() {
  const u = upd(), w = u.f.a[SEL.pos], s = seqSel(), mx = Math.max(...w);
  syncPass();
  $('mix-wh').innerHTML = `How much the vector at ${wchip(inTok(), 'v')} draws on each token`;
  $('mix-w').innerHTML = w.map((x, i) => `<span class="mixc"><span class="mixbar"><i style="height:${(100 * x / Math.max(mx, 1e-9)).toFixed(1)}%"></i></span><span class="mixp">${Math.round(100 * x)}%</span>${wchip(s[i], i === SEL.pos ? 'v' : '')}</span>`).join('');
  const even = 1 / w.length, spread = Math.max(...w.map(x => Math.abs(x - even)));
  const hiI = w.indexOf(Math.max(...w)), loI = w.indexOf(Math.min(...w));
  $('mix-wnote').innerHTML = w.length === 1 ? 'The first position has only itself to draw on: 100%.'
    : spread < 0.03 ? `Nearly even: ${w.length} tokens at about ${Math.round(100 * even)}% each. The layer hasn’t learned yet which earlier tokens matter.`
    : `The weights add up to 100%. Most from ${wchip(s[hiI], hiI === SEL.pos ? 'v' : '')} (${Math.round(100 * w[hiI])}%), least from ${wchip(s[loI], loI === SEL.pos ? 'v' : '')} (${Math.round(100 * w[loI])}%).`;
  EQM.refresh();
}

// ---------- 6 score every token ----------
const H_SYM = '<i class="ch-h">h</i>';
// h belongs to one position: name it by that position's token as well as its number
// on two lines: "h at", then the token as a chip with its position
// in grey: h belongs to a position but is neither an input embedding nor a candidate
const hChip = () => `<span class="hlab"><span>h at</span><span>${wchip(inTok())} <span class="hpos">(${SEL.pos})</span></span></span>`;
// The candidates for the next token, as a vertical stack at the next position, in the token table's
// own order (by token ID: the toy stores its rows most common first, so they're listed by their GPT-2
// IDs instead). Scrolling the stack picks which candidate is scored against h below.
const S6 = { cand: null, sbMode: 'track' };
const ORDER6 = V.ids.map((_, i) => i).sort((a, b) => V.ids[a] - V.ids[b]);
const ROWH = 32, PAD6 = 2;   // 7's stack shows 2 × 2 + 1 candidates, the picked one in line with the sentence
const rowH = vp => +vp.dataset.rh || ROWH;   // the header's squashed copy has shorter rows (set in CSS and here)
function stack6() {
  if (S6.cand == null) S6.cand = ORDER6[0];
  return `<span class="m6x" data-i="${SEL.pos + 1}"><span class="m6vp" tabindex="0" aria-label="candidates for the next token; scroll to pick one">${ORDER6.map(i =>
    `<span class="m6c${i === S6.cand ? ' sel' : ''}" data-t="${i}">${wchip(i, i === S6.cand ? 'vo' : '')}</span>`).join('')}</span><span class="m6slot"></span></span>`;
}
const EQ6 = DotEq($('m6-eq'), () => {
  const u = upd(), t = S6.cand;
  return { a: { label: 'after the layers', sym: H_SYM, chip: hChip(), vals: arr(u.f.h2[SEL.pos]), cls: 'ch' },
    b: { label: 'candidate embedding', sym: '<i class="cvo">E</i>', chip: wchip(t, 'vo'), vals: arr(rowOf(u.m, 'E', t)), cls: 'cvo' } };
}, 3, 4);
function render6() {
  const u = upd(), p = u.f.p[SEL.pos], tg = tgTok();
  TRACK6.render(); syncPass();
  bindStack($('m6-track').querySelector('.m6vp'));
  render6eq();
}
function render6eq() {
  const u = upd(), c = S6.cand, p = u.f.p[SEL.pos][c];
  $('m6-eqh').innerHTML = `One of the ${V.n} scores: h · the embedding of ${wchip(c, 'vo')}`;
  EQ6.refresh();
  $('m6-p').innerHTML = `After the softmax over all ${V.n} scores, ${wchip(c, 'vo')} gets a probability of <b>${p < 0.001 ? (100 * p).toFixed(3) : (100 * p).toFixed(1)}%</b>.`;
  // the error for the same candidate: label 1 if it really came next, 0 if not
  const tg = tgTok(), lab = c === tg ? 1 : 0, pc = lab ? 'tg' : 'neg';
  const term = (cls, l, v, f) => `<div class="term ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="f">${f}</div></div>`;
  $('m6-err').innerHTML = `<div class="m5eq">${term(lab ? 'lab1' : 'lab0', 'label', lab, wchip(c, pc) + (lab ? ' came next' : ' didn’t come next'))}<div class="op">−</div>
    ${term('', 'probability p', p.toFixed(3), `of ${wchip(c, pc)}`)}<div class="op">=</div>${term('err', 'error', num(lab - p, 3), '&nbsp;')}</div>
    <p class="note" style="margin:-4px 0 0">${lab ? `Only ${wchip(c, pc)} came next, so it is the only candidate with label 1. Every other candidate has label 0, and an error of −p.`
      : `${wchip(tg, 'tg')} came next, so it has label 1. ${wchip(c, pc)} and every other candidate have label 0, so their error is −p.`}</p>`;
}
// The stacks (7's, and the header's once you're past 7) share one candidate, S6.cand. Scrolling
// either picks it; it is scored in 7 and, unless it's the right answer, is 8's wrong answer.
const stacks = () => [...document.querySelectorAll('.m6vp')];
function bindStack(vp) {
  if (!vp) return;
  if (vp.closest('#statebar')) vp.dataset.rh = 22;
  S6.quiet = true; vp.scrollTop = ORDER6.indexOf(S6.cand) * rowH(vp);
  requestAnimationFrame(() => { S6.quiet = false; scroll6({ target: vp }); });   // in case it moved meanwhile
  vp.addEventListener('scroll', scroll6, { passive: true });
  vp.addEventListener('keydown', e => { if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return; e.preventDefault(); pick6(vp, ORDER6.indexOf(S6.cand) + (e.key === 'ArrowDown' ? 1 : -1)); });
}
function scroll6(e) {
  if (S6.quiet || S6.raf) return;
  S6.raf = requestAnimationFrame(() => {
    S6.raf = 0; const vp = e.target, c = ORDER6[Math.max(0, Math.min(ORDER6.length - 1, Math.round(vp.scrollTop / rowH(vp))))];
    if (c !== S6.cand) setCand(c, vp);
  });
}
function setCand(c, from = null) {
  S6.cand = c;
  stacks().forEach(vp => {
    vp.querySelectorAll('.m6c').forEach(r => { const on = +r.dataset.t === c; r.classList.toggle('sel', on); r.querySelector('.chip').classList.toggle('vo', on); });
    if (vp !== from) { S6.quiet = true; vp.scrollTop = ORDER6.indexOf(c) * rowH(vp); requestAnimationFrame(() => { S6.quiet = false; }); }
  });
  render6eq(); render7();
}
const pick6 = (vp, k) => vp.scrollTo({ top: Math.max(0, Math.min(ORDER6.length - 1, k)) * rowH(vp), behavior: 'smooth' });
['m6-track', 'sb-track'].forEach(id => $(id).addEventListener('click', e => { const r = e.target.closest('.m6c'); if (r) pick6(r.closest('.m6vp'), ORDER6.indexOf(+r.dataset.t)); }));

// ---------- 7 one prediction's updates ----------
// the parts of one prediction: its right answer's row pulled toward h, a wrong answer's pushed
// away, and the input token's row moved by the gradient that reaches the bottom
const S7 = { wrong: null };
function g7(i) { const u = upd(), p = u.f.p[SEL.pos][i], lab = i === tgTok() ? 1 : 0; return { p, lab, g: (lab - p) * u.alpha }; }
const gchain = (i, cls) => {
  const u = upd(), { p, lab, g } = g7(i), sc = (c, l, v, f) => `<div class="term ${c}"><div class="l">${l}</div><div class="v">${v}</div><div class="f">${f}</div></div>`;
  return sc('err', 'error', num(lab - p, 4), `${lab} − ${p.toFixed(4)}, for ${wchip(i, cls)}`) + '<div class="op">×</div>'
    + sc('', 'learning rate α', u.alpha.toFixed(4), 'at this point in training') + '<div class="op">=</div>'
    + sc('gval', 'scaling factor <span class="cg">g</span>', num(g, 5), g > 0 ? 'positive: toward h' : 'negative: away from h');
};
// Section 8 works out parts only: nothing is applied until section 9 adds every part up.
// A candidate's part is g × h; the input token's part is α × the direction sent back to its input vector.
const upEq = (i, cls) => () => {
  const u = upd(), h = arr(u.f.h2[SEL.pos]), { g } = g7(i);
  return { left: [{ t: 'scal', label: '<span class="cg">g</span>', val: num(g, 4), cls: 'cg' }, { t: 'op', s: '×' }, { t: 'vec', label: 'after the layers', sym: H_SYM, chip: hChip(), vals: h, cls: 'ch' }],
    result: { label: 'Δ from being scored', chip: `<span class="chip ${cls}">Δ ${tstr(i)}</span>`, vals: h.map(x => g * x), cls: 'cvo' } };
};
const EQ7U = VecEq($('m7-up'), () => upEq(tgTok(), 'tg')(), 4), EQ7D = VecEq($('m7-down'), () => upEq(S7.wrong, 'neg')(), 4);
const EQ7I = VecEq($('m7-in'), () => {
  const u = upd(), t = inTok(), sig = arr(u.one[SEL.pos].dh0[SEL.pos]).map(x => -x);
  return { left: [{ t: 'scal', label: 'α', val: u.alpha.toFixed(4) }, { t: 'op', s: '×' }, { t: 'vec', label: 'error-reducing nudge', sym: '<i class="ch-h">∇</i>', chip: `<span class="hlab"><span>for the input at</span><span>${wchip(t)} <span class="hpos">(${SEL.pos})</span></span></span>`, vals: sig, cls: 'ch' }],
    result: { label: 'Δ from being the input', chip: `<span class="chip v">Δ ${tstr(t)}</span>`, vals: sig.map(x => u.alpha * x), cls: 'cv' } };
}, 4);
function render7() {
  const u = upd(), p = u.f.p[SEL.pos], tg = tgTok(), wrongs = V.ids.map((_, i) => i).filter(i => i !== tg).sort((a, b) => p[b] - p[a]);
  // the wrong answer is the candidate picked in 7's stack (or the header's), unless that is the right answer
  if (S6.cand !== tg) S7.wrong = S6.cand; else if (S7.wrong == null || S7.wrong === tg) S7.wrong = wrongs[0];
  $('m7-g').innerHTML = gchain(tg, 'tg'); $('m7-gw').innerHTML = gchain(S7.wrong, 'neg');
  EQ7U.refresh(); EQ7D.refresh(); EQ7I.refresh();
  $('m7-wrong').innerHTML = wrongs.map(i => `<option value="${i}">${esc(optLabel(i))} (${(100 * p[i]).toFixed(1)}%)</option>`).join(''); $('m7-wrong').value = S7.wrong;
}
$('m7-wrong').addEventListener('change', e => setCand(+e.target.value));

// ---------- 8 adding up one sentence's update for one row ----------
const S8 = { row: null };
// The chosen row's parts, one line per prediction (each is what section 8 works out, for this row),
// then the input parts, then their sum: the row's whole change in this step. The row gets parts in
// both roles, so here it is shown in neither color.
const SROW = '<i class="crow">E</i>';
const EQ8A = VecEq($('m8-apply'), () => {
  const u = upd(), r = arr(rowOf(u.m, 'E', S8.row)), dl = delta8();
  return { left: [{ t: 'vec', label: 'the embedding', sym: SROW, chip: wchip(S8.row), vals: r, cls: 'cp' }, { t: 'op', s: '+' }, { t: 'vec', label: 'total change', sym: `Δ${SROW}`, chip: `<span class="chip">total Δ ${tstr(S8.row)}</span>`, vals: dl, cls: 'cp' }],
    result: { label: 'new embedding', chip: wchip(S8.row), vals: r.map((x, d) => x + dl[d]), cls: 'cp' } };
}, 4);
function parts8() {
  const u = upd(), f = u.f, r = S8.row, out = [];
  for (let t = 0; t < f.n; t++) {
    const lab = f.tg[t] === r ? 1 : 0, g = (lab - f.p[t][r]) * u.alpha;
    out.push({ t, kind: lab ? 'came' : 'not', what: `${wchip(f.inp[t])} → ?`, p: f.p[t][r], vals: arr(f.h2[t]).map(x => g * x) });
  }
  // as the input at t: what backprop sends back to the input vector there, from every prediction that can
  // see position t (mostly the one at t itself; later ones add a little through attention)
  for (let t = 0; t < f.n; t++) if (f.inp[t] === r) out.push({ t, kind: 'input', what: `${wchip(r)} is the input at ${t}`, vals: arr(u.full.dh0[t]).map(x => -u.alpha * x) });
  return out;
}
const ROLE8 = { came: ['tg', 'came next'], not: ['neg', 'didn’t come next'], input: ['v', 'the input'] };
function renderSum8() {
  const u = upd(), f = u.f, parts = parts8(), tot = delta8(), r = S8.row, top = Math.max(1e-12, ...parts.flatMap(p => p.vals.map(Math.abs)), ...tot.map(Math.abs));
  const cells = v => v.map(x => `<span class="rc" style="background:${heatColor(x, top * 1.4)}">${num(x, 4)}</span>`).join('');
  const pct = p => p < 0.001 ? (100 * p).toFixed(3) + '%' : (100 * p).toFixed(1) + '%';
  // what you're looking at: the sentence's predictions, and what each does to this one embedding
  const came = f.tg.map((x, t) => x === r ? t : -1).filter(t => t >= 0), inp = f.inp.map((x, t) => x === r ? t : -1).filter(t => t >= 0), n = f.n, rc = wchip(r);
  const at = ts => ts.length === 1 ? `position ${ts[0]}` : `positions ${ts.join(' and ')}`;
  const intro = !came.length && !inp.length
    ? `${rc} isn’t in this sentence. But each of its ${n} predictions still scores every token, ${rc} included, so each pushes ${rc}’s embedding away a little, by how likely it thought ${rc} was.`
    : `${rc} ${came.length ? `comes next at ${at(came)}` : 'never comes next'}${inp.length ? ` and is the input at ${at(inp)}` : ''}. The other predictions still score it, and push its embedding away a little, by how likely they thought it was.`;
  $('m8-sum').innerHTML = `<p class="m8intro">${intro}</p><div class="m8t"><div class="m8r m8h"><span>prediction</span><span class="pc" title="the probability the model gave ${esc(plain(r))} there">p</span><span>came next?</span><span>its Δ, one number per dimension</span></div>
    ${parts.map(p => { const [c, l] = ROLE8[p.kind];
      const ans = p.kind === 'input' ? '<span class="badge inb">input</span>' : p.kind === 'came' ? '<span class="badge pos">yes</span>' : '<span class="badge neg">no</span>';
      return `<div class="m8r${p.kind === 'input' ? ' inp' : ''}"><span class="w"><span class="pn">${p.t}</span>${p.what}</span><span class="pc">${p.kind === 'input' ? '' : pct(p.p)}</span><span class="m8ans">${ans}</span><span class="cells">${cells(p.vals)}</span></div>`; }).join('')}
    <div class="m8r m8tot"><span class="w">total Δ ${rc}: all ${parts.length} added up</span><span></span><span></span><span class="cells">${cells(tot)}</span></div></div>
    <p class="note" style="margin:8px 0 0">At each position the model gives every token a probability; this table follows one of them, ${rc}. Each line is the same calculation as section 8: (label − p) × α × h at that position, where the label is 1 if ${rc} came next and 0 if not. An input line is α × the error-reducing nudge for the input vector at that position: mostly from the prediction made there, plus a little from later predictions, whose attention can look back at it.</p>`;
}
function delta8() { const u = upd(), D0 = LAYOUT.E[0] + S8.row * D; return Array.from({ length: D }, (_, d) => -u.alpha * u.full.g[D0 + d]); }
function render8() {
  const u = upd(); if (S8.row == null) S8.row = tgTok(); syncPass();
  $('m8-row').innerHTML = V.ids.map((_, i) => `<option value="${i}">${esc(optLabel(i))}</option>`).join(''); $('m8-row').value = S8.row;
  renderSum8(); EQ8A.refresh();
  // every row's step, largest first: colored by its role in this sentence
  const s = new Set(seqSel()), tgs = new Set(u.f.tg), mag = V.ids.map((_, i) => Math.hypot(...Array.from({ length: D }, (_, d) => u.full.g[LAYOUT.E[0] + i * D + d])) * u.alpha);
  const ord = mag.map((x, i) => i).sort((a, b) => mag[b] - mag[a]), mx = mag[ord[0]] || 1, W0 = 700, H0 = 170, l = 8, bt = 70, top = 8, bw = (W0 - l - 4) / V.n, Y = v => top + (1 - v / mx) * (H0 - bt - top);
  let svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block">`;
  ord.forEach((i, r) => {
    const x = l + r * bw, c = u.f.inp.includes(i) ? 'var(--violet)' : tgs.has(i) ? 'var(--blue)' : '#f6c3ad', sel = i === S8.row;
    svg += `<g data-r="${i}" style="cursor:pointer"><title>${esc(plain(i))}: moves ${mag[i].toExponential(2)}</title><rect x="${x + 1.5}" y="${top}" width="${bw - 3}" height="${H0 - bt - top}" fill="transparent"/><rect x="${x + 1.5}" y="${Y(mag[i])}" width="${bw - 3}" height="${Y(0) - Y(mag[i])}" fill="${c}" ${sel ? 'stroke="var(--ink)" stroke-width="1.5"' : ''}/>
      <text transform="translate(${x + bw / 2 + 3},${Y(0) + 6}) rotate(60)" font-size="10" font-family="var(--mono)" fill="${sel ? 'var(--ink)' : 'var(--ink-2)'}" font-weight="${sel ? 700 : 400}">${esc(optLabel(i).replace('<|endoftext|>', '<eot>'))}</text></g>`;
  });
  $('m8-all').innerHTML = svg + `<line x1="${l}" x2="${W0}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/></svg>
    <div class="legend"><span><i style="background:var(--violet)"></i>an input in this sentence (and maybe an answer too)</span><span><i style="background:var(--blue)"></i>only an answer</span><span><i style="background:#f6c3ad"></i>not in the sentence: only pushed away</span></div>`;
}
$('m8-row').addEventListener('change', e => { S8.row = +e.target.value; render8(); renderStatebar(); });
$('sb-sentview').addEventListener('click', e => { const c = e.target.closest('[data-t]'); if (c) { S8.row = +c.dataset.t; render8(); renderStatebar(); } });
$('m8-all').addEventListener('click', e => { const g = e.target.closest('[data-r]'); if (g) { S8.row = +g.dataset.r; render8(); renderStatebar(); } });

// ---------- 9 two tokens drawing together ----------
const S9 = { a: tokIx(' coffee'), b: tokIx(' tea'), key: '', series: null };
const angle = (x, y) => Math.acos(Math.max(-1, Math.min(1, cosine(x, y)))) * 180 / Math.PI;
const logx = step => Math.log10(1 + step) / Math.log10(1 + G.stream.total);
const LOG_TICKS = () => [[10, '10'], [100, '100'], [1000, '1,000'], [G.stream.total, 'end']];
function angleSeries(a, b) {   // [step, angle] at log-spaced steps
  const key = [G.seed, CFG.ALPHA0, a, b].join(':'); if (S9.key === key) return S9.series;
  const N = G.stream.total, at = [...new Set([0, ...Array.from({ length: 120 }, (_, k) => Math.round(Math.pow(N, (k + 1) / 120)))])];
  S9.series = at.map(s => [s, angle(arr(rowOf(G.snaps[s], 'E', a)), arr(rowOf(G.snaps[s], 'E', b)))]); S9.key = key; return S9.series;
}
function render9() {
  const a = S9.a, b = S9.b === a ? (a === tokIx(' coffee') ? tokIx(' tea') : tokIx(' coffee')) : S9.b, N = G.stream.total, at = G.step, m = G.snaps[at];
  $('m9-plane').innerHTML = pairPlane(arr(rowOf(m, 'E', a)), arr(rowOf(m, 'E', b)), { aLabel: plain(a), bLabel: plain(b), bCls: 'a2' });
  const ser = angleSeries(a, b), cur = angle(arr(rowOf(m, 'E', a)), arr(rowOf(m, 'E', b)));
  const cw = 340, ch = 190, cl = 40, cr = 10, ct = 12, cb = 30, X = v => cl + logx(v) * (cw - cl - cr), Y = v => ct + (1 - v / 180) * (ch - ct - cb);
  const grid = [0, 45, 90, 135, 180].map(v => `<line x1="${cl}" x2="${cw - cr}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${cl - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${v}°</text>`).join('');
  const ticks = LOG_TICKS().map(([x, l]) => `<text x="${X(x)}" y="${ch - 14}" text-anchor="${x === N ? 'end' : 'middle'}" font-size="9">${l}</text>`).join('');
  $('m9-chart').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${grid}${ticks}
    <path d="${ser.map(([x, v], i) => (i ? 'L' : 'M') + X(x).toFixed(1) + ',' + Y(v).toFixed(1)).join('')}" fill="none" stroke="var(--violet)" stroke-width="2"/>
    <line x1="${X(at)}" x2="${X(at)}" y1="${ct}" y2="${ch - cb}" stroke="var(--ink-2)" stroke-dasharray="3 3"/>
    <circle cx="${X(at)}" cy="${Y(cur)}" r="5" fill="var(--violet)" stroke="white" stroke-width="2"/>
    <text x="${X(at) + (at > N * 0.8 ? -8 : 8)}" y="${Y(cur) - 10}" text-anchor="${at > N * 0.8 ? 'end' : 'start'}" font-size="11" font-weight="600" fill="var(--violet)">${Math.round(cur)}°</text>
    <text x="${cl + (cw - cl - cr) / 2}" y="${ch - 2}" text-anchor="middle" font-size="10" fill="var(--ink-2)">training steps (log scale)</text></svg>`;
  ['m9-a', 'm9-b'].forEach(id => { if (!$(id).options.length) V.ids.forEach((_, i) => $(id).add(new Option(optLabel(i), i))); });
  $('m9-a').value = a; $('m9-b').value = b; syncStep();
}
$('m9-a').addEventListener('change', e => { S9.a = +e.target.value; render9(); });
$('m9-b').addEventListener('change', e => { S9.b = +e.target.value; render9(); });

// ---------- 10 training at scale ----------
// the loss in equal-width bins along training, as [x 0..1, mean loss, last step in bin]
const NB = 120;
function lossBins() {
  const N = G.stream.total, w = Math.ceil(N / NB), out = [];
  for (let b = 0; b * w < N; b++) { let s = 0, n = 0; for (let i = b * w; i < Math.min(N, (b + 1) * w); i++) if (isFinite(G.loss[i])) { s += G.loss[i]; n++; } if (n) out.push([(b + 0.5) * w / N, s / n, Math.min(N, (b + 1) * w)]); }
  return out;
}
function snapshotRun() { return G.loss ? { bins: lossBins(), alpha: CFG.ALPHA0, seed: G.seed } : null; }
function render10l() {
  $('m10-seed').value = G.seed;
  $('m10-lr').querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.v === CFG.ALPHA0));
  const cw = 340, ch = 160, cl = 46, cb = 34, ct = 10, N = G.stream.total, X = f => cl + f * (cw - cl - 6);
  const L = lossBins(), lmax = 4.5, LY = v => ct + (1 - Math.min(v, lmax) / lmax) * (ch - ct - cb);
  const lp = A => A.map(([x, v], i) => (i ? 'L' : 'M') + X(x).toFixed(1) + ',' + LY(v).toFixed(1)).join('');
  const axis = (fy, ticks, f) => ticks.map(v => `<line x1="${cl}" x2="${cw - 6}" y1="${fy(v)}" y2="${fy(v)}" stroke="var(--grid)"/><text x="${cl - 5}" y="${fy(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${f(v)}</text>`).join('');
  const xticks = [[0, '0'], [N / 2, 'middle'], [N, 'end']].map(([v, l]) => `<line x1="${X(v / N)}" x2="${X(v / N)}" y1="${ch - cb}" y2="${ch - cb + 4}" stroke="var(--axis)"/><text x="${X(v / N)}" y="${ch - cb + 14}" text-anchor="${v === N ? 'end' : v ? 'middle' : 'start'}" font-size="9">${l}</text>`).join('')
    + `<text x="${cl + (cw - cl - 6) / 2}" y="${ch - 3}" text-anchor="middle" font-size="10">training steps</text>`;
  const now = X(G.step / N), prev = G.prev;
  const ghost = prev ? `<path d="${lp(prev.bins)}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>` : '';
  $('m10-loss').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${axis(LY, [0, 1, 2, 3, 4], v => v)}${xticks}${ghost}
    <line x1="${cl}" x2="${cw - 6}" y1="${LY(Math.log(V.n))}" y2="${LY(Math.log(V.n))}" stroke="var(--axis)" stroke-dasharray="2 3"/><text x="${cw - 8}" y="${LY(Math.log(V.n)) - 5}" text-anchor="end" font-size="9" fill="var(--muted)">uniform guess, log 41</text>
    ${G.step ? `<path d="${lp(L.filter(([, , e]) => e <= G.step))}" fill="none" stroke="var(--blue)" stroke-width="2"/>` : ''}
    <line x1="${now}" x2="${now}" y1="${ct}" y2="${ch - cb}" stroke="var(--ink-2)" stroke-dasharray="3 3"/></svg>`;
  const AMAX = 0.05, AY = v => ct + (1 - v / AMAX) * (ch - ct - cb), aline = a0 => `M${X(0)},${AY(a0)}L${X(1)},${AY(0)}`;
  $('m10-alpha').innerHTML = `<svg viewBox="0 0 ${cw} ${ch}" width="100%" style="display:block">${axis(AY, [0, 0.01, 0.02, 0.03, 0.04, 0.05], v => v ? String(v) : '0')}${xticks}
    ${prev && prev.alpha !== CFG.ALPHA0 ? `<path d="${aline(prev.alpha)}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>` : ''}
    <path d="${aline(CFG.ALPHA0)}" fill="none" stroke="var(--blue)" stroke-width="2"/>
    <circle cx="${now}" cy="${AY(alphaAt(G.step, N))}" r="4" fill="var(--blue)" stroke="white" stroke-width="2"/></svg>`;
  const bad = !G.loss.every(isFinite);
  $('m10-ghost').innerHTML = (prev ? `<span class="ghostkey"></span> previous run: seed ${prev.seed}, α ${prev.alpha}. ` : '') + (bad ? '<b>This run blew up:</b> the steps were so large that the numbers overflowed.' : '');
}
const S10 = { sel: tokIx(' coffee'), showPos: false, zoom: 1, pan: [0, 0], speed: 12 };   // speed: steps per frame (a full run plays in about 12 s)
function render10() {
  const m = G.snaps[G.step], E = rows(m), Pr = rows(m, 'P'), sel = S10.sel;
  if (!E.flat().every(isFinite)) { $('m10-scatter').innerHTML = '<p class="note" style="padding:20px">The numbers overflowed at this learning rate, so there is nothing to draw.</p>'; $('m10-nn').innerHTML = ''; return; }
  const pc = pca2(E, G.pca && G.pca.comps); G.pca = pc;
  const ptsIn = E.map(pc.project), ptsPos = S10.showPos ? Pr.map(pc.project) : [];
  const want = 1.12 * Math.max(1e-6, ...ptsIn.concat(ptsPos).map(([x, y]) => Math.max(Math.abs(x), Math.abs(y))));
  G.rad = !G.rad || want > G.rad ? want : G.rad * 0.92 + want * 0.08;   // grow at once, shrink gently
  const box = $('m10-scatter').clientWidth || 700, W0 = 700, H0 = box < 420 ? 700 : box < 560 ? 560 : 440;
  const cx = W0 / 2, cy = H0 / 2, k = Math.min(W0, H0) / 2 / G.rad * S10.zoom, P = ([x, y]) => [cx + (x - S10.pan[0]) * k, cy - (y - S10.pan[1]) * k];
  S10.k = k; S10.W0 = W0; S10.selPt = ptsIn[sel];
  const nb = neighbors(E, sel, 5), nbs = new Set(nb.map(x => x.i)), [ox, oy] = P([0, 0]), z = (W0 / box) * S10.zoom;
  let svg = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block;--z:${z}" id="scatter">
    <line x1="0" x2="${W0}" y1="${oy}" y2="${oy}" stroke="var(--grid)"/><line x1="${ox}" x2="${ox}" y1="0" y2="${H0}" stroke="var(--grid)"/>`;
  nb.forEach(({ i }) => { const [x1, y1] = P(ptsIn[sel]), [x2, y2] = P(ptsIn[i]); svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--violet)" stroke-width="${z}" opacity="0.35"/>`; });
  if (ptsPos.length) svg += `<path d="${ptsPos.map((q, i) => (i ? 'L' : 'M') + P(q).join(',')).join('')}" fill="none" stroke="var(--gold)" stroke-width="${z}" opacity="0.5"/>`;
  ptsPos.forEach((q, i) => { const [x, y] = P(q); svg += `<g class="pt pos"><circle cx="${x}" cy="${y}" r="${3.5 * z}"/><text x="${x + 5 * z}" y="${y + 3 * z}">pos ${i}</text></g>`; });
  ptsIn.forEach((q, i) => { const [x, y] = P(q); svg += `<g class="pt${i === sel ? ' sel' : nbs.has(i) ? ' nb' : ''}" data-t="${i}"><circle cx="${x}" cy="${y}" r="${(i === sel ? 6 : 4) * z}"/><text x="${x + 7 * z}" y="${y + 4 * z}">${esc(optLabel(i).replace('<|endoftext|>', '<eot>'))}</text></g>`; });
  $('m10-scatter').innerHTML = svg + '</svg>';
  $('m10-scatter').classList.toggle('zoomed', S10.zoom > 1);
  $('m10-nn').innerHTML = `<table class="data"><tbody>${nb.map(({ i, c }) => `<tr data-t="${i}"><td class="mono">${tstr(i)}</td><td class="n">${c.toFixed(2)}</td><td class="bar"><i style="width:${Math.max(0, c) * 100}%"></i></td></tr>`).join('')}</tbody></table>`;
  const s10 = $('m10-word'); if (!s10.options.length) V.ids.forEach((_, i) => s10.add(new Option(optLabel(i), i)));
  s10.value = sel;
  const sc = $('m10-scrub'); sc.max = G.stream.total; sc.value = G.step;
  $('m10-at').textContent = `${commas(G.step)} of ${commas(G.stream.total)} steps · pass ${Math.min(CFG.EPOCHS, G.stream.epoch(Math.max(0, G.step - 1)) + 1)} of ${CFG.EPOCHS}`;
}
function renderTimeline() { render10l(); render10(); renderStep(); refreshBars(); }
const bar10 = PlayBar($('m10-bar'), {
  ms: 40, canStep: () => G.step < G.stream.total, canReset: () => G.step > 0, rewind: () => { G.step = 0; },
  step: () => { G.step = Math.min(G.stream.total, G.step + S10.speed); renderTimeline(); }, end: () => { G.step = G.stream.total; renderTimeline(); },
  reset: () => { G.step = 0; renderTimeline(); },
});
$('m10-train').addEventListener('click', () => { bar10.stop(); G.step = 0; renderTimeline(); bar10.act('play'); });
$('m10-seed').addEventListener('change', e => { const prev = snapshotRun(); G.seed = Math.max(1, Math.min(9999, Math.round(+e.target.value) || 1)); retrain(prev); });
$('m10-seedrand').addEventListener('click', () => { const prev = snapshotRun(); G.seed = 1 + Math.floor(Math.random() * 9999); retrain(prev); });
$('m10-lr').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { const prev = snapshotRun(); CFG.ALPHA0 = +b.dataset.v; retrain(prev); } });
$('m10-scrub').addEventListener('input', e => { stopAll(); G.step = +e.target.value; renderTimeline(); });
$('m10-pos').addEventListener('change', e => { S10.showPos = e.target.checked; G.rad = null; render10(); });
$('m10-word').addEventListener('change', e => { S10.sel = +e.target.value; render10(); });
$('m10-scatter').addEventListener('click', e => { if (S10.dragged) return; const g = e.target.closest('[data-t]'); if (g) { S10.sel = +g.dataset.t; render10(); } });
$('m10-nn').addEventListener('click', e => { const g = e.target.closest('[data-t]'); if (g) { S10.sel = +g.dataset.t; render10(); } });
document.querySelector('#m10 .m8zoom').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.z === 'in' && S10.zoom === 1 && S10.selPt) S10.pan = S10.selPt.slice();   // the first zoom-in centres on the selected token
  if (b.dataset.z === 'fit') { S10.zoom = 1; S10.pan = [0, 0]; } else S10.zoom = Math.max(1, Math.min(12, S10.zoom * (b.dataset.z === 'in' ? 1.6 : 1 / 1.6)));
  if (S10.zoom === 1) S10.pan = [0, 0];
  render10();
});
(function pan10() {
  const el = $('m10-scatter'); let d = null;
  el.addEventListener('pointerdown', e => { if (e.button || S10.zoom === 1) return; d = { x: e.clientX, y: e.clientY, pan: S10.pan.slice() }; S10.dragged = false; });
  window.addEventListener('pointermove', e => {
    if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y; if (!S10.dragged && Math.hypot(dx, dy) < 5) return;
    S10.dragged = true; el.setPointerCapture?.(e.pointerId);
    const px = S10.W0 / el.clientWidth / S10.k;
    S10.pan = [d.pan[0] - dx * px, d.pan[1] + dy * px]; render10();
  });
  window.addEventListener('pointerup', () => { d = null; setTimeout(() => { S10.dragged = false; }, 0); });
})();

// ---------- 11 nearest neighbours in GPT-2's real table ----------
const NN = (() => { const ids = b64(GPT2.nn.ids, Uint16Array), cos = b64(GPT2.nn.cos, Uint8Array), at = new Map(GPT2.nn.q.map((id, r) => [id, r])); return { ids, cos, at, k: GPT2.nn.k }; })();
const M11_EX = ['coffee', 'dog', 'king', 'Paris', 'Monday', 'happy', 'blue'];
function render11() {
  const w = $('m11-word').value.trim(), out = $('m11-out');
  document.querySelectorAll('#m11-ex button').forEach(b => b.classList.toggle('on', b.dataset.w === w));
  if (!w) { out.innerHTML = '<p class="note">Type a word.</p>'; return; }
  const cap = w[0].toUpperCase() + w.slice(1);
  const variants = [...new Set([' ' + w, w, ' ' + cap, ' ' + w.toUpperCase()])];
  const vrow = variants.map(v => { const ids = BPE.encode(v); return `<span class="m11v">${ids.map(id => `<span class="chip">${gstr(id)}</span>`).join('')}<span class="note">${ids.length === 1 ? 'ID ' + ids[0] : ids.length + ' tokens'}</span></span>`; }).join('');
  const ids = BPE.encode(' ' + w), r = ids.length === 1 ? NN.at.get(ids[0]) : undefined;
  let nn;
  if (ids.length > 1) nn = `<p class="note">“${vis(' ' + w)}” is ${ids.length} tokens in GPT-2, so it has no embedding of its own.</p>`;
  else if (r === undefined) nn = `<p class="note">“${vis(' ' + w)}” is token ${ids[0]}, but its neighbours aren’t included on this page. Try a more common word.</p>`;
  else nn = `<table class="data"><tbody>${Array.from({ length: NN.k }, (_, j) => { const id = NN.ids[r * NN.k + j], c = NN.cos[r * NN.k + j] / 255; return `<tr><td class="n rk">${j + 1}</td><td class="mono">${gstr(id)}</td><td class="n mono">${id}</td><td class="n">${c.toFixed(2)}</td><td class="bar"><i style="width:${c * 100}%"></i></td></tr>`; }).join('')}</tbody></table>`;
  out.innerHTML = `<h4>The word as GPT-2 tokens</h4><div class="m11vs">${vrow}</div><h4 style="margin-top:14px">The 10 embeddings nearest to ${ids.length === 1 ? `“${gstr(ids[0])}”` : 'it'}, out of 50,257</h4>${nn}`;
}
$('m11-ex').innerHTML = M11_EX.map(w => `<button data-w="${w}">${w}</button>`).join('');
$('m11-ex').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { $('m11-word').value = b.dataset.w; render11(); } });
$('m11-word').addEventListener('input', render11);

// ---------- 11 every row's distance from the average row ----------
const DMU = (() => { const q = b64(GPT2.dmu.q, Uint8Array), { lo, hi } = GPT2.dmu; return Float32Array.from(q, x => lo + x / 255 * (hi - lo)); })();
// three rows from the 2023 list of “glitch tokens”, which sit nearest the average row, and two ordinary ones
const MARK11 = [[30212, 'red'], [36173, 'red'], [42089, 'red'], [262, 'violet'], [6891, 'violet']];
function render11g() {
  const host = $('m11g-plot'), W0 = host.clientWidth || 700, H0 = W0 < 500 ? 240 : 280, l = 44, r = 10, t = 12, b = 34, dpr = devicePixelRatio || 1;
  const lo = 0, hi = Math.ceil(GPT2.dmu.hi), X = id => l + id / 50256 * (W0 - l - r), Y = v => t + (1 - (v - lo) / (hi - lo)) * (H0 - t - b);
  host.innerHTML = `<div style="position:relative"><canvas width="${W0 * dpr}" height="${H0 * dpr}" style="width:${W0}px;height:${H0}px;display:block"></canvas><svg viewBox="0 0 ${W0} ${H0}" width="${W0}" height="${H0}" style="position:absolute;inset:0;pointer-events:none" id="m11g-svg"></svg></div>`;
  const cv = host.querySelector('canvas'), cx = cv.getContext('2d'); cx.scale(dpr, dpr);
  cx.fillStyle = 'rgba(82,81,78,0.28)';
  for (let id = 0; id < DMU.length; id++) cx.fillRect(X(id) - 0.6, Y(DMU[id]) - 0.6, 1.3, 1.3);
  let svg = [0, 2, 4, 6].filter(v => v <= hi).map(v => `<line x1="${l}" x2="${W0 - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" opacity="0.7"/><text x="${l - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${v}</text>`).join('')
    + [0, 10000, 20000, 30000, 40000, 50000].map(v => `<text x="${X(v)}" y="${H0 - 18}" text-anchor="middle" font-size="10">${commas(v)}</text>`).join('')
    + `<text x="${l + (W0 - l - r) / 2}" y="${H0 - 3}" text-anchor="middle" font-size="10">token ID (lower IDs were merged earlier, from more common pairs)</text>`;
  // labels alternate above and below their dots so neighbours don't collide
  const mark = (id, c, k) => `<circle cx="${X(id)}" cy="${Y(DMU[id])}" r="4" fill="none" stroke="var(--${c})" stroke-width="1.6"/><text x="${X(id)}" y="${Y(DMU[id]) + (k % 2 ? 17 : -9)}" text-anchor="middle" font-size="10.5" font-family="var(--mono)" fill="var(--${c})">${esc(BPE.str(id).replace(/^ /, '·'))}</text>`;
  MARK11.forEach(([id, c], k) => { svg += mark(id, c, k); });
  svg += `<text transform="translate(10,${t + (H0 - t - b) / 2}) rotate(-90)" text-anchor="middle" font-size="10">distance from the average embedding</text>`;
  svg += `<g id="m11g-hov"></g>`;
  $('m11g-svg').innerHTML = svg;
  const read = () => { $('m11g-read').innerHTML = `Each dot is one of the 50,257 token embeddings. Height is its distance from the average embedding. The red ones sit nearest the middle: glitch tokens GPT-2 almost never saw in training. The dots at the far left, just as low, are byte values that never occur in ordinary text. Hover to read any dot.`; $('m11g-hov').innerHTML = ''; };
  read();
  cv.addEventListener('pointermove', e => {
    const rc = cv.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top, span = 3 / (W0 - l - r) * 50256, c = (mx - l) / (W0 - l - r) * 50256;
    let best = -1, bd = Infinity;
    for (let id = Math.max(0, Math.floor(c - span)); id <= Math.min(50256, Math.ceil(c + span)); id++) { const d = Math.hypot(X(id) - mx, Y(DMU[id]) - my); if (d < bd) { bd = d; best = id; } }
    if (best < 0 || bd > 14) { read(); return; }
    $('m11g-hov').innerHTML = `<circle cx="${X(best)}" cy="${Y(DMU[best])}" r="5" fill="none" stroke="var(--ink)" stroke-width="1.5"/>`;
    $('m11g-read').innerHTML = `token <b>${best}</b>, <span class="mono">${gstr(best)}</span>: distance from the average embedding <b>${DMU[best].toFixed(2)}</b>`;
  });
  cv.addEventListener('pointerleave', read);
}

// ---------- 12 one word, two meanings, through the layers ----------
const S12 = { k: 0 };
function render12() {
  const s = GPT2.senses[S12.k], w = s.word;
  $('m12-word').innerHTML = GPT2.senses.map((x, k) => `<button data-k="${k}" class="${k === S12.k ? 'on' : ''}">${x.word}</button>`).join('');
  const phr = (t, cls) => `<span class="m12p ${cls}">${esc(t).replace(new RegExp(`\\b${w}\\b`), `<b>${w}</b>`)}</span>`;
  $('m12-phr').innerHTML = `${phr(s.same[0], 'a')} <span class="note">compared with</span> ${phr(s.same[1], 'a2')} <span class="note">(same meaning) and</span> ${phr(s.diff, 'b')} <span class="note">(another meaning)</span>`;
  const W0 = 700, H0 = 260, l = 46, r = 16, t = 14, b = 40, X = L => l + L / 12 * (W0 - l - r), Y = v => t + (1 - v) * (H0 - t - b);
  const line = (a, c) => `<path d="${a.map((v, L) => (L ? 'L' : 'M') + X(L).toFixed(1) + ',' + Y(v).toFixed(1)).join('')}" fill="none" stroke="${c}" stroke-width="2.2"/>` + a.map((v, L) => `<circle cx="${X(L)}" cy="${Y(v)}" r="3.5" fill="${c}"><title>layer ${L}: ${v.toFixed(2)}</title></circle>`).join('');
  $('m12-chart').innerHTML = `<svg viewBox="0 0 ${W0} ${H0}" width="100%" style="display:block">
    ${[0, 0.25, 0.5, 0.75, 1].map(v => `<line x1="${l}" x2="${W0 - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${l - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle" font-size="10">${v}</text>`).join('')}
    ${Array.from({ length: 13 }, (_, L) => `<text x="${X(L)}" y="${H0 - 24}" text-anchor="middle" font-size="10">${L}</text>`).join('')}
    <text x="${X(0)}" y="${H0 - 10}" text-anchor="middle" font-size="10" fill="var(--violet)" font-weight="600">the table</text>
    <text x="${l + (W0 - l - r) / 2 + 30}" y="${H0 - 6}" text-anchor="middle" font-size="11" font-weight="600" fill="var(--ink-2)">layer</text>
    <text transform="translate(14,${t + (H0 - t - b) / 2}) rotate(-90)" text-anchor="middle" font-size="11" font-weight="600" fill="var(--ink-2)">how alike the two “${w}” vectors are</text>
    ${line(s.cosSame, 'var(--blue)')}${line(s.cosDiff, 'var(--orange)')}
    <text x="${X(12) - 4}" y="${Y(s.cosSame[12]) - 10}" text-anchor="end" font-size="11" font-weight="600" fill="var(--blue)">${s.cosSame[12].toFixed(2)}</text>
    <text x="${X(12) - 4}" y="${Y(s.cosDiff[12]) + 16}" text-anchor="end" font-size="11" font-weight="600" fill="var(--orange)">${s.cosDiff[12].toFixed(2)}</text></svg>`;
  $('m12-legend').innerHTML = `<span><i style="background:var(--blue)"></i>“${esc(s.same[0])}” vs “${esc(s.same[1])}”</span><span><i style="background:var(--orange)"></i>“${esc(s.same[0])}” vs “${esc(s.diff)}”</span><span>cosine similarity after subtracting each layer’s average vector</span>`;
}
$('m12-word').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S12.k = +b.dataset.k; render12(); } });

// ---------- everything ----------
function renderAll() { renderSel(); render10l(); render10(); if (S4G.stage === 'start') render4g(); }
G.stream = makeStream(G.streamSeed);
resetTraining(); follow3v();
render1(); render2(); load2g(); render2g(); render4g(); render11(); render11g(); render12(); renderAll();
})();
