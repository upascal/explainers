(function () {
// gpt2data.js, model.js (from ../gpt2-embeddings), attndata.js and toy.js are concatenated ahead
// of this file by build.py: GPT2, BPE, V, CFG, initModel, …, ATTN, headAt, seqOf
const $ = id => document.getElementById(id);
const D = CFG.D;
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function num(x, n = 2) { const s = Math.abs(x).toFixed(n); return (x < 0 && +s !== 0 ? '−' : '') + s; }
const pct = (x, n = 0) => (100 * x).toFixed(n) + '%';
const arr = a => Array.from(a);

// transport icons, the same as on the other explainers
const I_RESET = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><rect x="1" y="2" width="2" height="12" fill="currentColor"/><polygon points="11,2 11,14 4,8" fill="currentColor"/><polygon points="18,2 18,14 11,8" fill="currentColor"/></svg>';
const I_PLAY = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><polygon points="4,2 4,14 13,8" fill="currentColor"/></svg>';
const I_PAUSE = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" fill="currentColor"/><rect x="9.5" y="2" width="3.5" height="12" fill="currentColor"/></svg>';
const I_STEP = '<svg viewBox="0 0 18 16" width="12" height="10" aria-hidden="true"><polygon points="2,2 2,14 11,8" fill="currentColor"/><rect x="12" y="2" width="2.5" height="12" fill="currentColor"/></svg>';
const I_END = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><polygon points="2,2 2,14 9,8" fill="currentColor"/><polygon points="9,2 9,14 16,8" fill="currentColor"/><rect x="17" y="2" width="2" height="12" fill="currentColor"/></svg>';

// ---------- showing tokens ----------
// a space shows as a faint dot, so " cat" and "cat" look different
const EOT_HTML = '<span class="eot">&lt;|endoftext|&gt;</span>';
const vis = s => esc(s).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="sp">↵</span>');
function gstr(id) { if (id === EOT) return EOT_HTML; const s = BPE.str(id); return s === null ? `<span class="hex">${BPE.hex(id)}</span>` : vis(s); }
const tstr = i => gstr(V.ids[i]);   // a toy token, by its index in the toy vocabulary
const chip = (i, cls = '', attrs = '') => `<span class="chip ${cls}" ${attrs}>${tstr(i)}</span>`;
const gplain = id => id === EOT ? '<|endoftext|>' : (BPE.str(id) ?? BPE.hex(id)).replace(/ /g, '·');

// a vector as a row of squares: blue for positive, orange for negative, stronger for larger
function heat(x, top) { const a = Math.min(1, Math.abs(x) / (top || 1)); return x >= 0 ? `rgba(42,120,214,${(0.05 + 0.9 * a).toFixed(3)})` : `rgba(235,104,52,${(0.05 + 0.9 * a).toFixed(3)})`; }
const absmax = (...vs) => Math.max(1e-9, ...vs.flatMap(v => arr(v).map(Math.abs)));
function vec(v, top, o = {}) {
  return `<span class="vec${o.nums ? ' nums' : ''}${o.big ? ' big' : ''}">${arr(v).map(x => {
    const a = Math.abs(x) / top;
    return `<i style="background:${heat(x, top)}" class="${o.nums && a > 0.6 ? 'dk' : ''}" title="${num(x, 3)}">${o.nums ? num(x, 2) : ''}</i>`;
  }).join('')}</span>`;
}
const blue = w => `rgba(42,120,214,${(0.04 + 0.96 * Math.min(1, w)).toFixed(3)})`;

// predictions: the top few tokens, as bars; `right` is the token that really came next
function pred(p, right, n = 4) {
  const top = p.map((x, j) => [x, j]).sort((a, b) => b[0] - a[0]).slice(0, n);
  if (right != null && !top.some(([, j]) => j === right)) top[n - 1] = [p[right], right];
  return top.map(([x, j]) => `${chip(j, j === right ? 'tg' : '')}<span class="bar${j === right ? ' ok' : ''}"><i style="width:${(100 * x).toFixed(1)}%"></i></span><span class="p">${pct(x)}</span>`).join('');
}
const argmax = a => a.indexOf(Math.max(...a));

// ---------- the toy, trained at load ----------
// The same toy and training order as the GPT-2 embeddings page, from initialization seed 2. Every
// 10th step's model is kept, for section 8.
const SEED = 2, STREAM = 2019, EVERY = 10;
const TRAIN = (() => {
  const st = makeStream(STREAM), m = initModel(SEED), frames = [m.slice()];
  for (let i = 0; i < st.total; i++) { trainStep(m, V.seqs[st.sent(i)], alphaAt(i, st.total)); if ((i + 1) % EVERY === 0) frames.push(m.slice()); }
  return { frames, total: st.total, M: m };
})();
const M = TRAIN.M;

// ---------- the shared play bar ----------
const BARS = []; let ACTIVE = null;
function PlayBar(host, o) {
  const bar = { o, host, playing: false, timer: null };
  host.classList.add('seg', 'transport', 'playbar');
  host.innerHTML = `<button data-a="reset" title="reset (shift + ←)">${I_RESET}</button><button data-a="play" title="play / pause (space)">${I_PLAY}</button><button data-a="step" title="step (→)">${I_STEP}</button><button data-a="end" title="skip to end (shift + →)">${I_END}</button>`;
  const tick = () => {
    if (!bar.playing) return;
    if (!o.canStep()) { bar.stop(); return; }
    o.step(); refreshBars();
    if (!o.canStep()) { bar.stop(); return; }
    bar.timer = setTimeout(tick, o.ms);
  };
  bar.stop = () => { bar.playing = false; clearTimeout(bar.timer); bar.refresh(); };
  bar.act = a => {
    ACTIVE = bar;
    BARS.forEach(b => { if (b !== bar && b.playing) b.stop(); });
    if (a === 'play') {
      if (bar.playing) bar.stop();
      else { if (!o.canStep()) o.reset(); if (o.canStep()) { bar.playing = true; tick(); } }
    }
    else { bar.stop(); if (a === 'step' && o.canStep()) o.step(); else if (a === 'end' && o.canStep()) o.end(); else if (a === 'reset') o.reset(); }
    refreshBars();
  };
  bar.refresh = () => {
    const can = o.canStep(), b = k => host.querySelector(`[data-a="${k}"]`);
    b('step').disabled = !can || bar.playing; b('end').disabled = !can;
    b('play').innerHTML = bar.playing ? I_PAUSE : I_PLAY;
    b('play').classList.toggle('on', bar.playing);
  };
  host.addEventListener('click', e => { const b = e.target.closest('button'); if (b && !b.disabled) bar.act(b.dataset.a); });
  BARS.push(bar); bar.refresh();
  return bar;
}
function refreshBars() { BARS.forEach(b => b.refresh()); }
function pickBar() {
  const vis = BARS.filter(b => { const r = b.host.closest('.module').getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && b.host.offsetParent; });
  if (ACTIVE && vis.includes(ACTIVE)) return ACTIVE;
  return vis[0] || null;
}
document.addEventListener('keydown', e => {
  if (document.body.classList.contains('present')) return;   // present.js has the keys while presenting
  if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input:not([type=checkbox]), select, textarea')) return;
  if (e.target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
  let a = null;
  if (e.key === ' ') a = 'play';
  else if (e.key === 'ArrowRight') a = e.shiftKey ? 'end' : 'step';
  else if (e.key === 'ArrowLeft' && e.shiftKey) a = 'reset';
  const bar = a && pickBar(); if (!bar) return;
  e.preventDefault(); bar.act(a);
});
window.EXPLAINER = { BARS, active: () => ACTIVE };   // for present.js

// ---------- formulas: in words or in symbols ----------
function setFx(m) {
  document.body.classList.toggle('fx-sym', m === 's');
  document.querySelectorAll('.fxsw button').forEach(b => b.classList.toggle('on', b.dataset.fx === m));
  try { localStorage.setItem('attn-fx', m); } catch (e) {}
}
document.querySelectorAll('.fxsw').forEach(sw => sw.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setFx(b.dataset.fx); }));
setFx((() => { try { return localStorage.getItem('attn-fx') || 'w'; } catch (e) { return 'w'; } })());

// a segmented control: mark the button whose data-[key] is `val`
function seg(host, key, val) { host.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset[key] === val)); }

// ---------- 1 the same vector, two sentences ----------
const DOG = seqOf('the dog can bark'), CAT = seqOf('the cat can purr'), CAN = 3;
function render1() {
  const cards = [DOG, CAT].map(s => ({ s, on: headAt(M, s, CAN), off: headAt(M, s, CAN, { off: true }) }));
  const top = absmax(cards[0].on.x[CAN]);
  $('m1-cards').innerHTML = cards.map(({ s, on, off }) => `<div class="m1card">
      <div class="sent">${[0, 1, 2, 3].map(u => chip(s[u], u === CAN ? 'v sm' : 'sm')).join(' ')} <span class="blank"></span></div>
      <div><div class="sub">the vector at ${chip(s[CAN], 'v sm')}, before attention</div>${vec(on.x[CAN], top, { nums: true })}</div>
      <div><div class="sub">next token, attention off</div><div class="pred">${pred(off.p, s[CAN + 1], 3)}</div></div>
      <div><div class="sub">next token, attention on</div><div class="pred">${pred(on.p, s[CAN + 1], 3)}</div></div>
    </div>`).join('');
  const [d, c] = cards, gd = argmax(d.off.p);
  $('m1-note').innerHTML = `The two vectors at ${chip(CAT[CAN], 'v sm')} are identical, number for number, so with attention off both sentences get the same guess: ${chip(gd, 'sm')}, at ${pct(d.off.p[gd])}. With attention on, ${chip(DOG[CAN + 1], 'tg sm')} gets ${pct(d.on.p[DOG[CAN + 1]])} after “the dog can” and ${chip(CAT[CAN + 1], 'tg sm')} gets ${pct(c.on.p[CAT[CAN + 1]])} after “the cat can”.`;
}

// ---------- 2 set the weights yourself ----------
const S2 = { s: 'dog', raw: [25, 25, 25, 25] };
const seq2 = () => S2.s === 'dog' ? DOG : CAT;
// hand-written lines, by the token with most of the weight; never says where the answer is
const FB2 = {
  0: 'Staring at the very start of the text. Peaceful, but it says nothing about who is doing the can-ing.',
  1: '“the” goes with every noun there is. It isn’t about to take sides.',
  3: '“can” is admiring itself in the mirror. It knows something can do something.',
  spread: 'A little of everything, and not much of anything.',
  near: 'Warmer. The model isn’t convinced yet.',
  dog: 'Woof. That’s the one.',
  cat: 'Purrfect.',
};
$('m2-sent').innerHTML = `<button data-s="dog">the dog can</button><button data-s="cat">the cat can</button>`;
$('m2-sent').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S2.s = b.dataset.s; render2(); } });
$('m2-preset').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  S2.raw = b.dataset.p === 'even' ? [25, 25, 25, 25] : headAt(M, seq2(), CAN).own.map(w => Math.round(100 * w));
  render2();
});
$('m2-rows').addEventListener('input', e => { const i = +e.target.dataset.i; if (!isNaN(i)) { S2.raw[i] = +e.target.value; render2(true); } });
function render2(fromSlider = false) {
  const s = seq2(), tot = S2.raw.reduce((a, b) => a + b, 0), w = tot ? S2.raw.map(r => r / tot) : [0.25, 0.25, 0.25, 0.25];
  seg($('m2-sent'), 's', S2.s);
  if (!fromSlider) $('m2-rows').innerHTML = [0, 1, 2, 3].map(u => `${chip(s[u], u === CAN ? 'v' : '')}<input type="range" min="0" max="100" value="${S2.raw[u]}" data-i="${u}" aria-label="how much ·can draws on token ${u}"><span class="pc" id="m2-pc${u}"></span>`).join('');
  else $('m2-rows').querySelectorAll('.chip').forEach((c, u) => { c.innerHTML = tstr(s[u]); });
  w.forEach((x, u) => { $('m2-pc' + u).textContent = pct(x); });
  const h = headAt(M, s, CAN, { w }), right = s[CAN + 1], won = argmax(h.p) === right;
  $('m2-pred').innerHTML = pred(h.p, right, 4);
  const big = argmax(w), fb = $('m2-fb');
  let line;
  if (w[big] < 0.5) line = FB2.spread;
  else if (big === 2) line = won ? FB2[S2.s] : FB2.near;
  else line = FB2[big];
  fb.textContent = line; fb.classList.toggle('ok', won && big === 2 && w[big] >= 0.5);
}

// ---------- the selected sentence and query position, shared by 3–6 ----------
const SEL = { sent: CORPUS.indexOf('the dog can bark'), pos: CAN };
const selSeq = () => V.seqs[SEL.sent];
const nIn = () => selSeq().length - 1;   // the inputs: every token but the last <|endoftext|>
function selRow(host) {
  const s = selSeq();
  host.innerHTML = `<label class="m3pick"><select aria-label="sentence">${CORPUS.map((c, i) => `<option value="${i}"${i === SEL.sent ? ' selected' : ''}>${esc(c)}</option>`).join('')}</select></label>
    <span class="lab">query at</span><span class="toks">${s.slice(0, nIn()).map((t, u) => chip(t, (u === SEL.pos ? 'v ' : '') + 'sm click', `data-u="${u}"`)).join('')}</span>`;
}
document.querySelectorAll('#m3-sel, #m4-sel, #m5-sel, #m6-sel').forEach(host => {
  host.addEventListener('change', e => { SEL.sent = +e.target.value; SEL.pos = Math.min(SEL.pos, nIn() - 1); renderSel(); });
  host.addEventListener('click', e => { const c = e.target.closest('[data-u]'); if (c) { SEL.pos = +c.dataset.u; renderSel(); } });
});
const selHead = () => headAt(M, selSeq(), SEL.pos);
function renderSel() { render3(); render4(); render5(); render6(); }

// ---------- 3 queries and keys ----------
const S3 = { row: null };
$('m3-t').addEventListener('click', e => { const r = e.target.closest('tr[data-u]'); if (r && +r.dataset.u <= SEL.pos) { S3.row = +r.dataset.u; render3(); } });
function render3() {
  selRow($('m3-sel'));
  const s = selSeq(), h = selHead(), n = nIn();
  if (S3.row == null || S3.row > SEL.pos) S3.row = argmax(h.score);
  const top = absmax(h.q, ...h.k);
  $('m3-q').innerHTML = `<span class="vlab"><span class="cq">q</span>, the query of ${chip(s[SEL.pos], 'v sm')}</span>${vec(h.q, top, { nums: true })}`;
  const allX = headAt(M, s, n - 1).x;   // for the later rows, shown faded
  $('m3-t').innerHTML = `<tr><th>token</th><th>its key <span class="ck lc">k</span></th><th style="text-align:right"><span class="cq lc">q</span> · <span class="ck lc">k</span></th><th style="text-align:right">÷ √6 = score</th></tr>` +
    s.slice(0, n).map((t, u) => {
      if (u > SEL.pos) return `<tr class="later" data-u="${u}"><td>${chip(t, 'sm')}</td><td>${vec(mv(allX[u], view(M, 'Wk'), D), top)}</td><td class="n" colspan="2">later: see section 4</td></tr>`;
      return `<tr class="clk${u === S3.row ? ' pick' : ''}" data-u="${u}"><td>${chip(t, (u === SEL.pos ? 'v ' : '') + 'sm')}</td><td>${vec(h.k[u], top)}</td><td class="n">${num(h.raw[u], 3)}</td><td class="n"><b>${num(h.score[u], 3)}</b></td></tr>`;
    }).join('');
  const k = h.k[S3.row];
  $('m3-math').innerHTML = `<span class="cq">q</span> · <span class="ck">k</span>(${tstr(s[S3.row])}) = ${arr(h.q).map((x, d) => `(${num(x)} × ${num(k[d])})`).join(' + ')} = <b>${num(h.raw[S3.row], 3)}</b>; ÷ √6 = <b>${num(h.score[S3.row], 3)}</b>`;
}

// ---------- 4 every score, and the mask ----------
const S4 = { mask: 'on' };
$('m4-mask').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S4.mask = b.dataset.m; render4(); } });
$('m4-grid').addEventListener('click', e => { const r = e.target.closest('[data-u]'); if (r) { SEL.pos = +r.dataset.u; renderSel(); } });
function render4() {
  selRow($('m4-sel')); seg($('m4-mask'), 'm', S4.mask);
  const s = selSeq(), n = nIn(), X = headAt(M, s, n - 1).x;
  const Q = X.map(x => mv(x, view(M, 'Wq'), D)), K = X.map(x => mv(x, view(M, 'Wk'), D));
  const S = Q.map(q => K.map(k => dot(q, k) / Math.sqrt(D)));
  const top = Math.max(...S.flat().map(Math.abs)), g = $('m4-grid');
  g.style.gridTemplateColumns = `auto repeat(${n}, minmax(46px, 1fr))`;
  let html = '<span></span>' + s.slice(0, n).map(t => `<span class="cl">${chip(t, 'sm')}</span>`).join('');
  for (let t = 0; t < n; t++) {
    html += `<span class="rl" data-u="${t}">${chip(s[t], (t === SEL.pos ? 'v ' : '') + 'sm click')}</span>`;
    for (let u = 0; u < n; u++) {
      const fut = u > t, cls = fut ? (S4.mask === 'on' ? 'mask' : 'peek') : '';
      const txt = fut && S4.mask === 'on' ? '−∞' : num(S[t][u], 2);
      html += `<span class="c ${cls}${t === SEL.pos ? ' row-on' : ''}" data-u="${t}" style="${fut && S4.mask === 'on' ? '' : `background:${heat(S[t][u], top)}`}">${txt}</span>`;
    }
  }
  g.innerHTML = html;
  const next = s[SEL.pos + 1];
  $('m4-note').innerHTML = S4.mask === 'on'
    ? `Each row is one query. Row ${chip(s[SEL.pos], 'v sm')} has ${SEL.pos + 1} score${SEL.pos ? 's' : ''}, one for each token up to it; everything to its right is hidden.`
    : (SEL.pos + 1 < n ? `With the mask off, row ${chip(s[SEL.pos], 'v sm')} also scores ${chip(next, 'tg sm')}, the very token it is supposed to predict. The outlined cells are the ones the mask hides.` : `With the mask off, every row also scores the tokens after it (outlined). The last row has nothing after it, so it’s the same either way.`);
}

// ---------- 5 softmax ----------
const S5 = { k: 0 };   // log2 of the factor the scores are multiplied by
$('m5-k').addEventListener('input', e => { S5.k = +e.target.value; render5(); });
$('m5-k1').addEventListener('click', () => { S5.k = 0; $('m5-k').value = 0; render5(); });
function render5() {
  selRow($('m5-sel'));
  const s = selSeq(), h = selHead(), f = 2 ** S5.k;
  const sc = h.score.map(x => x * f), ex = sc.map(Math.exp), Z = ex.reduce((a, b) => a + b, 0), w = ex.map(e => e / Z);
  $('m5-kv').textContent = '×' + (f >= 1 ? +f.toFixed(f < 10 ? 2 : 1) : '1/' + +(1 / f).toFixed(2));
  $('m5-t').innerHTML = `<tr><th>token</th><th style="text-align:right">score${f !== 1 ? ` × ${+f.toFixed(2)}` : ''}</th><th style="text-align:right"><span class="lc">e<sup>score</sup></span></th><th style="text-align:right">÷ ${num(Z, 3)}</th><th>weight</th></tr>` +
    s.slice(0, SEL.pos + 1).map((t, u) => `<tr><td>${chip(t, (u === SEL.pos ? 'v ' : '') + 'sm')}</td><td class="n">${num(sc[u], 3)}</td><td class="n">${num(ex[u], 3)}</td><td class="n"><b>${pct(w[u], 1)}</b></td><td style="width:40%"><span class="wbar" style="width:${(100 * w[u]).toFixed(1)}%"></span></td></tr>`).join('');
  const big = argmax(w);
  $('m5-note').innerHTML = SEL.pos === 0 ? 'The first position has only itself to look at, so it gets 100% whatever the score.'
    : `${chip(s[big], 'sm')} gets the most weight, ${pct(w[big])}.` + (S5.k === 0 ? ' Stretch the scores apart with the slider, or squash them together, and watch the weights.' : S5.k > 0 ? ' Stretched apart, the highest score takes more and more of the weight.' : ' Squashed together, the weights even out towards ' + pct(1 / (SEL.pos + 1)) + ' each.');
}

// ---------- 6 values ----------
function render6() {
  selRow($('m6-sel'));
  const s = selSeq(), h = selHead(), P = SEL.pos;
  const wv = h.v.map((v, u) => arr(v).map(x => x * h.a[u]));
  const tv = absmax(...h.v, h.z), tx = absmax(h.x[P], h.out, h.h1);
  const row = (a, b, c, d = '') => `<tr><td>${a}</td><td class="n">${b}</td><td class="vcol">${c}</td><td class="vcol">${d}</td></tr>`;
  $('m6-t').innerHTML = `<tr><th>token</th><th style="text-align:right">weight</th><th>its value <span class="cvv lc">v</span></th><th>weight × value</th></tr>` +
    s.slice(0, P + 1).map((t, u) => row(chip(t, (u === P ? 'v ' : '') + 'sm'), pct(h.a[u]), vec(h.v[u], tv), vec(wv[u], tv))).join('') +
    `<tr class="m6tot"><td colspan="3">added up: the blend</td><td>${vec(h.z, tv)}</td></tr>
     <tr class="m6tot"><td colspan="3">× <i>W</i><sub>O</sub>: attention’s output</td><td>${vec(h.out, tx)}</td></tr>
     <tr class="m6tot"><td colspan="3">+ the vector at ${chip(s[P], 'v sm')}</td><td>${vec(h.x[P], tx)}</td></tr>
     <tr class="m6tot"><td colspan="3">= the new vector at ${chip(s[P], 'v sm')}</td><td>${vec(h.h1, tx)}</td></tr>`;
  const right = s[P + 1];
  $('m6-pred').innerHTML = pred(h.p, right, 4);
  $('m6-ph').innerHTML = `What comes after ${chip(s[P], 'v sm')}`;
  const off = headAt(M, s, P, { off: true });
  $('m6-note').innerHTML = `After the MLP adds its change, the new vector is scored against every token. ${chip(right, 'tg sm')} came next, and gets ${pct(h.p[right])}; without attention’s output it would get ${pct(off.p[right])}.`;
}

// ---------- 8 training ----------
const PAIRS = {
  can: { seqs: [DOG, CAT], pos: CAN },
  likes: { seqs: [seqOf('the dog likes to play'), seqOf('the cat likes to sleep')], pos: 4 },
};
const S8 = { f: 0, pair: 'can', series: {} };
const NF = TRAIN.frames.length - 1;
function series8(k) {
  if (S8.series[k]) return S8.series[k];
  const P = PAIRS[k];
  return (S8.series[k] = TRAIN.frames.map(m => P.seqs.map(s => { const h = headAt(m, s, P.pos); return { w: h.own, look: h.own[2], p: h.p[s[P.pos + 1]] }; })));
}
const BAR8 = PlayBar($('m8-bar'), {
  ms: 60, canStep: () => S8.f < NF,
  step: () => { S8.f = Math.min(NF, S8.f + 4); render8(); },
  end: () => { S8.f = NF; render8(); },
  reset: () => { S8.f = 0; render8(); },
});
$('m8-step').max = NF;
$('m8-step').addEventListener('input', e => { BAR8.stop(); S8.f = +e.target.value; render8(); });
$('m8-pair').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S8.pair = b.dataset.p; render8(); } });
function render8() {
  seg($('m8-pair'), 'p', S8.pair);
  const P = PAIRS[S8.pair], ser = series8(S8.pair), now = ser[S8.f];
  $('m8-step').value = S8.f;
  $('m8-stepv').textContent = `step ${(S8.f * EVERY).toLocaleString('en')} of ${TRAIN.total.toLocaleString('en')}`;
  $('m8-rows').innerHTML = P.seqs.map((s, i) => `<div class="m8card">
      <div class="sent">${s.slice(0, P.pos + 1).map((t, u) => chip(t, (u === P.pos ? 'v ' : '') + 'sm')).join(' ')}</div>
      <div class="note" style="margin-bottom:8px">chance of ${chip(s[P.pos + 1], 'tg sm')} next: <b>${pct(now[i].p)}</b></div>
      <div class="attrow">${now[i].w.map((x, u) => `<span class="col"><span class="bar"><i style="height:${(100 * x).toFixed(1)}%"></i></span><span class="pc">${pct(x)}</span>${chip(s[u], 'sm')}</span>`).join('')}</div>
    </div>`).join('');
  // the chart: the weight on the animal, and the chance of the right answer, over training
  const W = 640, H = 170, L = 34, R = 10, T = 8, B = 22, x = f => L + (W - L - R) * f / NF, y = v => T + (H - T - B) * (1 - v);
  const line = (k, i) => ser.map((r, f) => `${f ? 'L' : 'M'}${x(f).toFixed(1)},${y(r[i][k]).toFixed(1)}`).join('');
  const C = { look: '#2a78d6', p: '#1baf7a' };
  $('m8-chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="weight on the animal and chance of the right answer over training">
    ${[0, 0.5, 1].map(v => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e0d9"/><text x="${L - 6}" y="${y(v) + 4}" font-size="11" text-anchor="end">${pct(v)}</text>`).join('')}
    ${[0, 1200, 2400, 3600].map(s => `<text x="${x(s / EVERY)}" y="${H - 4}" font-size="11" text-anchor="middle">${s.toLocaleString('en')}</text>`).join('')}
    ${['look', 'p'].map(k => [0, 1].map(i => `<path d="${line(k, i)}" fill="none" stroke="${C[k]}" stroke-width="1.6" ${i ? 'stroke-dasharray="5 4"' : ''}/>`).join('')).join('')}
    <line x1="${x(S8.f)}" x2="${x(S8.f)}" y1="${T}" y2="${H - B}" stroke="#0b0b0b" stroke-width="1"/>
  </svg>`;
  const [s0, s1] = P.seqs;
  $('m8-leg').innerHTML = `<span><i style="background:${C.look}"></i>weight from ${tstr(s0[P.pos])} on ${tstr(s0[2])} (solid) or ${tstr(s1[2])} (dashed)</span><span><i style="background:${C.p}"></i>chance of ${tstr(s0[P.pos + 1])} (solid) or ${tstr(s1[P.pos + 1])} (dashed)</span>`;
}

// ---------- GPT-2's heads ----------
const NL = ATTN.layers, NH = ATTN.heads;
const TXT = Object.fromEntries(Object.entries(ATTN.texts).map(([k, t]) => {
  const bin = atob(t.att), b = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i);
  const T = t.ids.length, NT = T * (T + 1) / 2;
  return [k, { ...t, T, at: (l, h, q, u) => u > q ? 0 : b[(l * NH + h) * NT + q * (q + 1) / 2 + u] / 255 }];
}));
// a head's grid on a canvas: row q, column u, darker for more weight
function drawHead(cv, txt, l, h, px) {
  px = cv.parentElement.clientWidth || px;   // draw at the size it's shown, so one-token columns stay crisp
  const T = txt.T, dpr = devicePixelRatio || 1, s = px / T;
  cv.width = px * dpr; cv.height = px * dpr; cv.style.width = '100%'; cv.style.height = 'auto';
  const c = cv.getContext('2d'); c.scale(dpr, dpr); c.fillStyle = '#f6f5f1'; c.fillRect(0, 0, px, px);
  for (let q = 0; q < T; q++) for (let u = 0; u <= q; u++) { c.fillStyle = blue(txt.at(l, h, q, u)); c.fillRect(u * s, q * s, Math.ceil(s), Math.ceil(s)); }
}
const headName = (l, h) => `${l}.${h}`;

// ---------- 9 one layer's 12 heads ----------
const S9 = { l: 4 };
$('m9-l').addEventListener('input', e => { S9.l = +e.target.value; render9(); });
$('m9-heads').addEventListener('click', e => { const c = e.target.closest('[data-h]'); if (c) { S10.l = S9.l; S10.h = +c.dataset.h; render10(); render9(); } });
function render9() {
  $('m9-lv').textContent = S9.l;
  const txt = TXT.nat;
  $('m9-heads').innerHTML = Array.from({ length: NH }, (_, h) => `<div class="h${S10.l === S9.l && S10.h === h ? ' on' : ''}" data-h="${h}" title="head ${headName(S9.l, h)}"><canvas></canvas><div class="t">${headName(S9.l, h)}</div></div>`).join('');
  $('m9-heads').querySelectorAll('canvas').forEach((cv, h) => drawHead(cv, txt, S9.l, h, 120));
  const first = Array.from({ length: NH }, (_, h) => ATTN.score.first[S9.l * NH + h]).filter(x => x > 0.5).length;
  $('m9-note').innerHTML = `On “${esc(GPT2_TEXT)}”, preceded by &lt;|endoftext|&gt;. In layer ${S9.l}, ${first} of the 12 heads put more than half their weight on the first token. Click a head to open it in section 10.`;
}
const GPT2_TEXT = TXT.nat.ids.slice(1).map(id => BPE.str(id)).join('');

// ---------- 10 any of the 144 ----------
const S10 = { l: 4, h: 11, text: 'nat', hover: null };
$('m10-text').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S10.text = b.dataset.t; render10(); } });
$('m10-grid').addEventListener('click', e => { const c = e.target.closest('[data-i]'); if (c) { S10.l = Math.floor(+c.dataset.i / NH); S10.h = +c.dataset.i % NH; render10(); } });
let G10 = '';
function render10() {
  seg($('m10-text'), 't', S10.text);
  const txt = TXT[S10.text];
  if (G10 !== S10.text) {   // the 144 thumbnails, redrawn only when the text changes
    G10 = S10.text;
    let html = '<span></span>' + Array.from({ length: NH }, (_, h) => `<span class="lab">${h}</span>`).join('');
    for (let l = 0; l < NL; l++) html += `<span class="lab">${l}</span>` + Array.from({ length: NH }, (_, h) => `<span class="hc" data-i="${l * NH + h}" title="head ${headName(l, h)}"><canvas></canvas></span>`).join('');
    $('m10-grid').innerHTML = html;
    $('m10-grid').querySelectorAll('canvas').forEach((cv, i) => drawHead(cv, txt, Math.floor(i / NH), i % NH, 40));
  }
  $('m10-grid').querySelectorAll('.hc').forEach(c => c.classList.toggle('on', +c.dataset.i === S10.l * NH + S10.h));
  $('m10-h').innerHTML = `Head ${headName(S10.l, S10.h)}: layer ${S10.l}, head ${S10.h}`;
  const T = txt.T, big = $('m10-big');
  big.style.gridTemplateColumns = `auto repeat(${T}, minmax(${T > 30 ? 9 : 14}px, 1fr))`;
  let html = '';
  for (let q = 0; q < T; q++) {
    html += `<span class="rl" data-q="${q}">${esc(gplain(txt.ids[q]))}</span>`;
    for (let u = 0; u < T; u++) html += `<span class="c" data-q="${q}" data-u="${u}" style="background:${u > q ? '#f6f5f1' : blue(txt.at(S10.l, S10.h, q, u))}"></span>`;
  }
  html += '<span></span>' + txt.ids.map((id, u) => `<span class="cl" data-u="${u}">${esc(gplain(id))}</span>`).join('');
  big.innerHTML = html;
  $('m10-hover').innerHTML = 'Hover over a square: its row is the position looking, its column the token looked at.';
}
$('m10-big').addEventListener('mousemove', e => {
  const c = e.target.closest('.c'); if (!c) return;
  const txt = TXT[S10.text], q = +c.dataset.q, u = +c.dataset.u;
  $('m10-big').querySelectorAll('.on').forEach(x => x.classList.remove('on'));
  $('m10-big').querySelector(`.rl[data-q="${q}"]`).classList.add('on'); $('m10-big').querySelector(`.cl[data-u="${u}"]`).classList.add('on');
  $('m10-hover').innerHTML = u > q ? `<b>${esc(gplain(txt.ids[q]))}</b> (position ${q}) can’t see <b>${esc(gplain(txt.ids[u]))}</b>, which comes later.`
    : `<b>${esc(gplain(txt.ids[q]))}</b> (position ${q}) puts <b>${pct(txt.at(S10.l, S10.h, q, u), 1)}</b> of its weight on <b>${esc(gplain(txt.ids[u]))}</b> (position ${u}).`;
});

// ---------- 11 previous-token and induction heads ----------
const REP = TXT.rep, RN = ATTN.repN;
const S11 = { q: 1 + RN + 2 };   // the second " mask"
const HEADS11 = [[4, 11, 'a previous-token head'], [5, 5, 'an induction head']];
['m11-s1', 'm11-s2'].forEach(id => $(id).addEventListener('click', e => { const t = e.target.closest('[data-u]'); if (t) { S11.q = +t.dataset.u; render11(); } }));
function render11() {
  const q = S11.q;
  HEADS11.forEach(([l, h, what], i) => {
    const w = REP.ids.map((_, u) => REP.at(l, h, q, u)), big = argmax(w);
    $(`m11-h${i + 1}`).innerHTML = `<b>Head ${headName(l, h)}</b>, ${what}: from <b>${esc(gplain(REP.ids[q]))}</b>, ${pct(w[big])} of its weight goes to <b>${esc(gplain(REP.ids[big]))}</b>`;
    $(`m11-s${i + 1}`).innerHTML = REP.ids.map((id, u) => `<span data-u="${u}" class="${u === q ? 'q' : ''}" style="background:${u > q ? 'transparent' : blue(w[u])};${u > q ? 'color:#c3c2b7' : w[u] > 0.55 ? 'color:white' : ''}">${u === 0 ? '&lt;|endoftext|&gt;' : esc(BPE.str(id).trim())}</span>`).join('');
  });
  const pn = REP.pNext, nxt = REP.ids[q + 1];
  $('m11-h3').innerHTML = q + 1 < REP.T ? `GPT-2’s chance of the word that really comes next, at every position. After <b>${esc(gplain(REP.ids[q]))}</b>, it gives <b>${esc(gplain(nxt))}</b> ${pct(pn[q], 1)}.` : 'GPT-2’s chance of the word that really comes next, at every position.';
  $('m11-p').innerHTML = pn.map((p, t) => `<i class="${t === q ? 'q' : ''}" style="height:${Math.max(1, 100 * p).toFixed(1)}%" title="after ${esc(gplain(REP.ids[t]))}: ${pct(p, 1)} for ${esc(gplain(REP.ids[t + 1]))}"></i>`).join('');
}

// ---------- 12 the attention sink ----------
const S12 = { text: 'nat' };
$('m12-text').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S12.text = b.dataset.t; render12(); } });
function render12() {
  seg($('m12-text'), 't', S12.text);
  const v = S12.text === 'nat' ? ATTN.score.first : ATTN.firstNoEot, n = v.filter(x => x > 0.5).length;
  const W = 640, H = 300, L = 54, R = 14, T = 26, B = 30, x = a => L + (W - L - R) * a, y = l => T + (H - T - B) * (l + 0.5) / NL;
  const dots = v.map((a, i) => { const l = Math.floor(i / NH), h = i % NH; return `<circle cx="${x(a).toFixed(1)}" cy="${(y(l) + ((h % 3) - 1) * 4).toFixed(1)}" r="5" fill="${a > 0.5 ? '#2a78d6' : '#c3c2b7'}" fill-opacity="0.75" data-i="${i}"><title>head ${headName(l, h)}: ${pct(a)}</title></circle>`; }).join('');
  $('m12-plot').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="each head's average weight on the first token, by layer">
    <rect x="${x(0.5)}" y="${T - 6}" width="${x(1) - x(0.5)}" height="${H - T - B + 12}" fill="#cde2fb" fill-opacity="0.35"/>
    <text x="${x(0.75)}" y="${T - 10}" font-size="12" text-anchor="middle" fill="#1f5ea8">${n} heads over 50%</text>
    ${Array.from({ length: NL }, (_, l) => `<text x="${L - 8}" y="${y(l) + 4}" font-size="11" text-anchor="end">layer ${l}</text><line x1="${L}" x2="${W - R}" y1="${y(l)}" y2="${y(l)}" stroke="#eceae4"/>`).join('')}
    ${[0, 0.25, 0.5, 0.75, 1].map(a => `<text x="${x(a)}" y="${H - 10}" font-size="11" text-anchor="middle">${pct(a)}</text>`).join('')}
    ${dots}
  </svg>`;
  $('m12-hover').innerHTML = `Each dot is a head: how much of its weight, averaged over the sentence’s positions, goes to the first token${S12.text === 'nat' ? ', &lt;|endoftext|&gt;' : `, “${esc(BPE.str(TXT.natNoEot.ids[0]))}”`}. Hover over one for its name.`;
}
$('m12-plot').addEventListener('mousemove', e => {
  const c = e.target.closest('circle'); if (!c) return;
  const i = +c.dataset.i, l = Math.floor(i / NH), h = i % NH, v = S12.text === 'nat' ? ATTN.score.first : ATTN.firstNoEot;
  $('m12-hover').innerHTML = `Head <b>${headName(l, h)}</b> puts <b>${pct(v[i])}</b> of its weight on the first token` + (S12.text === 'nat' ? `; its value there is <b>${num(ATTN.score.vratio[i], 2)}</b> times the length of its values elsewhere.` : '.');
});

// ---------- go ----------
render1(); render2(); renderSel(); render8(); render9(); render10(); render11(); render12();
})();
