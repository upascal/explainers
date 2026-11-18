(function () {
// resdata.js is concatenated ahead of this file by build.py: RES, everything from GPT-2 small
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function num(x, n = 2) { const s = Math.abs(x).toFixed(n); return (x < 0 && +s !== 0 ? '−' : '') + s; }
const sgn = (x, n = 2) => (x > 0 && +Math.abs(x).toFixed(n) !== 0 ? '+' : '') + num(x, n);
const NL = RES.layers, NH = RES.heads, NP = NL * NH;
const name = i => `${Math.floor(i / NH)}.${i % NH}`;
const idx = s => { const [l, h] = s.split('.').map(Number); return l * NH + h; };
const argmax = a => a.indexOf(Math.max(...a));
const rows = (flat, n) => Array.from({ length: flat.length / n }, (_, i) => flat.slice(i * n, i * n + n));

// ---------- tokens ----------
const EOT = 50256;
const tstr = id => RES.str[id] ?? `#${id}`;
const vis = s => esc(s).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="sp">↵</span>');
const tok = id => id === EOT ? '<span class="eot">&lt;|endoftext|&gt;</span>' : vis(tstr(id));
const chip = (id, cls = '') => `<span class="chip sm ${cls}">${tok(id)}</span>`;
const plain = id => id === EOT ? '<|endoftext|>' : tstr(id).trim();

// ---------- colours ----------
const blue = a => `rgba(42,120,214,${(0.05 + 0.95 * Math.max(0, Math.min(1, a))).toFixed(3)})`;
function div(x, top) { const a = Math.min(1, Math.abs(x) / (top || 1)); return x >= 0 ? `rgba(42,120,214,${(0.05 + 0.9 * a).toFixed(3)})` : `rgba(235,104,52,${(0.05 + 0.9 * a).toFixed(3)})`; }
const ramp = (lo, hi, f) => `<span>${lo}</span><span class="ramp" style="background:linear-gradient(90deg,${[0, 0.25, 0.5, 0.75, 1].map(f).join(',')})"></span><span>${hi}</span>`;

// a 12 × 12 grid of heads. o: color(i), title(i), on (selected), src (outlined in pink), off(i), click(i)
function headGrid(host, o) {
  let html = '<span></span>' + Array.from({ length: NH }, (_, h) => `<span class="lab">${h}</span>`).join('');
  for (let l = 0; l < NL; l++) {
    html += `<span class="lab">${l}</span>`;
    for (let h = 0; h < NH; h++) {
      const i = l * NH + h, off = o.off && o.off(i);
      html += `<span class="hc${off ? ' off' : ''}${i === o.on ? ' on' : ''}${i === o.src ? ' src' : ''}" data-i="${i}" title="${off ? `head ${name(i)}` : esc(o.title(i))}" style="${off ? '' : `background:${o.color(i)}`}"></span>`;
    }
  }
  host.innerHTML = html;
  if (!host.dataset.bound) {
    host.dataset.bound = 1;
    host.addEventListener('click', e => { const c = e.target.closest('.hc'); if (c) host._click(+c.dataset.i); });
  }
  host._click = o.click;
}

// a strip of tokens to pick a position from
function strip(host, ids, q, click, upto = ids.length) {
  host.innerHTML = ids.slice(0, upto).map((id, u) => `<span data-u="${u}" class="${u === q ? 'q' : ''}">${u === 0 && id === EOT ? '&lt;|endoftext|&gt;' : esc(tstr(id).trim() || tstr(id))}</span>`).join('');
  if (!host.dataset.bound) { host.dataset.bound = 1; host.addEventListener('click', e => { const t = e.target.closest('[data-u]'); if (t) host._click(+t.dataset.u); }); }
  host._click = click;
}

// diverging bars: name | negative half | positive half | value
function bars(items, top, totalRow) {
  const w = x => (100 * Math.min(1, Math.abs(x) / top)).toFixed(1) + '%';
  return items.map(([nm, x, cls = '']) => `<span class="nm ${cls}">${nm}</span><span class="neg">${x < 0 ? `<i style="width:${w(x)}"></i>` : ''}</span><span class="pos">${x > 0 ? `<i style="width:${w(x)}"></i>` : ''}</span><span class="v ${cls}">${sgn(x)}</span>`).join('') +
    (totalRow ? `<span class="nm tot">${totalRow[0]}</span><span></span><span></span><span class="v tot">${sgn(totalRow[1])}</span>` : '');
}

// formulas in words or symbols
function setFx(m) {
  document.body.classList.toggle('fx-sym', m === 's');
  document.querySelectorAll('.fxsw button').forEach(b => b.classList.toggle('on', b.dataset.fx === m));
  try { localStorage.setItem('res-fx', m); } catch (e) {}
}
document.querySelectorAll('.fxsw').forEach(sw => sw.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setFx(b.dataset.fx); }));
setFx((() => { try { return localStorage.getItem('res-fx') || 'w'; } catch (e) { return 'w'; } })());
function seg(host, key, val) { host.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset[key] === val)); }

const TX = Object.fromEntries(Object.entries(RES.texts).map(([k, t]) => {
  const T = t.ids.length;
  return [k, { ...t, T, contrib: rows(t.contrib, 25), norm: rows(t.norm, 25), heads: rows(t.heads, NP) }];
}));
const RN = RES.repN, MASK2 = 1 + RN + 2;   // the second " mask"
const PART = ['embeddings', ...Array.from({ length: NL }, (_, l) => [`attn ${l}`, `MLP ${l}`]).flat()];

// ---------- 1 every part's push ----------
const S1 = { text: 'rep', q: { rep: MASK2, nat: 4 } };
$('m1-text').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S1.text = b.dataset.t; render1(); } });
function render1() {
  seg($('m1-text'), 't', S1.text);
  const t = TX[S1.text], q = S1.q[S1.text];
  strip($('m1-strip'), t.ids, q, u => { S1.q[S1.text] = Math.min(u, t.T - 2); render1(); }, t.T - 1);
  const c = t.contrib[q], nxt = t.ids[q + 1], tot = c.reduce((a, b) => a + b, 0);
  $('m1-h').innerHTML = `At ${chip(t.ids[q])}, how much each part pushes the score of ${chip(nxt, 'tg')}, the token that really came next, above the average token’s. GPT-2 gives it ${(100 * t.pNext[q]).toFixed(1)}%.`;
  const top = Math.max(...c.map(Math.abs));
  $('m1-parts').innerHTML = bars(c.map((x, i) => [PART[i], x]), top, ['all 25', tot]);
  // the running vector's length after each part
  const n = t.norm[q], W = 300, H = 150, L = 34, B = 20, x = i => L + (W - L - 6) * i / 24, ly = v => Math.log10(Math.max(v, 1)), mx = Math.ceil(ly(Math.max(...n)) * 2) / 2;
  const y = v => 6 + (H - 6 - B) * (1 - ly(v) / mx);
  $('m1-norm').innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="length of the vector after each part">
    ${[1, 10, 100, 1000].filter(v => ly(v) <= mx).map(v => `<line x1="${L}" x2="${W - 6}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e0d9"/><text x="${L - 5}" y="${y(v) + 4}" font-size="10" text-anchor="end">${v}</text>`).join('')}
    <path d="${n.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')}" fill="none" stroke="#4a3aa7" stroke-width="2"/>
    ${[0, 12, 24].map(i => `<text x="${x(i)}" y="${H - 4}" font-size="10" text-anchor="middle">${i === 0 ? 'embed' : i === 12 ? 'layer 6' : 'layer 12'}</text>`).join('')}
  </svg>`;
  $('m1-nnote').innerHTML = `From ${n[0].toFixed(1)} after the embeddings to ${Math.round(n[24])} after the last layer (a log scale).`;
}

// ---------- 2 the hidden dial ----------
// W_V' = W_V M and W_O' = M⁻¹ W_O, with M stretching each of the 64 channels by its own amount
const DIAL = RES.dial, CH = 5;   // show 5 of the 64 channels
const stretch = Array.from({ length: RES.dHead }, (_, j) => Math.sin(12.9898 * (j + 1)) * 1.6);   // fixed, different for each channel
let S2 = 0;
$('m2-t').value = 0;   // a reload can restore the slider's old position
$('m2-t').addEventListener('input', e => { S2 = +e.target.value; render2(); });
$('m2-0').addEventListener('click', () => { S2 = 0; $('m2-t').value = 0; render2(); });
function mat(vals, r, c, top, lab, cls = '') {
  return `<div><div class="matlab">${lab}</div><div class="mat" style="grid-template-columns:repeat(${c}, auto)">${vals.map(v => `<i class="${cls}${Math.abs(v) / top > 0.6 ? ' dk' : ''}" style="background:${div(v, top)}" title="${num(v, 4)}">${num(v, 2)}</i>`).join('')}</div></div>`;
}
// a 4 × 64 corner of W_V times a 64 × 4 corner of W_O
const prod = (wv, wo) => Array.from({ length: 16 }, (_, k) => { const a = Math.floor(k / 4), b = k % 4; let t = 0; for (let j = 0; j < RES.dHead; j++) t += wv[a * RES.dHead + j] * wo[j * 4 + b]; return t; });
function render2() {
  const m = stretch.map(s => Math.exp(S2 * s)), dh = RES.dHead;
  const wv = DIAL.wv.map((v, k) => v * m[k % dh]);                       // 4 × 64
  const wo = DIAL.wo.map((v, k) => v / m[Math.floor(k / 4)]);            // 64 × 4
  const ov = prod(wv, wo);
  const sub = (a, r, c, C) => Array.from({ length: r * c }, (_, k) => a[Math.floor(k / c) * C + k % c]);
  const tv = Math.max(...DIAL.wv.map(Math.abs)) * 1.2, to = Math.max(...DIAL.wo.map(Math.abs)) * 1.2, tov = Math.max(...DIAL.ov.map(Math.abs));
  $('m2-mats').innerHTML = mat(sub(wv, 4, CH, dh), 4, CH, tv, `<i>W</i><sub>V</sub>: 4 rows × ${CH} of 64 columns`, 'ch') + '<span class="op">·</span>' +
    mat(sub(wo, CH, 4, 4), CH, 4, to, `<i>W</i><sub>O</sub>: ${CH} of 64 rows × 4 columns`, 'ch') + '<span class="op">=</span>' +
    mat(ov, 4, 4, tov, '<i>W</i><sub>V</sub><i>W</i><sub>O</sub>: 4 × 4 of 768 × 768');
  const base = prod(DIAL.wv, DIAL.wo), err = Math.max(...ov.map((v, k) => Math.abs(v - base[k])));
  $('m2-note').innerHTML = `Corners of head 5.5’s tables. ${S2 === 0 ? 'These are GPT-2’s own numbers.' : `Every channel of W<sub>V</sub> is stretched by its own amount, from ×${Math.min(...m).toFixed(2)} to ×${Math.max(...m).toFixed(2)}, and W<sub>O</sub> shrunk by the same.`} The product, summed over all 64 channels, ${S2 === 0 ? 'is what the head actually does.' : `hasn’t moved${err < 1e-9 ? '' : `: the largest change is ${err.toExponential(0)}`}.`}`;
}

// ---------- 3 read and write ----------
const S3 = { i: idx('4.7') };
const top3 = RES.overlap.map(o => o[0]), RO = RES.randOverlap;
function render3() {
  const lo = RO[0], col = v => blue((v - lo) / (1 - lo));
  headGrid($('m3-grid'), { color: i => col(top3[i]), title: i => `head ${name(i)}: largest cosine ${num(top3[i])}`, on: S3.i, click: i => { S3.i = i; render3(); } });
  $('m3-scale').innerHTML = ramp(`${num(lo)} (chance)`, '1', a => blue(a));
  const sv = RES.sv[S3.i], ov = RES.overlap[S3.i], W = 360, H = 120, L = 30, B = 18;
  const x = j => L + (W - L - 4) * j / 63, y1 = v => 4 + (H - 4 - B) * (1 - v / sv[0]), y2 = v => 4 + (H - 4 - B) * (1 - v);
  const line = (a, y, c, dash = '') => `<path d="${a.map((v, j) => `${j ? 'L' : 'M'}${x(j).toFixed(1)},${y(v).toFixed(1)}`).join('')}" fill="none" stroke="${c}" stroke-width="2" ${dash}/>`;
  $('m3-d').innerHTML = `<h5>Head ${name(S3.i)}</h5>
    <p class="note">Its 64 gains, largest first: how strongly each read direction is copied to its write direction.</p>
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="singular values">${sv.map((v, j) => `<rect x="${(x(j) - 2).toFixed(1)}" y="${y1(v).toFixed(1)}" width="3.6" height="${(H - B - y1(v)).toFixed(1)}" fill="#4a3aa7"/>`).join('')}
      <text x="${L - 4}" y="${y1(sv[0]) + 8}" font-size="10" text-anchor="end">${sv[0].toFixed(0)}</text><text x="${L - 4}" y="${H - B}" font-size="10" text-anchor="end">0</text></svg>
    <p class="note">How far its read and write subspaces line up: the 64 cosines between them, largest first, against two random subspaces of the same size (dashed).</p>
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="principal cosines">
      ${[0, 0.5, 1].map(v => `<line x1="${L}" x2="${W - 4}" y1="${y2(v)}" y2="${y2(v)}" stroke="#e1e0d9"/><text x="${L - 4}" y="${y2(v) + 4}" font-size="10" text-anchor="end">${v}</text>`).join('')}
      ${line(RO, y2, '#898781', 'stroke-dasharray="4 4"')}${line(ov, y2, '#2a78d6')}
    </svg>
    <p class="note">${ov[0] > 0.9 ? `Its largest cosine is ${num(ov[0])}: at least one direction it writes is almost exactly one it reads.` : ov[0] < RO[0] + 0.05 ? `Its largest cosine, ${num(ov[0])}, is about what two random subspaces give: it writes somewhere other than where it reads.` : `Its largest cosine is ${num(ov[0])}, against ${num(RO[0])} for random subspaces.`}</p>`;
}

// ---------- 4 tokens to tokens ----------
const S4 = { i: idx('7.2'), ex: 0 };
const EXS = RES.examples, CP = RES.copy;
$('m4-ex').addEventListener('click', e => { const c = e.target.closest('[data-e]'); if (c) { S4.ex = +c.dataset.e; render4(); } });
function render4() {
  headGrid($('m4-grid'), { color: i => div(CP[i], 1), title: i => `head ${name(i)}: copying score ${sgn(CP[i])}`, on: S4.i, click: i => { S4.i = i; render4(); } });
  $('m4-scale').innerHTML = ramp('−1 suppresses', '+1 copies', a => div(2 * a - 1, 1));
  $('m4-h').innerHTML = `Head ${name(S4.i)}: copying score ${sgn(CP[S4.i])}`;
  $('m4-ex').innerHTML = EXS.map((id, e) => `<span class="chip sm click${e === S4.ex ? ' v' : ''}" data-e="${e}">${tok(id)}</span>`).join('');
  const topT = RES.ovTop[S4.i][S4.ex], self = RES.ovSelf[S4.i][S4.ex], w = EXS[S4.ex];
  $('m4-top').innerHTML = `<span class="note">read</span><span>${chip(w, 'v')}</span><span></span><span class="note">writes most</span><span></span><span></span>` +
    topT.map((t, r) => `<span class="n">${r + 1}</span><span>${chip(t, t === w ? 'tg' : '')}</span><span></span>`).join('');
  const rank = (self + 1).toLocaleString('en');
  $('m4-note').innerHTML = self === 0 ? `${chip(w)} comes out on top: what this head reads, it writes.`
    : self < 50 ? `${chip(w)} itself comes ${rank}${ord(self + 1)} of 50,257.`
    : self > 50257 - 1000 ? `${chip(w)} itself comes ${rank}${ord(self + 1)} of 50,257, near the very bottom: this head pushes down what it reads.`
    : `${chip(w)} itself comes ${rank}${ord(self + 1)} of 50,257: this head doesn’t pass the token on as it is.`;
}
const ord = n => (n % 100 >= 11 && n % 100 <= 13) ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th';

// ---------- 5 composition ----------
const S5 = { i: idx('4.11'), k: 'k' };
const RC = RES.randComp[0];
$('m5-kind').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S5.k = b.dataset.k; render5(); } });
function render5() {
  seg($('m5-kind'), 'k', S5.k);
  const M = RES[S5.k + 'comp'][S5.i], L0 = Math.floor(S5.i / NH);
  const later = M.map((v, j) => [v, j]).filter(([, j]) => Math.floor(j / NH) > L0), mx = Math.max(RC * 3, ...later.map(([v]) => v));
  headGrid($('m5-grid'), { color: j => blue((M[j] - RC) / (mx - RC)), title: j => `${name(S5.i)} → ${name(j)}: ${num(M[j], 3)}`, src: S5.i, off: j => Math.floor(j / NH) <= L0 && j !== S5.i, click: j => { S5.i = j; render5(); } });
  $('m5-scale').innerHTML = ramp(`${num(RC, 3)} (chance)`, num(mx, 3), a => blue(a));
  const what = { k: 'keys', q: 'queries', v: 'values' }[S5.k];
  $('m5-h').innerHTML = later.length ? `What ${name(S5.i)} writes, read by later heads’ ${what}` : `Head ${name(S5.i)} is in the last layer: nothing reads it`;
  later.sort((a, b) => b[0] - a[0]);
  $('m5-top').innerHTML = later.slice(0, 6).map(([v, j], r) => `<span class="n">${r + 1}</span><span>head <b>${name(j)}</b></span><span class="n">${num(v, 3)} <span class="note">(${(v / RC).toFixed(1)}× chance)</span></span>`).join('');
  $('m5-note').innerHTML = later.length ? `Two random tables of the same shape score ${num(RC, 3)}, so that is the level that means nothing.` : '';
}

// ---------- 6 who writes the answer ----------
const S6 = { q: MASK2, h: '7.2' };
const REP = TX.rep, SPLIT = RES.texts.rep.split;
$('m6-head').innerHTML = Object.keys(SPLIT).map(k => `<button data-h="${k}">${k}</button>`).join('');
$('m6-head').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S6.h = b.dataset.h; render6(); } });
function render6() {
  const q = S6.q, hd = REP.heads[q], nxt = REP.ids[q + 1];
  strip($('m6-strip'), REP.ids, q, u => { S6.q = Math.min(u, REP.T - 2); render6(); }, REP.T - 1);
  $('m6-h').innerHTML = `At ${chip(REP.ids[q])}, each head’s own push on the score of ${chip(nxt, 'tg')}, which really comes next, against the average token’s. GPT-2 gives it ${(100 * REP.pNext[q]).toFixed(1)}%.`;
  const top = Math.max(1, ...hd.map(Math.abs));
  headGrid($('m6-grid'), { color: i => div(hd[i], top), title: i => `head ${name(i)}: ${sgn(hd[i])}`, on: idx(S6.h), click: i => { if (SPLIT[name(i)]) { S6.h = name(i); render6(); } } });
  $('m6-scale').innerHTML = ramp(`−${num(top, 1)}`, `+${num(top, 1)}`, a => div(2 * a - 1, 1));
  seg($('m6-head'), 'h', S6.h);
  const sp = SPLIT[S6.h], T = REP.T, src = sp.src.slice(q * T, q * T + q + 1), att = sp.att.slice(q * T, q * T + q + 1), bias = sp.bias[q];
  const order = src.map((v, s) => [Math.abs(v), s]).sort((a, b) => b[0] - a[0]).slice(0, 5).map(([, s]) => s).sort((a, b) => a - b);
  const rest = src.reduce((a, v, s) => order.includes(s) ? a : a + v, 0);
  const items = order.map(s => [esc(plain(REP.ids[s])), src[s]]).concat([[`${q + 1 - order.length} others`, rest], ['value bias', bias]]);
  const tt = Math.max(0.05, ...items.map(([, v]) => Math.abs(v)));
  $('m6-sh').innerHTML = `Head ${S6.h}’s push, by the token it came from`;
  $('m6-src').innerHTML = bars(items, tt, ['total', src.reduce((a, b) => a + b, 0) + bias]);
  const big = argmax(att);
  $('m6-note').innerHTML = `It puts ${(100 * att[big]).toFixed(0)}% of its weight on ${chip(REP.ids[big])}. Pick one of four heads above; the grid also shows which heads push hardest.`;
}

render1(); render2(); render3(); render4(); render5(); render6();
})();
