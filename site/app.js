(async function () {
// ---------- data ----------
// inline JSON when built (dist/), fetched from the served directory in dev (site/)
const load = async (id, file) => { const t = document.getElementById(id).textContent.trim(); return t ? JSON.parse(t) : (await fetch(file)).json(); };
const [DATA, SAMPLES] = await Promise.all([load('data-inline', 'data.json'), load('samples-inline', 'samples.json').catch(() => null)]);

const H = 4;
const MAIN_IDS = ['fox', 'australia', 'jordan', 'president', 'economy', 'humanity'];
const P = Object.fromEntries(DATA.prompts.map(p => [p.id, p]));
// display order: measured first-step entropy, fixed -> open
const ORDER = MAIN_IDS.slice().sort((a, b) => P[a].entropy - P[b].entropy);
let KEYS_FIXED;   // assigned once keysFrom() is defined, below
const KEY_NAMES = ['Key A', 'Key B', 'Key C'];
const KEY_COLORS = ['var(--blue)', 'var(--orange)', 'var(--aqua)'];
const KEY_HEX = ['#2a78d6', '#eb6834', '#1baf7a'];

// ---------- state ----------
const S = {
  pid: ORDER[0],
  steps: [],           // one per emitted position: {slot, how: 'manual'|'emit'|'plain', bracket?, draw?}
  cur: 0,              // position being viewed, 0 … steps.length
  wm: false,           // stepper: watermark on/off (on needs a key)
  key: null,
  m: 3,
  temp: 1.0,
  draw: null,          // pending draw (array of slots) at the viewed position
  tokView: 'text',
  docLen: 100,
  thr: 0.56,
  smp: { id: null, keyMode: 'demo', view: 'text', custom: null },   // 3: real-text detector
};

// ---------- maths ----------
// The recipe below is the one in Google's open-source implementation
// (transformers, SynthIDTextWatermarkLogitsProcessor): one linear-congruential
// step per ingredient, then a lookup in a fixed table of pre-flipped coins.
const MULT = 6364136223846793005n, MASK64 = (1n << 64n) - 1n, TABLE_SIZE = 65536;
function accum(h, d) { return ((h + BigInt(d)) * MULT + 1n) & MASK64; }
const TABLE = (() => {              // built once from a fixed, published seed
  let a = 0x9e3779b9; const t = new Uint8Array(TABLE_SIZE);
  for (let i = 0; i < TABLE_SIZE; i++) {
    a = a + 0x6D2B79F5 | 0;
    let x = Math.imul(a ^ a >>> 15, 1 | a);
    x = x + Math.imul(x ^ x >>> 7, 61 | x) ^ x;
    t[i] = ((x ^ x >>> 14) >>> 0) & 1;
  }
  return t;
})();
let ctxMemo = { k: null, v: null };
function ctxHash(w) {               // hash of the window, reused across candidates
  const k = w.join(','); if (ctxMemo.k === k) return ctxMemo.v;
  let h = 1n; for (const t of w) h = accum(h, t);
  ctxMemo = { k, v: h }; return h;
}
function rowFor(keys, w, cand, layer) {
  let h = accum(ctxHash(w), cand);          // window, then the candidate…
  h = accum(h, keys[layer % keys.length]);  // …then the secret key for this layer
  return Number(h % BigInt(TABLE_SIZE));
}
function coin(keys, w, cand, layer) { return TABLE[rowFor(keys, w, cand, layer)]; }
function softmax(logits, T) {
  const mx = Math.max(...logits); const w = logits.map(l => Math.exp((l - mx) / T)); const s = w.reduce((a, b) => a + b, 0);
  return w.map(x => x / s);
}
function sample(probs) { let r = Math.random(); for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) return i; } return probs.length - 1; }
const N_KEYS = 30;   // Google's production setting: one secret key per tournament layer
function randKey() { const a = new Uint32Array(N_KEYS); crypto.getRandomValues(a); return [...a].map(x => x % 10000); }
function keysFrom(seed) { const o = []; let a = seed >>> 0; for (let i = 0; i < N_KEYS; i++) { a = (Math.imul(a, 1103515245) + 12345) >>> 0; o.push(a % 10000); } return o; }
function commas(v) { return String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
KEYS_FIXED = [keysFrom(0x5E1F2A9C), keysFrom(0xC0DE4B17), keysFrom(0x3B8D77E4)];
function fmtTok(s) { return s.replace(/\n/g, '⏎').replace(/^ /, '·'); }
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
// plain-language description of a token, for tooltips
function tokDesc(s) {
  if (s === '<bos>') return 'start-of-sequence marker: the model always sees this first';
  if (/^\n+$/.test(s)) return s.length === 1 ? 'a line break' : `${s.length} line breaks (a paragraph break)`;
  if (/^ +$/.test(s)) return s.length === 1 ? 'a single space' : `${s.length} spaces`;
  if (s.startsWith(' ')) return `the word “${s.slice(1)}” with a space in front (how most words are stored)`;
  return `“${s}”, attached directly to the previous token`;
}
const GLYPH_KEY = `<span class="note">Reading tokens: <span class="mono">·</span> = a space in front of the word &nbsp;·&nbsp; <span class="mono">⏎</span> = a line break (<span class="mono">⏎⏎</span> = paragraph break) &nbsp;·&nbsp; <code>&lt;bos&gt;</code> = start marker. Hover any token for a description.</span>`;
function pct(x, d = 0) { return (100 * x).toFixed(d) + '%'; }

function prompt() { return P[S.pid]; }
function pathTo(k) { return S.steps.slice(0, k).map(st => st.slot); }
function node(pid = S.pid, path = pathTo(S.cur)) { return P[pid].nodes[path.join('-')]; }
function curStep() { return S.steps[S.cur] || null; }
function atEnd() { return S.cur >= DATA.depth; }   // no distribution beyond depth 3
function seqIdsTo(k) {
  const p = prompt(); const ids = p.tokens.map(t => t.id);
  let cur = [];
  for (const slot of pathTo(k)) { ids.push(P[S.pid].nodes[cur.join('-')].cand[slot].id); cur.push(slot); }
  return ids;
}
function seqIds() { return seqIdsTo(S.steps.length); }
function ctxIds() { return seqIdsTo(S.cur); }       // tokens before the viewed position
function seqToks() {
  const p = prompt(); const out = p.tokens.map(t => ({ id: t.id, s: t.s, how: 'prompt' }));
  let cur = [];
  S.steps.forEach(st => { const c = P[S.pid].nodes[cur.join('-')].cand[st.slot]; out.push({ id: c.id, s: c.s, how: st.how, slot: st.slot }); cur.push(st.slot); });
  return out;
}
function win(ids) { return ids.slice(-H); }
function candProbs(n = node(), T = S.temp) { return softmax(n.cand.map(c => c.logit), T); }
function coinsFor(n, w, key, m) { return n.cand.map(c => Array.from({ length: m }, (_, l) => coin(key, w, c.id, l))); }
// exact winner distribution of an m-layer tournament (random tiebreaks)
function exactWinner(probs, coins, m) {
  let p = probs.slice();
  for (let l = 0; l < m; l++) {
    const mass = p.reduce((a, q, i) => a + q * coins[i][l], 0);
    p = p.map((q, i) => q * (1 + coins[i][l] - mass));
  }
  return p;
}
function runBracket(slots, coins, m) {
  const rounds = []; let cur = slots.slice();
  for (let l = 0; l < m; l++) {
    const matches = []; const next = [];
    for (let i = 0; i < cur.length; i += 2) {
      const a = cur[i], b = cur[i + 1]; const ca = coins[a][l], cb = coins[b][l];
      let w, side, tie = false;
      if (ca > cb) { w = a; side = 'a'; }
      else if (cb > ca) { w = b; side = 'b'; }
      else { tie = true; side = Math.random() < 0.5 ? 'a' : 'b'; w = side === 'a' ? a : b; }
      matches.push({ a, b, ca, cb, w, side, tie }); next.push(w);
    }
    rounds.push(matches); cur = next;
  }
  return { rounds, winner: cur[0], slots: slots.slice() };
}
function tokenScore(ids, idx, key, m) {
  // score of token at position idx in ids: mean coin over layers using the H tokens before it
  const w = ids.slice(Math.max(0, idx - H), idx); const c = ids[idx];
  let s = 0; for (let l = 0; l < m; l++) s += coin(key, w, c, l); return s / m;
}
// log binomial pmf
const LG = (() => { const c = [0]; for (let i = 1; i < 20005; i++) c.push(c[i - 1] + Math.log(i)); return n => c[n]; })();
function binomTail(n, k) { // P(X >= k), X ~ Bin(n, 1/2)
  if (k <= 0) return 1; if (k > n) return 0;
  let s = 0; const ln2 = n * Math.log(2);
  for (let i = k; i <= n; i++) s += Math.exp(LG(n) - LG(i) - LG(n - i) - ln2);
  return Math.min(1, s);
}
function normTail(z) { // P(Z >= z)
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z >= 0 ? p : 1 - p;
}

// ---------- rendering ----------
const $ = id => document.getElementById(id);
function chip(t, cls = '') {
  const s = S.tokView === 'id' ? String(t.id) : fmtTok(t.s);
  const c = t.s === '<bos>' ? 'bos' : '';
  return `<span class="chip ${cls} ${c} ${S.tokView === 'id' ? 'id' : ''}" title="${esc(tokDesc(t.s))}">${esc(s)}</span>`;
}

// ---- the state bar, ported from the design-canvas sketch (Canvas-2)
// Written tokens lead off to the left; the last H sit in the dotted token window; the blank
// is the position being viewed. A token chosen there sits on the blank until ▶ moves it into
// the window and opens a fresh blank.
const HOW = { bos: 'beginning-of-sequence token', prompt: 'prompt token', emit: 'emitted by the watermark tournament', manual: 'chosen by hand in 2.2', plain: 'sampled with the watermark off' };
const SB = { cur: null, step: null };   // what the bar showed last time, so arrivals can animate
const TRI_L = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><polygon points="11,2 11,14 3,8" fill="currentColor"/></svg>';
const TRI_R = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><polygon points="5,2 5,14 13,8" fill="currentColor"/></svg>';
const TRI_START = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><rect x="1" y="2" width="2" height="12" fill="currentColor"/><polygon points="11,2 11,14 4,8" fill="currentColor"/><polygon points="18,2 18,14 11,8" fill="currentColor"/></svg>';
const TRI_END = '<svg viewBox="0 0 20 16" width="13" height="10" aria-hidden="true"><polygon points="2,2 2,14 9,8" fill="currentColor"/><polygon points="9,2 9,14 16,8" fill="currentColor"/><rect x="17" y="2" width="2" height="12" fill="currentColor"/></svg>';
function sbChip(t, extra = '') {
  const how = t.s === '<bos>' ? 'bos' : t.how;
  const cls = (how === 'prompt' ? '' : how) + (S.tokView === 'id' ? ' id' : '') + extra;
  const s = S.tokView === 'id' ? String(t.id) : fmtTok(t.s);
  return `<span class="chip ${cls}" title="${esc(tokDesc(t.s))} — ${HOW[how]}">${esc(s)}</span>`;
}
function sbNarrow() { $('statebar').classList.toggle('narrow', $('statebar-in').clientWidth < 525); }   // below this the pair cannot sit beside the toggle
function renderStatebar() {
  $('statebar').style.display = 'block'; sbNarrow();
  const toks = seqToks(); const np = prompt().tokens.length;
  const split = np + S.cur;                        // the blank sits here; everything before it is written
  const winStart = Math.max(0, split - H);
  const st = curStep(); const ends = atEnd(); const wm = S.wm && S.keys;
  const advanced = S.cur === SB.cur + 1;           // ▶ (or a click in 2.2) moved the last token into the window
  const landed = st && S.cur === SB.cur && st !== SB.step;   // a token was just generated at the viewed position
  SB.cur = S.cur; SB.step = st;
  const lead = toks.slice(0, winStart).map(t => sbChip(t)).join('');
  const win = toks.slice(winStart, split).map((t, i, a) => sbChip(t, advanced && i === a.length - 1 ? ' slide' : '')).join('');
  const fwdTitle = st ? 'next position' : ends ? 'the tree ends here' : wm ? 'generate the next token with the watermark' : 'generate the next token without the watermark';
  const pair = `<span class="sb-pair"><button data-act="prev" title="previous position" ${S.cur === 0 ? 'disabled' : ''}>${TRI_L}</button><button data-act="fwd" title="${fwdTitle}" ${ends ? 'disabled' : ''}>${TRI_R}</button></span>`;
  const toggle = `<span class="sb-wm${wm ? ' on' : ''}${S.keys ? '' : ' nokeys'}" data-act="wm" role="switch" aria-checked="${!!wm}" title="${S.keys ? 'generate the next token with or without the watermark' : 'needs the provider\'s keys — generate them in 2.3'}"><span class="sb-track"><span class="sb-knob"></span></span>Add Watermark</span>`;
  const promptSel = `<select class="sb-prompt" data-act="prompt" title="the prompt the sentence starts from — same list as 2.1">${ORDER.map((id, i) => `<option value="${id}"${id === S.pid ? ' selected' : ''}>${i + 1}. ${esc(P[id].text)}</option>`).join('')}</select>`;
  const keys = S.keys
    ? `<a class="sb-key" href="#s2-3" title="the provider's keys — set in 2.3">${S.keys.slice(0, 3).map(commas).join(' · ')} …</a>`
    : `<a class="sb-genkeys" href="#s2-3" title="the provider has not generated keys yet — generate them in 2.3">Generate Keys</a>`;
  $('statebar-in').innerHTML = `
    <div class="sb-set">
      <span class="sb-vals">
        <span class="sb-keys">keys ${keys}</span>
        <a href="#s2-5" title="tournament layers — set in 2.5">m <b>${S.m}</b></a>
        <a href="#s2-2" title="temperature — set in 2.2">temp <b>${S.temp.toFixed(2)}</b></a>
      </span>
      ${toggle}
    </div>
    <span class="sb-hr"></span>
    <div class="sb-row">
      ${promptSel}
      <div class="sb-lead">${lead}</div>
      <div class="sb-cluster">
        ${split > winStart ? `<span class="sb-win"><span class="sb-winbox">${win}</span><span class="sb-winlbl">token window</span></span>` : ''}
        <span class="sb-end"><span class="sb-slot${ends ? ' ends' : ''}"><span class="sb-blank" title="the next token goes here"><i class="t"></i><i class="l"></i><i class="t"></i>${st ? `<span class="sb-land">${sbChip(toks[split], landed ? ' pop' : '')}</span>` : ''}</span><span class="sb-slotlbl">next token</span></span>${pair}</span>
      </div>
    </div>
    <div class="sb-narrow">${toggle}${pair}</div>`;
}

function render21() {
  const sel = $('prompt-select');
  if (!sel.options.length) {
    ORDER.forEach((id, i) => { const o = document.createElement('option'); o.value = id; o.textContent = `${i + 1}. ${P[id].text}`; sel.appendChild(o); });
  }
  sel.value = S.pid;
  $('prompt-chips').innerHTML = prompt().tokens.map(t => chip(t)).join('');
}

function renderDist(chartId, legendId, opts) {
  const outcome = !!(opts && opts.outcome), clickable = !!(opts && opts.clickable);
  const n = node(); const probs = candProbs();
  const W = 700, rowH = 30, left = 130, right = 70, H0 = 8 * rowH + 10;
  const maxp = Math.max(...probs);
  const scale = (W - left - right) / Math.max(maxp, 1e-9);
  const st = curStep(); const b = st && st.bracket; const drawn = new Set(b ? st.draw : (S.draw || []));
  const atDepth = atEnd();
  let svg = `<svg viewBox="0 0 ${W} ${H0}" width="100%" style="display:block">`;
  n.cand.forEach((c, i) => {
    const y = i * rowH + 5; const w = Math.max(2, probs[i] * scale);
    let cls = 'bar';
    if (outcome) { if (b) cls += b.winner === i ? ' win' : drawn.has(i) ? ' lost' : ' idle'; else if (drawn.has(i)) cls += ' drawn'; }
    if (st && st.slot === i) cls += ' chosen';
    if (atDepth && clickable) cls += ' dis';
    svg += `<g class="${cls}" data-slot="${i}"><title>${esc(tokDesc(c.s))} — ${pct(probs[i], 1)}</title>
      <rect x="0" y="${y}" width="${W}" height="${rowH - 4}" fill="transparent"/>
      <text x="${left - 10}" y="${y + rowH / 2 + 1}" text-anchor="end" dominant-baseline="middle" font-family="var(--mono)" font-size="13" fill="var(--ink)">${esc(fmtTok(c.s))}</text>
      <text x="${left - 10}" y="${y + rowH / 2 + 1}" text-anchor="end" dominant-baseline="middle" font-size="11" fill="var(--muted)" class="idlab" style="display:none">${c.id}</text>
      <rect class="fill" x="${left}" y="${y + 4}" width="${w}" height="${rowH - 12}" rx="3"/>
      <text x="${left + w + 8}" y="${y + rowH / 2 + 1}" dominant-baseline="middle" font-size="12">${pct(probs[i], 1)}</text>
    </g>`;
  });
  svg += `<line x1="${left}" y1="0" x2="${left}" y2="${H0 - 2}" stroke="var(--axis)"/></svg>`;
  $(chartId).innerHTML = svg;
  if (clickable) document.querySelectorAll('#' + chartId + ' .bar').forEach(g => g.addEventListener('click', () => { if (atDepth) return; chooseManual(+g.dataset.slot); }));
  $(legendId).innerHTML = outcome
    ? (b ? `<span><i style="background:var(--blue)"></i>emitted by the tournament</span><span><i style="background:var(--orange)"></i>drawn, lost</span><span><i style="background:#c3c2b7"></i>not drawn</span>`
         : drawn.size ? `<span><i style="background:var(--blue-250)"></i>drawn as a candidate (2.6)</span><span class="note">Run the tournament to see which one wins.</span>`
         : `<span class="note">Run the tournament above to colour the outcome in here.</span>`)
    : st ? `<span><i style="background:var(--blue);outline:2px solid var(--ink)"></i>chosen at this position ${st.how === 'manual' ? 'by you' : 'by plain sampling, no watermark'}. Click another bar to branch here.</span>`
    : `<span>Click a bar to append that token, or use ▶ in the bar at the top.</span>`;
  $(legendId).innerHTML += `<span style="flex-basis:100%">${GLYPH_KEY}</span>`;
}

function render22() {
  const n = node(); const ids = ctxIds(); const atDepth = atEnd();
  $('seq-reset').disabled = S.steps.length === 0 && S.cur === 0;
  $('seq-prev').disabled = S.cur === 0;
  $('seq-fwd').disabled = atDepth;
  $('seq-end').disabled = atDepth;
  renderDist('dist-chart', 'dist-legend', { outcome: false, clickable: true });
  renderDist('dist-chart-27', 'dist-legend-27', { outcome: true, clickable: false });
  $('pmass-note').innerHTML = `In SynthID-Text the model always draws from the complete distribution over the entire vocabulary; the top eight shown here merely illustrate how the choice operates. For the given sentence these eight tokens cover <b>${pct(n.pmass8, 0)}</b> of the model's probability at this position (full-vocabulary entropy ${n.entropy.toFixed(1)} bits). Window of the last ${H} token IDs used by the watermark: <span class="mono">${win(ids).join(', ')}</span>.`;
  $('depth-note').textContent = atDepth ? 'Tree ends here (depth 3). Reset or step back to continue.' : `position ${S.cur + 1} of ${DATA.depth}${S.cur < S.steps.length ? ' (viewing an earlier position)' : ''}`;
}

function render23() {
  if (S.keys) { $('key-display').innerHTML = `<span style="color:var(--violet)" class="mono">${S.keys.slice(0, 8).join(', ')}, …</span> <span class="note">(${N_KEYS} keys, pinned to the top of the page)</span>`; $('gen-key').disabled = true; $('gen-key').textContent = 'Keys generated'; $('reroll-row').style.display = 'flex'; }
}

// ---- 2.4: long-form arithmetic, ported from the design-canvas sketch
const ROW_H = 30, LOOK_BEFORE = 132, LOOK_COUNT = 152, HEAD_Y = 129, IDLE_IDX = 4;
const INK = '#0b0b0b', MUTED = '#898781', INK2 = '#52514e', LIVE = '#2a78d6', SPENT = '#a8a6a0';
const HA = { cand: 0, layer: 0, step: 1, playing: false, look: 0, expand: false, timer: null, lookTimer: null, els: [], drawn: 0 };

function ingredients() {
  const n = node(), w = win(ctxIds()), c = n.cand[HA.cand], toks = seqToks(), base = ctxIds().length - H;
  const out = w.map((id, i) => ({ glyph: '+', label: `window token ${i + 1} · ${fmtTok(toks[base + i].s)}`, v: BigInt(id) }));
  out.push({ glyph: '+', label: `the candidate · ${fmtTok(c.s)}`, v: BigInt(c.id) });
  out.push({ glyph: '+', label: `secret key #${HA.layer + 1}`, v: BigInt(S.keys[HA.layer % S.keys.length]) });
  return out;
}
// One beat per line of the worked sum. The rhythm inside a block is:
// operation sign → the note → the second number → the rule → the result.
// The three operand beats mutate one line in place; only rules and results append.
function units() {
  const ing = ingredients(); const T = BigInt(TABLE_SIZE); const u = [];
  u.push({ kind: 'res', label: 'start at', num: 1n, muted: true });
  const block = (glyph, label, val, span, resLabel, resNum, bold) => {
    u.push({ kind: 'op', glyph, label: '', num: null, slot: val });
    u.push({ kind: 'op', glyph, label, num: null, slot: val, mutate: 'label' });
    u.push({ kind: 'op', glyph, label, num: val, slot: val, mutate: 'num' });
    u.push({ kind: 'rule', span });
    u.push({ kind: 'res', label: resLabel, num: resNum, bold });
  };
  let h = 1n;
  for (const g of ing) {
    const sum = h + g.v, product = sum * MULT, res = (product + 1n) & MASK64;
    block('+', g.label, g.v, [h, g.v, sum], 'sum so far', sum, false);
    if (HA.expand) {
      block('×', 'the fixed constant M', MULT, [sum, MULT, product], 'product · 39 digits', product, false);
      block('+', 'then add one', 1n, [product, 1n, product + 1n], 'keep the remainder after dividing by 2⁶⁴', res, true);
    } else {
      block('×', 'M, then + 1, keeping 64 bits', MULT, [sum, MULT, res], 'running total', res, true);
    }
    h = res;
  }
  block('÷', `remainder after dividing by ${commas(TABLE_SIZE)}`, T, [h, T], 'the row to read', h % T, true);
  return u;
}
function renderTableauControls() {
  if (!$('t-reset')) return;
  const last = units().length;
  $('t-reset').disabled = HA.step <= 1;
  $('t-back').disabled = HA.step <= 1;
  $('t-fwd').disabled = HA.step >= last;
  $('t-skip').disabled = HA.step >= last;
}
function stepDelay(u) {
  if (!u) return 400;
  if (u.kind === 'rule') return 260;
  if (u.kind === 'res') return 360;
  return u.mutate === 'num' ? 320 : u.mutate === 'label' ? 170 : 120;
}
function valText(u) {
  const glyph = u.glyph ? u.glyph + '  ' : '';
  return glyph + (u.num == null ? ' '.repeat(commas(u.slot).length) : commas(u.num));
}
function fadeIn(el, ms = 220) { if (el && el.animate) el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: ms, easing: 'ease' }); }

// Append or mutate exactly one line. Nothing already on screen is redrawn.
function applyUnit(u, i, animate) {
  const grid = $('t-grid'); if (!grid) return;
  if (u.mutate) {
    const line = HA.els[HA.els.length - 1]; if (!line) return;
    if (u.mutate === 'label') { const lab = line.querySelector('.tlab'); lab.textContent = u.label || ''; if (animate) fadeIn(lab, 200); }
    else { const val = line.querySelector('.tval'); val.textContent = valText(u); if (animate) fadeIn(val, 260); }
    line.dataset.uEnd = i; return;
  }
  const line = document.createElement('div');
  line.className = 'tline'; line.dataset.u = i; line.dataset.uEnd = i;
  if (u.kind === 'rule') {
    const w = Math.max(...u.span.map(n => commas(n).length)) + 3;
    line.innerHTML = `<div class="tnum"><span class="trule" style="width:${w}ch"></span></div><div class="tlab"></div>`;
    grid.appendChild(line);
    const rule = line.querySelector('.trule');
    if (animate && rule.animate) rule.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: 300, easing: 'cubic-bezier(.2,.7,.2,1)' });
  } else {
    line.innerHTML = `<div class="tnum"><span class="tval" style="font-weight:${u.bold ? 600 : 400}${u.muted ? ';color:' + MUTED : ''}">${esc(valText(u))}</span></div>` +
                     `<div class="tlab">${esc(u.label || '')}</div>`;
    grid.appendChild(line);
    if (animate) fadeIn(line.querySelector('.tnum')), fadeIn(line.querySelector('.tlab'));
  }
  HA.els.push(line);
}
function renderTableau(full) {
  const all = units(); const k = Math.min(HA.step, all.length); const done = k >= all.length;
  const grid = $('t-grid'); if (!grid) return;
  if (full || k < HA.drawn) { grid.innerHTML = ''; HA.els = []; HA.drawn = 0; }
  for (let i = HA.drawn; i < k; i++) applyUnit(all[i], i, !full && i === k - 1);
  HA.drawn = k;
  // everything above the newest running total is spent; the newest line is live
  let anchor = 0; for (let i = 0; i < k; i++) if (all[i].kind === 'res' && all[i].bold) anchor = i;
  HA.els.forEach(el => {
    el.classList.toggle('spent', +el.dataset.uEnd < anchor);
    el.classList.toggle('live', !done && +el.dataset.u <= k - 1 && +el.dataset.uEnd >= k - 1);
  });
  const sc = $('tableau-scroll'); if (sc) sc.scrollTop = sc.scrollHeight;
  let folded = 0; for (let i = 0; i < k; i++) if (all[i].kind === 'res' && all[i].bold) folded++;
  $('t-play').textContent = done ? '↻ replay' : HA.playing ? '… working' : '▶ work it out';
  renderTableauControls();
  $('t-progress').textContent = done || k <= 1 ? '' : `number ${Math.min(folded + 1, ingredients().length)} of ${ingredients().length}`;
}
function tickTableau() {
  const all = units(); HA.step++;
  const done = HA.step >= all.length; HA.playing = !done;
  renderTableau(false);
  if (!done) HA.timer = setTimeout(tickTableau, stepDelay(all[HA.step]));
}
function playHash() { clearTimeout(HA.timer); HA.step = 1; HA.playing = true; renderTableau(true); HA.timer = setTimeout(tickTableau, 220); }
function stepTableau(d) {
  clearTimeout(HA.timer); HA.playing = false;
  const prev = HA.step; HA.step = Math.max(1, Math.min(units().length, HA.step + d));
  renderTableau(HA.step < prev);
  renderTableauControls();
}
function targetRow() { const n = node(); return rowFor(S.keys, win(ctxIds()), n.cand[HA.cand].id, HA.layer); }
function lookRows(target) {
  const first = Math.max(0, Math.min(TABLE_SIZE - LOOK_COUNT, target - LOOK_BEFORE));
  let html = '';
  for (let i = 0; i < LOOK_COUNT; i++) {
    const a = first + i, v = TABLE[a], hit = a === target;
    html += `<div class="lrow"${hit ? ' data-hit' : ''}><span class="laddr">${commas(a)}</span><span class="lval ${v ? 'one' : ''}">${v}</span></div>`;
  }
  return { html, first };
}
function renderLookChrome(target) {
  const l = HA.look;
  $('look-btn').textContent = l === 2 ? '↻ look it up again' : l === 1 ? '… scanning the table' : `Is row ${commas(target)} a 0 or a 1?`;
  $('look-note').innerHTML = l === 2
    ? `row ${commas(target)} holds a <b>${TABLE[target]}</b> — that is the value for “${esc(fmtTok(node().cand[HA.cand].s))}” at layer ${HA.layer + 1}`
    : l === 1 ? '' : `${commas(TABLE_SIZE)} rows; the arithmetic above chose one`;
}
function doLook() {
  const rail = $('look-rail'); if (!rail || HA.look === 1) return;
  clearTimeout(HA.lookTimer);
  const target = targetRow(); const { first } = lookRows(target);
  const idleY = -(IDLE_IDX * ROW_H) + HEAD_Y, tgtY = -((target - first) * ROW_H) + HEAD_Y;
  rail.querySelector('[data-hit]').classList.remove('settled');
  HA.look = 0; renderLookChrome(target);
  rail.style.transition = 'none'; rail.style.filter = 'none'; rail.style.transform = `translateY(${idleY}px)`;
  requestAnimationFrame(() => {
    HA.look = 1; renderLookChrome(target);
    rail.style.transition = 'transform 2.5s cubic-bezier(0.1,0.66,0.02,1)';
    rail.style.filter = 'blur(0.5px)';
    rail.style.transform = `translateY(${tgtY}px)`;
    HA.lookTimer = setTimeout(() => {
      HA.look = 2; rail.style.filter = 'none';
      rail.querySelector('[data-hit]').classList.add('settled');
      renderLookChrome(target);
    }, 2500);
  });
}
function renderHash() {
  const el = $('hash-viz'); if (!S.keys) { el.innerHTML = ''; return; }
  clearTimeout(HA.timer); clearTimeout(HA.lookTimer); HA.playing = false; HA.look = 0;
  const target = targetRow(); const { html, first } = lookRows(target);
  const idleY = -(IDLE_IDX * ROW_H) + HEAD_Y;
  el.innerHTML = `
    <div class="card">
      <div class="note" style="margin-bottom:12px">Worked as long-form arithmetic. Six numbers go in, one line at a time — the four window tokens, the candidate, and this layer's secret key. Each one is added under the running total, the bar closes, and the sum is multiplied by <span class="mono2">M = ${commas(MULT)}</span> and bumped by one. Multiplying by an eighteen-digit number six times is what makes the result unguessable.</div>
      <div id="tableau-scroll" class="tableau"><div class="tgrid" id="t-grid"></div></div>
      <div class="row" style="margin-top:14px">
        <button id="t-play"></button>
        <span class="seg transport">
          <button id="t-reset" title="back to the first line"></button>
          <button id="t-back" title="previous line"></button>
          <button id="t-fwd" title="next line"></button>
          <button id="t-skip" title="show the whole calculation"></button>
        </span>
        <span class="note" id="t-progress"></span>
        <label class="ctl" style="margin-left:auto"><input type="checkbox" id="t-expand" ${HA.expand ? 'checked' : ''}> show the multiplication in full</label>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <div class="note">The arithmetic ended on an address. The table it addresses is public — ${commas(TABLE_SIZE)} coins flipped once and never touched again — so the lookup is the last step: go to that row and read what is written there.</div>
      <div class="look">
        <div class="look-head"></div>
        <div class="look-rail" id="look-rail" style="transform:translateY(${idleY}px)">${html}</div>
        <div class="look-fade top"></div><div class="look-fade bot"></div>
      </div>
      <div class="row" style="margin-top:14px"><button id="look-btn"></button><span class="note" id="look-note"></span></div>
    </div>`;
  HA.step = 1; HA.els = []; HA.drawn = 0; renderTableau(true); renderLookChrome(target);
  $('t-reset').innerHTML = TRI_START; $('t-back').innerHTML = TRI_L; $('t-fwd').innerHTML = TRI_R; $('t-skip').innerHTML = TRI_END;
  $('t-play').addEventListener('click', playHash);
  $('t-reset').addEventListener('click', () => stepTableau(-Infinity));
  $('t-back').addEventListener('click', () => stepTableau(-1));
  $('t-fwd').addEventListener('click', () => stepTableau(1));
  $('t-skip').addEventListener('click', () => stepTableau(Infinity));
  $('t-expand').addEventListener('change', e => { HA.expand = e.target.checked; HA.step = 1; renderTableau(true); });
  $('look-btn').addEventListener('click', doLook);
}
function hashSelect(cand, layer, scroll) {
  HA.cand = cand; HA.layer = layer;
  render24();     // refreshes the inputs grid (which names the key), the selects and the machine
  if (scroll) $('m24').scrollIntoView({ behavior: 'smooth', block: 'start' });
  playHash();
}
function renderHashControls() {
  const n = node(); const cs = $('hash-cand'), ls = $('hash-layer');
  cs.innerHTML = n.cand.map((c, i) => `<option value="${i}">${esc(fmtTok(c.s))} (${c.id})</option>`).join('');
  ls.innerHTML = Array.from({ length: N_KEYS }, (_, l) => `<option value="${l}">layer ${l + 1} · key #${l + 1}</option>`).join('');
  cs.value = HA.cand; ls.value = HA.layer; cs.disabled = ls.disabled = !S.keys;
}

function render24() {
  const ids = ctxIds(); const w = win(ids); const n = node();
  const keyTxt = S.keys ? `<span style="color:var(--violet)" class="mono">key #${HA.layer + 1} = ${commas(S.keys[HA.layer % S.keys.length])}</span><span class="note"> — the one this layer uses, of ${N_KEYS}</span>` : `<span class="note">— generate them in 2.3 —</span>`;
  $('coin-inputs').innerHTML = `
    <span class="k">secret keys</span><span>${keyTxt}</span>
    <span class="k">window (last ${H} tokens)</span><span class="chips">${w.map((id, i) => {
      const t = { id, s: (prompt().tokens.concat(seqToks().slice(prompt().tokens.length)))[ids.length - H + i] ? seqToks()[ids.length - H + i].s : '' };
      return `<span class="chip mono" title="${esc(tokDesc(t.s))}">${S.tokView === 'id' ? id : esc(fmtTok(t.s))}</span>`;
    }).join('')}<span class="seg" id="tok-toggle-2"><button class="${S.tokView === 'text' ? 'on' : ''}" data-v="text">text</button><button class="${S.tokView === 'id' ? 'on' : ''}" data-v="id">token IDs</button></span></span>
    <span class="k">candidate</span><span>one row per candidate token, hashed by its ID</span>
    <span class="k">layer</span><span>picks which secret key is used — layer 1 uses key #1, layer 2 uses key #2, and so on</span>`;
  if (!S.keys) { $('coin-table').innerHTML = `<p class="note">The table appears once the key exists.</p>`; $('hash-viz').innerHTML = `<p class="note">The machine starts once the key exists (2.3).</p>`; renderHashControls(); return; }
  const coins = coinsFor(n, w, S.keys, S.m); const probs = candProbs();
  renderHashControls(); renderHash();
  let t = `<table class="coins"><tr><th>candidate</th><th>token ID</th><th>p</th>${Array.from({ length: S.m }, (_, l) => `<th style="text-align:center">layer ${l + 1}</th>`).join('')}</tr>`;
  n.cand.forEach((c, i) => {
    t += `<tr><td class="mono">${esc(fmtTok(c.s))}</td><td class="mono" style="color:var(--muted)">${c.id}</td><td style="color:var(--muted)">${pct(probs[i], 0)}</td>${coins[i].map((v, l) => `<td class="c sel" data-c="${i}" data-l="${l}" title="click to watch this coin being computed"><span class="coin ${v ? 'one' : 'zero'}">${v}</span></td>`).join('')}</tr>`;
  });
  $('coin-table').innerHTML = t + '</table>';
  document.querySelectorAll('#coin-table td.sel').forEach(td => td.addEventListener('click', () => hashSelect(+td.dataset.c, +td.dataset.l, true)));
}

// ---- 2.6: the draw, ported from the design-canvas sketch.
// A selector band sweeps the distribution, overshoots, then eases onto the row
// it lands on; that row darkens and its token drops into the next empty slot.
const DROW_H = 30;
const DA = { band: 0, dur: 0, ease: 'ease-out', running: false, showBand: false, settled: null, timers: [], drawn: 0 };
function daStop() { DA.timers.forEach(clearTimeout); DA.timers = []; DA.running = false; }
function daAt(ms, fn) { DA.timers.push(setTimeout(fn, ms)); }
function nSlots() { return 2 ** S.m; }
function drawnSlots() { const st = curStep(); return S.draw || (st && st.draw) || []; }

function updateBand() {
  const el = $('draw-band'); if (!el) return;
  el.style.transform = `translateY(${DA.band * DROW_H}px)`;
  el.style.transition = `transform ${DA.dur || 0}ms ${DA.ease || 'ease-out'}, opacity 200ms ease`;
  el.style.opacity = DA.showBand ? 1 : 0;
}
function updateBars() {
  const slots = drawnSlots(); const counts = {};
  slots.forEach(i => counts[i] = (counts[i] || 0) + 1);
  document.querySelectorAll('#draw-rows .dfill').forEach((el, i) => {
    el.style.background = i === DA.settled ? '#0b0b0b' : counts[i] ? '#86b6ef' : '#2a78d6';
  });
}
function renderDrawRows() {
  const n = node(); const probs = candProbs(); const maxp = Math.max(...probs);
  $('draw-rows').innerHTML = n.cand.map((c, i) => `<div class="drow">
      <span class="dtok" title="${esc(tokDesc(c.s))}">${esc(fmtTok(c.s))}</span>
      <span class="dtrack"><span class="dfill" style="width:${(probs[i] / maxp * 100).toFixed(2)}%"></span></span>
      <span class="dpct">${pct(probs[i], 1)}</span></div>`).join('');
  updateBars();
}
function renderDrawSlots(full) {
  const n = node(); const slots = drawnSlots(); const total = nSlots();
  const chips = $('draw-chips'); if (!chips) return;
  if (full || slots.length < DA.drawn) { chips.innerHTML = ''; DA.drawn = 0; }
  for (let i = DA.drawn; i < slots.length; i++)
    chips.insertAdjacentHTML('beforeend', `<span class="dchip" title="${esc(tokDesc(n.cand[slots[i]].s))}">${esc(fmtTok(n.cand[slots[i]].s))}</span>`);
  DA.drawn = slots.length;
  $('draw-empties').innerHTML = '<span class="dempty"></span>'.repeat(Math.max(0, total - slots.length));
  const counts = {}; slots.forEach(i => counts[i] = (counts[i] || 0) + 1);
  const distinct = Object.keys(counts).length;
  const probs = candProbs(); const st = curStep();
  $('draw-note').innerHTML = slots.length === 0
    ? 'Nothing drawn yet at this position.'
    : `${distinct} distinct token${distinct > 1 ? 's' : ''} in ${slots.length} slot${slots.length > 1 ? 's' : ''}. “${esc(fmtTok(n.cand[0].s))}” (${pct(probs[0], 0)}) filled ${counts[0] || 0}.${!S.draw && st ? ' (the draw that decided this position)' : ''}`;
  $('draw-progress').textContent = DA.running ? `draw ${Math.min(slots.length + 1, total)} of ${total}`
    : slots.length ? `${slots.length} of ${total} drawn` : '';
  const fresh = !DA.running && slots.length === 0;
  $('draw-btn').innerHTML = fresh ? `${TRI_R} Draw 2<sup>${S.m}</sup> = ${total} candidates` : `${TRI_START} reset`;
  $('draw-btn').disabled = atEnd();
  $('draw-back').disabled = DA.running || slots.length === 0;
  $('draw-fwd').disabled = DA.running || slots.length >= total || atEnd();
  $('draw-skip').disabled = slots.length >= total || atEnd();
}
function render25(full = true) {
  $('m-v3').textContent = S.m; $('m2').value = S.m;
  $('slot-count').innerHTML = `2<sup>${S.m}</sup> = ${nSlots()} slots`;
  $('slot-heading').textContent = `${nSlots()} slots, one per draw`;
  if (full) { daStop(); DA.showBand = false; DA.settled = null; renderDrawRows(); renderDrawSlots(true); }
  else { updateBars(); renderDrawSlots(false); }
  updateBand();
}

function oneDraw(single) {
  const total = nSlots();
  if (!S.draw) S.draw = [];
  if (S.draw.length >= total) { DA.running = false; DA.showBand = false; updateBand(); renderDrawSlots(false); return; }
  const beat = 110;
  const target = sample(candProbs());
  const runUp = Math.round(beat * 2.6), settle = Math.round(beat * 6);
  const out = target < 4 ? 8.4 : -1.4;   // swing past the ends, then ease back
  DA.band = out; DA.dur = runUp; DA.ease = 'cubic-bezier(0.45,0.05,0.55,0.95)'; updateBand();
  daAt(runUp + 40, () => { DA.band = target; DA.dur = settle; DA.ease = 'cubic-bezier(0.22,1.15,0.36,1)'; updateBand(); });
  daAt(runUp + 40 + settle, () => { DA.settled = target; updateBars(); });
  daAt(runUp + 40 + settle + Math.round(beat * 2.4), () => {
    S.draw = S.draw.concat(target); renderDrawSlots(false); updateBars(); render22();
    if (single) { daAt(Math.round(beat * 1.2), () => { DA.running = false; DA.showBand = false; DA.settled = null; updateBand(); updateBars(); renderDrawSlots(false); }); return; }
    daAt(Math.round(beat * 2.2), () => { DA.settled = null; updateBars(); oneDraw(); });
  });
}
function daStart() {
  if (atEnd()) return;
  if (DA.running || drawnSlots().length) { daReset(); return; }
  daStop(); S.draw = []; DA.running = true; DA.showBand = true; DA.band = 0; DA.dur = 0; DA.settled = null;
  renderDrawSlots(true); updateBand(); updateBars();
  daAt(120, () => oneDraw());
}
function daFwd() {
  if (DA.running || drawnSlots().length >= nSlots() || atEnd()) return;
  daStop(); if (!S.draw) S.draw = drawnSlots().slice();
  DA.running = true; DA.showBand = true; DA.dur = 0;
  renderDrawSlots(false); updateBand();
  daAt(60, () => oneDraw(true));
}
function daBack() {
  if (DA.running || !drawnSlots().length) return;
  daStop(); S.draw = drawnSlots().slice(0, -1); DA.settled = null; DA.showBand = false;
  renderDrawSlots(true); updateBand(); updateBars(); render22();
}
function daSkip() {
  daStop(); const probs = candProbs(); S.draw = drawnSlots().slice();
  while (S.draw.length < nSlots()) S.draw.push(sample(probs));
  DA.running = false; DA.showBand = false; DA.settled = null;
  renderDrawSlots(false); updateBand(); updateBars(); render22();
}
function daReset() { daStop(); S.draw = []; DA.showBand = false; DA.settled = null; renderDrawSlots(true); updateBand(); updateBars(); render22(); }

function bracketHTML(b, n, revealRound) {
  if (!b) return `<span class="note">Draw candidates (2.6) or press Run — Run draws for you.</span>`;
  const m = b.rounds.length, N = 1 << m;
  const PITCH = N > 8 ? 30 : 38, SLOTH = 26, SLOTW = 118, COLW = 162, TOP = 26;
  const W = m * COLW + SLOTW + 6, HGT = TOP + N * PITCH + 6;
  // vertical centre of every slot: column 0 evenly spaced, later columns midway between their feeders
  const cen = [Array.from({ length: N }, (_, i) => TOP + i * PITCH + PITCH / 2)];
  for (let c = 1; c <= m; c++) cen[c] = Array.from({ length: N >> c }, (_, j) => (cen[c - 1][2 * j] + cen[c - 1][2 * j + 1]) / 2);
  const colX = c => c * COLW;
  const col = [b.slots.slice()];
  for (let c = 1; c <= m; c++) col[c] = b.rounds[c - 1].map(mt => mt.w);

  let paths = '', winPaths = '', slots = '', labels = '';   // blue drawn last, so it sits over the joins
  for (let c = 0; c <= m; c++) {
    const vis = c <= revealRound;
    labels += `<div class="brt" style="left:${colX(c)}px;width:${SLOTW}px">${c === m ? 'output' : 'layer ' + (c + 1)}</div>`;
    col[c].forEach((cand, j) => {
      const y = cen[c][j] - SLOTH / 2;
      let cls = 'bslot';
      if (c === m) cls += ' champ';
      else {
        const mt = b.rounds[c][j >> 1];
        const iAmA = j % 2 === 0;
        if (c < revealRound) { const won = mt.side === (iAmA ? 'a' : 'b'); cls += won ? ' w' : ' l'; if (mt.tie) cls += ' tied'; }
      }
      const coin = c < m ? b.rounds[c][j >> 1][j % 2 === 0 ? 'ca' : 'cb'] : null;
      slots += `<div class="${cls}" style="left:${colX(c)}px;top:${y}px;width:${SLOTW}px;height:${SLOTH}px;${vis ? '' : 'opacity:0;'}" title="${esc(tokDesc(n.cand[cand].s))}">
        <span class="bt">${esc(fmtTok(n.cand[cand].s))}</span>${coin == null ? '' : `<span class="coin ${coin ? 'one' : 'zero'}">${coin}</span>`}</div>`;
    });
    if (c === m) break;
    // connectors from column c into column c+1
    b.rounds[c].forEach((mt, j) => {
      const drawn = c < revealRound;
      const xA = colX(c) + SLOTW, xMid = colX(c + 1) - (COLW - SLOTW) / 2, xB = colX(c + 1);
      const yTop = cen[c][2 * j], yBot = cen[c][2 * j + 1], yW = cen[c + 1][j];
      const topWins = mt.side === 'a';
      const seg = (d, win) => `<path d="${d}" class="bl ${drawn ? (win ? 'win' + (mt.tie ? ' tied' : '') : 'lose') : 'idle'}"/>`;
      // the winner's route: its own stub, its half of the vertical, then out to the next slot
      const bits = [
        [`M${xA},${yTop} H${xMid}`, topWins],
        [`M${xA},${yBot} H${xMid}`, !topWins],
        [`M${xMid},${yTop} V${yW}`, topWins],
        [`M${xMid},${yW} V${yBot}`, !topWins],
        [`M${xMid},${yW} H${xB}`, true],
      ];
      bits.forEach(([d, win]) => { const h = seg(d, win); if (drawn && win) winPaths += h; else paths += h; });
      if (mt.tie && drawn) winPaths += `<text x="${xMid + 5}" y="${yW - 5}" class="btielab">tie</text>`;
    });
  }
  return `<div class="bwrap" style="width:${W}px;height:${HGT}px">
    <svg class="blines" width="${W}" height="${HGT}" viewBox="0 0 ${W} ${HGT}">${paths}${winPaths}</svg>${labels}${slots}</div>`;
}
function render26(reveal = 99) {
  $('m-v2').textContent = S.m; $('temp-v2').textContent = S.temp.toFixed(2);
  $('run-btn').disabled = !S.keys || atEnd();
  const st = curStep();
  $('run-note').textContent = !S.keys ? 'Generate the key in 2.3 first.' : atEnd() ? 'Tree ends at depth 3 — step back or reset to run again.' : st && !st.bracket ? `This position was ${st.how === 'manual' ? 'chosen by hand' : 'sampled without the watermark'}; running replaces it.` : '';
  $('bracket').innerHTML = bracketHTML(st && st.bracket, node(), reveal);
  const bb = st && st.bracket;
  if (bb) {
    const all = bb.rounds.flat(), ties = all.filter(mt => mt.tie).length;
    $('tie-note').innerHTML = ties === 0
      ? `Every match was settled by the values the tokens read.`
      : `<b>${ties} of ${all.length}</b> match${all.length > 1 ? 'es' : ''} ${ties === 1 ? 'was a tie' : 'were ties'} — both tokens read the same value, so the winner was picked by a fair flip and the watermark had no say. ${ties === all.length ? 'With every match tied, this position carries no watermark at all.' : ''}`;
  } else $('tie-note').innerHTML = '';
  renderScores();
}
function renderScores() {
  const ids = seqIds(); const toks = seqToks(); const np = prompt().tokens.length;
  if (!S.keys || toks.length === np) { $('score-strip').innerHTML = `<span class="note">No tokens emitted yet.</span>`; return; }
  let html = ''; let sum = 0, k = 0;
  for (let i = np; i < ids.length; i++) {
    const sc = tokenScore(ids, i, S.keys, S.m); sum += sc; k++;
    html += `<span class="sc" style="${toks[i].how === 'manual' ? 'background:#fbf3e3' : toks[i].how === 'plain' ? 'background:#e6e5e0' : ''}"><span class="tk">${esc(fmtTok(toks[i].s))}</span><span class="v">${sc.toFixed(2)}</span></span>`;
  }
  html += `<span class="mean">running mean ${(sum / k).toFixed(3)} <span class="note">over ${k} token${k > 1 ? 's' : ''}</span></span>`;
  $('score-strip').innerHTML = html;
}

function miniHist(n, probs, counts, total, color, avg) {
  // horizontal bars: grey outline = true prob, fill = observed share
  const W = 220, rowH = 20, left = 78; const HH = 8 * rowH + 4;
  const maxv = Math.max(...probs, ...counts.map(c => c / total), ...(avg || [0])) * 1.05;
  const sc = (W - left - 34) / maxv;
  let s = `<svg viewBox="0 0 ${W} ${HH}" width="100%" style="display:block">`;
  n.cand.forEach((c, i) => {
    const y = i * rowH + 2; const obs = counts[i] / total;
    s += `<text x="${left - 6}" y="${y + rowH / 2}" text-anchor="end" dominant-baseline="middle" font-family="var(--mono)" font-size="11" fill="var(--ink)">${esc(fmtTok(c.s)).slice(0, 10)}</text>`;
    s += `<rect x="${left}" y="${y + 3}" width="${Math.max(1, obs * sc)}" height="${rowH - 8}" rx="2" fill="${color}"/>`;
    s += `<rect x="${left}" y="${y + 3}" width="${Math.max(1, probs[i] * sc)}" height="${rowH - 8}" rx="2" fill="none" stroke="var(--ink-2)" stroke-width="1.2"/>`;
    if (avg) s += `<line x1="${left + avg[i] * sc}" x2="${left + avg[i] * sc}" y1="${y + 1}" y2="${y + rowH - 3}" stroke="var(--violet)" stroke-width="2"/>`;
    s += `<text x="${left + Math.max(obs, probs[i], avg ? avg[i] : 0) * sc + 4}" y="${y + rowH / 2}" dominant-baseline="middle" font-size="10">${pct(obs, 0)}</text>`;
  });
  return s + '</svg>';
}

function render27() {
  const n = node(); const probs = candProbs(); const ids = ctxIds(); const w = win(ids);
  const RUNS = 600;
  const showAvg = $('proof-toggle').checked;
  let avg = null;
  if (showAvg) { // exact average over many random keys
    const K = 400; avg = new Array(8).fill(0);
    for (let k = 0; k < K; k++) { const c = coinsFor(n, w, randKey(), S.m); const d = exactWinner(probs, c, S.m); d.forEach((v, i) => avg[i] += v / K); }
  }
  $('panels-27').innerHTML = KEYS_FIXED.map((key, ki) => {
    const coins = coinsFor(n, w, key, S.m); const counts = new Array(8).fill(0);
    for (let r = 0; r < RUNS; r++) { const slots = Array.from({ length: 2 ** S.m }, () => sample(probs)); counts[runBracket(slots, coins, S.m).winner]++; }
    const l1 = counts.reduce((a, c, i) => a + Math.abs(c / RUNS - probs[i]), 0);
    return `<div class="panel"><div class="pt"><i style="background:${KEY_HEX[ki]}"></i>${KEY_NAMES[ki]} <span class="note mono" style="font-weight:400">${key.slice(0, 3).join(', ')}…</span></div>${miniHist(n, probs, counts, RUNS, KEY_HEX[ki], avg)}<div class="note" style="margin-top:4px">distance from model: ${l1.toFixed(2)}</div></div>`;
  }).join('');
  $('legend-27').innerHTML = `<span><i style="border:1.5px solid var(--ink-2);background:none"></i>the model's probability</span><span><i style="background:linear-gradient(90deg,#2a78d6,#eb6834,#1baf7a)"></i>what the tournament emitted under that key (${RUNS} runs)</span>${showAvg ? `<span><i style="background:var(--violet);width:3px"></i>average over 400 keys</span>` : ''}<span class="note">“distance” = sum of absolute differences; 0 would be perfect agreement.</span><span style="flex-basis:100%">${GLYPH_KEY}</span>`;
}

// ---------- Section 4: does watermarking degrade quality? ----------
function render3() {
  const p = P['australia']; const n = p.nodes['']; const probs = softmax(n.cand.map(c => c.logit), S.temp);
  const wi = n.cand.findIndex(c => c.s === ' Sydney'); const ci = n.cand.findIndex(c => c.s === ' Canberra');
  const ids = p.tokens.map(t => t.id); const w = win(ids);
  const pw = probs[wi];
  const rows = [1, 2, 3, 4].map(m => {
    const reach = 1 - Math.pow(1 - pw, 2 ** m);
    let under = null;
    if (S.keys) { const c = coinsFor(n, w, S.keys, m); under = exactWinner(probs, c, m)[wi]; }
    // averaged over random keys (exact per key, averaged)
    let avgk = 0; const K = 300; for (let k = 0; k < K; k++) { const c = coinsFor(n, w, randKey(), m); avgk += exactWinner(probs, c, m)[wi] / K; }
    return { m, reach, under, avgk };
  });
  const keyCol = `<th class="n">under the current keys</th>`;
  $('s3-body').innerHTML = `
    <div class="row" style="margin-bottom:6px"><span>Prompt: <b>${esc(p.text)}</b></span><span class="note">temperature ${S.temp.toFixed(2)} (from 2.2)</span></div>
    <div class="chips" style="margin-bottom:12px">${n.cand.map((c, i) => `<span class="chip ${i === wi ? 'wrong' : ''}" title="${esc(tokDesc(c.s))}">${esc(fmtTok(c.s))} <span class="sp">${pct(probs[i], 1)}</span></span>`).join('')}</div>
    <div class="stat">
      <div class="s"><div class="l">P(“Canberra”), unwatermarked</div><div class="v">${pct(probs[ci], 1)}</div></div>
      <div class="s"><div class="l">P(“Sydney”), unwatermarked</div><div class="v">${pct(pw, 1)}</div></div>
      <div class="s"><div class="l">P(“Sydney”) under your key, m = ${S.m}</div><div class="v">${S.keys ? pct(rows[S.m - 1].under, 1) : '—'}</div></div>
    </div>
    <table class="data"><tr><th>layers m</th><th class="n">slots drawn</th><th class="n">P(“Sydney” reaches the bracket)</th>${keyCol}<th class="n">averaged over 300 keys</th></tr>
    ${rows.map(r => `<tr style="${r.m === S.m ? 'background:#f3f6fb' : ''}"><td>${r.m}</td><td class="n">${2 ** r.m}</td><td class="n">${pct(r.reach, 1)}</td><td class="n" style="font-weight:600">${r.under == null ? '—' : pct(r.under, 1)}</td><td class="n">${pct(r.avgk, 1)}</td></tr>`).join('')}</table>
    <p class="note">“Under the current keys” shows the percentages for this key given the token window. “Averaged over 300 keys” is the mean percentage this token will occur if the key was changed every time a token was sampled. Notice how under this condition the watermarked rate approaches the unwatermarked rate. Re-roll the key in 2.3 to see the percentages under your key change.</p>`;
}

// ---------- Section 3: detection ----------
let detCache = null;
function detStats() {
  // per-token score mean/var under watermark for this prompt tree, key, m, temp
  const sig = [S.pid, S.keys && S.keys.join(','), S.m, S.temp].join('|');
  if (detCache && detCache.sig === sig) return detCache;
  const nodes = Object.values(prompt().nodes); const N = 3000; let s1 = 0, s2 = 0;
  for (let i = 0; i < N; i++) {
    const n = nodes[Math.floor(Math.random() * nodes.length)];
    const probs = candProbs(n, S.temp);
    const w = Array.from({ length: H }, () => Math.floor(Math.random() * 256000));
    const coins = coinsFor(n, w, S.keys, S.m);
    const slots = Array.from({ length: 2 ** S.m }, () => sample(probs));
    const wnr = runBracket(slots, coins, S.m).winner;
    const sc = coins[wnr].reduce((a, b) => a + b, 0) / S.m; s1 += sc; s2 += sc * sc;
  }
  const mu = s1 / N, va = s2 / N - mu * mu;
  detCache = { sig, mu, va }; return detCache;
}
function render4() {
  const ids = seqIds(); const np = prompt().tokens.length; const k = ids.length - np;
  if (!S.keys) { $('det-current').innerHTML = `<p class="note">Generate the key in 2.3 to enable the detector.</p>`; $('det-chart').innerHTML = ''; $('det-stats').innerHTML = ''; return; }
  if (k === 0) $('det-current').innerHTML = `<p class="note"><b>Your document:</b> nothing emitted yet. Run the tournament in 2.7 a few times.</p>`;
  else {
    let sum = 0; for (let i = np; i < ids.length; i++) sum += tokenScore(ids, i, S.keys, S.m);
    const nCoins = k * S.m; const pv = binomTail(nCoins, Math.round(sum * S.m)); const ev = evidence(pv);
    $('det-current').innerHTML = `<div class="stat">
      <div class="s"><div class="l">Your document</div><div class="v">${k} <small>tokens</small></div></div>
      <div class="s"><div class="l">Mean score</div><div class="v">${(sum / k).toFixed(3)}</div></div>
      <div class="s" title="${ev.line} (one-sided p-value ${ev.pct} at a mean score of ${(sum / k).toFixed(3)})"><div class="l">Evidence of a watermark</div><div class="v">${ev.lab}</div></div></div>
      <p class="note">${k} token${k > 1 ? 's' : ''} can't tell you much: even a perfect 1.000 has a ${pct(Math.pow(0.5, k * S.m), 1)} chance of happening by luck. Length is everything — below.</p>`;
  }
  // chart: null vs watermarked score distributions for docLen
  const N = S.docLen; const nC = N * S.m; const { mu, va } = detStats();
  const sdW = Math.sqrt(va / N), sd0 = Math.sqrt(0.25 / nC);
  const x0 = 0.4, x1 = 0.85; const W = 700, HH = 200, left = 40, bottom = 30;
  const xs = x => left + (x - x0) / (x1 - x0) * (W - left - 10);
  const dens0 = x => Math.exp(-0.5 * ((x - 0.5) / sd0) ** 2) / sd0;
  const densW = x => Math.exp(-0.5 * ((x - mu) / sdW) ** 2) / sdW;
  const pk = Math.max(dens0(0.5), densW(mu));
  const ys = d => HH - bottom - d / pk * (HH - bottom - 20);
  const path = f => { let d = ''; for (let i = 0; i <= 200; i++) { const x = x0 + (x1 - x0) * i / 200; d += (i ? 'L' : 'M') + xs(x).toFixed(1) + ',' + ys(f(x)).toFixed(1); } return d; };
  const thr = S.thr;
  const fpr = nC <= 20000 ? binomTail(nC, Math.ceil(thr * nC)) : normTail((thr - 0.5) / sd0);
  const tpr = normTail((thr - mu) / sdW);
  let svg = `<svg viewBox="0 0 ${W} ${HH}" width="100%" style="display:block">`;
  svg += `<path d="${path(dens0)} L${xs(x1)},${HH - bottom} L${xs(x0)},${HH - bottom} Z" fill="#c3c2b7" fill-opacity="0.35"/><path d="${path(dens0)}" fill="none" stroke="var(--ink-2)" stroke-width="1.5"/>`;
  svg += `<path d="${path(densW)} L${xs(x1)},${HH - bottom} L${xs(x0)},${HH - bottom} Z" fill="var(--blue)" fill-opacity="0.25"/><path d="${path(densW)}" fill="none" stroke="var(--blue)" stroke-width="2"/>`;
  svg += `<line x1="${xs(thr)}" x2="${xs(thr)}" y1="10" y2="${HH - bottom}" stroke="var(--red)" stroke-width="2" stroke-dasharray="4 3"/>`;
  svg += `<text x="${xs(thr) + 5}" y="${HH - bottom - 6}" font-size="11" fill="var(--red)">threshold ${thr.toFixed(3)}</text>`;
  svg += `<line x1="${left}" x2="${W - 10}" y1="${HH - bottom}" y2="${HH - bottom}" stroke="var(--axis)"/>`;
  for (let x = 0.4; x <= 0.851; x += 0.05) svg += `<text x="${xs(x)}" y="${HH - bottom + 16}" text-anchor="middle" font-size="11">${x.toFixed(2)}</text>`;
  svg += `<text x="${W / 2}" y="${HH - 2}" text-anchor="middle" font-size="11">mean score over the document</text>`;
  svg += `<text x="${xs(0.5)}" y="${ys(dens0(0.5)) - 6}" text-anchor="middle" font-size="11" fill="var(--ink-2)">unwatermarked</text>`;
  svg += `<text x="${xs(mu)}" y="${ys(densW(mu)) - 6}" text-anchor="middle" font-size="11" fill="var(--blue)">watermarked</text>`;
  $('det-chart').innerHTML = svg + '</svg>';
  const perK = fpr * 50000, perNews = fpr * 2000;
  $('det-stats').innerHTML = `
    <div class="s"><div class="l">True positives caught</div><div class="v">${pct(tpr, 1)}</div></div>
    <div class="s"><div class="l">Innocent docs flagged</div><div class="v">${fpr < 0.0001 ? '<0.01%' : pct(fpr, 2)}</div></div>
    <div class="s"><div class="l">Of 50,000 student essays</div><div class="v">${Math.round(perK).toLocaleString()} <small>flagged</small></div></div>
    <div class="s"><div class="l">Of 2,000 newsroom submissions</div><div class="v">${Math.round(perNews).toLocaleString()} <small>flagged</small></div></div>`;
  $('det-note').innerHTML = `Watermarked-document curve: positions sampled from the model's own continuations of the current prompt (${Object.keys(prompt().nodes).length} positions), your key, <i>m</i> = ${S.m}, temperature ${S.temp.toFixed(2)}; expected per-token score ${mu.toFixed(3)}. Unwatermarked curve: exact — fair coins, ${N} × ${S.m} of them. Change the prompt to see how a text's openness sets the separation.`;
}

function evidence(pv) {
  // A p-value is hard to read off a card. Say how strong the evidence is, and restate it as
  // a natural frequency ("about 1 in 4"), which people parse far more reliably than "24.6%".
  const lab = pv > 0.05 ? 'None' : pv > 0.01 ? 'Weak' : pv > 0.001 ? 'Moderate' : 'Strong';
  const many = pv >= 0.4 ? 'Most' : pv >= 0.25 ? 'Many' : `About 1 in ${commas(Math.round(1 / pv))}`;
  const line = pv < 0.001 ? 'Fewer than 1 in 1,000 texts this long score this high without the watermark.'
    : `${many} text${pv >= 0.25 ? 's' : 's'} this long score this high without the watermark${pv >= 0.25 ? ', so this is just what chance looks like' : ''}.`;
  return { lab, line, pct: pv < 0.001 ? '<0.1%' : pct(pv, 1) };
}

// ---------- Section 3: real text (precomputed samples + in-browser tokenizer) ----------
const KINDS = [['human', 'Human, public domain'], ['gemma-plain', 'Gemma-2-2b, unwatermarked'], ['gemma-wm', 'Gemma-2-2b, watermarked under the demo key']];
const REGS = [['oratory', 'oratory'], ['prose', 'fiction'], ['law', 'law'], ['code', 'code']];
let TOKP = null;           // tokenizer promise, created on first use
let smpCache = { sig: null, rows: null };
function smpList() { const l = SAMPLES ? SAMPLES.samples.slice() : []; if (S.smp.custom) l.push(S.smp.custom); return l; }
function smpCurrent() { const l = smpList(); return l.find(x => x.id === S.smp.id) || l[0]; }
function smpKeys() { return S.smp.keyMode === 'demo' ? SAMPLES.demo_key : S.keys; }
function scoreSample(sm, keys, m = SAMPLES.m) {
  // Repeated-context masking, as in the paper's detector: a window seen earlier in the text
  // yields the same coins again, so it is not fresh evidence and is skipped (score null).
  const scores = []; const seen = new Set(); let sum = 0, n = 0, skipped = 0;
  for (let i = 1; i < sm.ids.length; i++) {
    const w = sm.ids.slice(Math.max(0, i - H), i).join(','); const fresh = !seen.has(w); seen.add(w);
    if (i < sm.prompt_ntok) continue;
    if (!fresh) { scores.push(null); skipped++; continue; }
    const v = tokenScore(sm.ids, i, keys, m); scores.push(v); sum += v; n++;
  }
  return { scores, mean: sum / n, n, skipped, pv: binomTail(n * m, Math.round(sum * m)) };
}
function smpShade(v) { return `hsl(212 60% ${96 - 44 * v}%)`; }
function render4s() {
  if (!$('smp-select')) return;
  if (!SAMPLES) { $('smp-stats').innerHTML = `<p class="note">samples.json is missing — run <span class="mono">precompute/samples.py</span>.</p>`; return; }
  const sel = $('smp-select'); const list = smpList();
  if (sel.options.length !== list.length) {
    sel.innerHTML = KINDS.map(([k, lab]) => `<optgroup label="${lab}">${list.filter(x => x.kind === k).map(x => `<option value="${x.id}">${esc(x.label)}</option>`).join('')}</optgroup>`).join('')
      + (S.smp.custom ? `<optgroup label="Pasted"><option value="custom">${esc(S.smp.custom.label)}</option></optgroup>` : '');
  }
  const sm = smpCurrent(); S.smp.id = sm.id; sel.value = sm.id;
  $('smp-key').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === S.smp.keyMode));
  $('smp-view').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === S.smp.view));
  const keys = smpKeys();
  if (!keys) {
    $('smp-stats').innerHTML = `<p class="note">Generate the key in 2.3 to score these under your own key — or switch back to the demo key, the one these Gemma samples were actually watermarked with.</p>`;
    $('smp-table').innerHTML = ''; $('smp-strip').innerHTML = ''; $('smp-note').innerHTML = ''; return;
  }
  const r = scoreSample(sm, keys); const ev = evidence(r.pv);
  $('smp-stats').innerHTML = `
    <div class="s"><div class="l">${esc(sm.label)}</div><div class="v">${r.n} <small>tokens</small></div></div>
    <div class="s"><div class="l">Mean score</div><div class="v">${r.mean.toFixed(3)}</div></div>
    <div class="s" title="${ev.line} (one-sided p-value ${ev.pct} at a mean score of ${r.mean.toFixed(3)})"><div class="l">Evidence of a watermark</div><div class="v">${ev.lab}</div></div>`;
  // overview: every sample under this key, so the entropy story is visible at a glance
  const sig = S.smp.keyMode + '|' + keys.join(',');
  if (smpCache.sig !== sig) { const rows = {}; SAMPLES.samples.forEach(x => rows[x.id] = scoreSample(x, keys).mean); smpCache = { sig, rows }; }
  const cell = id => { const x = SAMPLES.samples.find(y => y.id === id); if (!x) return '<td></td>'; const v = smpCache.rows[id]; return `<td class="n${sm.id === id ? ' cur' : ''}" style="background:${smpShade(Math.max(0, Math.min(1, (v - 0.4) / 0.4)))};cursor:pointer" data-id="${id}">${v.toFixed(3)}</td>`; };
  const shortKind = { 'human': 'Human, public domain', 'gemma-plain': 'Gemma-2-2b, unwatermarked', 'gemma-wm': 'Gemma-2-2b, watermarked' };
  const idFor = (reg, kind) => kind === 'human' ? SAMPLES.samples.find(x => x.register === reg && x.kind === 'human')?.id : `${reg}-${kind === 'gemma-wm' ? 'wm' : 'plain'}`;
  $('smp-table').innerHTML = `<table class="data"><tr><th rowspan="2">Source</th><th class="n grp" colspan="${REGS.length}">Mean score under ${S.smp.keyMode === 'demo' ? 'the demo key' : 'your key'}</th></tr><tr>${REGS.map(([, lab]) => `<th class="n">${lab}</th>`).join('')}</tr>
    ${KINDS.map(([kind]) => `<tr><td>${shortKind[kind]}</td>${REGS.map(([reg]) => cell(idFor(reg, kind))).join('')}</tr>`).join('')}</table>`;
  $('smp-table').querySelectorAll('td[data-id]').forEach(td => td.addEventListener('click', () => { S.smp.id = td.dataset.id; render4s(); }));
    // the passage itself: readable prose, or one chip per token shaded by score
  let html = '';
  if (S.smp.view === 'text') {
    html = `<span class="pr">${esc(sm.toks.slice(1, sm.prompt_ntok).join(''))}</span><span class="md">${esc(sm.toks.slice(sm.prompt_ntok).join(''))}</span>`;
  } else {
    for (let i = 0; i < sm.prompt_ntok; i++) html += `<span class="sc pr" title="${esc(tokDesc(sm.toks[i]))} — prompt, not scored"><span class="tk">${esc(fmtTok(sm.toks[i]))}</span><span class="v">prompt</span></span>`;
    r.scores.forEach((v, j) => { const t = sm.toks[sm.prompt_ntok + j];
      html += v == null ? `<span class="sc pr" title="${esc(tokDesc(t))} — its 4-token context already occurred earlier, so it is not fresh evidence"><span class="tk">${esc(fmtTok(t))}</span><span class="v">repeat</span></span>`
        : `<span class="sc${v >= 0.75 ? ' hi' : ''}" style="background:${smpShade(v)}" title="${esc(tokDesc(t))} — score ${v.toFixed(2)}"><span class="tk">${esc(fmtTok(t))}</span><span class="v">${v.toFixed(2)}</span></span>`; });
  }
  $('smp-strip').className = S.smp.view === 'text' ? `passage${sm.register === 'code' ? ' code' : ''}` : 'scores';
  $('smp-strip').innerHTML = html;
  const bodyLab = sm.kind === 'human' ? 'human text' : sm.kind === 'custom' ? 'your text' : 'Gemma-2-2b response';
  $('smp-note').innerHTML = S.smp.view === 'text'
    ? `${sm.prompt_ntok > 1 ? '<span class="lg pr">prompt</span> &nbsp; ' : ''}<span class="lg md">${bodyLab}</span>`
    : `<span class="lg" style="background:${smpShade(1)}"></span> more 1s &nbsp; <span class="lg" style="background:${smpShade(0)}"></span> more 0s &nbsp; <span class="lg pr">not scored</span> &nbsp; <i>m</i> = 4`;
}
async function loadTokenizer() {
  if (location.protocol === 'file:') throw new Error('needs http');
  return TOKP ??= (async () => {
    const { AutoTokenizer, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5/dist/transformers.min.js');
    env.allowRemoteModels = false; env.allowLocalModels = true;
    env.localModelPath = new URL('tokenizer/', document.baseURI).href;   // tokenizer/gemma-2-2b/{tokenizer.json,tokenizer_config.json}
    return AutoTokenizer.from_pretrained('gemma-2-2b');
  })();
}
async function scorePasted() {
  const text = $('smp-text').value; if (!text.trim()) return;
  $('smp-go').disabled = true; $('smp-tokstatus').textContent = 'loading tokenizer (17 MB, once)…';
  try {
    const tok = await loadTokenizer();
    const ids = Array.from(tok.encode(text), Number);          // adds the leading <bos>, as data.json does
    S.smp.custom = { id: 'custom', label: 'Your text', kind: 'custom', register: '', source: 'pasted', prompt_ntok: 1, ids, toks: ids.map(i => tok.decode([i])) };
    S.smp.id = 'custom'; $('smp-tokstatus').textContent = `${ids.length - 1} tokens`; $('smp-go').disabled = false; render4s();
  } catch (e) {
    TOKP = null; console.warn(e);
    $('smp-tokstatus').textContent = 'Tokenizer unavailable: this needs the page served over HTTP with the tokenizer/ folder next to it.';
  }
}

function renderAll() {
  $('temp-v').textContent = S.temp.toFixed(2); $('m-v').textContent = S.m;
  renderStatebar(); render21(); render22(); render23(); render24(); render25(); render26(); if (S.keys) render27(); else $('panels-27').innerHTML = `<p class="note">Generate the key in 2.3 first.</p>`;
  render3(); render4(); render4s();
}

// ---------- actions ----------
// Decide the viewed position; anything after it is discarded (you are branching).
function setStep(step) { if (animating) return; S.steps = S.steps.slice(0, S.cur); S.steps.push(step); S.draw = null; }
function chooseManual(slot) { if (atEnd()) return; setStep({ slot, how: 'manual' }); S.cur++; renderAll(); }
function plainSample() { if (atEnd()) return; setStep({ slot: sample(candProbs()), how: 'plain' }); renderAll(); }
function stepNext() { if (animating) return; if (curStep()) { S.cur++; S.draw = null; renderAll(); } else if (!atEnd()) { if (S.wm && S.keys) runTournament(ROUND_MS_FAST); else plainSample(); } }
function stepPrev() { if (animating || S.cur === 0) return; S.cur--; S.draw = null; renderAll(); }
function resetSeq() { S.steps = []; S.cur = 0; S.draw = null; renderAll(); }
// ⏩ : fill in every position left in the tree at once (no bracket animation)
function stepToEnd() {
  if (animating) return;
  while (!atEnd()) {
    if (curStep()) { S.cur++; S.draw = null; continue; }
    if (S.wm && S.keys) { const n = node(), probs = candProbs(), w = win(ctxIds()), coins = coinsFor(n, w, S.keys, S.m);
      const draw = Array.from({ length: 2 ** S.m }, () => sample(probs)); const b = runBracket(draw, coins, S.m);
      setStep({ slot: b.winner, how: 'emit', bracket: b, draw }); }
    else setStep({ slot: sample(candProbs()), how: 'plain' });
    S.cur++;
  }
  S.cur = Math.min(S.cur, S.steps.length); renderAll();
}
function setKey(k) {
  const first = !S.keys; S.keys = k; S.draw = null;
  S.steps.forEach(st => { delete st.bracket; delete st.draw; });   // coins changed: old brackets no longer apply
  if (first) S.wm = true;
  renderAll();
}
function doDraw() { if (animating) return; daSkip(); render26(); }
let animating = false;
// 2.7's Run button walks the bracket at reading pace, because watching it resolve is the point
// there. Stepping the sentence along runs the same animation much faster: you still see the
// rounds, but you are not kept waiting for a token.
const ROUND_MS = 620, ROUND_MS_FAST = 130;
function runTournament(ms = ROUND_MS) {
  if (!S.keys || animating || atEnd()) return;
  if (!S.draw || S.draw.length !== 2 ** S.m) doDraw();
  const n = node(); const coins = coinsFor(n, win(ctxIds()), S.keys, S.m);
  const b = runBracket(S.draw, coins, S.m); const draw = S.draw;
  animating = true; $('run-btn').disabled = true;
  let r = 0;
  $('bracket').innerHTML = bracketHTML(b, n, 0);   // show the draw before any round is judged
  const tick = () => {
    r++; $('bracket').innerHTML = bracketHTML(b, n, r);
    if (r <= S.m) setTimeout(tick, ms);
    else { animating = false; setStep({ slot: b.winner, how: 'emit', bracket: b, draw }); renderAll(); }
  };
  setTimeout(tick, ms);
}

// ---------- wiring ----------
$('prompt-select').addEventListener('change', e => { S.pid = e.target.value; resetSeq(); });
function setTokView(v) {
  S.tokView = v;
  document.querySelectorAll('#tok-toggle button, #tok-toggle-2 button').forEach(x => x.classList.toggle('on', x.dataset.v === v));
  render21(); renderStatebar(); render24();
}
$('tok-toggle').querySelectorAll('button').forEach(b => b.addEventListener('click', () => setTokView(b.dataset.v)));
$('coin-inputs').addEventListener('click', e => { const b = e.target.closest('#tok-toggle-2 button'); if (b) setTokView(b.dataset.v); });
$('temp').addEventListener('input', e => { S.temp = +e.target.value; S.draw = null; renderAll(); });
$('statebar-in').addEventListener('click', e => {
  const el = e.target.closest('[data-act]'); if (!el || el.disabled) return;
  if (el.dataset.act === 'prompt') return;                                       // handled on 'change', not 'click'
  if (el.dataset.act === 'prev') stepPrev();
  else if (el.dataset.act === 'fwd') stepNext();
  else if (el.dataset.act === 'wm' && S.keys) { S.wm = !S.wm; renderStatebar(); }   // no keys: the toggle is inert until 2.3
});
$('statebar-in').addEventListener('change', e => { const el = e.target.closest('[data-act="prompt"]'); if (el) { S.pid = el.value; resetSeq(); } });
if (window.ResizeObserver) new ResizeObserver(sbNarrow).observe($('statebar-in'));
$('gen-key').addEventListener('click', () => { setKey(randKey()); window.scrollTo({ top: $('m23').offsetTop - 80 }); });
['reroll-key', 'reroll-key-2', 'reroll-key-3', 'reroll-key-4'].forEach(id => $(id).addEventListener('click', () => { if (S.keys) setKey(randKey()); }));
$('m').addEventListener('input', e => { S.m = +e.target.value; S.draw = null; renderAll(); });
$('draw-btn').addEventListener('click', daStart);
$('draw-back').addEventListener('click', daBack);
$('draw-fwd').addEventListener('click', daFwd);
$('draw-skip').addEventListener('click', daSkip);
$('m2').addEventListener('input', e => { $('m').value = e.target.value; $('m').dispatchEvent(new Event('input')); });
$('hash-cand').addEventListener('change', e => hashSelect(+e.target.value, HA.layer, false));
$('hash-layer').addEventListener('change', e => hashSelect(HA.cand, +e.target.value, false));
$('run-btn').addEventListener('click', () => runTournament());
$('rerun-27').addEventListener('click', () => { if (S.keys) render27(); });
$('proof-toggle').addEventListener('change', () => { if (S.keys) render27(); });
$('doclen').addEventListener('input', e => { S.docLen = +e.target.value; $('doclen-v').textContent = S.docLen; $('len-presets').querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.n === S.docLen)); render4(); });
$('len-presets').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { S.docLen = +b.dataset.n; $('doclen').value = S.docLen; $('doclen-v').textContent = S.docLen; $('len-presets').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); render4(); }));
$('thr').addEventListener('input', e => { S.thr = +e.target.value; $('thr-v').textContent = S.thr.toFixed(3); render4(); });

if ($('smp-select')) {
  $('smp-select').addEventListener('change', e => { S.smp.id = e.target.value; render4s(); });
  $('smp-key').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { S.smp.keyMode = b.dataset.v; render4s(); }));
  $('smp-view').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { S.smp.view = b.dataset.v; render4s(); }));
  if ($('smp-go')) $('smp-go').addEventListener('click', scorePasted);
}
document.querySelectorAll('.glyph-key').forEach(el => el.innerHTML = GLYPH_KEY);
$('seq-reset').innerHTML = TRI_START; $('seq-prev').innerHTML = TRI_L; $('seq-fwd').innerHTML = TRI_R; $('seq-end').innerHTML = TRI_END;
$('seq-reset').addEventListener('click', resetSeq);
$('seq-prev').addEventListener('click', stepPrev);
$('seq-fwd').addEventListener('click', stepNext);
$('seq-end').addEventListener('click', stepToEnd);
$('draw-back').innerHTML = TRI_L; $('draw-fwd').innerHTML = TRI_R; $('draw-skip').innerHTML = TRI_END;
$('run-btn').querySelector('.ico').innerHTML = TRI_R; $('rerun-27').querySelector('.ico').innerHTML = TRI_START;
renderAll();
})();
