(function () {
// ---------- presentation mode ----------
// Press P (or open #present, #present/N) to show the page as 16:9 slides. Each heading gets a
// title slide, and each widget, callout and table a slide of its own. The live nodes are moved
// into the slides and back on Esc, so every widget keeps its state. The prose after a slide, up
// to the next one, becomes its speaker notes (N opens them in a second window).
// Shared by the explainers. The page sets it up on <main>: data-statebar-from="m3" shows the
// status bar from that widget's slide on; without it the page handles the bar itself, on the
// 'present:slide' event that every slide change sends (detail: { i, slide, slides }, or null on exit).
// The page's app.js may expose window.EXPLAINER = { BARS, active() } for the play bars.
const main = document.querySelector('main'), body = document.body, root = document.documentElement;
const PCFG = { statebarFrom: main.dataset.statebarFrom || null };
const on = () => body.classList.contains('present');
let deck, stage, slides = [], cur = 0, moved = [], notesWin = null, t0 = 0, clock = null, sbFrom = Infinity;

// "Present" button under the contents
const toc = document.querySelector('nav.toc');
if (toc) toc.insertAdjacentHTML('afterend', '<p class="note present-cta"><button class="present-btn">Present as slides</button> or press P</p>');
document.querySelector('.present-btn')?.addEventListener('click', () => enter());

function collect() {
  const out = []; let sec = null, say = null, last = null;
  const add = (kind, el, o = {}) => { last = { kind, anchor: el, els: kind === 'title' ? [] : [el], sec, say, notes: [], ...o }; say = null; out.push(last); };
  for (const el of main.children) {
    if (el.matches('.slide-say')) say = el.innerHTML;
    else if (el.matches('h1')) add('title', el, { title: el.innerHTML, num: '' });
    else if (el.matches('h2, h3')) {
      const n = el.querySelector('.num');
      sec = { id: el.id, num: n ? n.textContent : '', title: [...el.childNodes].filter(c => c !== n).map(c => c.textContent).join('') };
      add('title', el, { title: sec.title, num: sec.num });
    }
    else if (el.matches('.module')) add('widget', el);
    else if (el.matches('.callout, table')) add('box', el);
    else if (el.matches('.formula') && last) last.els.push(el);   // formulas go up on the slide they follow
    else if (el.matches('nav.toc, footer, .present-cta')) continue;
    else if (last) last.notes.push(el);
  }
  return out;
}

const label = s => {
  if (s.kind === 'title') return (s.num ? s.num + ' ' : '') + s.title.replace(/<.*?>/g, '');
  const h = s.anchor.querySelector('.m3title, .t, h4'), sec = s.sec ? s.sec.num + ' ' + s.sec.title : '';
  return h ? `${sec} · ${h.textContent}` : sec || 'Table';
};

function enter(n) {
  if (on()) return go(n ?? cur);
  slides = collect();
  if (n == null) {   // start at the slide being read
    const y = innerHeight * 0.4; n = 0;
    slides.forEach((s, i) => { if (s.anchor.getBoundingClientRect().top <= y) n = i; });
  }
  deck = document.createElement('div'); deck.id = 'deck';
  deck.innerHTML = '<div class="stage"><div class="sb-slot"></div><div class="slides"></div><div class="pfoot"></div></div>';
  stage = deck.firstChild;
  const park = el => { const ph = document.createComment('slide'); el.before(ph); moved.push([el, ph]); return el; };
  slides.forEach((s, i) => {
    const node = document.createElement('section');
    node.className = 'pslide k-' + s.kind; node.setAttribute('aria-label', label(s));
    const say = s.say ? `<p class="psay">${s.say}</p>` : '';
    if (s.kind === 'title') node.innerHTML = `<div class="pbody">${s.num ? `<div class="pnum">${s.num}</div>` : ''}<h2 class="ptitle">${s.title}</h2>${say}</div>`;
    else node.innerHTML = `<header class="phead">${s.sec ? `<span class="num">${s.sec.num}</span>${s.sec.title}` : ''}</header><div class="pbody"></div>${say}`;
    const pb = node.querySelector('.pbody');
    s.els.forEach(el => pb.append(park(el)));
    if (PCFG.statebarFrom && s.els.some(el => el.id === PCFG.statebarFrom)) sbFrom = Math.min(sbFrom, i);
    s.node = node; stage.querySelector('.slides').append(node);
  });
  const sb = document.getElementById('statebar');
  if (sb) stage.querySelector('.sb-slot').append(park(sb));
  body.append(deck); body.classList.add('present');
  addEventListener('resize', fit); fit(); go(n);
}

function exit() {
  if (!on()) return;
  const anchor = slides[cur].anchor;
  moved.forEach(([el, ph]) => ph.replaceWith(el)); moved = [];
  body.classList.remove('present'); deck.remove(); deck = null; root.style.fontSize = ''; sbFrom = Infinity;
  removeEventListener('resize', fit);
  if (document.fullscreenElement) document.exitFullscreen();
  history.replaceState(null, '', location.pathname + location.search);
  dispatchEvent(new Event('resize'));
  scrollTo({ top: anchor.getBoundingClientRect().top + scrollY - 70, behavior: 'instant' });
  if (notesWin && !notesWin.closed) notesWin.close();
  document.dispatchEvent(new CustomEvent('present:slide', { detail: null }));
}

// a 16:9 stage that fits the window; everything inside is sized in rem, so it scales with it
let fitting = false;
function fit() {
  if (fitting) return;   // our own resize event below
  const w = Math.min(innerWidth, innerHeight * 16 / 9);
  stage.style.width = w + 'px'; stage.style.height = w * 9 / 16 + 'px';
  root.style.fontSize = (w / 100 * 1.35).toFixed(2) + 'px';
  fitting = true; dispatchEvent(new Event('resize')); fitting = false;   // widgets that measure themselves re-render at the new size
}

function go(n) {
  cur = Math.max(0, Math.min(slides.length - 1, n));
  slides.forEach((s, i) => s.node.classList.toggle('on', i === cur));
  const E = window.EXPLAINER, sl = slides[cur].node;
  if (E) E.BARS.forEach(b => { if (b.playing && !sl.contains(b.host)) b.stop(); });
  if (PCFG.statebarFrom) document.getElementById('statebar')?.classList.toggle('shown', cur >= sbFrom && slides[cur].kind !== 'title');
  stage.querySelector('.pfoot').textContent = `${cur + 1} / ${slides.length}`;
  history.replaceState(null, '', `#present/${cur + 1}`);
  renderNotes();
  document.dispatchEvent(new CustomEvent('present:slide', { detail: { i: cur, slide: slides[cur], slides } }));
}

// the play bar on this slide: the one used last if it is here, else the first visible one
function bar() {
  const E = window.EXPLAINER; if (!E) return null;
  const sl = slides[cur].node, bars = E.BARS.filter(b => sl.contains(b.host) && b.host.offsetParent);
  return bars.includes(E.active()) ? E.active() : bars[0] || null;
}

function key(e) {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const typing = e.target.closest && e.target.closest('input:not([type=checkbox]), select, textarea');
  if (!on()) { if ((e.key === 'p' || e.key === 'P') && !typing) { e.preventDefault(); enter(); } return; }
  if (typing && e.key !== 'Escape') return;
  if (e.target.closest && e.target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
  const b = bar(), k = e.key;
  let done = true;
  if (k === 'PageDown' || k === 'ArrowDown') go(cur + 1);
  else if (k === 'PageUp' || k === 'ArrowUp') go(cur - 1);
  else if (k === 'ArrowRight' && e.shiftKey) { if (b) b.act('end'); }
  else if (k === 'ArrowLeft' && e.shiftKey) { if (b) b.act('reset'); }
  else if (k === 'ArrowRight') { if (b && b.o.canStep()) b.act('step'); else go(cur + 1); }   // a clicker steps the widget, then moves on
  else if (k === 'ArrowLeft') go(cur - 1);
  else if (k === ' ') { if (b) b.act('play'); else go(cur + 1); }
  else if (k === 'Home') go(0);
  else if (k === 'End') go(slides.length - 1);
  else if (k === 'Escape' || k === 'p' || k === 'P') exit();
  else if (k === 'n' || k === 'N') openNotes();
  else if (k === 'r' || k === 'R') slides[cur].node.querySelector('.reveal-btn')?.click();
  else if (k === 'f' || k === 'F') document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen?.();
  else done = false;
  if (done) e.preventDefault();
}
document.addEventListener('keydown', key);

// links to a section (the status bar has them) go to its title slide
document.addEventListener('click', e => {
  const a = on() && e.target.closest('a[href^="#s"]'); if (!a) return;
  const i = slides.findIndex(s => s.kind === 'title' && s.sec && '#' + s.sec.id === a.getAttribute('href'));
  if (i >= 0) { e.preventDefault(); go(i); }
});

// ---------- speaker notes, in a second window ----------
const NOTES_CSS = `body.pnotes { margin: 0; padding: 22px 28px 40px; font-size: 19px; background: var(--page); }
.pnotes .top { display: flex; justify-content: space-between; font-family: var(--sans); font-size: 14px; color: var(--muted); }
.pnotes .top b { color: var(--ink); }
.pnotes h2 { font-size: 26px; margin: 10px 0 14px; }
.pnotes .txt { max-width: 46em; }
.pnotes .txt table { font-family: var(--sans); font-size: 14px; border-collapse: collapse; }
.pnotes .txt td, .pnotes .txt th { padding: 3px 6px; border-bottom: 1px solid var(--grid); text-align: left; }
.pnotes .none { font-family: var(--sans); color: var(--muted); font-size: 15px; }
.pnotes .next { font-family: var(--sans); font-size: 15px; color: var(--muted); border-top: 1px solid var(--grid); padding-top: 10px; margin-top: 24px; }
.pnotes .keys { font-family: var(--sans); font-size: 13px; color: var(--muted); margin-top: 6px; }`;

function openNotes() {
  if (notesWin && !notesWin.closed) { notesWin.focus(); return; }
  notesWin = open('', 'explainer-notes', 'width=760,height=860');
  if (!notesWin) return;
  let css = '';
  for (const s of document.styleSheets) { try { css += [...s.cssRules].map(r => r.cssText).join('\n'); } catch (_) {} }   // cross-origin font sheets can't be read
  const d = notesWin.document;
  d.open();
  d.write(`<!doctype html><html><head><meta charset="utf-8"><title>Speaker notes</title><style>${css}\n${NOTES_CSS}</style></head><body class="pnotes"><div class="top"><span><b class="cnt"></b> &nbsp; ${esc(document.title)}</span><span class="clock"></span></div><h2 class="now"></h2><div class="txt"></div><p class="next"></p><p class="keys">PageDown / PageUp: next / previous slide · →: step the widget · Esc: leave presentation</p></body></html>`);
  d.close();
  d.addEventListener('keydown', key);
  t0 = t0 || Date.now();
  clearInterval(clock); clock = setInterval(tickClock, 1000);
  renderNotes();
}
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
function tickClock() {
  if (!notesWin || notesWin.closed) { clearInterval(clock); return; }
  const s = Math.floor((Date.now() - t0) / 1000);
  notesWin.document.querySelector('.clock').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
function renderNotes() {
  if (!notesWin || notesWin.closed || !slides.length) return;
  const d = notesWin.document, s = slides[cur], nx = slides[cur + 1];
  d.querySelector('.cnt').textContent = `${cur + 1} / ${slides.length}`;
  d.querySelector('.now').textContent = label(s);
  d.querySelector('.txt').innerHTML = s.notes.map(el => el.outerHTML).join('\n') || '<p class="none">No notes for this slide.</p>';
  d.querySelector('.next').textContent = nx ? 'Next: ' + label(nx) : 'Last slide';
  tickClock();
}

// dev check: slides whose content doesn't fit (PRESENT.check() in the console). Each slide is
// measured while showing, since the status bar takes room on some.
function check() {
  const was = cur;
  const out = slides.map((s, i) => {
    go(i); const b = s.node.querySelector('.pbody');
    return { slide: i + 1, label: label(s), over: b.scrollHeight - b.clientHeight, wide: b.scrollWidth - b.clientWidth };
  }).filter(r => r.over > 1 || r.wide > 1);
  go(was); return out;
}
window.PRESENT = { enter, exit, go, check, get slides() { return slides; } };

const fromHash = () => { const m = location.hash.match(/^#present(?:\/(\d+))?$/); if (m) enter(m[1] ? m[1] - 1 : 0); };
addEventListener('hashchange', fromHash);
setTimeout(fromHash, 0);
})();
