# SynthID-Text explainer

A single-page, self-contained explainer of how tournament-sampling text
watermarks (SynthID-Text) work, aimed at policy and journalism readers.
Everything on the page is computed client-side from precomputed model data.

Companion to the `watermark-testing` repo, whose n=500 Gemma-2-9b entity-shift
results are the data for the findings page.

## Layout

| path | what |
|---|---|
| `src/index.md` | **the prose.** Markdown; `## 2.4 Title` headings get their ids/numbers and the TOC automatically; `<!-- module: m21 -->` pulls in a widget |
| `src/modules/*.html` | one interactive widget per file (`m21` … `m27`, `m4s`), placed by the markers above |
| `src/app.js` | the interactive code (hash, tournament, detector, charts) |
| `src/template.html`, `src/style.css` | page shell and styles |
| `src/findings.html` | the research findings page (hand-written HTML): 9B entity-shift results, the VIIIth-Olympiad case, entropy-shape argument |
| `precompute/precompute.py` | walks Gemma-2-2b's top-8 continuations to depth 3 for each prompt; writes `site/data.json` (raw logits, so temperature is client-side) |
| `precompute/samples.py` | tokenises four public-domain passages (`precompute/texts/`) and generates Gemma-2-2b continuations in the same registers, plain and watermarked under a fixed demo key, with a bit-exact Python port of the page's hash/tournament; writes `site/samples.json` |
| `precompute/probe.py` | quick first-step top-12 for candidate prompts, used to pick the prompt set |
| `site/` | **generated** dev output (`index.html`, `findings.html`, `style.css`, `app.js`) plus the data files above; serve this directory |
| `dist/` | **generated** self-contained `index.html` / `findings.html` (CSS and JSON inlined) — the publishable files |
| `build.py` | `src/` → `site/` and `dist/` |

## Working on it

```bash
pip install -r ../requirements.txt                     # markdown-it-py
python build.py                                        # src/ -> site/ (dev) and dist/ (self-contained)
python -m http.server 8765 --directory site            # then open http://localhost:8765/
```

Edit the text in `src/index.md`, run `python build.py`, refresh. Don't edit
`site/index.html` or `dist/` by hand — they are overwritten.

Regenerating the model data (needs `google/gemma-2-2b`, Apple-silicon MPS by default):

```bash
pip install -r precompute/requirements.txt
python precompute/precompute.py --out site/data.json   # ~1 min
python precompute/samples.py                           # ~10 min; --selftest prints the coin-table fingerprint
```

The `site/` pages must be served over HTTP for their dev fetches to work; the
`dist/` pages open from disk with no server at all.

## Publishing

The page is served at [ulyssespascal.com/apps/synthid](https://ulyssespascal.com/apps/synthid).
The site repo (`upascal/ulyssespascal-com`) does not carry a copy: its
`external-apps.json` names this repo and a tag, and `scripts/fetch-apps.mjs`
downloads that tag's `dist/` into `public/apps/synthid/` at build time.

`dist/` is committed here, so a release is *build, commit, then tag* — tagging
without rebuilding ships a stale page:

```bash
python build.py
git commit -am "..."            # must include the dist/ changes
git tag -a v1.1.0 -m "..." && git push --follow-tags
```

Then bump `ref` for the `synthid` app in the site's `external-apps.json` and
push that; Vercel redeploys and picks up the new tag.
