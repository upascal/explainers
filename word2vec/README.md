# word2vec explainer

A single-page, self-contained explainer of how word2vec (skip-gram with negative
sampling) learns. A sibling of the SynthID-Text explainer in the parent directory, built
the same way. Nothing is precomputed: a 5-dimensional toy model trains on a hand-written
30-sentence corpus in the browser, so every number on the page is computed live.

| path | what |
|---|---|
| `src/index.md` | **the prose.** `## 4 Title` headings get their ids/numbers and the TOC automatically; `<!-- module: m4 -->` pulls in a widget |
| `src/modules/*.html` | one interactive widget per section (`m1` … `m10`) |
| `src/model.js` | the toy model: corpus, vocabulary, seeded training stream, updates, PCA |
| `src/app.js` | the interactives, the shared play bar, the sticky status bar |
| `src/template.html`, `src/style.css` | page shell and styles |
| `site/` | **generated** dev output |
| `dist/index.html` | **generated** self-contained page — the publishable file |

```bash
pip install -r ../requirements.txt     # markdown-it-py
python build.py                        # src/ -> site/ and dist/
open dist/index.html                   # works straight from disk
```

The training stream (window shrink, subsampling, negative draws) uses a fixed seed; the
initialization seed can be re-rolled on the page. `CFG` in `src/model.js` holds the toy
settings. Changing the corpus or settings can change whether the clusters form, so re-check
the outcomes quoted in the prose (§8 and footnote 5) after editing them.
