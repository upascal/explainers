# GPT-2 embeddings explainer

A single-page, self-contained explainer of GPT-2's first layer: byte-pair encoding, the token
table, the position table, and how next-token prediction trains the table (through the tied
output). A sibling of `../word2vec/`, built the same way and trained on the same 30 sentences.

Two kinds of numbers are on the page:

- **Live.** GPT-2's real tokenizer runs in the browser from its own merge list, and a toy
  one-layer, one-head GPT (6 dimensions, tied embeddings, no LayerNorm, plain SGD) trains on the
  corpus at load. Every toy number on the page is computed there.
- **Precomputed from GPT-2 small** (`src/gpt2data.js`): the merge list, a few real rows, nearest
  neighbours for 14,940 word tokens, every row's distance from the average row, the position
  table's similarity and main directions, GPT-2's next-token guesses for the corpus, and the
  layer-by-layer "plot" / "cell" / … minimal pairs of section 12.

| path | what |
|---|---|
| `src/index.md` | **the prose.** `## 4 Title` headings get their ids/numbers and the TOC automatically; `<!-- module: m4 -->` pulls in a widget |
| `src/modules/*.html` | one interactive widget per file (`m1` … `m12`) |
| `src/model.js` | GPT-2's tokenizer (byte-level BPE) and the toy model: forward pass, hand-written backprop, training stream |
| `src/app.js` | the interactives, the shared play bar, the sticky status bar |
| `src/gpt2data.js` | **generated** by `precompute/precompute.py` |
| `src/template.html`, `src/style.css` | page shell and styles (started from word2vec's) |
| `site/` | **generated** dev output |
| `dist/index.html` | **generated** self-contained page — the publishable file |

```bash
pip install -r ../requirements.txt                 # markdown-it-py
python build.py                                    # src/ -> site/ and dist/
open dist/index.html                               # works straight from disk

pip install -r precompute/requirements.txt         # only to regenerate src/gpt2data.js
python precompute/precompute.py                    # ~1 min; uses the cached `gpt2` checkpoint if present
```

The training stream (sentence order) uses a fixed seed; the initialization seed can be re-rolled
on the page. `CFG` in `src/model.js` holds the toy settings. Learning rates above about 0.06 make
the toy blow up, which is why the page offers 0.05 at most. Changing the corpus or settings can
change whether the clusters form, so re-check the outcomes quoted in the prose (§10 and footnote 6)
after editing them.
