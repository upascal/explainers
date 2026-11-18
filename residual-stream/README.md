# Residual stream explainer

A single-page, self-contained explainer of GPT-2's residual stream: every layer reads from and
adds to one vector per position, so what one attention head reads and what it writes can be
compared directly. It covers splitting the final vector into its 25 parts, the OV circuit
(W_V W_O) and why its factors alone mean nothing, read and write subspaces against chance,
heads as token-to-token maps (copying and suppression), composition between heads, and which
heads write the answer on real text. It follows `../attention/`, built the same way.

Everything on the page is precomputed from GPT-2 small (`src/resdata.js`); nothing is trained
in the browser. Weights are used as GPT-2 applies them, to row vectors (`x @ W`), so a head reads
through col(W_V) and writes into row(W_O). Each head's LayerNorm gain is folded into W_Q, W_K
and W_V; the normalization itself is applied to inputs where an input is used. The final
LayerNorm's scale is frozen at its value in the real run, which makes the split of a logit into
parts exact (the precompute asserts it).

| path | what |
|---|---|
| `src/index.md` | **the prose.** `## 4 Title` headings get their ids/numbers and the TOC automatically; `<!-- module: m4 -->` pulls in a widget |
| `src/modules/*.html` | one interactive widget per file |
| `src/app.js`, `src/style.css`, `src/template.html` | the interactives, styles (started from attention's) and page shell |
| `src/resdata.js` | **generated** by `precompute/precompute.py`, with the text of every token shown |
| `site/`, `dist/index.html` | **generated**: dev output, and the self-contained publishable page |

```bash
pip install -r ../requirements.txt                 # markdown-it-py
python build.py                                    # src/ -> site/ and dist/
open dist/index.html                               # works straight from disk

pip install -r precompute/requirements.txt         # only to regenerate src/resdata.js
python precompute/precompute.py                    # a few minutes on CPU; prints every number the prose quotes
```

The precompute prints the numbers the prose relies on. Re-check the prose against them after
changing the texts, the example words or the token sets.
