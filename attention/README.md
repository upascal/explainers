# Attention explainer

A single-page, self-contained explainer of attention in GPT-2: one head taken apart on a toy
model (query, key, score, causal mask, softmax, value, the residual add), watched while it
trains, and then GPT-2 small's own 144 heads, with its previous-token head, induction heads and
attention sink. A sibling of `../gpt2-embeddings/`, and built the same way.

Two kinds of numbers are on the page:

- **Live.** The toy model from `../gpt2-embeddings/` (one layer, one head, 6 dimensions, the same 30
  sentences and training order) trains in the browser at load, from initialization seed 2. Its
  code is not copied: `build.py` pulls in `../gpt2-embeddings/src/model.js` and `gpt2data.js`
  (which carries GPT-2's merge list for the tokenizer).
- **Precomputed from GPT-2 small** (`src/attndata.js`): every head's attention weights on three
  texts, GPT-2's next-token chances on the repeated random words, and per-head scores
  (previous-token, induction, weight on the first token, the first token's value length).

| path | what |
|---|---|
| `src/index.md` | **the prose.** `## 4 Title` headings get their ids/numbers and the TOC automatically; `<!-- module: m4 -->` pulls in a widget |
| `src/modules/*.html` | one interactive widget per file |
| `src/toy.js` | one position's way through the toy's head, with attention as learned, set by hand, or off |
| `src/app.js` | the interactives |
| `src/attndata.js` | **generated** by `precompute/precompute.py` |
| `src/template.html`, `src/style.css` | page shell and styles (started from gpt2-embeddings') |
| `site/`, `dist/index.html` | **generated**: dev output, and the self-contained publishable page |

```bash
pip install -r ../requirements.txt                 # markdown-it-py
python build.py                                    # src/ -> site/ and dist/
open dist/index.html                               # works straight from disk

pip install -r precompute/requirements.txt         # only to regenerate src/attndata.js
python precompute/precompute.py                    # seconds; uses the cached `gpt2` checkpoint if present
```

Changing the toy model in `../gpt2-embeddings/src/model.js`, or the seed here, can change what the
head learns. Re-check the numbers quoted in sections 1, 8 and the last footnote after doing so
(seed 2: " can" puts about 87% / 83% on " dog" / " cat"; across 100 seeds, 79 get both answers
right, 54 look mostly at the animal, 47 have a clean value, 14 learn "likes to").
