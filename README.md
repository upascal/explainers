# Explainers

Single-page, self-contained interactive explainers. Each one lives in its own
folder with the same layout: prose in `src/index.md`, one widget per file in
`src/modules/`, and a `build.py` that writes a dev `site/` and a self-contained
`dist/index.html`.

| folder | explainer |
|---|---|
| [`synthid/`](synthid/) | how tournament-sampling text watermarks (SynthID-Text) work |
| [`word2vec/`](word2vec/) | how word2vec (skip-gram with negative sampling) learns, trained live in the browser |
| [`gpt2-embeddings/`](gpt2-embeddings/) | GPT-2's first layer: tokenizer, token and position tables, and how next-token prediction trains them |

```bash
pip install -r requirements.txt     # markdown-it-py, shared by every build.py
cd word2vec && python build.py      # or any other explainer folder
```

See each folder's README for details.
