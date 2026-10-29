# How GPT-2 turns text into vectors

<p class="lede">Before a language model can do anything with a sentence, it has to turn the text into numbers. In GPT-2, released by OpenAI in 2019, that is the job of the first layer: it cuts the text into <em>tokens</em> and looks up a list of numbers for each one. Every large language model since starts the same way. This page takes that first layer apart: where the tokens come from, what the table of numbers holds, and how training fills it in.</p>

<p class="note">Everything on this page is computed live in your browser. GPT-2’s real tokenizer runs here, from its own list of 50,000 merges, and a few things are taken from its real table of numbers. To watch the table being learned, a toy model with GPT-2’s structure, shrunk to 6 numbers per token and one layer, trains on the same 30 sentences as the <a href="../word2vec/">word2vec page</a>. From section 5 on, a bar at the top shows the prediction you have selected. Every interactive has the same controls: reset, play, step, and skip to end. The keyboard drives the one you used last: → steps, space plays or pauses, shift + → skips to the end, and shift + ← resets.</p>

<!-- toc -->

## 1 The problem

A model handles numbers, not letters. word2vec’s answer was to give every *word* a list of numbers. But a fixed list of words never covers all text. New words, names, typos, other languages and code keep turning up, and a word that isn’t on the list has no numbers at all. Going down to single letters fixes that, but then a sentence becomes a long string of pieces that each mean almost nothing.

GPT-2 splits the difference. Its text is cut into *tokens*: common words are a single token, rarer ones are built from a few pieces, and anything at all can fall back to single bytes. Its vocabulary has 50,257 tokens, and each one has a number, its *token ID*.

<!-- module: m1 -->

Notice the spaces. A space belongs to the word after it, so <span class="tok"><span class="sp">·</span>coffee</span> (with its space) and “coffee” (at the start of a line) are different tokens, with different IDs. Capitals make different tokens too: <span class="tok"><span class="sp">·</span>coffee</span> is one token, <span class="tok"><span class="sp">·</span>COFFEE</span> is three. A dot marks the space: on this page a token written <span class="tok"><span class="sp">·</span>coffee</span> starts with a space.

The toy model on this page uses GPT-2’s own tokens, only far fewer of them. The settings below compare the two.

| Setting | This page | GPT-2 small |
|---|---|---|
| Vocabulary | 41 tokens, all of them GPT-2’s own | 50,257 tokens |
| Numbers per token | 6 | 768 |
| Positions | 8 | 1,024 |
| Layers | 1, with one attention head | 12, with 12 heads each |
| Parameters | 756 | 124 million, 38.6 million of them in the token embedding table |
| Training text | 30 sentences, 157 tokens | WebText: about 40 GB, 8 million web pages |
| Text per update | one sentence | 512 sequences of 1,024 tokens |
| Initialization | random, standard deviation 0.02; positions 0.01 | the same, in GPT-2’s released code |
| Optimizer | plain gradient descent, learning rate 0.05, decaying linearly | not given in the paper; its predecessor used Adam |
| LayerNorm | none | before each sublayer, and once more after the last layer |

## 2 Building the vocabulary

GPT-2’s vocabulary was not written by hand. It was learned from text by *byte-pair encoding* (BPE), a compression trick from the 1990s. Start with every text written as single bytes. Count every pair of neighbouring pieces, merge the most common pair into a new piece, and repeat. Each merge adds one token to the vocabulary.

Before counting, the text is split into words, numbers and punctuation, with each space kept on the word after it, and pairs are never counted across those pieces. That is why tokens are whole words or parts of words, and why <span class="tok"><span class="sp">·</span>the</span> and “the” come out as different tokens.

<!-- module: m2 -->

GPT-2 started from the 256 possible byte values, ran 50,000 merges over its training text, and added one special token, <code>&lt;|endoftext|&gt;</code>, which marks where one document ends and the next begins: 256 + 50,000 + 1 = 50,257.

To tokenize new text, GPT-2 does not count anything. It replays its list of merges: within each word, it repeatedly applies whichever available merge it learned earliest, until none applies. A token’s ID is simply 256 plus the place of its merge in that list, so a low ID means a merge learned early, from a very common pair.

<!-- module: m2g -->

## 3 Token embeddings

A token ID is just a row number. The first layer of GPT-2 is a table with one row for each of the 50,257 tokens and 768 numbers in each row. For every token in the text, it looks up that token’s row. Nothing is calculated: it is a lookup. A token’s row is called its *embedding*: the 768 numbers in row 6891 are <span class="tok"><span class="sp">·</span>coffee</span>’s embedding. GPT-2’s code calls the whole table `wte`, for word token embeddings; formulas usually call it *E*, and token *j*’s embedding *E*<sub>*j*</sub>. The table holds 38.6 million numbers, almost a third of GPT-2 small’s parameters.

Here is the toy corpus, cut into GPT-2’s tokens, and its vocabulary.

<!-- module: m3 -->

Before training, every number in the table is random, drawn from a normal distribution with a standard deviation of 0.02. Training then adjusts the embeddings, as sections 7 to 9 show. Here is one toy embedding, at any step of its training, next to the same token’s real embedding from GPT-2.

<!-- module: m3v -->

word2vec gave every word two vectors, an input vector and an output vector, kept in two tables. GPT-2 has one token embedding table, and uses it twice: at the start, to look up each token, and at the end, to score which token comes next (section 7). Sharing one table this way is called *weight tying*.

<div class="callout reveal">
  <div class="t">Why can GPT-2 get away with one table, when word2vec needed two?</div>
  <button class="reveal-btn" aria-expanded="false">Reveal answer</button>
  <div class="reveal-body" hidden>In word2vec, one table would score a word against its own vector. That score is a squared length, always positive, so every word would be predicted next to itself. In GPT-2 the vector scored at the end is not the embedding that went in at the start: it has passed through 12 layers, which have mixed in the rest of the text. So a token’s embedding is not automatically its own best match. One table also saves parameters: a second 50,257 × 768 table would add 38.6 million.</div>
</div>

## 4 Position

The layers that come next, attention layers, see the text as a set of vectors with nothing to say which came first. “the dog walks” and “walks the dog” would look the same. So GPT-2 has a second table, of *position embeddings*: one for each position from 0 to 1,023 (`wpe` in GPT-2’s code).

These position embeddings are learned, just like the token embeddings. Nobody told GPT-2 that position 10 is close to position 11. But after training, the embeddings of nearby positions are nearly the same, and they drift apart smoothly with distance.

<!-- module: m4g -->

The table holds position this precisely, but GPT-2 seems to use only a rough version of it. Averaging the embeddings of every 10 neighbouring positions, so that they share one, barely changes GPT-2’s predictions. Its precise sense of order seems to come from the attention layers that follow, which can see which tokens came before.

For each token in the text, GPT-2 adds its position embedding to the token’s embedding. The sum is the vector that goes into the first layer. Here is that addition in the toy model, which has 8 positions.

<!-- module: m4 -->

## 5 Predicting the next token

GPT-2 learns its tables from one task: predict the next token from the prior context, every token up to that point. Every token in a text ends one such context, and makes one prediction. The vector at position *t* can look at the tokens up to *t*, never ahead, and has to say which token comes next. Each sentence here starts and ends with <code>&lt;|endoftext|&gt;</code>, as documents do in GPT-2’s training text, so a sentence of 7 tokens makes 8 predictions: its first token, from <code>&lt;|endoftext|&gt;</code> alone, each token after that, and the end. All of them are trained at once.

<!-- module: m5 -->

Compare word2vec’s window from the other page: it made pairs of neighbouring words, in both directions. Here the context is everything that came before, and the order matters.

## 6 Mixing in the context

The input vector at a position only knows its own token and where it sits. To predict what comes next, it needs the rest of the prior context: after “ hot”, it matters whether “a cup of” came before. That is what the layers between looking up the embeddings and scoring them do.

Each layer has two parts. An *attention* part lets each position draw on the vectors at its own and every earlier position, more on some than on others, and adds the mix to its own vector. An *MLP* part then works on each position’s vector by itself and adds its result too. Because each part adds to the vector instead of replacing it, the vector that comes out of the last layer, the final *hidden state* *h*, is exactly a sum of the input vector and what each part added to it:

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">final hidden state</span><span class="g"><span class="o">=</span><span class="t tv">input vector</span></span><span class="g"><span class="o">+</span><span class="t">attention’s mix</span></span><span class="g"><span class="o">+</span><span class="t">MLP’s change</span></span></span><span class="s"><i>h</i> = <span class="sv"><i>x</i></span> + attn(<i>x</i><sub>0</sub> … <i>x</i><sub><i>t</i></sub>) + mlp(…)</span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

The toy has one such layer, so here it is in full for the selected prediction.

<!-- module: mix -->

GPT-2 repeats this 12 times, with 12 attention parts working side by side in each layer, and normalizes the result once more after the last one. This page doesn’t open those layers up; what matters here is that *h* is built from the input vectors of the whole prior context, so training can reach every one of their embeddings.

## 7 Scoring every token

After GPT-2’s last layer, each position has one vector of 768 numbers. Call it *h*. As section 6 showed, it starts as the input vector of the token at that position, and the layers mix into it information from all the tokens before it. It belongs to one position but carries the whole context. To predict the next token, GPT-2 scores *h* against **every token embedding in the table**, the same table as in section 3, with the dot product, as word2vec did. That makes 50,257 scores, one for each token. The *softmax* turns them into probabilities that add up to 1:

<div class="formula">p<sub><i>j</i></sub> = e<sup>score<sub><i>j</i></sub></sup> / (e<sup>score<sub>1</sub></sup> + e<sup>score<sub>2</sub></sup> + … + e<sup>score<sub><i>n</i></sub></sup>)</div>

A higher score means a higher probability, and raising one token’s probability lowers all the others. On this page an embedding scored this way is a *candidate embedding*, in green; one looked up for a token in the text, as in section 4, is an *input embedding*, in purple. It is the same table either way, and the same token’s embedding can play both parts. word2vec turned each score into a probability on its own, with the sigmoid. The softmax makes the tokens compete.

<!-- module: m6 -->

As in word2vec, the **error** is the right answer minus the guess. Now there is one for every token: the label is 1 for the token that really came next and 0 for every other token, so the error is label − p. At the start of training every probability is about 1/41, the model’s best guess when it knows nothing.

## 8 One update

Now the table learns from the error. One prediction gives token embeddings a change, Δ, in two different ways: as candidates that were scored at the end, and, for the input token, as the input at the start. They use different formulas.

### 8.1 The Δ from being scored

Each token gets a scaling factor, its error times the learning rate, exactly as in word2vec:

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t tg">scaling factor</span><span class="g"><span class="o">=</span><span class="t terr">error</span></span><span class="g"><span class="o">×</span><span class="t">learning rate</span></span></span><span class="s"><span class="sg"><i>g</i></span> = <span class="serr">(label − p)</span> × α</span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

Each token’s embedding then gets a change, Δ, of its scaling factor times *h*:

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t tvo">change to the embedding</span><span class="g"><span class="o">=</span><span class="t tg">scaling factor</span></span><span class="g"><span class="o">×</span><span class="t">h</span></span></span><span class="s">Δ<span class="svo"><i>E</i><sub><i>j</i></sub></span> = <span class="sg"><i>g</i><sub><i>j</i></sub></span> · <i>h</i></span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

- The token that really came next has a positive scaling factor, so its Δ points toward *h*. Once applied, *h* will score it higher next time.
- **Every other token** has a negative scaling factor, so its Δ points away from *h*, in proportion to how likely the model thought it was. word2vec drew 5 random words to push away. GPT-2 pushes away all 50,256 wrong tokens, the likely ones hardest.

<!-- module: m7 -->

### 8.2 The Δ from being the input

The embeddings looked up at the start, the input token’s and the position’s, learn too, but they are harder to reach: they affect the error only through the layers. To find which way each of them should move, GPT-2 works backwards from the error through the same steps that made the prediction, the scores, then each layer’s MLP and attention parts, down to the input vectors. At each step backwards it uses the chain rule of calculus. This is *backpropagation*. It gives a direction for every number in the model. The embeddings scored at the end are the first step back, which is why their change is so simple. word2vec never needed more than that first step, since nothing sits between its vectors and the score.

Further back, backpropagation gives directions for the layers’ own tables too, which this page doesn’t show, and what arrives at the input vector is an *error-reducing nudge*. It is a vector the same size as an embedding: its direction says which way to move the input vector to lower this prediction’s error fastest, and its length how steeply the error falls that way. The input token’s embedding and the position embedding each get a change of the learning rate times that nudge. On this page the whole backwards pass is computed exactly, through the toy’s one layer.

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t tv">change to the input embedding</span><span class="g"><span class="o">=</span><span class="t">learning rate</span></span><span class="g"><span class="o">×</span><span class="t">error-reducing nudge</span></span></span><span class="s">Δ<span class="sv"><i>E</i><sub><i>x</i></sub></span> = α · ∇</span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

<!-- module: m7b -->

## 9 Adding up an update

A sentence makes several predictions, and every one of them moves every token embedding a little. The model works out all of these parts first, with the table as it stands, then adds them up and changes each embedding once. This is one training step.

A token’s embedding can get three kinds of change from one sentence:

| Where the token is | Its change | The size depends on |
|---|---|---|
| it came next, at some position | a pull toward *h*: (1 − p) × α × *h* | how surprised the model was: small p, big pull |
| it didn’t come next, at every other position | a push away from *h*: −p × α × *h* | how tempting it was: almost none if p is near 0 |
| it is the input, at some position | α × the error-reducing nudge there | how much that input affected the errors |

A token that isn’t in the sentence at all gets only the second kind.

Only now does anything change. Each embedding moves once, by all of its Δs added up:

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">new embedding</span><span class="g"><span class="o">=</span><span class="t">embedding</span></span><span class="g"><span class="o">+</span><span class="t tvo">every change from being scored</span></span><span class="g"><span class="o">+</span><span class="t tv">every change from being the input</span></span></span><span class="s"><i>E</i><sub><i>j</i></sub> ← <i>E</i><sub><i>j</i></sub> + <span class="svo">Σ<sub><i>t</i></sub> <i>g</i><sub><i>t</i>,<i>j</i></sub> · <i>h</i><sub><i>t</i></sub></span> + <span class="sv">Σ<sub><i>t</i> : input is <i>j</i></sub> α · ∇<sub><i>t</i></sub></span></span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

<!-- module: m8 -->

An embedding gets parts from three roles: as the right answer at some positions, as a wrong answer at all the others, and as an input wherever its token appears in the sentence. One GPT-2 step adds up the parts from 512 × 1,024 = 524,288 predictions.

## 10 Two tokens drawing together

“coffee” and “tea” never appear in the same sentence. But the same tokens come before them, <span class="tok"><span class="sp">·</span>hot</span>, <span class="tok"><span class="sp">·</span>the</span> and <span class="tok"><span class="sp">·</span>my</span>, and the same tokens follow, <span class="tok"><span class="sp">·</span>with</span>, <span class="tok"><span class="sp">·</span>every</span> and <span class="tok"><span class="sp">·</span>is</span>. So the vectors *h* that should predict them are alike, which pulls both embeddings the same way when they are scored, and as inputs both embeddings should lead to the same predictions. Their embeddings drift together, through the same kind of shared company word2vec used.

Any two embeddings lie together in one flat plane, so they can be drawn at their true lengths with the true angle between them. Drag through training, and try a token that shares no company with <span class="tok"><span class="sp">·</span>coffee</span>, such as <span class="tok"><span class="sp">·</span>dog</span>.

<!-- module: m9 -->

## 11 Training at scale

Now run the toy over the whole corpus, 120 times, one sentence per step: 3,600 steps. The *loss* is the error as the model measures it: −log p of the token that really came next, averaged over the sentence’s predictions. It starts at log 41 ≈ 3.7, a uniform guess among 41 tokens.

<!-- module: m10l -->

The loss cannot fall to zero, because the corpus doesn’t always say what comes next. After <span class="tok"><span class="sp">·</span>hot</span>, the corpus has <span class="tok"><span class="sp">·</span>coffee</span> twice, <span class="tok"><span class="sp">·</span>tea</span> twice, <span class="tok"><span class="sp">·</span>drink</span> once and the end of the sentence twice, so the best possible guess there splits its probability.

To see the table itself, the map below projects the 6-number embeddings down to 2 numbers with PCA, which keeps the two directions in which the embeddings spread out most. The picture is approximate; the neighbour list beside it is computed in all 6 dimensions. Drag the slider under it to any point in training; playing above moves it too.

<!-- module: m10 -->

Watch <span class="tok"><span class="sp">·</span>coffee</span> and <span class="tok"><span class="sp">·</span>tea</span>, and <span class="tok"><span class="sp">·</span>dog</span> and <span class="tok"><span class="sp">·</span>cat</span>. The sentence-starting tokens “the”, “a”, “i”, “my” and “pour” also gather, apart from <span class="tok"><span class="sp">·</span>the</span>, <span class="tok"><span class="sp">·</span>a</span> and <span class="tok"><span class="sp">·</span>my</span> with their spaces: they are different tokens, used in different places. Turn on the position embeddings to see where the 8 positions sit among the tokens.

## 12 Inside GPT-2’s table

The real table, after training on 40 GB of text, holds the same kind of structure at a far larger scale. Here are the nearest neighbours of any word in it, by the angle between embeddings. The embeddings themselves come straight from GPT-2’s released weights.

<!-- module: m11 -->

Many of the neighbours are the same word in another form: capitalized, plural, or without its space. GPT-2 has to learn for itself that <span class="tok"><span class="sp">·</span>coffee</span> and <span class="tok"><span class="sp">·</span>Coffee</span> mean the same thing, and puts their embeddings close together. The rest are related words: for <span class="tok"><span class="sp">·</span>coffee</span>, <span class="tok"><span class="sp">·</span>espresso</span>, <span class="tok"><span class="sp">·</span>caffeine</span> and, as in the toy, <span class="tok"><span class="sp">·</span>tea</span>.

The table has other habits. Its embeddings are not spread evenly in every direction: they all share a large common offset, so two random embeddings have a cosine similarity of about 0.27 instead of 0. Subtract the average embedding and that drops to about 0. And some embeddings barely moved at all. A token’s embedding only gets the pull toward *h* when that token comes next, which some tokens almost never did. The tokenizer seems to have been built from different text than the model was trained on, so it has tokens like <span class="tok"><span class="sp">·</span>TheNitrome</span>, <span class="tok"><span class="sp">·</span>RandomRedditor</span> and <span class="tok"><span class="sp">·</span>externalToEVA</span>: strings common in the tokenizer’s text that hardly appear in the model’s. They were found in 2023 as “glitch tokens”, which confuse the model. Their embeddings got almost no training, and sit close to the average embedding.

<!-- module: m11g -->

## 13 The table is static

A token’s embedding is the same wherever the token appears. “plot” gets the same numbers in “the garden plot” and in “the secret plot”. Telling the two apart is the job of the 12 layers that follow: each layer mixes information from the other tokens into each vector. Coming out of the tables the two “plot” vectors are identical; layer by layer they move apart.

<!-- module: m12 -->

So the first layer is a starting point. It gives each token what is true of it everywhere, and the rest of the network builds what it means here.

<footer>
  <h4 class="fn-h">Footnotes</h4>
  <ol class="fn">
    <li>Radford, Wu, Child, Luan, Amodei and Sutskever, <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">“Language Models are Unsupervised Multitask Learners”</a> (2019). GPT-2: byte-level BPE with merges kept within character categories, a 1,024-token context, batches of 512, WebText.</li>
    <li>Sennrich, Haddow and Birch, <a href="https://arxiv.org/abs/1508.07909">“Neural Machine Translation of Rare Words with Subword Units”</a>, ACL (2016), brought byte-pair encoding, first a compression method (Gage, 1994), to language modelling.</li>
    <li>Press and Wolf, <a href="https://arxiv.org/abs/1608.05859">“Using the Output Embedding to Improve Language Models”</a>, EACL (2017), on tying the input and output tables. GPT-2’s released code computes its scores with the token embedding table itself.</li>
    <li>Wang and Chen, <a href="https://arxiv.org/abs/2010.04903">“What Do Position Embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding”</a>, EMNLP (2020). The same comparison of every pair of positions as in section 4, for GPT-2, BERT and RoBERTa: only GPT-2’s shows clear periodic patterns. A weighted sum of GPT-2’s position embeddings recovers the position to within about 1 on average, as the readout in section 4 does.</li>
    <li>Yedidia, <a href="https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix">“GPT-2’s positional embedding matrix is a helix”</a> (2023): three directions hold about 90% of the position embeddings’ variation, and plotted in them the 1,024 embeddings trace a helix, with positions 0 and 1,023 as outliers. And <a href="https://www.lesswrong.com/posts/zRA8B2FJLtTYRgie6/the-positional-embedding-matrix-and-previous-token-heads-how">“The positional embedding matrix and previous-token heads: how do they actually work?”</a> (2023), on averaging neighbouring position embeddings with little effect on GPT-2’s predictions.</li>
    <li>Rumbelow and Watkins, <a href="https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation">“SolidGoldMagikarp (plus, prompt generation)”</a> (2023), on GPT-2 and GPT-3’s “glitch tokens”, such as <span class="tok"><span class="sp">·</span>SolidGoldMagikarp</span> (a single token, as section 2 shows), and their embeddings near the centre of the table.</li>
    <li>Ethayarajh, <a href="https://arxiv.org/abs/1909.00512">“How Contextual are Contextualized Word Representations?”</a>, EMNLP (2019), on how GPT-2’s vectors share a common direction and grow more context-specific in higher layers. Section 13 subtracts each layer’s average vector, taken over a dozen ordinary sentences, before comparing, so the shared offset doesn’t make every pair look alike. Its layer 12 is the last layer’s output before GPT-2’s final LayerNorm.</li>
    <li>The toy model, corpus and seeds are this page’s own. The toy adds up its loss over a sentence’s predictions, where GPT-2 averages over a batch; that only changes the scale of the learning rate. It uses GELU in its MLP, as GPT-2 does, and no LayerNorm. Sections 6 to 9 look at the model just before the selected sentence’s own update, in whichever of its 120 passes is chosen; sections 3, 4, 10 and 11 share one training step, from 0 to 3,600. Across 100 initialization seeds, <span class="tok"><span class="sp">·</span>coffee</span> and <span class="tok"><span class="sp">·</span>tea</span> ended up as each other’s nearest neighbours 99 times, and <span class="tok"><span class="sp">·</span>dog</span> and <span class="tok"><span class="sp">·</span>cat</span> 99 times too.</li>
  </ol>
</footer>
