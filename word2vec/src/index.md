# How word2vec learns

<p class="lede">In 2013 a team at Google released <a href="https://code.google.com/archive/p/word2vec/">word2vec</a>, a small program that reads a lot of text and gives every word a list of numbers. Words used in similar ways end up with similar lists. The idea underlies much of modern language technology, including the first layer of every large language model. This page takes the training procedure apart one step at a time.</p>

<p class="note">Everything on this page is computed live in your browser. A tiny model with 5 dimensions trains on a hand-written corpus of 30 sentences, so the numbers on screen are the real numbers. Each section builds on the last, and they all share one model: the vectors created in section 2 are the ones trained in 6, 7 and 8. The bar at the top shows where training stands in section 8. Every interactive has the same controls: reset, play, step, and skip to end. The keyboard drives the one you used last: → steps, space plays or pauses, shift + → skips to the end, and shift + ← resets.</p>

<!-- toc -->

## 1 The problem

Computers handle numbers, not meanings. To do anything useful with language, a program needs some numeric description of a word that captures what the word means. The simplest approach gives each word an arbitrary ID: “coffee” is 17, “tea” is 18. But the IDs say nothing about meaning. 17 is no closer to 18 than it is to 9,000.

word2vec starts from an old observation in linguistics, the *distributional hypothesis*: words that appear in similar contexts tend to have similar meanings. As the linguist J.R. Firth put it, “you shall know a word by the company it keeps.” You may never have seen “coffee” and “tea” in the same sentence, but you have seen both next to “hot”, “cup” and “drink”. Their shared company is the clue.

You can use the same clue yourself. Each sentence below, taken from the toy corpus on this page, is missing a word.

<!-- module: m1 -->

“purr” and “fur” each fit one sentence, next to “cat” and “dog”. “coffee” and “tea” fit either of the last two equally well, because they keep the same company.

word2vec has only this kind of evidence: which words appear near which. In the corpus’s 14 sentences about coffee and tea, the two words never appear together and have nearly the same neighbours. By the end of the page, the model will have placed “coffee” and “tea” close together through those shared neighbours alone.

The settings below compare the toy model with the real thing. They are small enough to train in under a second, and the training procedure is the same one the real program uses.

| Setting | This page | word2vec (paper / code) |
|---|---|---|
| Vocabulary | 36 words | 692K words (paper); 3M words and phrases (Google News vectors) |
| Corpus | 30 sentences, 156 words | about 1B words (paper); about 100B (Google News vectors) |
| Dimensions | 5 | 300 (paper, Google News vectors); the code defaults to 100 |
| Window | up to 5, randomly shrunk | up to 5, randomly shrunk |
| Negative samples *k* | 5 | 5 by default; the paper suggests 5–20 for small data, 2–5 for large |
| Learning rate | 0.025, decaying linearly | 0.025 for skip-gram, 0.05 for CBOW |
| Subsampling threshold *t* | 0.01 | 10⁻⁵ (paper’s formula); 10⁻³ (code’s formula and default) |
| Minimum count | 1 | 5 |
| Passes over the corpus | 200 | 5 by default |

## 2 The vocabulary and two tables

Training begins with bookkeeping. The program splits the corpus into words, counts each one, and builds the *vocabulary*: the list of every distinct word. The real program drops words that appear fewer than five times. The toy corpus keeps them all.

<!-- module: m2 -->

Then every word gets **two** vectors, each a row of numbers, one per dimension. This page uses 5 dimensions, so each vector holds 5 numbers. The original paper used 300.

The two vectors are:

- The **input vector *v*** describes the word when it is the center of attention. Each number in the vector starts out random, drawn uniformly from the range ±0.5/*d*, where *d* is the number of dimensions. That is ±0.1 here, and ±0.0017 at 300 dimensions.
- The **output vector *v*′** describes the word when it appears in another word’s context. It starts out as all zeros.

Here is one input vector, as it starts and as it ends up after training.

<!-- module: m2v -->

All the input vectors together form one table, and all the output vectors form another. Training adjusts both tables. At the end, the input table is usually the one people keep and call “the word embeddings.”

<div class="callout reveal">
  <div class="t">Why two tables?</div>
  <button class="reveal-btn" aria-expanded="false">Reveal answer</button>
  <div class="reveal-body" hidden>With a single table, a word would be scored against itself using the same vector twice. The score would be |<i>v</i>|², a squared length, which is never negative, so the probability of section 5 would never drop below 0.5. Every word would be predicted to sit next to itself. Two tables also change what “close” means. A word’s input vector is pulled toward the output vectors of its neighbours. So two words end up with similar input vectors when they share neighbours, not when they sit side by side.</div>
</div>

## 3 The window

Training walks through the text **in order**, word by word, and each word takes a turn as the *center*. The words around it are its *context*. word2vec looks up to five words to either side. But for each center word, it first shrinks the window at random to somewhere between 1 and 5. A word right next to the center is inside the window every time; a word five places away only one time in five. So nearer words count more.

Before the window is placed, very frequent words are randomly dropped. Words like “the” and “a” appear next to everything and say little about any particular word. Dropping some of their occurrences speeds up training and gives the rarer, more informative words more room in each window. In the paper, a word that makes up a fraction *f* of the corpus is dropped with probability 1 − √(*t*/*f*), with *t* around 10⁻⁵. The released code uses a slightly different formula, with a default *t* of 10⁻³. This page uses the code’s formula, with *t* raised to 0.01 so that it has an effect on so small a corpus.

Each (center, context) pair inside the window becomes one training example.

<!-- module: m3 -->

During training the window slides by itself: every word in the corpus takes a turn as the center, once per pass, 200 passes in all. The window size and the dropped words are drawn fresh each time, so the same sentence yields different pairs on every pass.

## 4 The dot product

For each pair, the model needs a score: a single number saying how strongly it believes the two words belong together. word2vec uses the simplest score there is, the *dot product*. Multiply the center word’s input vector and the context word’s output vector dimension by dimension, then add up the results.

<!-- module: m4 -->

When the two vectors point the same way, most of the products are positive and the score is large. When they point in opposite directions, the score is negative. When they are unrelated, the positive and negative products cancel out and the score is near zero. At the start of training every output vector is zero, so every score is exactly zero.

After training, try “coffee” with “cup”, then with “fur”. The real neighbour scores far higher. In this small corpus almost every score ends up below zero, because each word is also drawn as a fake “noise” word thousands of times (section 7), which pushes scores down. What matters is the ranking: words that really appear together score higher than words that don’t.

## 5 The sigmoid

A score can be any number. To read it as a probability, word2vec passes it through the *sigmoid* function, σ(*x*) = 1 / (1 + e<sup>−*x*</sup>). It is a smooth step. Large negative scores go to almost 0, large positive scores go to almost 1, and a score of 0 goes to exactly 0.5.

The result is the model’s probability that this pair really appeared together in the text. For a real pair, the right answer is 1, so the **error** is 1 − σ(score).

<!-- module: m5 -->

At the start of training every score is 0, so σ is 0.5 and the first error is exactly 0.5: the model is a coin toss.

<div class="callout">
  <div class="t">Why not ask about every word at once?</div>
  The natural question is “which word, out of the whole vocabulary, is the context?” That calls for a <em>softmax</em>, which scores every word in the vocabulary, millions of them, for every training pair. The first word2vec paper avoided that cost with <em>hierarchical softmax</em>, a chain of sigmoids down a binary tree of the vocabulary. The second paper replaced it with <em>negative sampling</em>, the approach on this page: one sigmoid for the real word and one for each of a handful of randomly drawn ones.
</div>

## 6 One positive update

Now the model learns from its error. The update has a single scaling factor:

<p class="formula"><i>g</i> = (label − σ) × α</p>

The label is 1 for a real pair. α is the *learning rate*, which sets the size of each step. It starts at 0.025 and shrinks linearly to almost zero over the course of training, so early steps are large and late steps are fine adjustments.

Both vectors then move:

- The output vector moves toward the input vector: ***v*′ ← *v*′ + *g* · *v***. This is applied right away.
- The input vector should move toward the output vector by *g* · *v*′, using *v*′ as it was before the update. But this change is **not applied yet**. It goes into a pending row, for reasons that section 7 makes clear.

<!-- module: m6 -->

Pick any real pair from the corpus, and a moment in training: the numbers are the model’s actual state just before that pair. Early on, a context word’s *v*′ may still be all zeros, so the pending change to *v* is zero too and only *v*′ moves. Later the vectors are larger, the error is smaller and α has shrunk, so the steps are finer.

## 7 Negative samples

A positive update alone would be useless. The easiest way to score every pair high is to make every vector point the same way, and then the vectors carry no information. The model also needs examples of words that do *not* belong together.

Negative sampling makes them up. For each real pair, word2vec draws *k* = 5 “noise” words at random and treats each as a fake pair with the center word. A word’s chance of being drawn is its frequency raised to the power 0.75. This flattens the distribution: common words are still drawn most often, but rare words get more of a chance than their raw frequency would give them. If a draw happens to be the real context word, it is skipped.

For a fake pair the label is 0, so *g* = (0 − σ) × α is negative, and the same update rule pushes the two vectors apart. Each noise word’s output vector moves at once, and each adds its share to the center word’s pending row. Only after the real pair and all the noise words is **the accumulated pending change applied to the input vector, once**. Every target in a pair is scored against the same, unchanged *v*.

<!-- module: m7 -->

This is the whole of training. Every pair, in every pass, gets one positive update and up to five negative ones.

## 8 Training at scale

Now run the same loop over the whole corpus, 200 times. The chart below projects the 5-dimensional input vectors down to 2 dimensions with PCA (principal component analysis), which keeps the two directions in which the vectors spread out most. This picture is approximate. Distances on it are only roughly right, and the neighbour list beside it is computed in all 5 dimensions. Drag the timeline to go back to any point in training.

<!-- module: m8 -->

The loss is the error measured as the model sees it, summed over the real word and its noise words: −log σ(score) for the real word and −log(1 − σ(score)) for each noise word. It starts at about 6 × log 2 ≈ 4.2, one coin toss for each of the six targets, and falls as the vectors arrange themselves.

Watch “coffee” and “tea”. They never appear in the same sentence, yet they end up nearly on top of each other. Their input vectors were both pulled toward the same output vectors, those of “hot”, “cup”, “drink” and “milk”. Most of the time the same happens to “dog” and “cat”. Turn on the output vectors to see those context anchors that the input vectors arrange themselves around.

Below the map, two input vectors are drawn exactly. Any two vectors, in 5 dimensions or 300, lie together in one flat plane, so they can be drawn at their true lengths with the true angle between them. At the start of training the angle between “coffee” and “tea” is close to 90°; by the end it has nearly closed.

## 9 Aside: CBOW

The first word2vec paper proposed a second model, *continuous bag-of-words* (CBOW), and it is the default in the released code. It flips the question around. Instead of using the center word to predict each context word, CBOW averages the input vectors of all the context words and uses that average to predict the center word.

<!-- module: m9 -->

That makes one update per center word instead of one per pair, so CBOW trains faster. The cost is that every context word receives the same nudge, whatever its distance or role. The machinery is otherwise identical: the same sigmoid, the same negative samples, the same pending row. The fill-in-the-blank framing makes CBOW a distant ancestor of the masked language modelling later used to train BERT.

## 10 The bridge to LLMs

The first layer of a large language model is the same kind of object as word2vec’s input table: a lookup table with one vector per token, started at random and learned by gradient descent. Three things differ.

- **Tokens, not words.** LLMs split text into subword pieces, so rare and new words are built from parts instead of being dropped.
- **A different training signal.** The table is trained end to end on next-token prediction, through many layers, not directly on which words appear near which.
- **Messier geometry.** LLM embeddings tend to crowd into a narrow cone (*anisotropy*), and word frequency distorts their distances, so nearest neighbours are less clean than in word2vec.

One thing is the same. The table is **static**: a word gets one vector, whatever sentence it is in. “Bank” comes out of the table with the same numbers in “the river bank” and in “she paid it into the bank”. Meaning in context is built by the transformer layers that follow.

<!-- module: m10 -->

<footer>
  <h4 class="fn-h">Footnotes</h4>
  <ol class="fn">
    <li>Mikolov, Sutskever, Chen, Corrado and Dean, <a href="https://arxiv.org/abs/1310.4546">“Distributed Representations of Words and Phrases and their Compositionality”</a>, NeurIPS (2013). Skip-gram with negative sampling, subsampling of frequent words, and phrases.</li>
    <li>Mikolov, Chen, Corrado and Dean, <a href="https://arxiv.org/abs/1301.3781">“Efficient Estimation of Word Representations in Vector Space”</a> (2013). Skip-gram and CBOW, trained with hierarchical softmax.</li>
    <li>The released code (<a href="https://code.google.com/archive/p/word2vec/"><span class="mono">word2vec.c</span></a>) differs from the papers in a few details. In skip-gram it flips the roles: the context word’s input vector predicts the center word, which comes to the same thing on average, since a pair inside a window also appears the other way round. Its subsampling formula keeps a word with probability (√(<i>f</i>/<i>t</i>) + 1) · <i>t</i>/<i>f</i>, with a default <i>t</i> of 10⁻³. CBOW is the default model. σ comes from a precomputed table and is clipped at ±6. This page follows the code’s update order, window shrinking, initialization, learning-rate decay and subsampling formula, but uses the papers’ center-predicts-context framing and an exact σ.</li>
    <li>The 3-million-word Google News vectors, trained on about 100 billion words, come from the word2vec project’s release, not from either paper.</li>
    <li>The toy corpus, the seeds and the 200 passes are this page’s own. Re-rolling the initialization changes the final picture. Across 100 initialization seeds, “coffee” and “tea” ended up as each other’s nearest neighbours every time, and “dog” and “cat” 87 times.</li>
  </ol>
</footer>
