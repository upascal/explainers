<!-- slide: Attention is how each token in GPT-2 reads the tokens before it. Here is how one attention head works, and what GPT-2’s 144 heads do. -->
# How attention lets GPT-2 look back

<p class="lede">A token’s embedding is the same wherever the token appears. To predict what comes next, GPT-2 needs to know what came before, and the part that brings it in is called <em>attention</em>. At every position, attention picks which earlier tokens matter and copies some of what they carry. This page takes one attention head apart, watches it learn, and then looks at what the 144 heads in GPT-2 have learned to do.</p>

<p class="note">This page follows on from <a href="../gpt2-embeddings/">How GPT-2 turns text into vectors</a>, and uses the same toy model: GPT-2’s structure, shrunk to 6 numbers per token and one layer with one attention head, trained on 30 short sentences. It trains in your browser when the page loads, and every number from the toy model is computed from it. GPT-2’s own attention weights, from section 9 on, were computed ahead of time from its released weights. Interactives with a play bar can be driven from the keyboard: → steps, space plays or pauses, shift + → skips to the end, and shift + ← resets.</p>

<!-- toc -->

<!-- slide: Same word, same position, different next word. -->
## 1 The problem

Two of the toy model’s sentences are “the dog can bark” and “the cat can purr”. After “the dog can”, the next token should be <span class="tok"><span class="sp">·</span>bark</span>; after “the cat can”, it should be <span class="tok"><span class="sp">·</span>pur</span>, the first half of “purr”, which GPT-2 cuts in two.

The vector at <span class="tok"><span class="sp">·</span>can</span> starts out as the embedding of <span class="tok"><span class="sp">·</span>can</span> plus the embedding of position 3, and both are the same in the two sentences. Nothing in that vector says which animal came before. If each position only worked on its own vector, the two sentences would get exactly the same prediction.

<!-- slide: Without attention, the two sentences look the same to “can”. -->
<!-- module: m1 -->

With attention switched off, the toy model gives both sentences the same guess, and gets one of them wrong. With it on, the vector at <span class="tok"><span class="sp">·</span>can</span> has taken something from the animal, and each sentence gets its own answer.

<!-- slide: Where should “can” look? -->
## 2 Where to look

Attention works in two steps. First, the vector at <span class="tok"><span class="sp">·</span>can</span> gives each token up to and including itself a *weight*: how much to take from it. The weights are between 0 and 1 and add up to 1. Then it takes a blend of what those tokens carry, mixed in those proportions, and adds the blend to its own vector.

Before seeing how the model picks the weights, pick them yourself. The sliders set how much <span class="tok"><span class="sp">·</span>can</span> draws on each token; the toy model’s own layer does the rest.

<!-- slide: Set the weights yourself, and watch the prediction. -->
<!-- module: m2 -->

A good set of weights depends on the sentence, but the rule behind it doesn’t. The model can’t store one set of weights for “the dog can” and another for “the cat can”; it has to *compute* them, from the vectors at hand, in a way that works for every sentence. The next three sections show how.

<!-- slide: Every token makes a query and a key. Their dot product is a score. -->
## 3 Queries and keys

Each position turns its vector *x* into two new vectors, by multiplying it with two small tables of numbers that the model learns: a *query* <span class="cq">q</span>, with the table *W*<sub>Q</sub>, and a *key* <span class="ck">k</span>, with the table *W*<sub>K</sub>. Loosely, the query says what this position is looking for, and the key says what this token has to offer. In the toy model both are 6 numbers long, like *x*.

To score how well an earlier token matches, the position takes the dot product of its own query with that token’s key, as word2vec did with its two vectors: multiply the numbers pair by pair and add them up. A large score means a good match. The score is then divided by √6, the square root of the vector’s length; section 5 says why.

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">score</span><span class="g"><span class="o">=</span><span class="t tq">query here</span></span><span class="g"><span class="o">·</span><span class="t tk">key there</span></span><span class="g"><span class="o">÷</span><span class="t">√6</span></span></span><span class="s">score<sub><i>t</i>,<i>u</i></sub> = <span class="sq"><i>q</i><sub><i>t</i></sub></span> · <span class="sk"><i>k</i><sub><i>u</i></sub></span> / √<i>d</i>, &nbsp; <span class="sq"><i>q</i><sub><i>t</i></sub></span> = <span class="sv"><i>x</i><sub><i>t</i></sub></span><i>W</i><sub>Q</sub>, &nbsp; <span class="sk"><i>k</i><sub><i>u</i></sub></span> = <span class="sv"><i>x</i><sub><i>u</i></sub></span><i>W</i><sub>K</sub></span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

Every position uses the same two tables. What differs from sentence to sentence is the vectors that go in.

<!-- slide: One query, scored against every key up to it. -->
<!-- module: m3 -->

The query of <span class="tok"><span class="sp">·</span>can</span> is the same in both sentences, since its vector is. The keys of <span class="tok"><span class="sp">·</span>dog</span> and <span class="tok"><span class="sp">·</span>cat</span> are different, because their embeddings are, and so are their scores.

<!-- slide: A position may only look back. -->
## 4 No peeking ahead

Section 3 only scored the tokens up to the query’s own position, and that is a rule, not a shortcut. GPT-2 trains every position of a text at once, each predicting the token after it. If <span class="tok"><span class="sp">·</span>can</span> could see the tokens after it, it could read the answer, <span class="tok"><span class="sp">·</span>pur</span>, straight off the next position, and would learn nothing useful for text it hasn’t seen.

So every score from a position to a later one is replaced by −∞ before the next step. That is the *causal mask*. Here is every score in one sentence, each row a query and each column a key.

<!-- slide: Every score in a sentence. The mask hides the future. -->
<!-- module: m4 -->

With the mask on, the grid is a triangle: the first position can only look at itself, the last at everything. This is also what lets GPT-2 write text one token at a time: a position’s vector never depends on what comes after it, so adding a token at the end leaves every earlier vector as it was.

<!-- slide: The softmax turns scores into weights that add up to 1. -->
## 5 From scores to weights

The scores can be any number, positive or negative. To turn them into weights, attention uses the *softmax*, the same function GPT-2 uses at the very end to turn its scores for every token into probabilities: raise *e* to the power of each score, then divide each by their total.

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">weight</span><span class="g"><span class="o">=</span><span class="t">e<sup>score</sup></span></span><span class="g"><span class="o">÷</span><span class="t">the sum of e<sup>score</sup> over every token up to here</span></span></span><span class="s"><i>a</i><sub><i>t</i>,<i>u</i></sub> = e<sup>score<sub><i>t</i>,<i>u</i></sub></sup> / (e<sup>score<sub><i>t</i>,0</sub></sup> + … + e<sup>score<sub><i>t</i>,<i>t</i></sub></sup>)</span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

The result is always between 0 and 1, the weights add up to 1, and a higher score always gets a higher weight. A masked score of −∞ becomes e<sup>−∞</sup> = 0: no weight at all.

<!-- slide: The softmax, step by step, and what scaling the scores does. -->
<!-- module: m5 -->

The softmax cares about the gaps between scores. Stretch the scores apart and the largest one takes nearly all the weight; squash them together and the weights even out. A dot product of two longer vectors adds up more products, so its scores tend to be spread wider: in GPT-2, where a head’s vectors have 64 numbers, they would be about 8 times as wide as for vectors of one number. Dividing by √64 = 8 brings them back, so that a head starts training with weights that are spread out, and can learn where to look, instead of starting stuck on one token.

<!-- slide: Values: what each token hands over. -->
## 6 Values: what gets passed along

The weights say how much to take from each token. What is taken is a third vector, the *value* <span class="cvv">v</span>, made from each position’s *x* with a third table, *W*<sub>V</sub>. Keeping the value apart from the key lets a token be found for one reason and hand over something else: the key of <span class="tok"><span class="sp">·</span>cat</span> only has to say “I’m the one to look at”, while its value can carry what follows from it.

The head multiplies each value by its weight and adds them up. That blend goes through one more table, *W*<sub>O</sub>, which turns it into a change for the position’s vector.

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">attention’s output</span><span class="g"><span class="o">=</span><span class="t">(weight × </span><span class="t tval">value</span><span class="o">, added up over every token up to here)</span></span><span class="g"><span class="o">×</span><span class="t">W<sub>O</sub></span></span></span><span class="s">attn<sub><i>t</i></sub> = (<i>a</i><sub><i>t</i>,0</sub><span class="svv"><i>v</i><sub>0</sub></span> + … + <i>a</i><sub><i>t</i>,<i>t</i></sub><span class="svv"><i>v</i><sub><i>t</i></sub></span>) <i>W</i><sub>O</sub>, &nbsp; <span class="svv"><i>v</i><sub><i>u</i></sub></span> = <span class="sv"><i>x</i><sub><i>u</i></sub></span><i>W</i><sub>V</sub></span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

<!-- slide: The weighted values, added up, become the change. -->
<!-- module: m6 -->

## 7 Adding it back in

Attention doesn’t replace the vector at a position: it adds its output to it. The vector keeps everything it had, its own token and position, and gains what the head took from the context.

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">new vector</span><span class="g"><span class="o">=</span><span class="t tv">old vector</span></span><span class="g"><span class="o">+</span><span class="t">attention’s output</span></span></span><span class="s"><i>h</i><sub><i>t</i></sub> = <span class="sv"><i>x</i><sub><i>t</i></sub></span> + attn<sub><i>t</i></sub></span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

In the toy model, an MLP then works on that vector, still at its own position, and adds its own change, and the result is scored against every token in the table, as on the <a href="../gpt2-embeddings/#s7">previous page</a>. GPT-2 does the same 12 times over. The running vector that every layer reads from and adds to is often called the *residual stream*: attention is the only part of the model that moves information from one position to another along it.

<!-- slide: Nobody tells the head where to look. Training finds out. -->
## 8 How training shapes the head

No one tells the toy model that <span class="tok"><span class="sp">·</span>can</span> should look at the animal. At the start, its four tables hold small random numbers, every score is close to 0, and the weights are nearly even. Each training step works out how every number in the model should change to make the right next token more likely, and changes it a little, as on the previous page. That includes *W*<sub>Q</sub>, *W*<sub>K</sub>, *W*<sub>V</sub> and *W*<sub>O</sub>.

When <span class="tok"><span class="sp">·</span>pur</span> should have been more likely after “the cat can”, the update asks for more of <span class="tok"><span class="sp">·</span>cat</span>’s value in the mix, and so for a higher score between the query of <span class="tok"><span class="sp">·</span>can</span> and the key of <span class="tok"><span class="sp">·</span>cat</span>; and asks for <span class="tok"><span class="sp">·</span>cat</span>’s value, through *W*<sub>O</sub>, to point towards <span class="tok"><span class="sp">·</span>pur</span>. Step through the toy model’s training and watch both happen.

<!-- slide: The weight on the animal, and the right guess, over training. -->
<!-- module: m8 -->

The other pair doesn’t work out. After “the dog likes to” and “the cat likes to”, the answers are <span class="tok"><span class="sp">·</span>play</span> and <span class="tok"><span class="sp">·</span>sleep</span>, and the clue is just as far back, but <span class="tok"><span class="sp">·</span>to</span> never learns to look at the animal, and the model ends up guessing about 50–50. One head in one layer has a single query table for every position; it can learn to send <span class="tok"><span class="sp">·</span>can</span> to the animal, but here it didn’t find a way to send <span class="tok"><span class="sp">·</span>to</span> there too. GPT-2 has far more room: many heads, in many layers.

<!-- slide: GPT-2 runs 12 heads side by side in every layer. -->
## 9 Many heads

GPT-2 small has 12 layers, and each has 12 attention heads working side by side, 144 in all. Each head has its own *W*<sub>Q</sub>, *W*<sub>K</sub> and *W*<sub>V</sub>, which turn the 768-number vector into a query, key and value of 64 numbers each, and so each head picks its own weights. The 12 heads’ blends, 12 × 64 = 768 numbers, are put side by side and go through one *W*<sub>O</sub> back into the vector.

Here are the 12 heads of one layer, on one sentence. Each square is a head’s grid of weights, drawn as in section 4: a row for each position, a column for each token it can look at, and darker for more weight.

<!-- slide: Twelve heads, twelve different ways to look. -->
<!-- module: m9 -->

Heads in the same layer read the same vectors, but they look in very different ways: some spread their weight out, some always look one token back, and many put almost all of it on the very first token.

<!-- slide: All 144 of GPT-2’s heads, on real text. -->
## 10 Inside GPT-2: 144 heads

Pick any of GPT-2’s 144 heads, by layer and head number, and see its weights on one of three texts: a sentence, the same sentence without the <code>&lt;|endoftext|&gt;</code> GPT-2 puts between documents, and a list of 20 random words, then the same 20 again. Hover over the big grid to read a weight.

<!-- slide: Pick a head. -->
<!-- module: m10 -->

Most heads are hard to name. A few do something simple and clear enough to have names, and the next two sections look at them.

<!-- slide: One head looks one token back. Another uses it to find repeats. -->
## 11 Previous-token and induction heads

Head 4.11, the twelfth head of the fifth layer (both counted from 0), puts almost all its weight, on average 99.9% on the sentence above, on the token just before. That sounds useless on its own, but it writes into each position’s vector what the token before it was.

That lets a head in a later layer do something cleverer. Take the random words, repeated. In the second copy, at <span class="tok"><span class="sp">·</span>mask</span>, the head can ask, with its query, for a position whose *previous* token was <span class="tok"><span class="sp">·</span>mask</span>, and the key it matches is the one 4.11 helped build, at <span class="tok"><span class="sp">·</span>artist</span>. The head copies <span class="tok"><span class="sp">·</span>artist</span>’s value, and GPT-2 predicts <span class="tok"><span class="sp">·</span>artist</span> again. A head that does this is an *induction head*: it completes a pattern that appeared earlier, “mask artist … mask → artist”, whatever the words. Heads 5.1, 5.5, 6.9, 7.2 and 7.10 are the clearest in GPT-2.

<!-- slide: Two heads working together to continue a pattern. -->
<!-- module: m11 -->

In the first copy the words are random, and GPT-2 can’t predict any of them. From the second word of the second copy on, its top guess is right every time. Nothing about these words was learned in training; the heads look the answer up in the text itself. Researchers have found that induction heads form at a particular point in training, at the same time as a model gets much better at using what came earlier in a long text.

<!-- slide: Most heads, most of the time, look at the first token. -->
## 12 The attention sink

In section 10, many heads were dark down the first column. Of GPT-2’s 144 heads, 90 put more than half their weight on the first token of the sentence, averaged over its positions. Without the <code>&lt;|endoftext|&gt;</code> at the start, 92 do the same with whatever token comes first.

The softmax explains why. The weights at a position have to add up to 1, so a head can’t give *no* weight to everything; it has to put it somewhere. A head that is only useful in some contexts, such as an induction head with no repeat to find, needs somewhere harmless to put it the rest of the time. The first token is visible from every position, and in these heads its value is small, a median of 0.11 times the size of the other values, so weight parked there adds almost nothing. Such a place is called an *attention sink*.

<!-- slide: Each dot is one of GPT-2’s heads. -->
<!-- module: m12 -->

So a dark first column usually means a head that is *idle* at that position, not one that thinks the first token matters. The weights say where a head looks; what it takes from there is the value, and a large weight on a small value is a small effect.

<!-- slide: Attention is the only way information moves between positions. -->
## 13 What attention adds

The first layer of GPT-2 gives every token the same vector wherever it appears. Attention is how that changes: each position asks a question with its query, every earlier token answers with its key, the softmax turns the match into weights, and the values, blended in those proportions, are added to the position’s vector. It is the only place in the model where one position reads another. Repeated over 144 heads in 12 layers, that is enough for a vector to come out of GPT-2 knowing which animal came before, what followed a word the last time it appeared, and a great deal else.

<p class="next-explainer">The next explainer, <a href="https://ulyssespascal.com/apps/residual-stream">What GPT-2’s heads read and write</a>, follows the residual stream: what each head takes from it, what it puts back, and how one head writes for another. This page continues <a href="https://ulyssespascal.com/apps/gpt2-embeddings">How GPT-2 turns text into vectors</a>, which follows the same toy model from its first layer, and <a href="https://ulyssespascal.com/apps/word2vec">How word2vec learns</a> before it.</p>

<footer>
  <h4 class="fn-h">Footnotes</h4>
  <ol class="fn">
    <li>Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin, <a href="https://arxiv.org/abs/1706.03762">“Attention Is All You Need”</a>, NeurIPS (2017): queries, keys and values, scaling by √<i>d</i>, the mask, and many heads side by side. Bahdanau, Cho and Bengio, <a href="https://arxiv.org/abs/1409.0473">“Neural Machine Translation by Jointly Learning to Align and Translate”</a>, ICLR (2015), introduced attention as a weighted blend picked by a softmax.</li>
    <li>Radford, Wu, Child, Luan, Amodei and Sutskever, <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">“Language Models are Unsupervised Multitask Learners”</a> (2019). GPT-2 small: 12 layers of 12 heads, 768 numbers per vector, 64 per head. Its weights on this page come from the released model (Hugging Face <code>gpt2</code>), on the texts shown.</li>
    <li>Elhage, Nanda, Olsson and others, <a href="https://transformer-circuits.pub/2021/framework/index.html">“A Mathematical Framework for Transformer Circuits”</a> (2021), on the residual stream (the subject of the <a href="../residual-stream/">next page</a>), attention as the only part that moves information between positions, and how a previous-token head and an induction head work together.</li>
    <li>Olsson, Elhage, Nanda and others, <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html">“In-context Learning and Induction Heads”</a> (2022), on induction heads, repeated random tokens as a test for them, and the point in training where they form. Wang, Variengien, Conmy, Shlegeris and Steinhardt, <a href="https://arxiv.org/abs/2211.00593">“Interpretability in the Wild”</a>, ICLR (2023), name 2.2 and 4.11 as GPT-2 small’s previous-token heads. The induction scores here are the average weight from each word in the second copy to the word after its first appearance.</li>
    <li>Xiao, Tian, Chen, Han and Lewis, <a href="https://arxiv.org/abs/2309.17453">“Efficient Streaming Language Models with Attention Sinks”</a>, ICLR (2024), named the attention sink. Kobayashi, Kuribayashi, Yokoi and Inui, <a href="https://arxiv.org/abs/2004.10102">“Attention is Not Only a Weight: Analyzing Transformers with Vector Norms”</a>, EMNLP (2020), found in BERT that heavy weights often fall on tokens with small values. The ratio in section 12 is, for each head with more than half its weight on the first token, the length of its value at the first position over the average length at the others, on the sentence of section 10; 99% of those heads are under 0.5.</li>
    <li>The toy model is the one on the <a href="../gpt2-embeddings/">GPT-2 embeddings page</a>, with initialization seed 2 and the same training order; that page uses seed 1. With seed 2 its head sends <span class="tok"><span class="sp">·</span>can</span> to the animal clearly, and each animal’s value alone points to the right answer. Across 100 seeds, 79 trained toy models predict both <span class="tok"><span class="sp">·</span>bark</span> and <span class="tok"><span class="sp">·</span>pur</span> right, 54 put more than half of <span class="tok"><span class="sp">·</span>can</span>’s weight on the animal in both sentences, and in 47 the animal’s value alone gives the right answer; in the rest, the head spreads the job over several tokens. 14 of the 100 learn “likes to” as well.</li>
  </ol>
</footer>
