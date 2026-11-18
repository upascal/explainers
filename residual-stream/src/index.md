<!-- slide: Every layer of GPT-2 reads from one running vector and adds to it. Here is how to see what each part reads and writes. -->
# What GPT-2’s heads read and write

<p class="lede">Inside GPT-2, each position has one running vector of 768 numbers. Every layer reads it and adds something to it, and nothing is ever taken out. That shared vector, the <em>residual stream</em>, is where one part of the model leaves information for another. This page uses it to answer a question the <a href="../attention/">attention page</a> left open: what does a head take from the stream, what does it put back, and how do the two relate?</p>

<p class="note">Everything on this page comes from GPT-2 small’s released weights, and from running it on two texts. It was computed ahead of time; nothing is trained here. It follows on from <a href="../attention/">How attention lets GPT-2 look back</a>, and uses the same two texts as its sections 10 and 11.</p>

<!-- toc -->

<!-- slide: The final vector is a sum, and every part of it can be measured. -->
## 1 One stream, many writers

GPT-2 starts each position’s vector as a token embedding plus a position embedding. Then each of its 12 layers adds two things: the output of its attention part, and the output of its MLP. So the vector after the last layer is exactly a sum of 25 parts.

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">final vector</span><span class="g"><span class="o">=</span><span class="t tv">embeddings</span></span><span class="g"><span class="o">+</span><span class="t">attention 0</span><span class="o">+</span><span class="t">MLP 0</span></span><span class="g"><span class="o">+ … +</span><span class="t">attention 11</span><span class="o">+</span><span class="t">MLP 11</span></span></span><span class="s"><i>x</i><sub>12</sub> = <span class="sv"><i>x</i><sub>0</sub></span> + attn<sub>0</sub> + mlp<sub>0</sub> + … + attn<sub>11</sub> + mlp<sub>11</sub></span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

GPT-2 scores the final vector against every token’s embedding to pick the next token, after one last *LayerNorm*, which subtracts the vector’s average and divides it by its length. The division depends on the whole vector, but for one prediction it is a single number, and with it fixed at its real value the score of any token splits exactly into 25 pieces. Only differences between scores matter to the softmax, so each piece is measured against the average token’s: how much that part pushed this token up or down compared with the rest.

<!-- slide: Each part’s push on the token that really came next. -->
<!-- module: m1 -->

At the second <span class="tok"><span class="sp">·</span>mask</span>, the answer, <span class="tok"><span class="sp">·</span>artist</span>, is a word GPT-2 has seen once, 19 tokens back. The 25 parts put its score 15.2 above the average token’s. Attention adds 10.5 of that, the MLPs 4.6, and the embeddings almost nothing: the answer isn’t at this position, so the parts that move information between positions do most of the work. Over the whole second copy, attention adds 11.1 on average; over the first, where there is no earlier copy to find, 1.4.

The vector also grows as it goes. Over the sentence’s positions it starts with a length of about 5 and ends near 450, so a part that adds a lot late on can outweigh everything before it.

<!-- slide: A head reads through a 64-number window. -->
## 2 A window into the stream

An attention head doesn’t read the whole stream. It reads it through its tables: the value it takes from a token is that token’s vector times *W*<sub>V</sub>, which keeps 64 numbers out of 768. Whatever is in the other 704 directions, the head can’t see. Then *W*<sub>O</sub> turns the 64 numbers back into 768 and adds them in. Put together, the two tables make one map from the stream to the stream, called the head’s *OV circuit*:

<div class="formula fxblock"><span class="fx"><span class="w"><span class="t">what the head adds, per token it looks at</span><span class="g"><span class="o">=</span><span class="t">weight</span><span class="o">×</span><span class="t tv">that token’s vector</span></span><span class="g"><span class="o">×</span><span class="t tval">W<sub>V</sub> W<sub>O</sub></span></span></span><span class="s"><i>a</i><sub><i>t</i>,<i>u</i></sub> <span class="sv"><i>x</i><sub><i>u</i></sub></span> <span class="svv"><i>W</i><sub>OV</sub></span>, &nbsp; <span class="svv"><i>W</i><sub>OV</sub></span> = <i>W</i><sub>V</sub><i>W</i><sub>O</sub> (768 × 768, rank ≤ 64)</span></span><span class="fxsw seg" role="group" aria-label="show formulas as"><button data-fx="w">words</button><button data-fx="s">symbols</button></span></div>

The queries and keys work the same way. A score is a query times a key, and both come from the stream, so the two tables fold into one *QK circuit*, *W*<sub>Q</sub>*W*<sub>K</sub><sup>T</sup>, that compares two vectors from the stream. The QK circuit decides *where* the head looks; the OV circuit decides *what* it moves from there.

Only these products mean anything. Stretch one of the 64 numbers in the middle by any amount, and shrink it by the same amount on the way out, and the head does exactly what it did before. So asking what *W*<sub>V</sub> alone reads, or how large the 64-number value is, has no single answer.

<!-- slide: Change both tables, and nothing changes. -->
<!-- module: m2 -->

<!-- slide: What a head reads, and where it writes, in the same space. -->
## 3 Reading and writing directions

Because the OV circuit goes from the stream back to the stream, what a head reads and what it writes live in the same 768-number space, and can be compared. Taking the circuit apart with the *singular value decomposition* gives up to 64 pairs: a direction the head reads, the direction it writes the result to, and a gain between them. Every other direction of the stream it ignores.

So does a head write back into the directions it reads, or somewhere else? Compare its 64 read directions with its 64 write directions: the cosines between the two sets, best match first, are their *principal angles*. A cosine of 1 means some direction is both read and written.

There is a catch. Two 64-direction subspaces of a 768-number space overlap by chance: picked at random, their best match has a cosine of about 0.54, not 0. GPT-2’s heads average 0.71, and 129 of the 144 are above what random subspaces reach 99 times in 100. They mostly write near where they read, but not into the same place. The highest is head 4.7, at 0.97: some direction it writes is almost exactly one it reads.

<!-- slide: Each head’s overlap between reading and writing. -->
<!-- module: m3 -->

High overlap doesn’t say what the head does with it. A head can read a direction and write it back as it was, or write it back reversed, which erases it. The next section tells those apart, and head 4.7 turns out to be the second kind.

<!-- slide: Feed a head a token, and see which tokens it pushes up. -->
## 4 Copying and suppressing

The cleanest way to see what an OV circuit does is in terms of tokens. Feed it a token, and score its output against every token in the vocabulary, as GPT-2 does at the end. One adjustment matters: a head after layer 0 doesn’t see a token’s raw embedding, but what layer 0 made of it, so the input here is the embedding plus layer 0’s MLP’s change to it. With the raw embedding, almost every head looks like noise.

A head that pushes up the very token it read *copies*. One that pushes it down *suppresses*. The *copying score* sums this up over 500 common words: +1 for a head that only copies, −1 for one that only suppresses, and about 0 for random tables.

<!-- slide: Copying, suppressing, and everything between. -->
<!-- module: m4 -->

Of GPT-2’s 144 heads, 62 score above +0.5 and 12 below −0.5. Head 7.2 is a clean copier: feed it <span class="tok"><span class="sp">·</span>coffee</span> and <span class="tok"><span class="sp">·</span>coffee</span> comes out on top, and the same for <span class="tok"><span class="sp">·</span>Paris</span>. Head 10.7 is the opposite: feed it <span class="tok"><span class="sp">·</span>artist</span>, and <span class="tok"><span class="sp">·</span>artist</span> comes out 50,256th of 50,257, second to last. Head 4.7, which writes back almost exactly where it reads, scores −0.47, and a typical word fed to it comes out 49,709th: it reads a token and pushes it down. Head 5.5, the induction head from the attention page, scores +0.83, yet <span class="tok"><span class="sp">·</span>artist</span> itself comes out only 502nd. The score sums up the circuit as a whole; a head can lean towards copying without handing a given token on cleanly. Section 6 shows what that means on text.

<!-- slide: One head writes where another reads. -->
## 5 One head writing for another

The same comparison works between two heads. Head A writes into the stream; a head B in a later layer reads from it. How much of what A writes lands where B reads? B reads in three ways, so there are three answers: through its keys (*K-composition*), its queries (*Q-composition*) or its values (*V-composition*). Each is measured from the two heads’ circuits alone, as a number between 0 and 1, and two random circuits of the same shape score about 0.036.

On the attention page, head 4.11 wrote into each position which token came before it, and induction heads used that with their keys to find a repeat. If that is right, 4.11’s OV circuit should line up with those heads’ keys. It does: of all heads after layer 4, the six that read most of 4.11’s writing with their keys are 6.9, 5.1, 5.5, 7.10, 5.0 and 7.2, at about 2.4 to 2.8 times chance. Five of them are the induction heads from the attention page. Nothing about any text went into this: it comes from the weights alone.

<!-- slide: Pick a head, and see who reads what it writes. -->
<!-- module: m5 -->

The scores are small, a tenth of the way to 1, because each head reads and writes only a sliver of the stream, and shares it with everything else. Their size only means something next to chance.

<!-- slide: On real text: who finds the answer, and who writes it. -->
## 6 Who finds the answer, and who writes it

The weights say what a head *can* do. On real text, the same split as section 1 works for every head: each head’s output at a position is a sum over the tokens it looked at, weight times that token’s vector through the OV circuit, plus a constant from the value’s bias, which arrives in full wherever the head looks, since the weights add up to 1. So each head’s push on the next token splits exactly by the token it came from.

Take the second <span class="tok"><span class="sp">·</span>mask</span> again. Head 5.5 puts 96% of its weight on the first <span class="tok"><span class="sp">·</span>artist</span>, as on the attention page, but its own push on <span class="tok"><span class="sp">·</span>artist</span> is only +0.34. Heads 7.2 and 9.9 look there too, with 66% and 85% of their weight, and push harder: +0.48 and +1.00. Head 10.7 looks there as well, with 81%, and pushes <span class="tok"><span class="sp">·</span>artist</span> *down*, by 1.76: the copy suppression section 4 read off its weights. Over the whole second copy, the heads that push the right word up most are 7.2, 9.9 and 9.6, and the one that pushes it down most is 10.7.

So finding the earlier copy and writing the answer are different jobs. Head 5.5 finds it, but writes little of the answer itself; whatever else it contributes has to reach the answer through the heads and MLPs after it.

<!-- slide: Each head’s push, and where it came from. -->
<!-- module: m6 -->

## 7 What the weights can’t tell you

Everything above treats a head as a straight line from what it reads to what it writes, and for the OV circuit that is exactly true: once the weights are fixed, the value and output tables are just multiplications. Three things keep it from being the whole story.

- **LayerNorm.** Each head reads the stream after a LayerNorm, which subtracts the average and divides by the length. The gain and bias can be folded into the head’s tables, as here, but the division can’t: the head sees each vector at the same length, however long it was.
- **The MLPs.** Half of what is written to the stream comes from the MLPs, which aren’t linear. Section 4 already needed layer 0’s MLP just to describe a head’s input.
- **Directions aren’t meanings.** A read direction is a direction in a space of 768 numbers that carries far more than 768 things at once. What one *means* has to be found from data, for example with sparse autoencoders trained on the stream or on heads’ outputs, which split it into many more, sparser directions.

Still, the stream is what makes this kind of measurement possible at all. Every part reads from the same space and writes into the same space, so any two can be compared directly, from the weights or on a text: one head’s writing with another’s reading, or a head’s writing with the vocabulary at the end.

<p class="next-explainer">This page follows <a href="https://ulyssespascal.com/apps/attention">How attention lets GPT-2 look back</a>, which opens up a single attention head.</p>

<footer>
  <h4 class="fn-h">Footnotes</h4>
  <ol class="fn">
    <li>Elhage, Nanda, Olsson and others, <a href="https://transformer-circuits.pub/2021/framework/index.html">“A Mathematical Framework for Transformer Circuits”</a> (2021): the residual stream, the OV and QK circuits, their freedom to rescale, eigenvalues as a measure of copying, and Q-, K- and V-composition. Composition scores here are their ‖<i>M</i><sub>B</sub><i>W</i><sub>OV</sub><sup>A</sup>‖ / (‖<i>M</i><sub>B</sub>‖ ‖<i>W</i><sub>OV</sub><sup>A</sup>‖), in Frobenius norm, with <i>M</i><sub>B</sub> the later head’s QK or OV circuit; the chance level is the same score for two products of random 768 × 64 and 64 × 768 tables, averaged over 50 draws.</li>
    <li>GPT-2 applies its tables to row vectors (<i>x</i> <i>W</i>), so a head reads through the columns of <i>W</i><sub>V</sub> and writes into the rows of <i>W</i><sub>O</sub>, and in <i>W</i><sub>OV</sub> = <i>U</i> Σ <i>V</i><sup>T</sup> the read directions are the columns of <i>U</i> and the write directions those of <i>V</i>: the other way round from the usual column-vector convention. Each head’s LayerNorm gain is folded into its <i>W</i><sub>Q</sub>, <i>W</i><sub>K</sub> and <i>W</i><sub>V</sub>. The overlaps of section 3 are the singular values of <i>Q</i><sub>read</sub><sup>T</sup><i>Q</i><sub>write</sub>, for orthonormal bases of the two subspaces; the chance level averages 200 pairs of random 64-dimensional subspaces, and its 99th percentile comes from 300 more.</li>
    <li>Section 4’s copying score is the sum of the eigenvalues of the token-to-token circuit divided by the sum of their sizes, as in Elhage et al., computed over 500 common single-token words: their effective embeddings (embedding plus layer 0’s MLP), normalized, in; centred scores against the same words’ embeddings out. Random tables of the same shape score about 0, give or take 0.09.</li>
    <li>McDougall, Conmy, Rushing, McGrath and Nanda, <a href="https://arxiv.org/abs/2310.04625">“Copy Suppression: Comprehensively Understanding an Attention Head”</a> (2023), on head 10.7 of GPT-2 small, which pushes down tokens that appear earlier in the context when the model predicts them. Wang, Variengien, Conmy, Shlegeris and Steinhardt, <a href="https://arxiv.org/abs/2211.00593">“Interpretability in the Wild”</a>, ICLR (2023), found heads of both kinds in GPT-2 small.</li>
    <li>Section 1 and 6’s split of a score is exact: the final LayerNorm’s division is fixed at its value in the real run, and its own bias adds a constant to each score that belongs to no part. Every score is taken relative to the average over all 50,257 tokens, since only differences between scores reach the softmax. The precompute checks that the 25 parts and the constant add up to GPT-2’s own score, and that each head’s split by source adds up to its whole push. Head-level pushes leave out the output bias, which belongs to the attention part as a whole.</li>
    <li>Kissane, Krzyzanowski, Bloom, Conmy and Nanda, <a href="https://arxiv.org/abs/2406.17759">“Interpreting Attention Layer Outputs with Sparse Autoencoders”</a> (2024), train sparse autoencoders on what attention heads write, and use them to name what each head moves.</li>
  </ol>
</footer>
