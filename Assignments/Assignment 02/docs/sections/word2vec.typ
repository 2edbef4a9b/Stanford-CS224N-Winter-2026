#import "../utils.typ": answer, mtext, proof

= Understanding word2vec (20 points)

Recall that the key insight behind `word2vec` is that _a word is known by the company it keeps_. Concretely, consider a "center" word $c$ surrounded before and after by a context of a certain length. We term words in this contextual window "outside words" ($O$). For example, in @fig-word2vec, the context window length is 2, the center word $c$ is "banking", and the outside words are "turning", "into", "crises", and "as":

#figure(
  image("../assets/word2vec.png", width: 60%),
  caption: [The word2vec skip-gram prediction model with window size 2],
) <fig-word2vec>

Skip-gram `word2vec` aims to learn the probability distribution $P(O | C)$. Specifically, given a specific word $o$ and a specific word $c$, we want to predict $P(O = o | C = c)$: the probability that word $o$ is an "outside" word for $c$ (i.e., that it falls within the contextual window of $c$). We model this probability by taking the softmax function over a series of vector dot-products:

$
  P(O = o | C = c) = (exp(bold(u)_o^T bold(v)_c)) / (sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c))
$ <word2vec-condprob>

For each word, we learn vectors $u$ and $v$, where $bold(u)_o$ is the "outside" vector representing outside word $o$, and $bold(v)_c$ is the "center" vector representing center word $c$. We store these parameters in two matrices, $bold(U)$ and $bold(V)$. The columns of $bold(U)$ are all the "outside" vectors $bold(u)_w$; the columns of $bold(V)$ are all of the "center" vectors $bold(v)_w$. Both $bold(U)$ and $bold(V)$ contain a vector for every $w in #mtext[Vocabulary]$.#footnote[
  Assume that every word in our vocabulary is matched to an integer number $k$. Bolded lowercase letters represent vectors. $bold(u)_k$ is both the $k^(#mtext[th])$ column of $bold(U)$ and the "outside" word vector for the word indexed by $k$. $bold(v)_k$ is both the $k^(#mtext[th])$ column of $bold(V)$ and the "center" word vector for the word indexed by $k$. *In order to simplify notation we shall interchangeably use* $k$ *to refer to word* $k$ *and the index of word* $k$.
]

Recall from lectures that, for a single pair of words $c$ and $o$, the loss is given by:

$
  bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U)) = -log P(O = o | C = c)
$ <naive-softmax>

We can view this loss as the cross-entropy#footnote[
  The *cross-entropy loss* between the true (discrete) probability distribution $p$ and another distribution $q$ is $-sum_i p_i log(q_i)$.
] between the true distribution $bold(y)$ and the predicted distribution $hat(bold(y))$, for a particular center word $c$ and a particular outside word $o$.
Here, both $bold(y)$ and $hat(bold(y))$ are vectors with length equal to the number of words in the vocabulary.
Furthermore, the $k^(#mtext[th])$ entry in these vectors indicates the conditional probability of the $k^(#mtext[th])$ word being an "outside word" for the given $c$.
The true empirical distribution $bold(y)$ is a one-hot vector with a 1 for the true outside word $o$, and 0 everywhere else, for this particular example of center word $c$ and outside word $o$.#footnote[
  Note that the true conditional probability distribution of context words for the entire training dataset would not be one-hot.
]
The predicted distribution $hat(bold(y))$ is the probability distribution $P(O | C = c)$ given by our model in @word2vec-condprob.

*Note:* Throughout this homework, when computing derivatives, please use the method reviewed during the lecture (i.e. no Taylor Series Approximations).

+ (2 points)
  Prove that the naive-softmax loss (@naive-softmax) is the same as the cross-entropy loss between $bold(y)$ and $hat(bold(y))$, i.e. (note that $bold(y)$ (true distribution), $hat(bold(y))$ (predicted distribution) are vectors and $hat(bold(y))_o$ is a scalar):

  $
    -sum_(w in#mtext[Vocab]) bold(y)_w log(hat(bold(y))_w) = -log(hat(bold(y))_o).
  $

  Your answer should be one line. You may describe your answer in words.

  #proof

  Since $bold(y)$ is a one-hot vector with a 1 for the true outside word $o$, and 0 everywhere else, the only term in the sum that is non-zero is the term where $w = o$. Therefore, we have:

  $
    -sum_(w in#mtext[Vocab]) bold(y)_w log(hat(bold(y))_w) = -bold(y)_o log(hat(bold(y))_o) = -log(hat(bold(y))_o).
  $

+ (6 points)
  + Compute the partial derivative of $bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U))$ with respect to $bold(v)_c$. _Please write your answer in terms of_ $bold(y)$, $hat(bold(y))$, $bold(U)$_, and show your work to receive full credit_.

    - *Note:* Your final answers for the partial derivative should follow the shape convention: the partial derivative of any function $f(x)$ with respect to $x$ should have the *same shape* as $x$.#footnote[
        This allows us to efficiently minimize a function using gradient descent without worrying about reshaping or dimension mismatching. While following the shape convention, we're guaranteed that $theta := theta - alpha (partial J(theta)) / (partial theta)$ is a well-defined update rule.
      ]
    - Please provide your answers for the partial derivative in vectorized form. For example, when we ask you to write your answers in terms of $bold(y)$, $hat(bold(y))$, and $bold(U)$, you may not refer to specific elements of these terms in your final answer (such as $bold(y)_1$, $bold(y)_2$, $dots$). You may also not refer to specific vectors such as $u_0$, $u_1$, etc.

    #answer

    $
      bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U)) &= -log P(O = o | C = c) \
      &= -log exp(bold(u)_o^T bold(v)_c) / (sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c)) \
      &= -bold(u)_o^T bold(v)_c + log sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c)
    $

    $
      nabla_(bold(v)_c) -bold(u)_o^T bold(v)_c = -bold(u)_o = -bold(u)_o bold(y)_o = -sum_(w in#mtext[Vocab]) bold(y)_w bold(u)_w = -bold(U) bold(y)
    $

    $
      nabla_(bold(v)_c) log sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c) &= 1 / (sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c)) dot sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c) dot bold(u)_w \
      &= sum_(w in#mtext[Vocab]) (exp(bold(u)_w^T bold(v)_c) / (sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c))) dot bold(u)_w \
      &= sum_(w in#mtext[Vocab]) hat(bold(y))_w dot bold(u)_w \
      &= bold(U) hat(bold(y))
    $

    Therefore, the partial derivative can be computed as:

    $
      nabla_(bold(v)_c) bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U))) = -bold(U) bold(y) + bold(U) hat(bold(y)) = bold(U) (hat(bold(y)) - bold(y))
    $

  + When is the gradient you computed equal to zero? Write a mathematical equation. *Hint:* You may wish to review and use some introductory linear algebra concepts.

    #answer

    $
      bold(U) (hat(bold(y)) - bold(y)) = 0 <=> (hat(bold(y)) - bold(y)) in #mtext[null] (bold(U))
    $

  + The gradient you found is the difference between the two terms. Provide an interpretation of how each of these terms improves the word vector when this gradient is subtracted from the word vector $v_c$.

    #answer

    The stochastic gradient descent update rule is

    $
      bold(v)_c^(t + 1) &= bold(v)_c^(t) - alpha nabla_(bold(v)_c^t) bold(J)_#mtext[naive-softmax] (bold(v)_c^t, o, bold(U))) \
      &= bold(v)_c^(t) - alpha bold(U) (hat(bold(y)) - bold(y)) \
      &= bold(v)_c^(t) - underbrace(alpha bold(U) hat(bold(y)), #mtext[Prediction]) + underbrace(alpha bold(U) bold(y), #mtext[Truth])
    $

    When we subtract the gradient (which results in the addition above), we are essentially performing two updates to the word vector $bold(v)_c$:

    - The "Truth" term $(alpha bold(U) bold(y))$: Since $bold(y)$ is a one-hot vector for the actual context word $o$, $bold(U) bold(y)$ simplifies to the vector of the actual context word, $bold(u)_o$. Adding this term pulls $bold(v)_c$ closer to the actual context word's vector. It increases the dot product $bold(u)_o^T bold(v)_c$, which effectively increases the probability the model assigns to the correct word in future iterations.

    - The "Prediction" term $(-alpha bold(U) hat(bold(y)))$: This term is the weighted average of all context word vectors, where the weights are the probabilities $hat(bold(y))_w$ that the model assigned to them. Subtracting this term pushes $bold(v)_c$ away from all words the model currently assigns high probability to. Essentially, it acts as a corrective force: if the model incorrectly predicted a word that didn't appear, this term nudges $bold(v)_c$ away from that word's vector, decreasing the dot product and lowering the probability for incorrect predictions.

+ (1 point)
  In many downstream applications using word embeddings, L2 normalized vectors (e.g. $bold(u) / norm(bold(u))$ where $norm(bold(u)) = sqrt(sum_i u_i^2)$) are used instead of their raw forms (e.g. $bold(u)$). Let’s consider a hypothetical downstream task of binary classification of phrases as being positive or negative, where you decide the sign based on the sum of individual embeddings of the words. When would L2 normalization take away useful information for the downstream task? When would it not?

  *Hint:* Consider the case where $bold(u)_x = alpha bold(u)_y$ for some words $x != y$ and some scalar $alpha$. When $alpha$ is positive, what will be the value of normalized $bold(u)_x$ and normalized $bold(u)_y$? How might $bold(u)_x$ and $bold(u)_y$ be related for such a normalization to affect or not affect the resulting classification?

  #answer

  L2 normalization takes away useful information when the magnitude of a word vector matters for the downstream task. For example, if $bold(u)_x = alpha bold(u)_y$ with $alpha > 0$, then $bold(u)_x / norm(bold(u)_x) = bold(u)_y / norm(bold(u)_y)$, so the two words become indistinguishable after normalization even though their original norms may encode different sentiment strength or importance. In that case, normalization can hurt binary phrase classification because summing normalized embeddings removes information about how strongly each word should influence the final sign.

  On the other hand, normalization does not hurt when only the direction of the embeddings matters and the norms contain no useful task-specific information. If two words that differ only by a positive scalar multiple should contribute identically to classification, then mapping them to the same normalized vector is acceptable or even desirable. Therefore, for this particular task, applying L2 normalization is only better if sentiment is encoded mainly by direction; if sentiment strength is also encoded by vector norm, then using the raw embeddings is preferable.

+ (1 point)
  Write down the partial derivative of $bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U))$ with respect to $bold(U)$. Please break down your answer in terms of the column vectors $(partial bold(J)(bold(v)_c, o, bold(U))) / (partial bold(u)_1)$, $(partial bold(J)(bold(v)_c, o, bold(U))) / (partial bold(u)_2)$, $dots.c$, $(partial bold(J)(bold(v)_c, o, bold(U))) / (partial bold(u)_(#mtext[|Vocab|]))$ (do not further expand these terms). No derivations are necessary, just an answer in the form of a matrix.

  #answer

  $
    (partial bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U))) / (partial bold(U))
    =
    [ (partial bold(J)(bold(v)_c, o, bold(U))) / (partial bold(u)_1),
      (partial bold(J)(bold(v)_c, o, bold(U))) / (partial bold(u)_2),
      dots,
      (partial bold(J)(bold(v)_c, o, bold(U))) / (partial bold(u)_(#mtext[|Vocab|])) ]
  $

+ (5 points)
  Compute the partial derivatives of $bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U))$ with respect to each of the "outside" word vectors, $bold(u)_w$'s. There will be two cases: when $w = o$, the true "outside" word vector, and $w != o$, for all other words. Please write your answer in terms of $bold(y)$, $hat(bold(y))$, and $bold(v)_c$. In this subpart, you may use specific elements within these terms as well (such as $bold(y)_1$, $bold(y)_2$, $dots$). Note that $bold(u)_w$ is a vector while $bold(y)_1$, $bold(y)_2$, $dots$ are scalars. Show your work to receive full credit.

  #answer

  $
    bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U)) = -bold(u)_o^T bold(v)_c + log sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c)
  $

  $
    nabla_(bold(u)_w) -bold(u)_o^T bold(v)_c = cases(
      -bold(v)_c & quad w = o,
      0 & quad w != o
    )
  $

  $
    nabla_(bold(u)_w) log sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c) = (exp(bold(u)_w^T bold(v)_c) / (sum_(w in#mtext[Vocab]) exp(bold(u)_w^T bold(v)_c))) dot bold(v)_c = hat(bold(y))_w dot bold(v)_c
  $

  Therefore, the partial derivative with respect to $bold(u)_w$ can be computed as:

  $
    nabla_(bold(u)_w) bold(J)_#mtext[naive-softmax] (bold(v)_c, o, bold(U)) = cases(
      -bold(v)_c + hat(bold(y))_o dot bold(v)_c & quad w = o,
      hat(bold(y))_w dot bold(v)_c & quad w != o
    ) = (hat(bold(y))_w - bold(y)_w) bold(v)_c
  $

+ (2 points)
  As an additional exercise for this problem, you will be taking the derivatives of some common loss functions, which may be used in variations of `word2vec` (such as the negative sampling variant). The Leaky ReLU (Leaky Rectified Linear Unit) activation function is given by @eq-leaky-relu and @fig-leaky-relu:

  $
    f(x) = max(alpha x, x)
  $ <eq-leaky-relu>

  #figure(
    image("../assets/leaky_relu_graph.png", width: 30%),
    caption: [Leaky ReLU],
  ) <fig-leaky-relu>

  Where $x$ is a scalar and $0 < alpha < 1$, please compute the derivative of $f(x)$ with respect to $x$. You may ignore the case where the derivative is not defined at 0.#footnote[
    If you're interested in how to handle the derivative at this point, you can read more about the notion of subderivatives.
  ]

  #answer

  $
    f'(x) = cases(
      alpha & quad x < 0,
      1 & quad x > 0
    )
  $

  This is because when $x < 0$, we have $alpha x < x$, so $f(x) = alpha x$ and therefore $f'(x) = alpha$. When $x > 0$, we have $x > alpha x$, so $f(x) = x$ and therefore $f'(x) = 1$. We ignore $x = 0$ as instructed, since the derivative is not defined there.

  At $x = 0$, the left-hand derivative is $alpha$ and the right-hand derivative is $1$, so they are not equal. Because of this, the ordinary derivative does not exist at $0$. A *subderivative* generalizes the notion of derivative for such non-smooth points: instead of a single slope, we allow a set of valid slopes. For Leaky ReLU at $x = 0$, the subderivatives form the interval $[alpha, 1]$. In optimization, one may pick any value in this interval as a generalized gradient.

+ (3 points)
  The sigmoid function is given by @eq-sigmoid:

  $
    sigma(x) = 1 / (1 + e^(-x)) = e^x / (e^x + 1)
  $ <eq-sigmoid>

  Please compute the derivative of $sigma(x)$ with respect to $x$, where $x$ is a scalar. Please write your answer in terms of $sigma(x)$. Show your work to receive full credit.

  #answer

  $
    sigma(x) = 1 / (1 + e^(-x))
  $

  $
    sigma'(x) & = -(1 + e^(-x))^(-2) dot (-e^(-x)) \
              & = e^(-x) / (1 + e^(-x))^2 \
              & = 1 / (1 + e^(-x)) dot e^(-x) / (1 + e^(-x)) \
              & = sigma(x) (1 - sigma(x))
  $

  Therefore,

  $
    sigma'(x) = sigma(x) (1 - sigma(x))
  $
