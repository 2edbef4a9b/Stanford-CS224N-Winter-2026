#import "../utils.typ": answer, mtext, proof

= Position Embeddings Exploration (6 points)

Position embeddings are an important component of the Transformer architecture, allowing the model to differentiate between tokens based on their position in the sequence.
In this question, we explore the need for positional embeddings in Transformers and how they can be designed.

Recall that the crucial components of the Transformer architecture are the self-attention layer and the feed-forward neural network layer.
Given an input tensor $bold(X) in RR^(T times d)$, where $T$ is the sequence length and $d$ is the hidden dimension, the self-attention layer computes

$
  bold(Q) = bold(X) bold(W)_Q, quad
  bold(K) = bold(X) bold(W)_K, quad
  bold(V) = bold(X) bold(W)_V
$

$
  bold(H) = #mtext[softmax] ((bold(Q) bold(K)^T) / sqrt(d)) bold(V)
$

where $bold(W)_Q, bold(W)_K, bold(W)_V in RR^(d times d)$ are weight matrices, and $bold(H) in RR^(T times d)$ is the output.

Next, the feed-forward layer applies

$
  bold(Z) = #mtext[ReLU] (bold(H) bold(W)_1 + bold(1) dot bold(b)_1) bold(W)_2 + bold(1) dot bold(b)_2
$

where $bold(W)_1, bold(W)_2 in RR^(d times d)$ and $bold(b)_1, bold(b)_2 in RR^(1 times d)$ are weights and biases; $bold(1) in RR^(T times 1)$ is a vector of ones#footnote[
  Outer product with $bold(1)$ represents broadcasting and makes the feed-forward notation mathematically sound.
]; and $bold(Z) in RR^(T times d)$ is the final output.

(Note that we have omitted some details of the Transformer architecture for simplicity.)

+ (4 points) *Permuting the input.*

  + (3 points)
    Suppose we permute the input sequence $bold(X)$ such that the tokens are shuffled randomly. This can be represented as multiplication by a permutation matrix $bold(P) in RR^(T times T)$, i.e. $bold(X)_#mtext[perm] = bold(P) bold(X)$. (See #link("https://en.wikipedia.org/wiki/Permutation_matrix")[Wikipedia] for a recap on permutation matrices.)

    *Show* that the output $bold(Z)_#mtext[perm]$ for the permuted input $bold(X)_#mtext[perm]$ will be $bold(Z)_#mtext[perm] = bold(P) bold(Z)$.

    You are given that for any permutation matrix $bold(P)$ and any matrix $bold(A)$, the following hold:

    $
      #mtext[softmax] (bold(P) bold(A) bold(P)^T) = bold(P) #mtext[softmax] (bold(A)) bold(P)^T
    $

    $
      #mtext[ReLU] (bold(P) bold(A)) = bold(P) #mtext[ReLU] (bold(A)).
    $

    #proof

    $
      Q_#mtext[perm] = bold(X)_#mtext[perm] bold(W)_Q = bold(P) bold(X) bold(W)_Q = bold(P) bold(Q)
    $

    $
      K_#mtext[perm] = bold(X)_#mtext[perm] bold(W)_K = bold(P) bold(X) bold(W)_K = bold(P) bold(K)
    $

    $
      V_#mtext[perm] = bold(X)_#mtext[perm] bold(W)_V = bold(P) bold(X) bold(W)_V = bold(P) bold(V)
    $

    $
      H_#mtext[perm] &= #mtext[softmax] ((Q_#mtext[perm] K_#mtext[perm]^T) / sqrt(d)) V_#mtext[perm] \
      &= #mtext[softmax] ((bold(P) bold(Q) (bold(P) bold(K))^T) / sqrt(d)) bold(P) bold(V) \
      &= #mtext[softmax] ((bold(P) bold(Q) bold(K)^T bold(P)^T) / sqrt(d)) bold(P) bold(V) \
      &= bold(P) #mtext[softmax] ((bold(Q) bold(K)^T) / sqrt(d)) bold(P)^T bold(P) bold(V) \
      &= bold(P) #mtext[softmax] ((bold(Q) bold(K)^T) / sqrt(d)) bold(V) \
      &= bold(P) bold(H)
    $

    $
      Z_#mtext[perm] &= #mtext[ReLU] (H_#mtext[perm] bold(W)_1 + bold(1) dot bold(b)_1) bold(W)_2 + bold(1) dot bold(b)_2 \
      &= #mtext[ReLU] (bold(P) bold(H) bold(W)_1 + bold(1) dot bold(b)_1) bold(W)_2 + bold(1) dot bold(b)_2 \
      &= #mtext[ReLU] (bold(P) (bold(H) bold(W)_1 + bold(1) dot bold(b)_1)) bold(W)_2 + bold(1) dot bold(b)_2 \
      &= bold(P) #mtext[ReLU] (bold(H) bold(W)_1 + bold(1) dot bold(b)_1) bold(W)_2 + bold(1) dot bold(b)_2 \
      &= bold(P) (#mtext[ReLU] (bold(H) bold(W)_1 + bold(1) dot bold(b)_1) bold(W)_2 + bold(1) dot bold(b)_2) \
      &= bold(P) bold(Z)
    $

  + (1 point)
    Think about the implications of the result you derived above. *Explain* why this property of the Transformer model could be problematic when processing text.

    #answer

    The mathematical result $bold(Z)_#mtext[perm] = bold(P) bold(Z)$ proves that the core Transformer architecture (without position embeddings) is permutation equivariant. This means that if you shuffle the order of the input words, the model will output the exact same contextualized representations, just shuffled in the exact same order.

    This property is highly problematic for processing natural language because word order defines meaning (syntax and grammar). For example, the sentence "The dog bit the man" and "The man bit the dog" contain the exact same words, but have completely different meanings. Without positional embeddings, the self-attention mechanism treats the sentence as a mere "bag of words" and computes the exact same representation for the word "dog" in both sentences. The model would be completely blind to sequence, directionality, and grammatical structure.

+ (2 points) *Position embeddings* are vectors that encode the position of each token in the sequence. They are added to the input word embeddings before feeding them into the Transformer.

  One approach is to generate position embeddings using a fixed function of position and embedding dimension.
  If the input word embeddings are $bold(X) in RR^(T times d)$, position embeddings $Phi in RR^(T times d)$ are generated as follows:

  $
    Phi_(t, 2i) = sin(t \/ 10000^(2i\/d))
  $

  $
    Phi_(t, 2i + 1) = cos(t \/ 10000^(2i\/d))
  $

  where $t in {0, 1, dots, T - 1}$ and $i in {0, 1, dots, d/2 - 1}$#footnote[
    Here $d$ is assumed even, which is typically the case for most models.
  ].

  Specifically, position embeddings are added to input word embeddings:

  $
    bold(X)_#mtext[pos] = bold(X) + Phi.
  $

  + (1 point)
    Do you think the position embeddings will help the issue identified in part (a)? If yes, explain how; if not, explain why not.

    #answer

    Yes. The position embeddings effectively break the permutation equivariance identified in part (a). By adding $Phi$ to the word embeddings $bold(X)$, the resulting input $bold(X)_#mtext[pos]$ becomes dependent on the absolute position $t$. Even if two words are identical (i.e., $bold(X)_(t_1) = bold(X)_(t_2)$), they will have different input representations after the addition ($bold(X)_(t_1) + Phi_(t_1) != bold(X)_(t_2) + Phi_(t_2)$) because $Phi$ is unique for each position. Consequently, self-attention can distinguish different orderings of the same tokens, allowing the model to capture sequence order and syntax.


  + (1 point)
    Can the position embeddings for two different tokens in the input sequence be the same? If yes, provide an example. If not, explain why not.

    #answer

    No. The position embedding for each position $t$ is a $d$-dimensional vector where each pair of dimensions $(2i, 2i + 1)$ encodes a sine/cosine signal at a different frequency. For two distinct positions $t_1$ and $t_2$ to have the same embedding, the values would need to match across all $d/2$ frequencies simultaneously. Because the wavelengths form a geometric progression (roughly from $2 pi$ up to $2 pi dot 10000$), it is (within any practical sequence length $T$) not possible for two distinct integers $t_1 != t_2$ to produce exactly the same vector across all dimensions.
