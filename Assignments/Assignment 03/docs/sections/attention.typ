#import "../utils.typ": answer, mtext

= Attention Exploration (14 points)

Multi-head self-attention is the core modeling component of Transformers. In this question, you'll get some practice working with the self-attention equations, and motivate why multi-headed self-attention can be preferable to single-headed self-attention.

Recall that attention can be viewed as an operation on a _query_ vector $q in RR^d$, a set of _value_ vectors ${v_1, dots, v_n}$, $v_i in RR^d$, and a set of _key_ vectors ${k_1, dots, k_n}$, $k_i in RR^d$, specified as follows:

$
  c = sum_(i=1)^n v_i alpha_i
$

$
  alpha_i = exp(k_i^T q) / (sum_(j=1)^n exp(k_j^T q))
$

with $alpha = {alpha_1, dots, alpha_n}$ termed the "attention weights". Observe that the output $c in RR^d$ is an average over the value vectors weighted with respect to $alpha$.

+ (3 points) *Copying in attention.* One advantage of attention is that it's particularly easy to "copy" a value vector to the output $c$. In this problem, you'll motivate why this is the case.

  + (2 points) The distribution $alpha$ is typically relatively "diffuse"; the probability mass is spread out between many different $alpha_i$. However, this is not always the case. *Describe* (in one sentence) under what conditions the categorical distribution $alpha$ puts almost all of its weight on some $alpha_j$, where $j in {1, dots, n}$ (i.e. $alpha_j >> sum_(i != j) alpha_i$). What must be true about the query $q$ and/or the keys ${k_1, dots, k_n}$?

    #answer

    This occurs when the dot product between the query $q$ and the key $k_j$ is significantly larger than the dot products with all other keys (i.e., $k_j^T q >> k_i^T q$ for all $i != j$), which geometrically means $q$ is highly aligned with $k_j$ and has a sufficiently large magnitude, while being roughly orthogonal to or pointing away from the other keys.

  + (1 point) Under the conditions you gave in (i), *describe* the output $c$.

    #answer

    Under these conditions, since $alpha_j approx 1$ and all other $alpha_i approx 0$, the output vector $c$ will be approximately equal to the value vector $v_j$ (i.e., $c approx v_j$), effectively "copying" the specific value vector to the output.

+ (2 points) *An average of two.* Instead of focusing on just one vector $v_j$, a Transformer model might want to incorporate information from _multiple_ source vectors.

  Consider the case where you instead want to incorporate information from *two* vectors $v_a$ and $v_b$, with corresponding key vectors $k_a$ and $k_b$. Assume that (1) all key vectors are orthogonal, so $k_i^T k_j = 0$ for all $i != j$; and (2) all key vectors have norm $1$. *Find an expression* for a query vector $q$ such that $c approx 1/2 (v_a + v_b)$, and *justify your answer*.#footnote[
    Hint: while the softmax function will never _exactly_ average the two vectors, you can get close by using a large scalar multiple in the expression.
  ]
  (Recall what you learned in part (a).)

  #answer

  $
    q = lambda (k_a + k_b)
  $

  where $lambda$ is a large scalar.

  This works because the dot products $k_a^T q$ and $k_b^T q$ will both be equal to $lambda$, while the dot products with all other keys will be zero. As $lambda$ becomes large, the softmax will assign approximately equal weight to $v_a$ and $v_b$, resulting in $c approx 1/2 (v_a + v_b)$.

+ (5 points) *Drawbacks of single-headed attention.* In the previous part, you saw how it was _possible_ for a single-headed attention to focus equally on two values. The same concept could easily be extended to any subset of values. In this question you'll see why it's not a _practical_ solution.

  Consider a set of key vectors ${k_1, dots, k_n}$ that are now randomly sampled, $k_i ~ cal(N)(mu_i, Sigma_i)$, where the means $mu_i in RR^d$ are known to you, but the covariances $Sigma_i$ are unknown (unless specified otherwise in the question). Further, assume that the means $mu_i$ are all perpendicular; $mu_i^T mu_j = 0$ if $i != j$, and unit norm, $norm(mu_i)=1$.

  + (2 points) Assume that the covariance matrices are $Sigma_i = alpha I$, for vanishingly small $alpha$, for all $i in {1, 2, dots, n}$. Design a query $q$ in terms of the $mu_i$ such that as before, $c approx 1/2 (v_a + v_b)$, and provide a brief argument as to why it works.

    #answer

    $
      q = lambda (mu_a + mu_b)
    $

    where $lambda$ is a large scalar.

    Because $alpha$ is vanishingly small, we can approximate $k_i approx mu_i$. Using the given orthogonal and unit norm properties, the dot products are $k_a^T q approx lambda(mu_a^T mu_a + mu_a^T mu_b) = lambda(1 + 0) = lambda$, and similarly $k_b^T q approx lambda$. For all other keys $i in.not {a, b}, k_i^T q approx 0$. When we pass these dot products through the softmax function, we get $alpha_a = alpha_b approx exp(lambda) / (2 exp(lambda) + n - 2)$. Since $lambda$ is a very large scalar, $exp(lambda)$ dominates the constant $n - 2$, causing both $alpha_a$ and $alpha_b$ to perfectly converge to $0.5$, and all other $alpha_i$ to converge to $0$. Thus, $c approx 1/2 (v_a + v_b)$.

  + (3 points) Though single-headed attention is resistant to small perturbations in the keys, some types of larger perturbations may pose a bigger issue. In some cases, one key vector $k_a$ may be larger or smaller in norm than the others, while still pointing in the same direction as $mu_a$.#footnote[
      Unlike the original Transformer, some newer Transformer models apply layer normalization before attention. In these pre-layernorm models, norms of keys cannot be too different which makes the situation in this question less likely to occur.
    ]

    As an example, consider a covariance for item $a$ as $Sigma_a = alpha I + 1/2 (mu_a mu_a^T)$ for vanishingly small $alpha$. This causes $k_a$ to point in roughly the same direction as $mu_a$, but with large variances in magnitude. Further, let $Sigma_i = alpha I$ for all $i != a$.

    #figure(
      image("../assets/images/ka_plausible.png", width: 35%),
      caption: [The vector $mu_a$ (shown here in 2D as an example), with the range of possible values of $k_a$ shown in red. As mentioned previously, $k_a$ points in roughly the same direction as $mu_a$, but may have larger or smaller magnitude.],
    )

    When you sample ${k_1, dots, k_n}$ multiple times, and use the $q$ vector that you defined in part (i), what do you expect the vector $c$ will look like qualitatively for different samples? Think about how it differs from part (i) and how $c$'s variance would be affected.

    #answer

    Under this new covariance, $k_a$ fluctuates significantly in magnitude along the direction of $mu_a$, meaning $k_a approx c_a mu_a$ where $c_a$ is a random scalar that can be larger or smaller than 1. When we use the same $q = lambda(mu_a + mu_b)$, the dot products become $k_a^T q approx lambda c_a$ and $k_b^T q approx lambda$.

    Because $lambda$ is an extremely large scalar, the softmax function becomes highly sensitive to even minor differences between $c_a$ and $1$. If $c_a > 1$ in a given sample, the softmax will assign almost 100% of the weight to $v_a$ ($c approx v_a$); if $c_a < 1$, it will assign almost 100% of the weight to $v_b$ ($c approx v_b$). Unlike part (i) where $c$ was a stable average of $0.5(v_a + v_b)$, the output $c$ here will randomly and erratically flip-flop between being almost entirely $v_a$ or almost entirely $v_b$ across different samples, causing the variance of $c$ to become extremely high.

+ (3 points) *Benefits of multi-headed attention.* Now you'll see some of the power of multi-headed attention. You'll consider a simple version of multi-headed attention which is identical to single-headed self-attention as presented above, except two query vectors ($q_1$ and $q_2$) are defined, which leads to a pair of vectors ($c_1$ and $c_2$), each the output of single-headed attention given its respective query vector. The final output of the multi-headed attention is their average, $1/2 (c_1 + c_2)$.

  As in question (c), consider a set of key vectors ${k_1, dots, k_n}$ that are randomly sampled, $k_i ~ cal(N)(mu_i, Sigma_i)$, where the means $mu_i$ are known to you, but the covariances $Sigma_i$ are unknown. Also as before, assume that the means $mu_i$ are mutually orthogonal; $mu_i^T mu_j = 0$ if $i != j$, and unit norm, $norm(mu_i)=1$.

  + (1 point) Assume that the covariance matrices are $Sigma_i = alpha I$, for vanishingly small $alpha$. Design $q_1$ and $q_2$ in terms of $mu_i$ such that $c$ is approximately equal to $1/2 (v_a + v_b)$. Note that $q_1$ and $q_2$ should have different expressions.

    #answer

    $
      q_1 = lambda mu_a quad q_2 = lambda mu_b
    $

    where $lambda$ is a very large scalar.

    With $q_1 = lambda mu_a$, the dot product $k_a^T q_1 approx mu_a^T mu_a lambda = lambda$, while for all other keys $i != a$, we have $k_i^T q_1 approx 0$ due to orthogonality. Since $lambda$ is very large, the softmax assigns almost all the weight to $v_a$, making $c_1 approx v_a$.

    Similarly, with $q_2 = lambda mu_b$, the dot product $k_b^T q_2 approx lambda$ and $k_i^T q_2 approx 0$ for all $i != b$, so the softmax assigns almost all the weight to $v_b$, making $c_2 approx v_b$.

    Averaging the two outputs yields

    $
      c = 1/2 (c_1 + c_2) approx 1/2 (v_a + v_b)
    $

  + (2 points) Assume that the covariance matrices are $Sigma_a = alpha I + 1/2 (mu_a mu_a^T)$ for vanishingly small $alpha$, and $Sigma_i = alpha I$ for all $i != a$. Take the query vectors $q_1$ and $q_2$ that you designed in part (i). What, qualitatively, do you expect the output $c$ to look like across different samples of the key vectors? Explain briefly in terms of variance in $c_1$ and $c_2$. You can ignore cases in which $k_a^T q_i < 0$.

    #answer

    Under this new covariance, $k_a$ fluctuates significantly in magnitude along the direction of $mu_a$. However, $c$ remains highly stable and consistently approximately $1/2 (v_a + v_b)$ across different samples, resulting in near-zero variance for $c_1$, $c_2$, and $c$.

    *Explanation:*

    - For $c_1$: We have $k_a^T q_1 approx lambda c_a$ (where $c_a > 0$ is the fluctuating magnitude along $mu_a$). Since we ignore cases where $k_a^T q_1 < 0$, the term $lambda c_a$ remains a very large positive number compared to the near-zero dot products of other keys. Thus, the softmax still confidently outputs $alpha_a approx 1$, keeping $c_1 approx v_a$ stably. The variance of $c_1$ is extremely low.
    - For $c_2$: The fluctuating key $k_a$ is orthogonal to $mu_b$, so $k_a^T q_2 approx 0$. Therefore the fluctuation of $k_a$ has essentially no effect on the second head. The dot product $k_b^T q_2 approx lambda$ remains dominant, so $c_2 approx v_b$ stably. The variance of $c_2$ is also extremely low.
    - Consequently, their average $c = 1/2 (c_1 + c_2)$ remains highly stable regardless of magnitude perturbations of $k_a$.

+ (1 point) Based on part (d), briefly summarize how multi-headed attention overcomes the drawbacks of single-headed attention that you identified in part (c).

  #answer

  Single-headed attention struggles to attend to multiple targets simultaneously because the softmax function's "winner-takes-all" nature makes it highly unstable and extremely sensitive to the natural magnitude fluctuations of key vectors (leading to high variance). Multi-headed attention overcomes this drawback by allowing specialization: each individual head can robustly focus on a single specific token (which softmax is perfectly suited for), and the network can safely aggregate (average or concatenate) these stable, low-variance outputs to successfully capture information from multiple sources at once.
