#import "../utils.typ": answer, mtext, proof

= Neural Networks Optimization (8 points)

+ (4 points) Adam Optimizer

  Recall the standard Stochastic Gradient Descent update rule:

  $
    bold(theta)_(t+1) <- bold(theta)_t - alpha nabla_(bold(theta)_t) J_#mtext[minibatch] (bold(theta)_t)
  $

  where $t + 1$ is the current timestep, $bold(theta)$ is a vector containing all of the model parameters, ($bold(theta)_t$ is the model parameter at time step $t$, and $bold(theta)_(t+1)$ is the model parameter at time step $t + 1$), $J$ is the loss function, $nabla_(bold(theta)) J_#mtext[minibatch] (bold(theta))$ is the gradient of the loss function with respect to the parameters on a minibatch of data, and $alpha$ is the learning rate. Adam Optimization#footnote[
    Kingma and Ba, 2015, #link("https://arxiv.org/pdf/1412.6980.pdf")
  ] uses a more sophisticated update rule with two additional steps.#footnote[
    The actual Adam update uses a few additional tricks that are less important, but we won't worry about them here. If you want to learn more about it, you can take a look at: #link("http://cs231n.github.io/neural-networks-3/#sgd")
  ]

  + (2 points) First, Adam uses a trick called _momentum_ by keeping track of $bold(m)$, a rolling average of the gradients:

    $
      bold(m)_(t+1) <- beta_1 bold(m)_t + (1 - beta_1) nabla_(bold(theta)_t) J_#mtext[minibatch] (bold(theta)_t)
    $

    $
      bold(theta)_(t+1) <- bold(theta)_t - alpha bold(m)_(t+1)
    $

    where $beta_1$ is a hyperparameter between 0 and 1 (often set to 0.9). *Briefly explain in 2--4 sentences* (you don't need to prove mathematically, just give an intuition) how using $bold(m)$ stops the updates from varying as much and why this low variance may be helpful to learning, overall.

    #answer

    Momentum reduces update variance because $bold(m)$ averages the current gradient with past gradients, so a single noisy minibatch has less influence on the update direction. As a result, the parameter updates become smoother and less likely to oscillate sharply from step to step. This lower-variance update is helpful because it allows learning to make steadier progress, especially in directions where minibatch gradients are noisy or inconsistent.

  + (2 points) Adam extends the idea of _momentum_ with the trick of _adaptive learning rates_ by keeping track of $bold(v)$, a rolling average of the magnitudes of the gradients:

    $
      bold(m)_(t+1) <- beta_1 bold(m)_t + (1 - beta_1) nabla_(bold(theta)_t) J_#mtext[minibatch] (bold(theta)_t)
    $
    $
      bold(v)_(t+1) <- beta_2 bold(v)_t + (1 - beta_2) (nabla_(bold(theta)_t) J_#mtext[minibatch] (bold(theta)_t) dot.o nabla_(bold(theta)_t) J_#mtext[minibatch] (bold(theta)_t))
    $
    $
      bold(theta)_(t+1) <- bold(theta)_t - alpha bold(m)_(t+1) / sqrt(bold(v)_(t+1))
    $

    where $dot.o$ and $\/$ denote elementwise multiplication and division (so $bold(z) dot.o bold(z)$ is elementwise squaring) and $beta_2$ is a hyperparameter between 0 and 1 (often set to 0.99). Since Adam divides the update by $sqrt(bold(v))$, which of the model parameters will get larger updates? Why might this help with learning? *Briefly explain in 2--4 sentences*.

    #answer

    The model parameters with smaller values in $bold(v)$ will receive larger updates, because dividing by $sqrt(bold(v))$ increases the effective step size for parameters whose recent gradients have been small. Conversely, parameters with consistently large gradients get smaller updates. This helps learning by preventing large-gradient parameters from dominating the optimization while allowing slower-changing parameters to make meaningful progress, leading to more balanced and stable training.

+ (4 points) Dropout#footnote[
    Srivastava et al., 2014, #link("https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf")
  ] is a regularization technique. During training, dropout randomly sets units in the hidden layer $bold(h)$ to zero with probability $p_"drop"$ (dropping different units each minibatch), and then multiplies $bold(h)$ by a constant $gamma$. We can write this as:

  $
    bold(h)_"drop" = gamma bold(d) dot.o bold(h)
  $

  where $bold(d) in {0, 1}^(D_h)$ ($D_h$ is the size of $bold(h)$) is a mask vector where each entry is 0 with probability $p_"drop"$ and 1 with probability $(1 - p_"drop")$. $gamma$ is chosen such that the expected value of $bold(h)_"drop"$ is $bold(h)$:

  $
    bb(E)_(p_"drop") [bold(h)_"drop"]_i = h_i
  $

  for all $i in {1, dots, D_h}$.

  + (2 points) What must $gamma$ equal in terms of $p_"drop"$? Briefly justify your answer or show your math derivation using the equations given above.

    #answer

    We can derive $gamma$ by calculating the expected value of $bold(h)_"drop"$:

    $
      bb(E)_(p_"drop") [bold(h)_"drop"]_i = bb(E)_(p_"drop") [gamma bold(d)_i h_i] = gamma h_i bb(E)_(p_"drop") [bold(d)_i] = gamma h_i (1 - p_"drop") = h_i
    $

    Solving for $gamma$ gives:

    $
      gamma = 1 / (1 - p_"drop")
    $

  + (2 points) Why should dropout be applied during training? Why should dropout *not* be applied during evaluation? *Briefly explain in 2--4 sentences.* *Hint:* It may help to look at the dropout paper linked.

  Dropout is applied during training to prevent neurons from "co-adapting" (relying too heavily on specific neighbors), which forces the network to learn more robust, independent features and acts as an efficient approximation of training an ensemble of exponentially many neural networks. It should not be applied during evaluation (testing) because we want the model to utilize its full capacity and learned weights to make deterministic, stable predictions, rather than introducing random noise that would hinder accuracy.
