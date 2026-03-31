#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm

#show: style-algorithm.with(
  caption-align: start,
  breakable: true,
)

#import "../utils.typ": answer, mtext, unjustified

= Neural Transition-Based Dependency Parsing (40 points)

In this section, you'll be implementing a neural-network based dependency parser with the goal of maximizing performance on the UAS (Unlabeled Attachment Score) metric.

Before you begin, please follow the README to install all the needed dependencies for the assignment. We will be using PyTorch 2.1.2 from #link("https://pytorch.org/get-started/locally/")[https://pytorch.org/get-started/locally/] with the `CUDA` option set to `None`, and the `tqdm` package -- which produces progress bar visualizations throughout your training process. The official PyTorch website is a great resource that includes tutorials for understanding PyTorch's Tensor library and neural networks.

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between _head_ words, and words which modify those heads. There are multiple types of dependency parsers, including transition-based parsers, graph-based parsers, and feature-based parsers. Your implementation will be a _transition-based_ parser, which incrementally builds up a parse one step at a time. At every step it maintains a _partial parse_, which is represented as follows:

- A _stack_ of words that are currently being processed.
- A _buffer_ of words yet to be processed.
- A list of _dependencies_ predicted by the parser.

Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words of the sentence in order. At each step, the parser applies a _transition_ to the partial parse until its buffer is empty and the stack size is 1. The following transitions can be applied:

- `SHIFT`: removes the first word from the buffer and pushes it onto the stack.
- `LEFT-ARC`: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack, adding a _first_word_ $arrow$ _second_word_ dependency to the dependency list.
- `RIGHT-ARC`: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack, adding a _second_word_ $arrow$ _first_word_ dependency to the dependency list.

On each step, your parser will decide among the three transitions using a neural network classifier.

+ (4 points) Go through the sequence of transitions needed for parsing the sentence _"I presented my findings at the NLP conference"_. The dependency tree for the sentence is shown below. At each step, give the configuration of the stack and buffer, as well as what transition was applied this step and what new dependency was added (if any). The first three steps are provided below as an example.

  #figure(
    image("../assets/example.png", width: 80%),
  )

  #unjustified[
    #table(
      columns: (1fr, 1.5fr, 1fr, 1fr),
      align: (left, left, left, left),
      stroke: 0.4pt + luma(180),
      [*Stack*], [*Buffer*], [*New dependency*], [*Transition*],

      [[ROOT]],
      [[I, presented, my, findings, at, the, NLP, conference]],
      [],
      [Initial Configuration],

      [[ROOT, I]],
      [[presented, my, findings, at, the, NLP, conference]],
      [],
      [`SHIFT`],

      [[ROOT, I, presented]],
      [[my, findings, at, the, NLP, conference]],
      [],
      [`SHIFT`],

      [[ROOT, presented]],
      [[my, findings, at, the, NLP, conference]],
      [presented $arrow$ I],
      [`LEFT-ARC`],

      [[ROOT, presented, my]],
      [[findings, at, the, NLP, conference]],
      [],
      [`SHIFT`],

      [[ROOT, presented, my, findings]],
      [[at, the, NLP, conference]],
      [],
      [`SHIFT`],

      [[ROOT, presented, my]],
      [[at, the, NLP, conference]],
      [findings $arrow$ my],
      [`LEFT-ARC`],

      [[ROOT, presented]],
      [[at, the, NLP, conference]],
      [presented $arrow$ findings],
      [`RIGHT-ARC`],

      [[ROOT, presented, at]], [[the, NLP, conference]], [], [`SHIFT`],

      [[ROOT, presented, at, the]], [[NLP, conference]], [], [`SHIFT`],

      [[ROOT, presented, at, the, NLP]], [[conference]], [], [`SHIFT`],

      [[ROOT, presented, at, the, NLP, conference]], [[]], [], [`SHIFT`],

      [[ROOT, presented, at, the, conference]],
      [[]],
      [conference $arrow$ NLP],
      [`LEFT-ARC`],

      [[ROOT, presented, at, conference]],
      [[]],
      [conference $arrow$ the],
      [`LEFT-ARC`],

      [[ROOT, presented, conference]],
      [[]],
      [conference $arrow$ at],
      [`LEFT-ARC`],

      [[ROOT, presented]], [[]], [presented $arrow$ conference], [`RIGHT-ARC`],

      [[ROOT]], [[]], [presented $arrow$ ROOT], [`RIGHT-ARC`],
    )
  ]

+ (2 points) A sentence containing $n$ words will be parsed in how many steps (in terms of $n$)? Briefly explain in 1--2 sentences why.

  A sentence with $n$ words will be parsed in $2n$ steps. This is because each word must be shifted from the buffer to the stack (which takes $n$ steps), and then each word must be removed from the stack through either a `LEFT-ARC` or `RIGHT-ARC` transition (which also takes $n$ steps), resulting in a total of $2n$ steps.

+ (6 points) Implement the `__init__` and `parse_step` functions in the `PartialParse` class in `parser_transitions.py`. This implements the transition mechanics your parser will use. You can run basic (non-exhaustive) tests by running `python parser_transitions.py part_c`.

+ (8 points) Our network will predict which transition should be applied next to a partial parse. We could use it to parse a single sentence by applying predicted transitions until the parse is complete. However, neural networks run much more efficiently when making predictions about _batches_ of data at a time (i.e., predicting the next transition for any different partial parses simultaneously). We can parse sentences in minibatches with the following algorithm.

  #algorithm-figure(
    [Minibatch Dependency Parsing],
    inset: 0.4em,
    breakable: true,
    {
      import algorithmic: *
      Line[*Input:* `sentences`, a list of sentences to be parsed, and `model`, our model that makes parse decisions]
      LineBreak
      Line[Initialize `partial_parses` as a list of PartialParses, one for each sentence in `sentences`]
      Line[Initialize `unfinished_parses` as a shallow copy of `partial_parses`]
      While(
        [`unfinished_parses` is not empty],
        {
          Line[Take the first `batch_size` parses in `unfinished_parses` as a minibatch]
          Line[Use the `model` to predict the next transition for each partial parse in the minibatch]
          Line[Perform a parse step on each partial parse in the minibatch with its predicted transition]
          Line[Remove the completed (empty buffer and stack of size 1) parses from `unfinished_parses`]
        },
      )

      Return[`dependencies` for each parse in `partial_parses`]
    },
  )


  #unjustified[
    Implement this algorithm in the `minibatch_parse` function in `parser_transitions.py`. You can run basic (non-exhaustive) tests by running `python parser_transitions.py part_d`.
  ]

  _Note:_ You will need `minibatch_parse` to be correctly implemented to evaluate the model you will build in part (e). However, you do not need it to train the model, so you should be able to complete most of part (e) even if `minibatch_parse` is not implemented yet.

+ (20 points) We are now going to train a neural network to predict, given the state of the stack, buffer, and dependencies, which transition should be applied next.

  First, the model extracts a feature vector representing the current state. We will be using the feature set presented in the original neural dependency parsing paper: _A Fast and Accurate Dependency Parser using Neural Networks_.#footnote[
    Chen and Manning, 2014, #link("https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf")
  ] The function extracting these features has been implemented for you in `utils/parser_utils.py`. This feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.). They can be represented as a list of integers $bold(w) = [w_1, w_2, dots, w_m]$ where $m$ is the number of features and each $0 <= w_i < abs(V)$ is the index of a token in the vocabulary ($abs(V)$ is the vocabulary size). Then our network looks up an embedding for each word and concatenates them into a single input vector:

  $
    bold(x) = [bold(E)_(w_1), ..., bold(E)_(w_m)] in RR^(d m)
  $

  where $bold(E) in RR^(abs(V) times d)$ is an embedding matrix with each row $bold(E)_w$ as the vector for a particular word $w$ with dimension $d$. We then compute our prediction as:

  $
    bold(h) = #mtext[ReLU] (bold(x) bold(W) + bold(b)_1)
  $

  $
    bold(l) = bold(h) bold(U) + bold(b)_2
  $

  $
    hat(bold(y)) = #mtext[softmax] (bold(l))
  $

  where $bold(h)$ is referred to as the hidden layer, $bold(l)$ is referred to as the logits, $hat(bold(y))$ is referred to as the predictions, and $#mtext[ReLU] (z) = max(z, 0)$). We will train the model to minimize cross-entropy loss:

  $
    J(theta) = #mtext[CE] (bold(y), hat(bold(y))) = -sum_(j = 1)^3 bold(y)_j log hat(bold(y))_j
  $

  where $bold(y)_j$ denotes the $j$th element of $bold(y)$. To compute the loss for the training set, we average this $J(theta)$ across all training examples.

  + Compute the derivative of $bold(h) = #mtext[ReLU] (bold(x) bold(W) + bold(b)_1)$ with respect to $bold(x)$. For simplicity, you only need to show the derivative $(partial h_i) / (partial x_j)$ for some index $i$ and $j$. You may ignore the case where the derivative is not defined at 0.

    #answer

    Let $z = (bold(x) bold(W) + bold(b)_1)$, then $z_i = sum_(j=1)^(d m) x_j W_(j i) + b_(1 i)$

    $
      (partial h_i) / (partial x_j) &= (partial #mtext[ReLU] (z_i)) / (partial z_i) dot (partial z_i) / (partial x_j) \
      &= cases(
        0 & quad z_i <= 0,
        W_(j i) & quad z_i > 0,
      )
    $

  + Recall in part 1b, we computed the partial derivative of $bold(J)_"naive-softmax"(bold(v)_c, o, bold(U))$. Likewise, please compute the partial derivative of $J(theta)$ with respect to the $i$th entry of $bold(l)$, which is denoted as $bold(l)_i$. Specifically, compute $(partial #mtext[CE] (bold(y), hat(bold(y)))) / (partial bold(l)_i)$, assuming that $bold(l) in RR^3$, $hat(bold(y)) in RR^3$, $bold(y) in RR^3$, and the true label is $c$ (i.e., $y_j = 1$ if $j = c$). *Hint:* Use the chain rule: $(partial J)/(partial bold(l)) = (partial J)/(partial hat(bold(y))) dot (partial hat(bold(y)))/(partial bold(l))$.

    #answer

    $
      J(theta) & = #mtext[CE] (bold(y), hat(bold(y))) \
               & = - log hat(bold(y))_c \
               & = - log(exp(bold(l)_c) / (sum_(j=1)^3 exp(bold(l)_j))) \
               & = - bold(l)_c + log(sum_(j=1)^3 exp(bold(l)_j))
    $

    $
      (partial #mtext[CE] (bold(y), hat(bold(y)))) / (partial bold(l)_i) &= - (partial bold(l)_c) / (partial bold(l)_i) + (partial log(sum_(j=1)^3 exp(bold(l)_j))) / (partial bold(l)_i) \
      &= cases(
        -1 + exp(bold(l)_c) / (sum_(j=1)^3 exp(bold(l)_j)) & quad i = c,
        exp(bold(l)_i) / (sum_(j=1)^3 exp(bold(l)_j)) & quad i != c,
      ) \
      &= hat(bold(y))_i - y_i
    $

  + We will use UAS score as our evaluation metric. UAS refers to Unlabeled Attachment Score, which is computed as the ratio between number of correctly predicted dependencies and the number of total dependencies despite of the relations (our model doesn't predict this).

    In `parser_model.py` you will find skeleton code to implement this simple neural network using PyTorch. Complete the `__init__`, `embedding_lookup` and `forward` functions to implement the model. Then complete the `train_for_epoch` and `train` functions within the `run.py` file.

    Finally execute `python run.py` to train your model and compute predictions on test data from Penn Treebank (annotated with Universal Dependencies).

    *Note:*

    - For this assignment, you are asked to implement Linear layer and Embedding layer. Please *DO NOT* use *torch.nn.Linear* or *torch.nn.Embedding* module in your code, otherwise you will receive deductions for this problem.
    - Please follow the naming requirements in our TODO if there are any, e.g. if there are explicit requirements about variable names you have to follow them in order to receive full credits. You are free to declare other variable names if not explicitly required.

    *Hints:*

    #unjustified[
      - Each of the variables you are asked to declare (`self.embed_to_hidden_weight`, `self.embed_to_hidden_bias`, `self.hidden_to_logits_weight`, `self.hidden_to_logits_bias`) corresponds to one of the variables above ($bold(W)$, $bold(b)_1$, $bold(U)$, $bold(b)_2$).
    ]
    - It may help to work backwards in the algorithm (start from $hat(bold(y))$) and keep track of the matrix/vector sizes.
    - Once you have implemented `embedding_lookup (e)` or `forward (f)` you can call `python parser_model.py` with flag `-e` or `-f` or both to run sanity checks with each function. These sanity checks are fairly basic and passing them doesn't mean your code is bug free.
    - When debugging, you can add a debug flag: `python run.py -d`. This will cause the code to run over a small subset of the data, so that training the model won't take as long. Make sure to remove the `-d` flag to run the full model once you are done debugging.
    - When running with debug mode, you should be able to get a loss smaller than 0.2 and a UAS larger than 65 on the dev set (although in rare cases your results may be lower, there is some randomness when training).
    - It should take up to *15 minutes* to train the model on the entire training dataset, i.e., when debug mode is disabled.
    - When debug mode is disabled, you should be able to get a loss smaller than 0.08 on the train set and an Unlabeled Attachment Score larger than 87 on the dev set. For comparison, the model in the original neural dependency parsing paper gets 92.5 UAS. If you want, you can tweak the hyperparameters for your model (hidden layer size, hyperparameters for Adam, number of epochs, etc.) to improve the performance (but you are not required to do so).

    *Deliverables:*

    - Working implementation of the transition mechanics that the neural dependency parser uses in `parser_transitions.py`.
    - Working implementation of minibatch dependency parsing in `parser_transitions.py`.
    - Working implementation of the neural dependency parser in `parser_model.py`. (We'll look at and run this code for grading).
    - Working implementation of the functions for training in `run.py`. (We'll look at and run this code for grading).
    - Please use efficient functions and *avoid for loops* when implementing `embedding_lookup`. *Otherwise, you may exceed the GradeScope test time limit.*
    - *Report the best UAS your model achieves on the dev set and the UAS it achieves on the test set in your written submission*. You can report it in the PDF and tag the page.
