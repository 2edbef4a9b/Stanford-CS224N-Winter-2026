#import "../utils.typ": answer, mtext, unjustified

= Coding a Transformer From Scratch (30 points)

In this question you will fill in code to implement a decoder only, GPT-2 style transformer, and a simple training loop.

For part (a), we have included unit tests for each sub problem that can run locally on your laptop. You will be awarded full points for the subproblem if you pass the unit test. Do not edit the unit test file as we will separately be running the tests when you submit your code.

The following tips might be useful during this part of the assignment:

- Add assert statements to check the shape of tensors matches what you think it should be.
- Consider the #link("https://docs.kidger.site/jaxtyping/")[Jaxtyping] package to type hint the shape of tensors.
- Consider the #link("https://einops.rocks/")[einops] package for manipulating tensors (`einops.rearrange` is particularly useful).

This will help you not only write less buggy code, but also make your code far more readable.

+ (20 points) In this part of the question we will implement a transformer in the `model_solution.py` file. The file contains a number of different classes that you will implement. In the end, you will have an implementation of the `Transformer` class with functioning `forward` and `generate` methods. In part (b), we will (start to) train your implementation of `Transformer`.

  + (0 points) Familiarize yourself with the classes in the `model_solution.py` file. We will ask you to implement them in the order `MLP`, `CausalAttention`, `DecoderBlock`, and finally `Transformer`. We will get you to implement the classes in this order because it is the order of dependence. `Transformer` depends on `DecoderBlock`, that in turn depends on `CausalAttention` and `MLP`.

  + (1 point) Implement `MLP.forward`. Check you pass the corresponding test.

  + (6 points) Implement `CausalAttention.forward`. Check you pass the corresponding test.

  + (2 points) Implement `DecoderBlock.forward`. Check you pass the corresponding test.

  + (6 points) Implement `Transformer.forward`. Check you pass the corresponding test.

  + (5 points) Implement `Transformer.generate`. Check you pass the corresponding test. Note: you should implement greedy decoding for this function.

+ (10 points) After finishing part (a), you now have a functioning Transformer model. If you look at `Transformer.__init__`  we can see that when you create an instance of the `Transformer` class, we initialize the model with random weights according to the `Transformer._init_weights` method. In this part of the question, you will implement a training loop, and start training a small model locally on your laptop.

  + (0 points) Look in the `train.py` file and familiarize yourself with the training loop. We will run this code to train the model.

  + (7 points) First, implement `Transformer.get_loss_on_batch`. This function maps a batch of tokens to a single loss value. We use this function in `train.py` to get the loss over a batch. Check you pass the corresponding test.

  #unjustified[
    + (3 points) Run `train.py`. This will train the model, using your `Transformer.get_loss_on_batch` on 100 batches of data. At the end of training it will save a graph of the training loss and gradient norm over training to `losses_and_grad_norms.png`, include an image of this below.

      If everything is correct, you should see a decreasing loss curve.
  ]

+ (9 points) *(Bonus)*
  In this optional bonus question, your goal is to speed up learning. We will keep the number of gradient steps fixed at 100, however you can change anything else in `train.py` or `model.py` to speed up training.

  We will consider a change successful if the final loss after 100 steps is lower than the baseline curve you reported in part (b)(iii).

  We will award 3 points for each different change that leads to a speedup. Thus, for full points, you need three different changes to the training file, each of which leads to a speedup. These changes should   compound. For example, you may begin by changing the learning rate, leading to a lower loss. You then might keep this better learning rate, and combine it with a second change (e.g., changing the optimizer or model architecture), that leads to an even lower loss. Note that changing the learning rate to three different values counts as one idea: we are looking for three different types of ideas.

  When you are done, submit:

  - A description of each change you made.
  - Up to three new learning curves, one for each change you made; and additionally include the baseline from (b)(iii).
  - The lowest loss you achieved after 100 steps.

  #answer

  + Baseline

    Results:

    - Final loss after 100 steps: `10.767414`
    - Best loss during training: `10.766797`
    - Final gradient norm: `1.461107`

    #figure(
      image("../optimization/baseline.png"),
      caption: [Baseline training curve from part (b)(iii).],
    )

  + Change 01: Increase the learning rate

    The first change was to increase the learning rate from `1e-5` to `1e-3`.

    Code Change:

    ```diff
         train(
    -        learning_rate=1e-5,
    +        learning_rate=1e-3,
             gradient_clipping=1,
             model_config=tiny_model_config,
             batch_size=16,
    ```

    Results:

    - Final loss after 100 steps: `6.451349`
    - Best loss during training: `6.451349`
    - Final gradient norm: `0.696417`

    This produced a large improvement over the baseline, suggesting that the original training configuration was optimization-limited.

    #figure(
      image("../optimization/optimization01.png"),
      caption: [Training curve after increasing the learning rate.],
    )

  + Change 02: Increase model size

    The second change was to increase the model size. The original model had `d_model=33` and `n_heads=3`. We increased this to `d_model=64` and `n_heads=4`.

    Code Change:

    ```diff
     if __name__ == "__main__":
         tiny_model_config = ModelConfig(
    -        d_model=33,
    -        n_heads=3,
    +        d_model=64,
    +        n_heads=4,
             n_layers=3,
             context_length=512,
             vocab_size=50257,
         )
    ```

    Results:

    - Final loss after 100 steps: `5.856569`
    - Best loss during training: `5.803541`
    - Final gradient norm: `0.582588`

    This further improved the final loss, indicating that the original model was also somewhat capacity-limited.

    #figure(
      image("../optimization/optimization02.png"),
      caption: [Training curve after increasing the model size.],
    )

  + Change 03: Reduce context length

    Finally, keeping the larger model and higher learning rate, we reduced the context length from `512` to `128`. This means the model has to learn to predict tokens based on a smaller context, which should make the task easier.

    Code Change:

    ```diff
     if __name__ == "__main__":
         tiny_model_config = ModelConfig(
             d_model=64,
             n_heads=4,
             n_layers=3,
    -        context_length=512,
    +        context_length=128,
             vocab_size=50257,
         )
    ```

    Results:

    - Final loss after 100 steps: `5.713151`
    - Best loss during training: `5.553332`
    - Final gradient norm: `0.747328`

    This produced the best final loss among all of the experiments.

    #figure(
      image("../optimization/optimization03.png"),
      caption: [Training curve after reducing the context length.],
    )

  + Lowest loss achieved

    The lowest loss achieved after 100 steps was `5.713151`, which was achieved after all three changes were implemented.
