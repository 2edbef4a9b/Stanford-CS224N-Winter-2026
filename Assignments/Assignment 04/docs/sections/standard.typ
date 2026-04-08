#import "@preview/lilaq:0.6.0" as lq
#import "../utils.typ": answer

= Standard Benchmarking (10 points)

In this problem, you will evaluate the properties of various different LLMs that we give you access to through an API using standard benchmarking techniques.

+ (0 points) Begin by reading the `query.py` file. This includes the `query_model` function that you will use to access the models in this question. The function accepts a `query` and `model_id`. It returns the response from the model corresponding to `model_id`. The cost is charged to your GCP billing account. *Everyone has a fixed amount of credits that should last the entire problem set. Keep a close eye on how much money you have used.* You can do this by accessing the billing page on the GCP dashboard.

  To avoid overspending, try to test your code on a small number of inputs. For example, in this question you will run your code on 100 questions, but you can test your code using a smaller number, say 10. This will both be quicker to run and iterate on for testing, and be cheaper. When you feel confident your implementation is correct, you can run it on all 100 examples. (Tip: when running on the full dataset, use `tqdm` for convenient progress bars.)

  In the following questions, we will refer to models by their `model_id`, which will simply be a single capital letter, e.g. model A or B.

+ (10 points) In this question, you will benchmark a model on the GSM8K benchmark.

  + (7 points) We have provided you with 100 problems from the GSM8K benchmark in the `gsm8k_first_100.jsonl` file. Benchmark models A and B on this data using a string-match evaluation. We have provided you with the `standard_prompt_template` function that you should use to prompt the models, and the `standard_output_extractor` that you should use to get the numerical answer from the model outputs.

    Include all the code you used in the `gsm8k.py` file. In addition, write the scores you get for models A and B.

    #answer

    Models A and B were evaluated on the 100 GSM8K examples using exact string-match on the extracted final answer. Outputs that could not be parsed by `standard_output_extractor` were counted as incorrect.

    *Scores*:

    - Model A: `93/100`
    - Model B: `51/100`

    *Invalid Output Rates*:

    - Model A: `0/100`
    - Model B: `8/100`

    Retry logic with exponential backoff was added in `client/query.py` so that temporary `429 RESOURCE_EXHAUSTED` API errors would be retried automatically instead of crashing the full evaluation run.

  + (3 points) Now develop your own prompt template to increase the performance of model A. Write your prompt template by completing the `superior_prompt_template` function in `gsm8k.py`. Use this function to evaluate the model by filling in `eval_model_on_gsm8k_with_improved_prompt`, and record the performance of model A. Below, include:

    - What your prompt was.
    - Why you thought this would improve model performance.
    - A bar chart of performance with the standard prompt and your proposed prompt. You can use the standard prompt performance number from part (a).
    - Whether performance improved. Discuss possible reasons why or why not.

    To be clear, your goal in this part is to develop one new prompt that leads to model A _outperforming_ its results when using our standard prompt.

    #answer

    *Prompt*:

    ```python
        prompt = f"""
    Solve the following math word problem.

    Problem: {question}

    Your last line must be exactly:
    `#### <numerical_answer>`
    Your final answer must be a single integer number without any units, commas, or decimal points.
    """.strip()
    ```

    *Scores*:

    - Standard Prompt: `93/100`
    - Superior Prompt: `99/100`

    *Invalid Output Rates*:

    - Standard Prompt: `0/100`
    - Superior Prompt: `0/100`

    *Reasons For Improvement*:

    Under string-match evaluation, the answer is wrong if the model's output does not match the expected answer exactly. For example, if the expected answer is `#### 5` but the model outputs `#### 5.00`, `#### $5` or `#### 5.`, then the answer would be counted as incorrect, even though it is mathematically correct. The superior prompt explicitly instructs the model to output a single integer, which helps to reduce the chances of the model outputting an correct answer in an incorrect format.

    *Performance Comparison*:

    #figure(
      lq.diagram(
        width: 12cm,
        height: 8cm,
        xaxis: (
          ticks: (
            (0, [Model A]),
            (1, [Model A Enhanced]),
            (2, [Model B]),
          ),
          subticks: none,
        ),
        yaxis: (
          ticks: (0, 20, 40, 60, 80, 100),
          label: [Accuracy],
        ),
        ylim: (0, 110),
        lq.bar(
          (0, 1, 2),
          (93, 99, 51),
          fill: (rgb("#C58DE9"), rgb("#FFB3CC"), rgb("#FF8863")),
          width: 60%,
        ),
        lq.place(0, 97, [93%]),
        lq.place(1, 103, [99%]),
        lq.place(2, 55, [51%]),
      ),
      caption: [Exact-match GSM8K accuracy for Model A, Model A with the improved prompt, and Model B.],
    )

    *Discussion*:

    Although the superior prompt led to an improvement in evaluation scores, this does not necessarily mean that the model's underlying reasoning abilities have improved. By observing the actual response of the model, we can find out that it can give the correct and reasonable answer to almost all questions with the standard prompt. Out prompt just made it more likely for the model to output the answer in the correct format, which is part of what the evaluation metric is checking for. This highlights a key limitation of string-match evaluation: it can be gamed by prompt engineering without actually improving the model's reasoning abilities.
