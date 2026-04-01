
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

  + (3 points) Now develop your own prompt template to increase the performance of model A. Write your prompt template by completing the `superior_prompt_template` function in `gsm8k.py`. Use this function to evaluate the model by filling in `eval_model_on_gsm8k_with_improved_prompt`, and record the performance of model A. Below, include:

    - What your prompt was.
    - Why you thought this would improve model performance.
    - A bar chart of performance with the standard prompt and your proposed prompt. You can use the standard prompt performance number from part (a).
    - Whether performance improved. Discuss possible reasons why or why not.

    To be clear, your goal in this part is to develop one new prompt that leads to model A _outperforming_ its results when using our standard prompt.

    #answer
