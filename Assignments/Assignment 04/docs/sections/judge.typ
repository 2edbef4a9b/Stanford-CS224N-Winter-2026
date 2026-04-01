#import "../utils.typ": answer, unjustified

= LLM Judge Benchmarking (13 points)

In this problem, you will evaluate the performance of different LLMs using another LLM as a judge. We will use the Alpaca Eval dataset for this evaluation. The questions in the dataset are designed to be representative of user interactions with chatbots.

#unjustified[
  + (0 points) We have included 30 problems from the Alpaca Eval dataset in `alpaca_eval_first_30.jsonl`. Start by inspecting some of the problems to get a feel for the dataset.
]

+ (5 points) Now we will evaluate the performance of models E and F on this subset of Alpaca Eval using LLM Z as judge. Recall that to run this evaluation, we will have to:

  - For each input prompt, sample a response from models E and F.
  - Prompt model Z to decide which response is best.
  - Aggregate the win rate between the models across the 30 problems.

  We have provided some starter code for you to complete in `llm_judge.py`. In addition to submitting the code, include the following in your writeup:

  - Details of how you conducted the evaluation. This should include how you prompted the LLM judge and extracted scores from it.
  - Plots of your results, and accordingly which model was better.

  *NOTE:* When you complete your evaluation, save the model responses and judge output to a file. You will be analyzing these outputs in the next two subproblems, and it will be easier if you do not have to regenerate the evaluation data.

+ (4 points) Now inspect 5 of the responses from models E and F to 5 of the Alpaca Eval questions. For each, decide which one you would rate as being a more useful response. Below include:

  - How many times you agreed with the judge, and how many times you disagreed.
  - Include 1 example from the dataset where you disagree with which output the judge preferred. For this example, include the Alpaca Eval query, model E response, model F response, and judge response.
  - Do you notice anything that is consistently different between the output of model E and model F?
  - Overall, do you think the LLM judge is doing a good job? If so, why. If not, why.

+ (4 points) Now across the 30 Alpaca Eval problems, plot the following two histograms on the same axis:

  - The histogram of lengths of output that the judge preferred.
  - The histogram of lengths of output that the model did not prefer.

  As we tested on 30 problems, each of which has a preferred and not preferred output according to the judge, each histogram will have 30 datapoints.

  Include the plot below. Comment on any trends you notice in the data. Now you have seen this, do you trust that the LLM-as-judge evaluation you ran is indeed reflecting which model users would prefer? Provide justification for the answer you give.
