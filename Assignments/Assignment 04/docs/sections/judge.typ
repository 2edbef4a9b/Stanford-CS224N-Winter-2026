#import "@preview/lilaq:0.6.0" as lq
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

  #answer

  The model Z is prompted with the following template to evaluate the responses from models E and F:

  ```python
      prompt = f"""
  You are an impartial judge comparing two answers to the same question.

  Judge the answers based on the following criteria:

  - Correctness: Does the answer correctly address the question?
  - Helpfulness: Does the answer provide useful information or insights?
  - Relevance: Is the answer relevant to the question asked?
  - Clarity: Is the answer clearly and coherently expressed?
  - Completeness: Does the answer cover all necessary aspects of the question?
  - Harmlessness: Is the answer free from harmful or inappropriate content?

  Do not favor an answer for being longer if it is not better.

  Question:
  {query}

  Answer from Model E:
  {response_E}

  Answer from Model F:
  {response_F}

  Explain your judgment.

  Explain your judgment briefly. Then on the final line, output exactly
  one of the following tags to indicate your preference:

  #### Preference: <MODEL_E_BETTER>
  #### Preference: <MODEL_F_BETTER>
      """.strip()
  ```

  The preference is extracted from the judge’s response by regex matching the final line of the response to determine which model the judge preferred.

  ```python
  PREFERENCE_RE = re.compile(
      r"#### Preference:\s*(<MODEL_E_BETTER>|<MODEL_F_BETTER>|<NO_PREFERENCE_FOUND>)"
  )

  match = PREFERENCE_RE.search(judge_output)
  if match:
      return match.group(1)

  return NO_PREFERENCE_FOUND_TAG
  ```

  The final results were:

  #figure(
    lq.diagram(
      width: 12cm,
      height: 8cm,
      xaxis: (
        ticks: (
          (0, [Model E]),
          (1, [Model F]),
          (2, [No preference]),
        ),
        subticks: none,
      ),
      yaxis: (
        ticks: (0, 20, 40, 60, 80, 100),
        label: [Percent Preferred],
      ),
      ylim: (0, 110),
      lq.bar(
        (0, 1, 2),
        (96.7, 3.3, 0.0),
        fill: (rgb("#C58DE9"), rgb("#FFB3CC"), rgb("#FF8863")),
        width: 60%,
      ),
      lq.place(0, 100.7, [96.7%]),
      lq.place(1, 7.3, [3.3%]),
      lq.place(2, 4, [0.0%]),
    ),
    caption: [LLM-as-a-judge preference rates on the 30 Alpaca Eval examples.],
  )

  - Model E wins: `29/30`
  - Model F wins: `1/30`
  - No preference extracted: `0/30`

  Thus, model Z judged *Model E* to be substantially better than *Model F* on this subset of Alpaca Eval.

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

  #answer

  #figure(
    image("../results/histogram_tokens.svg", width: 100%),
    caption: [Distribution of preferred vs. not-preferred response lengths measured in output tokens.],
  )

  #figure(
    image("../results/histogram_chars.svg", width: 100%),
    caption: [Distribution of preferred vs. not-preferred response lengths measured in characters.],
  )

  Both plots show a clear right-shift for the preferred outputs: responses preferred by the judge are generally longer than responses not preferred by the judge, in both token count and character count. This is largely because model E's responses are usually longer than model F's responses, and model E was preferred on 29 out of 30 examples. The only exception is the single example in which model F was preferred.

  This suggests a strong correlation between response length and judge preference. Part of this may reflect real quality differences, since more useful answers are often more complete and therefore longer. However, the strength of the pattern also suggests that length bias may be influencing the judge.

  Overall, it is clear that the LLM-as-judge evaluation cannot be fully trusted as a proxy for user preference. The result that model E is better than model F is probably meaningful, but the histograms indicate that the judge may systematically reward longer responses, which could exaggerate the margin between the two models.
