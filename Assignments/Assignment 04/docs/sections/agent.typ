#import "../utils.typ": answer

= Designing an Agent (15 points)

In this question, you will design an AI agent that can complete three different tasks. There are three problems your agent should be able to solve:

- If you stacked 10 Statues of Liberty on top of each other, how much taller than the Eiffel Tower would this be (in meters)?
- Find the GDP of the G7 countries and calculate the mean and standard deviation.
- Read `data.csv` and calculate the correlation coefficient between columns A and B.

You will fill in the `query_agent` function in `agent.py`. This function takes in a single text prompt, and returns a string answer. The code to test the agent is in `test_agent.py`. You can see that there are three test cases, one for each of the above tasks. Each one calls the `query_agent` function with a description of the task, and then passes the string outputted from the function and does a fuzzy match against the ground truth answer.

You will submit:

- A completed `agent.py` file.
- A text description of your agent design. This should be a few paragraphs in length, as a rule of thumb less than half a page of text. Make sure to include details of what tools you gave the agent.

For each test case that your agent passes, you will receive 5 points. We will deduct points if your text description does not match the implemented code. In addition, because we are showing you the test cases, it is easy for you to hard code the solutions. We will deduct points for any hard coded solutions.

#text(fill: red)[
  *Notes*

  We will provide them a blank template.

  Then have them run evals on the model.

  We will ask them not to cheat, but they can optimize their models for the eval.

  They will submit the code and an explanation of how their agent works.

  Current Problem Ideas:

  - Search + Calculator
    - Task: The Eiffel Tower is X meters tall. If you stacked Y Statues of Liberty, which would be taller and by how much?
    - Tools: `WikipediaQueryRun` for heights + `LLMMathChain` for calculation

  - Search + Python
    - "Find the GDP of the G7 countries and calculate the mean and standard deviation"
    - Tools: `DuckDuckGoSearchRun` + `PythonREPLTool`

  - File Reading + Code Execution
    - Task: Read `data.csv` and calculate the correlation coefficient between columns A and B
    - Tools: `read_file` + `PythonREPLTool`
]
