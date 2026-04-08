import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from dotenv import load_dotenv
from tqdm import tqdm

from client.models import Query
from client.query import query_model

INVALID_ANS = "[invalid]"


def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""
Output a numerical answer to the following problem with two or fewer
steps of reasoning. Output your numericalanswer as the only line of
your output in the format "#### <numerical_answer>."

Problem: {question}
    """.strip()

    return prompt


def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or
            "[invalid]" if no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


# ------------------------------------------- #
# TODO For you to fill in
# ------------------------------------------- #


@dataclass
class Metrics:
    correct_times: int = 0
    invalid_times: int = 0
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def print_metrics(metrics: Metrics, total_examples: int) -> None:
    print("=" * 100)
    print(f"Accuracy: {metrics.correct_times}/{total_examples}")
    print(f"Invalid Outputs Rate: {metrics.invalid_times}/{total_examples}")
    print(f"Total Cost: ${metrics.total_cost:.8f}")
    print(f"Total Input Tokens: {metrics.total_input_tokens}")
    print(f"Total Output Tokens: {metrics.total_output_tokens}")
    print(f"Average Cost: ${metrics.total_cost / total_examples:.8f}")
    print(f"Average Input Tokens: {metrics.total_input_tokens / total_examples}")
    print(f"Average Output Tokens: {metrics.total_output_tokens / total_examples}")
    print("=" * 100)


def load_gsm8k_data(path: str, num: int | None = None) -> list[dict[str, str | int]]:
    """Load GSM8K data from a jsonl file."""
    dataset = []
    count = 0
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            dataset.append(example)
            count += 1
            if num is not None and count >= num:
                break
    return dataset


def eval_model(
    model_id: str,
    prompt_template: Callable[[str], str] = standard_prompt_template,
    num_examples: int | None = None,
) -> None:
    """
    Benchmark a model on the GSM8K dataset and print out performance metrics.

    Args:
        model_id: the string identifier for the model to evaluate.
        prompt_template: a function that takes a gsm8k question and
            returns a prompt string for the model.
        num_examples: If not None, limits the number of examples to
            evaluate on.
    """
    metrics = Metrics()
    gsm8k_examples = load_gsm8k_data("./data/gsm8k_first_100.jsonl", num=num_examples)
    print(f"Evaluating Model {model_id} on GSM8K with {len(gsm8k_examples)} examples.")

    for example in tqdm(gsm8k_examples):
        prompt = prompt_template(str(example["question"]))
        query = Query(
            turns=[
                {"user": prompt},
            ]
        )
        response = query_model(model_id=model_id, query=query)
        model_answer = standard_output_extractor(response.text)
        correct_answer = str(example["numerical_answer"])
        if model_answer == correct_answer:
            metrics.correct_times += 1
        elif model_answer == INVALID_ANS:
            metrics.invalid_times += 1
        metrics.total_cost += response.cost
        metrics.total_input_tokens += response.input_tokens
        metrics.total_output_tokens += response.output_tokens

    print(f"Model {model_id} Evaluation Results:")
    print_metrics(
        metrics,
        total_examples=(
            num_examples if num_examples is not None else len(gsm8k_examples)
        ),
    )


def eval_model_on_gsm8k() -> None:
    """
    Benchmark models A and B on the GSM8K dataset using the standard prompt template.

    See example_usage.py for how to query models and handle responses.
    The data file (gsm8k_first_100.jsonl) contains 'question' and 'numerical_answer'
    fields.

    Think about: What metric will you use to evaluate performance? How will you
    handle cases where the model's output cannot be parsed?
    """
    eval_model(model_id="A", prompt_template=standard_prompt_template)
    eval_model(model_id="B", prompt_template=standard_prompt_template)


def superior_prompt_template(question: str) -> str:
    """
    Design your own prompt template that outperforms standard_prompt_template on model A.

    Args:
        question: gsm8k question.
    Returns:
        Your improved prompt for the model.

    Look at standard_prompt_template() to understand the baseline approach. What
    aspects of how you prompt the model might affect its reasoning or accuracy?

    NOTE: Your prompt must still produce output in the "#### <answer>" format
    so that standard_output_extractor() can parse the response.
    """
    prompt = f"""
Solve the following math word problem.

Problem: {question}

Your last line must be exactly:
`#### <numerical_answer>`
Your final answer must be a single integer number without any units,
commas, or decimal points.
    """.strip()

    return prompt


def eval_model_on_gsm8k_with_improved_prompt() -> None:
    """
    Evaluate model A using your superior_prompt_template.
    """
    eval_model(model_id="A", prompt_template=superior_prompt_template)


if __name__ == "__main__":
    load_dotenv()

    ## Uncomment to run your code
    # eval_model_on_gsm8k()
    # eval_model_on_gsm8k_with_improved_prompt()
