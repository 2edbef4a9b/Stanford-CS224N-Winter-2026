import json
import re
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

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

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


def load_gsm8k_examples(
    path: str, num: int | None = None
) -> list[dict[str, str | int]]:
    """Load GSM8K examples from a jsonl file."""
    gsm8k_examples = []
    count = 0
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            gsm8k_examples.append(example)
            count += 1
            if num is not None and count >= num:
                break
    return gsm8k_examples


def eval_model_on_gsm8k() -> None:
    """
    Benchmark models A and B on the GSM8K dataset using the standard prompt template.

    See example_usage.py for how to query models and handle responses.
    The data file (gsm8k_first_100.jsonl) contains 'question' and 'numerical_answer'
    fields.

    Think about: What metric will you use to evaluate performance? How will you
    handle cases where the model's output cannot be parsed?
    """
    metrics_a = Metrics()
    metrics_b = Metrics()
    gsm8k_examples = load_gsm8k_examples("data/gsm8k_first_100.jsonl")

    print(f"Evaluating Model A on GSM8K with {len(gsm8k_examples)} examples.")
    for example in tqdm(gsm8k_examples):
        prompt = standard_prompt_template(str(example["question"]))
        query = Query(
            turns=[
                {"user": prompt},
            ]
        )
        response = query_model(model_id="A", query=query)
        model_answer = standard_output_extractor(response.text)
        correct_answer = str(example["numerical_answer"])
        if model_answer == correct_answer:
            metrics_a.correct_times += 1
        elif model_answer == INVALID_ANS:
            metrics_a.invalid_times += 1
        metrics_a.total_cost += response.cost
        metrics_a.total_input_tokens += response.input_tokens
        metrics_a.total_output_tokens += response.output_tokens
    print("Model A Evaluation Results:")
    print_metrics(metrics_a, total_examples=len(gsm8k_examples))

    print(f"Evaluating Model B on GSM8K with {len(gsm8k_examples)} examples.")
    for example in tqdm(gsm8k_examples):
        prompt = standard_prompt_template(str(example["question"]))
        query = Query(
            turns=[
                {"user": prompt},
            ]
        )
        response = query_model(model_id="B", query=query)
        model_answer = standard_output_extractor(response.text)
        correct_answer = str(example["numerical_answer"])
        if model_answer == correct_answer:
            metrics_b.correct_times += 1
        elif model_answer == INVALID_ANS:
            metrics_b.invalid_times += 1
        metrics_b.total_cost += response.cost
        metrics_b.total_input_tokens += response.input_tokens
        metrics_b.total_output_tokens += response.output_tokens

    print("Model B Evaluation Results:")
    print_metrics(metrics_b, total_examples=len(gsm8k_examples))


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
    # TODO complete for question 2bii

    pass


def eval_model_on_gsm8k_with_improved_prompt() -> None:
    """
    Evaluate model A using your superior_prompt_template.
    """
    # TODO complete for question 2bii

    pass


if __name__ == "__main__":
    load_dotenv()

    ## Uncomment to run your code
    # eval_model_on_gsm8k()
    # eval_model_on_gsm8k_with_improved_prompt()
