import json
import os
import re

from dotenv import load_dotenv
from matplotlib import pyplot as plt
from tqdm import tqdm

from client.models import Query, QueryResponse
from client.query import query_model

# You may find these constants useful for structuring the judge's output.
MODEL_E_PREFERRED_TAG = "<MODEL_E_BETTER>"
MODEL_F_PREFERRED_TAG = "<MODEL_F_BETTER>"
NO_PREFERENCE_FOUND_TAG = "<NO_PREFERENCE_FOUND>"


def load_alpaca_data(path: str, num: int | None = None) -> list[dict[str, str]]:
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


def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
    """
    Construct a prompt for an LLM judge to evaluate two model responses.

    Args:
        query: the question given to the two models (from AlpacaEval)
        response_E: output from model E on query
        response_F: output from model F on query
    Returns:
        Prompt for the LLM judge.

    Consider: The judge is an LLM that will output free-form text. How will you
    design the prompt so that you can reliably determine which response it preferred?
    Your llm_judge_template and extract_llm_judge_preference should work together.
    """
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

    return prompt


def extract_llm_judge_preference(judge_output: str) -> str:
    """
    Extract the judge's preference from its output.

    Args:
        judge_output: the string sampled from the LLM judge.
    Returns:
        A string representing which response the judge preferred.

    This function should work in tandem with your llm_judge_template design.
    What if the judge's output is malformed or ambiguous?
    """
    PREFERENCE_RE = re.compile(
        r"#### Preference:\s*(<MODEL_E_BETTER>|<MODEL_F_BETTER>|<NO_PREFERENCE_FOUND>)"
    )

    match = PREFERENCE_RE.search(judge_output)
    if match:
        return match.group(1)

    return NO_PREFERENCE_FOUND_TAG


def eval_model(model_id: str) -> None:
    alpaca_data = load_alpaca_data(path="./data/alpaca_eval_first_30.jsonl")
    results = []

    print(
        f"Evaluating Model {model_id} on AlpacaEval with {len(alpaca_data)} questions."
    )
    for example in tqdm(alpaca_data):
        prompt = example["instruction"]
        query = Query(
            turns=[
                {"user": prompt},
            ]
        )
        response = query_model(model_id=model_id, query=query)

        print(f"Question:\n{prompt}\n")
        print(f"Model {model_id} response:\n{response.text}\n")

        results.append(response)

    save_responses(model_id=model_id, responses=results)


def save_responses(model_id: str, responses: list[QueryResponse]) -> None:
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/model_{model_id}_responses.jsonl", "w") as file:
        for response in responses:
            json.dump(
                {
                    "text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost": response.cost,
                },
                file,
            )
            file.write("\n")


def load_responses(model_id: str) -> list[QueryResponse]:
    responses = []
    with open(f"./results/model_{model_id}_responses.jsonl") as file:
        for line in file:
            data = json.loads(line)
            response = QueryResponse(
                text=data["text"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                cost=data["cost"],
            )
            responses.append(response)
    return responses


def run_llm_judge_eval():
    """
    Run the LLM-as-a-judge evaluation comparing models E and F on AlpacaEval data.
    Use model Z as the judge.

    For each AlpacaEval instruction, you'll need responses from both models E and F,
    then have the judge compare them.

    Remember to save your results (model responses + judge outputs) - you will
    need them for Parts C and D.
    """

    if not os.path.exists("./results/model_E_responses.jsonl"):
        eval_model("E")

    if not os.path.exists("./results/model_F_responses.jsonl"):
        eval_model("F")

    alpaca_data = load_alpaca_data(path="./data/alpaca_eval_first_30.jsonl")
    model_E_responses = load_responses("E")
    model_F_responses = load_responses("F")
    results = []

    print(
        f"Running LLM judge evaluation on AlpacaEval with {len(alpaca_data)} questions."
    )
    for i in tqdm(range(len(alpaca_data))):
        query = alpaca_data[i]["instruction"]
        response_E = model_E_responses[i].text
        response_F = model_F_responses[i].text

        judge_prompt = llm_judge_template(
            query=query,
            response_E=response_E,
            response_F=response_F,
        )

        judge_query = Query(
            turns=[
                {"user": judge_prompt},
            ]
        )

        judge_response = query_model(model_id="Z", query=judge_query)
        judge_output = judge_response.text
        preference = extract_llm_judge_preference(judge_output)

        print(f"Judge output:\n{judge_output}\n")
        print(f"Extracted preference: {preference}\n")

        results.append(
            {
                "judge_output": judge_output,
                "preference": preference,
                "model_E_output_tokens": model_E_responses[i].output_tokens,
                "model_F_output_tokens": model_F_responses[i].output_tokens,
            }
        )

    os.makedirs("./results", exist_ok=True)
    with open("./results/judge_results.jsonl", "w") as file:
        for result in results:
            json.dump(result, file)
            file.write("\n")


def plot_model_output_lengths() -> None:
    """
    For Part D: Plot histograms of response lengths for preferred vs. not-preferred outputs.
    """
    preferred_token_lengths = []
    not_preferred_token_lengths = []

    preferred_char_lengths = []
    not_preferred_char_lengths = []

    input_path = "./results/judge_results.jsonl"
    if not os.path.exists(input_path):
        print(f"Judge results file not found: {input_path}")
        return

    with open(input_path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            result = json.loads(line)
            preference = result["preference"]

            e_tokens = result["model_E_output_tokens"]
            f_tokens = result["model_F_output_tokens"]
            e_chars = result["model_E_output_chars"]
            f_chars = result["model_F_output_chars"]

            if preference == MODEL_E_PREFERRED_TAG:
                preferred_token_lengths.append(e_tokens)
                not_preferred_token_lengths.append(f_tokens)
                preferred_char_lengths.append(e_chars)
                not_preferred_char_lengths.append(f_chars)
            elif preference == MODEL_F_PREFERRED_TAG:
                preferred_token_lengths.append(f_tokens)
                not_preferred_token_lengths.append(e_tokens)
                preferred_char_lengths.append(f_chars)
                not_preferred_char_lengths.append(e_chars)
            else:
                continue

    if not preferred_token_lengths or not preferred_char_lengths:
        print("No preferred vs. not-preferred pairs found to plot.")
        return

    os.makedirs("./results", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(
        preferred_token_lengths,
        bins=10,
        alpha=0.6,
        label="Preferred",
        color="skyblue",
    )
    plt.hist(
        not_preferred_token_lengths,
        bins=10,
        alpha=0.6,
        label="Not Preferred",
        color="salmon",
    )
    plt.xlabel("Response Length (output tokens)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Response Lengths: Preferred vs Not Preferred (Tokens)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/histogram_tokens.svg")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(
        preferred_char_lengths,
        bins=10,
        alpha=0.6,
        label="Preferred",
        color="skyblue",
    )
    plt.hist(
        not_preferred_char_lengths,
        bins=10,
        alpha=0.6,
        label="Not Preferred",
        color="salmon",
    )
    plt.xlabel("Response Length (characters)")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Response Lengths: Preferred vs Not Preferred (Characters)"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/histogram_chars.svg")
    plt.close()

    print("Saved token-length histogram to ./results/histogram_tokens.svg")
    print("Saved character-length histogram to ./results/histogram_chars.svg")


if __name__ == "__main__":
    load_dotenv()

    ## Uncomment to run your code
    # run_llm_judge_eval()
    # plot_model_output_lengths()
