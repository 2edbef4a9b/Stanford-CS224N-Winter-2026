import time

from client.models import Query, QueryResponse
from client.providers import get_provider


def query_model(
    model_id: str,
    query: Query,
    max_retries: int = 5,
    initial_delay: float = 5.0,
) -> QueryResponse:

    # Get the appropriate provider
    provider_name, provider = get_provider(model_id=model_id)
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # Make the request to the LLM provider
            response: QueryResponse = provider.query(
                model_id=model_id,
                query=query,
            )
            return response

        except Exception as e:
            error_message = str(e)
            is_retryable = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "rate limit" in error_message.lower()
            )

            if not is_retryable or attempt == max_retries - 1:
                raise

            print(
                f"Rate limited when querying model {model_id}. "
                f"Retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Unreachable")
