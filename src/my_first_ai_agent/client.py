from openai import OpenAI


def create_client() -> OpenAI:
    """Build an OpenAI client pointed at the NeuralWatt API.

    Notes:
        (connect_timeout, read_timeout). No timeout before -> hung forever on
        slow/unresponsive API -> app froze.
        api_key is read from the OPENAI_API_KEY env var (see .env).

    Returns:
        OpenAI: configured client with a 30s timeout and 2 retries.
    """
    return OpenAI(
        base_url="https://api.neuralwatt.com/v1",
        timeout=30.0,
        max_retries=2,
    )
