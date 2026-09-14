import os

from openai import OpenAI, Timeout

API_KEY_ENV = "NEURALWATT_API_KEY"
BASE_URL = "https://api.neuralwatt.com/v1"


def create_client() -> OpenAI:
    """Build an OpenAI client pointed at the NeuralWatt API.

    The key is taken from NEURALWATT_API_KEY and passed explicitly so an
    ambient OPENAI_API_KEY is never sent to the NeuralWatt endpoint.

    Returns:
        OpenAI: configured client with a 60s read timeout, 5s connect
        timeout and 2 retries.

    Raises:
        RuntimeError: if NEURALWATT_API_KEY is not set.
    """
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing {API_KEY_ENV} environment variable")
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=Timeout(60.0, connect=5.0),
        max_retries=2,
    )
