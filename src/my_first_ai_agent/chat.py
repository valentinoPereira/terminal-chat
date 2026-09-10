import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from .client import create_client
from .loader import run_with_spinner

SYSTEM_PROMPT = (
    "You are a helpful assistant running in the terminal. So always provide "
    "pure text responses and never use any text formatting apart from spaces "
    "and new lines."
)

MODEL = "deepseek-v4-flash"

QUIT_COMMANDS = {"/quit", "/exit", "/q", "quit"}


def exchange(client: openai.OpenAI, context: list[ChatCompletionMessageParam], line: str) -> str | None:
    """Send one user message and append the answer to `context`.

    On API errors the user message is rolled back so it can be retried,
    and None is returned.
    """

    def send() -> ChatCompletion:
        return client.chat.completions.create(model=MODEL, messages=context)

    context.append({"role": "user", "content": line})
    try:
        response = run_with_spinner(send)
    except openai.APIError as exc:  # covers connection, timeout, rate limit
        print(f"[api error:{exc.__class__.__name__}] {exc}")
        del context[-1]  # drop stuck user msg so it can be retried
        return None
    response_text = response.choices[0].message.content or ""
    context.append({"role": "assistant", "content": response_text})
    return response_text


def run_repl() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    client = create_client()
    context: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    print("Type /quit or press Ctrl+C to exit.")
    history = InMemoryHistory()
    while True:
        try:
            line = prompt("> ", history=history).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nStopped.")
            break
        if not line:
            continue
        if line.lower() in QUIT_COMMANDS:
            print("Goodbye.")
            break
        try:
            result = exchange(client, context, line)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        if result is not None:
            print(f">>> {result}\n")
