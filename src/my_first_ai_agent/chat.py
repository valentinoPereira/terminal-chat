import re
from typing import Any

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
CLEAR_COMMAND = "/clear"

# Rough guardrails against an unbounded conversation. The provider's real
# context limit is unknown, so these keep the request small enough to work
# while leaving room for the reply. Tune to the deployed model if needed.
MAX_CONTEXT_CHARS = 16_000
MIN_RESPONSE_CHARS = 1_000
MAX_INPUT_CHARS = 8_000

# Strip C0 control characters and DEL so untrusted model output cannot
# manipulate the terminal. Ordinary text, tabs, newlines and carriage returns
# are preserved. Also removes complete ANSI escape sequences (CSI/OSC/single
# char) which are how terminals are actually controlled.
_ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b."
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


Message = dict[str, Any]


def clean_text(text: str) -> str:
    """Remove terminal-controlling escape sequences and control characters."""
    return _CONTROL_CHARS.sub("", _ANSI_ESCAPE.sub("", text))


def trim_context(
    context: list[ChatCompletionMessageParam],
    max_chars: int = MAX_CONTEXT_CHARS,
    reserve: int = MIN_RESPONSE_CHARS,
) -> list[ChatCompletionMessageParam]:
    """Trim `context` for one request, keeping the system prompt and the most
    recent turns that fit within `max_chars` (minus `reserve`)."""
    budget = max_chars - reserve
    if not context:
        return context

    system = context[0] if context[0].get("role") == "system" else None
    body = context[1:] if system else context
    size = len(str(system.get("content", ""))) if system else 0
    keep: list[ChatCompletionMessageParam] = []
    for message in reversed(body):
        content = str(message.get("content", ""))
        if len(content) > budget:
            content = content[:budget]  # truncate a single oversize message
        if size + len(content) > budget and keep:
            break
        keep.append({**message, "content": content})
        size += len(content)
    keep.reverse()

    return ([system] if system else []) + keep


def _extract_response_text(response: ChatCompletion) -> str:
    """Read the assistant text from a completion, rejecting malformed shapes."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("API response contained no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("API response contained no message")
    return message.content or ""


def _rollback_user_message(context: list[ChatCompletionMessageParam]) -> None:
    """Drop a pending user message so the exchange can be retried."""
    if context and context[-1].get("role") == "user":
        del context[-1]


def exchange(
    client: openai.OpenAI, context: list[ChatCompletionMessageParam], line: str
) -> str | None:
    """Send one user message and append the answer to `context`.

    On an API or malformed-response error the user message is rolled back so
    it can be retried, and None is returned.
    """

    def send() -> ChatCompletion:
        request = trim_context(context)
        return client.chat.completions.create(model=MODEL, messages=request)

    context.append({"role": "user", "content": line})
    try:
        response = run_with_spinner(send)
        response_text = clean_text(_extract_response_text(response))
    except openai.APIError as exc:  # covers connection, timeout, rate limit
        print(f"[api error:{exc.__class__.__name__}] {exc}")
        _rollback_user_message(context)
        return None
    except ValueError as exc:  # malformed API response
        print(f"[invalid response] {exc}")
        _rollback_user_message(context)
        return None
    context.append({"role": "assistant", "content": response_text})
    return response_text


def run_repl() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    print("Type /quit or press Ctrl+C to exit. Type /clear to reset the conversation.")
    try:
        client = create_client()
    except RuntimeError as exc:
        print(exc)
        print("Copy .env.example to .env and set your API key, then rerun.")
        return
    context: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
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
        if line.lower() == CLEAR_COMMAND:
            context = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared.")
            continue
        if len(line) > MAX_INPUT_CHARS:
            print(f"[input too long] max {MAX_INPUT_CHARS} characters per message")
            continue
        try:
            result = exchange(client, context, line)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        if result is not None:
            print(f">>> {result}\n")
