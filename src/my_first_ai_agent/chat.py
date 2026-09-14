import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import openai
from dotenv import dotenv_values
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from .client import API_KEY_ENV, create_client
from .loader import print_stream, run_with_spinner

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
# Hard cap on generated reply tokens, so latency and cost stay bounded.
MAX_RESPONSE_TOKENS = 2_000

# Strip control characters and DEL so untrusted model output cannot
# manipulate the terminal. C1 controls (U+0080-U+009F, incl. 8-bit CSI
# U+009B), bare carriage returns (line spoofing) and DEL are all removed;
# \r\n still collapses to \n. Ordinary text, tabs and newlines are kept.
# Also removes complete ANSI escape sequences (CSI/OSC/single char) which
# are how terminals are actually controlled, plus bidi overrides and
# zero-width/invisible characters that can disguise hostile text.
_ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b."
)
_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]"
)


Message = dict[str, Any]


def clean_text(text: str) -> str:
    """Remove terminal-controlling escape sequences and control characters."""
    return _CONTROL_CHARS.sub("", _ANSI_ESCAPE.sub("", text))


def _content_text(content: Any) -> str:
    """Normalize a message's `content` into plain text.

    Handles None (empty string), a list of content parts (text parts are
    joined) and plain strings.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content)


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
    size = len(_content_text(system.get("content"))) if system else 0
    keep: list[ChatCompletionMessageParam] = []
    for message in reversed(body):
        content = _content_text(message.get("content"))
        if len(content) > budget:
            content = content[:budget]  # truncate a single oversize message
        if size + len(content) > budget and keep:
            break
        keep.append({**message, "content": content})
        size += len(content)
    keep.reverse()

    return ([system] if system else []) + keep


def _iter_stream_text(
    stream: Generator[ChatCompletionChunk, None, None],
) -> Generator[str, None, None]:
    """Yield text deltas from a streaming completion, rejecting malformed chunks."""
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            raise ValueError("API response contained no choices")
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            raise ValueError("API response contained no message")
        if delta.content:
            yield delta.content


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

    def open_stream() -> Generator[ChatCompletionChunk, None, None]:
        request = trim_context(context)
        return client.chat.completions.create(
            model=MODEL,
            messages=request,
            stream=True,
            max_tokens=MAX_RESPONSE_TOKENS,
        )

    context.append({"role": "user", "content": line})
    try:
        stream = run_with_spinner(open_stream)
        response_text = clean_text(print_stream(_iter_stream_text(stream)))
    except openai.APIError as exc:  # covers connection, timeout, rate limit
        # str(exc) can include the raw server/proxy response body, so it
        # goes through the sanitizer and a length cap before printing.
        print(f"[api error:{exc.__class__.__name__}] {clean_text(str(exc))[:500]}")
        _rollback_user_message(context)
        return None
    except ValueError as exc:  # malformed API response
        print(f"[invalid response] {exc}")
        _rollback_user_message(context)
        return None
    context.append({"role": "assistant", "content": response_text})
    return response_text


def _load_api_key_env() -> None:
    """Populate os.environ with NEURALWATT_API_KEY from <project root>/.env.

    The .env path is resolved from this file's location, so loading never
    walks up parent folders and picks up a stray .env from elsewhere. Only
    the needed key is read, and only when it is not already set, so other
    .env variables never enter the process environment (the HTTP client
    trusts env vars like HTTPS_PROXY and SSL_CERT_FILE).
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if API_KEY_ENV not in os.environ:
        value = dotenv_values(env_path).get(API_KEY_ENV)
        if value:
            os.environ[API_KEY_ENV] = value


def run_repl() -> None:
    _load_api_key_env()
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
            # Cancel the in-flight request and roll back the pending user
            # message so the prompt comes back instead of the app exiting.
            print()
            _rollback_user_message(context)
            continue
        if result is not None:
            print()  # spacing after the streamed reply
