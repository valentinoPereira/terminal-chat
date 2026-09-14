"""Helpers for printing (streamed) API responses with a spinner.

`run_with_spinner` wraps a blocking call with a `rich` spinner. For streaming
requests, `print_stream` shows the spinner until the first chunk arrives, then
prints cleaned chunks as they stream in.
"""

import re
from collections.abc import Callable, Iterable
from typing import TypeVar

from rich.console import Console

T = TypeVar("T")

_console = Console()
_CONTROL_PART = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def run_with_spinner(fn: Callable[[], T], description: str = "Working...") -> T:
    """Run `fn` (no-arg callable) and return its result while showing a spinner.

    If `fn` raises, the exception propagates to the caller. The spinner is
    cleared either way (the status context manager handles it on exit).
    """
    with _console.status(description):
        return fn()


def print_stream(chunks: Iterable[str]) -> str:
    """Print an iterable of text chunks live, returning the joined text.

    A spinner shows until the first chunk arrives, then chunks print as they
    come. A newline is printed at the end unless the stream was empty.
    """
    parts: list[str] = []
    for chunk in chunks:
        if not parts:
            _console.log("Working...")
        # Control chars are safe to strip per chunk; escape sequences can span
        # chunk boundaries, so the joined text is cleaned again by the caller.
        parts.append(chunk)
        print(_CONTROL_PART.sub("", chunk), end="", flush=True)
    if parts:
        print()
    return "".join(parts)
