"""Run a blocking call while showing a spinner.

Delegates the loading indicator to `rich`, so it can be reused for any
slow/blocking operation (API calls, file IO, etc.) without custom threading
or terminal-escape logic.
"""

from collections.abc import Callable
from typing import TypeVar

from rich.console import Console

T = TypeVar("T")

_console = Console()


def run_with_spinner(fn: Callable[[], T], description: str = "Working...") -> T:
    """Run `fn` (no-arg callable) and return its result while showing a spinner.

    If `fn` raises, the exception propagates to the caller. The spinner is
    cleared either way (the status context manager handles it on exit).
    """
    with _console.status(description):
        return fn()
