from collections.abc import Generator
from types import SimpleNamespace

import openai

from my_first_ai_agent.chat import clean_text, exchange, trim_context
from my_first_ai_agent.client import create_client


def chunk(text: str | None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


class StubCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self._content = content
        self._error = error

    def create(self, model: str, messages: list, stream: bool, max_tokens: int):
        assert stream is True
        if self._error is not None:
            raise self._error

        def gen() -> Generator:
            yield chunk(self._content)
            yield chunk(None)  # final chunk carries no delta

        return gen()


def make_client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions(**kwargs)))


def test_exchange_appends_user_and_assistant_messages():
    context = []
    client = make_client(content="hello there")

    result = exchange(client, context, "hi")

    assert result == "hello there"
    assert context == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


def test_exchange_none_content_becomes_empty_string():
    context = []
    client = make_client(content=None)

    result = exchange(client, context, "hi")

    assert result == ""
    assert context[-1] == {"role": "assistant", "content": ""}


def test_exchange_api_error_rolls_back_user_message(capsys):
    context = [{"role": "system", "content": "sys"}]
    error = openai.APIError(
        "boom\x1b[2Jescaped\r\n" + "x" * 900,
        request=SimpleNamespace(),
        body=None,
    )
    client = make_client(error=error)

    result = exchange(client, context, "hi")

    assert result is None
    assert context == [{"role": "system", "content": "sys"}]
    out = capsys.readouterr().out
    assert "[api error:APIError]" in out
    assert "\x1b" not in out and "\r" not in out
    assert "x" * 900 not in out  # error text is capped at 500 chars


def test_exchange_empty_choices_chunk_is_tolerated(capsys):
    class KeepAlive:
        def create(self, model, messages, stream, max_tokens):
            assert stream is True
            # Providers stream legitimate chunks with no choices (keep-alive,
            # usage) alongside content chunks.
            yield SimpleNamespace(choices=[])
            yield chunk("hel")
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason="stop")])
            yield chunk(None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=KeepAlive()))
    context = [{"role": "system", "content": "sys"}]

    result = exchange(client, context, "hi")

    assert result == "hel"
    assert context[-1]["role"] == "assistant"
    assert "[invalid response]" not in capsys.readouterr().out


def test_exchange_builds_text_from_streamed_chunks(capsys):
    class ChunkStream:
        def create(self, model, messages, stream, max_tokens):
            assert stream is True
            yield chunk("hel")
            yield chunk("lo")
            yield chunk(None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=ChunkStream()))
    context = []

    result = exchange(client, context, "hi")

    assert result == "hello"
    assert context[-1] == {"role": "assistant", "content": "hello"}
    assert "hello" in capsys.readouterr().out


def test_exchange_strips_terminal_control_characters():
    class EscapeStream:
        def create(self, model, messages, stream, max_tokens):
            # The escape sequence is split across chunk boundaries.
            yield chunk("ok\x1b[")
            yield chunk("2J\x1b[Hbye")
            yield chunk(None)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=EscapeStream())
    )

    result = exchange(client, [], "hi")

    assert "\x1b" not in result
    assert result == "okbye"


def test_clean_text_keeps_whitespace_but_strips_escapes():
    assert clean_text("a\tb\n\x1b[2Jc\x07d") == "a\tb\ncd"
    # OSC sequence terminated by BEL is removed entirely
    assert clean_text("x\x1b]0;title\x07y") == "xy"
    # \r\n collapses to \n, a bare \r (line spoofing) is dropped
    assert clean_text("line\r\nspoof\red inline") == "line\nspoofed inline"
    # C1 controls dropped; an 8-bit CSI (U+009B + params) leaves inert text
    assert clean_text("a\u009bb\u009B2Jc") == "ab2Jc"
    # bidi overrides and zero-width characters are dropped
    assert clean_text("i\u202ed\u200bi\u2066t") == "idit"
    # DCS payload is left as inert text (escape removed, rest kept)
    assert clean_text("x\x1bP1;2qdata\x1b\\y") == "x1;2qdatay"

    # still fast on adversarial input (linear time)
    import time

    start = time.perf_counter()
    clean_text("\u009b\x1b[" * 100_000)
    assert time.perf_counter() - start < 5


def test_trim_context_keeps_system_and_recent_turns():
    context = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a" * 10},
        {"role": "assistant", "content": "b" * 10},
        {"role": "user", "content": "c" * 10},
    ]

    trimmed = trim_context(context, max_chars=30, reserve=5)

    assert trimmed[0] == {"role": "system", "content": "sys"}
    # budget = 25; system (3) leaves 22. The newest user (10) is kept, but
    # the older pair (20) would push the total to 30, so it is dropped
    # whole and the request stays at 13.
    assert [m["role"] for m in trimmed] == ["system", "user"]


def test_trim_context_drops_whole_turns_to_keep_pairs():
    context = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a" * 4},
        {"role": "assistant", "content": "b" * 4},
        {"role": "user", "content": "c" * 4},
        {"role": "assistant", "content": "d" * 4},
    ]

    trimmed = trim_context(context, max_chars=15, reserve=0)

    # budget = 15; system (3) leaves 12, exactly one pair (8) fits, and
    # the older minus the newest pair would not, so only the newest stays
    assert [m["role"] for m in trimmed] == ["system", "user", "assistant"]
    assert [len(m["content"]) for m in trimmed[1:]] == [4, 4]


def test_trim_context_truncates_oversize_single_message():
    context = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 100},
    ]

    trimmed = trim_context(context, max_chars=30, reserve=5)

    # the END of the message is kept; budget (25) minus system (3) = 22
    assert trimmed[-1]["content"] == "x" * 22


def test_trim_context_stays_within_budget_and_keeps_pairs():
    context = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a" * 30},
        {"role": "assistant", "content": "b" * 30},
    ]

    trimmed = trim_context(context, max_chars=33, reserve=0)

    # budget = 33; system (3) + oversize-truncated messages must fit
    assert sum(len(m["content"]) for m in trimmed) <= 33
    # dropping an old turn must never leave the request starting with an
    # assistant message after the system prompt
    assert [m["role"] for m in trimmed] == ["system", "user"]


def test_trim_context_handles_none_and_list_content():
    context = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": None},
        {"role": "assistant", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
        {"role": "user", "content": 42},
    ]

    trimmed = trim_context(context, max_chars=200, reserve=10)

    assert trimmed[1]["content"] == ""
    assert trimmed[2]["content"] == "part one\npart two"
    assert trimmed[3]["content"] == "42"


def test_create_client_uses_neuralwatt_key_and_ignores_ambient_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    monkeypatch.setenv("NEURALWATT_API_KEY", "nw-correct")

    client = create_client()
    client.close()

    assert client.api_key == "nw-correct"


def test_create_client_missing_key_raises(monkeypatch):
    monkeypatch.delenv("NEURALWATT_API_KEY", raising=False)

    try:
        create_client()
    except RuntimeError as exc:
        assert "NEURALWATT_API_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
