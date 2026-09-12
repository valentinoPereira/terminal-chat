from types import SimpleNamespace

import openai

from my_first_ai_agent.chat import clean_text, exchange, trim_context
from my_first_ai_agent.client import create_client


class StubCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self._content = content
        self._error = error

    def create(self, model: str, messages: list):
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


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
    error = openai.APIError("boom", request=SimpleNamespace(), body=None)
    client = make_client(error=error)

    result = exchange(client, context, "hi")

    assert result is None
    assert context == [{"role": "system", "content": "sys"}]
    assert "[api error:APIError]" in capsys.readouterr().out


def test_exchange_malformed_choices_rolls_back_user_message(capsys):
    class EmptyChoices:
        def create(self, model, messages):
            return SimpleNamespace(choices=[])

    client = SimpleNamespace(chat=SimpleNamespace(completions=EmptyChoices()))
    context = [{"role": "system", "content": "sys"}]

    result = exchange(client, context, "hi")

    assert result is None
    assert context == [{"role": "system", "content": "sys"}]
    assert "[invalid response]" in capsys.readouterr().out


def test_exchange_strips_terminal_control_characters():
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok\x1b[2J\x1b[Hbye"))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: reply))
    )

    result = exchange(client, [], "hi")

    assert "\x1b" not in result
    assert result == "okbye"


def test_clean_text_keeps_whitespace_but_strips_escapes():
    assert clean_text("a\tb\n\r\x1b[2Jc\x07d") == "a\tb\n\rcd"
    # OSC sequence terminated by BEL is removed entirely
    assert clean_text("x\x1b]0;title\x07y") == "xy"


def test_trim_context_keeps_system_and_recent_turns():
    context = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a" * 10},
        {"role": "assistant", "content": "b" * 10},
        {"role": "user", "content": "c" * 10},
    ]

    trimmed = trim_context(context, max_chars=30, reserve=5)

    assert trimmed[0] == {"role": "system", "content": "sys"}
    # budget = 25; the newest pair (20 chars) fits, the oldest user (10) does
    # not, so the tail is kept and the oldest turn is dropped
    assert [m["role"] for m in trimmed] == ["system", "assistant", "user"]


def test_trim_context_truncates_oversize_single_message():
    context = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 100},
    ]

    trimmed = trim_context(context, max_chars=30, reserve=5)

    assert len(trimmed[-1]["content"]) == 25


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
