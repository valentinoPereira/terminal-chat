from types import SimpleNamespace

import openai

from my_first_ai_agent.chat import exchange


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
