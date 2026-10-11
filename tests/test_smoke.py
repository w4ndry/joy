import io

from prompt_toolkit.filters import Always
from rich.console import Console

from joy_agent.config import Settings
from joy_agent.llm import LLMClient, model_name
from joy_agent.ui import (
    AGENT_NOT_CONNECTED,
    INPUT_PLACEHOLDER,
    create_prompt_session,
    run_repl,
)


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, *, model, messages):
        self.calls.append({"model": model, "messages": messages.copy()})
        message = type("Message", (), {"content": "# Hello"})()
        choice = type("Choice", (), {"message": message})()
        return type(
            "Response",
            (),
            {"choices": [choice]},
        )()


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_package_configuration_defaults(monkeypatch):
    monkeypatch.delenv("JOY_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("JOY_AGENT_MODEL", raising=False)

    settings = Settings.from_environment()

    assert settings.provider == "openai"
    assert settings.model == "gpt-4o-mini"
    assert model_name(settings) == "openai:gpt-4o-mini"


def test_llm_client_sends_and_remembers_messages(monkeypatch):
    fake_client = FakeClient()
    monkeypatch.setattr("joy_agent.llm.ai.Client", lambda: fake_client)

    client = LLMClient(Settings())

    assert client.send("Explain this code") == "# Hello"
    assert fake_client.chat.completions.calls[0] == {
        "model": "openai:gpt-4o-mini",
        "messages": [{"role": "user", "content": "Explain this code"}],
    }
    assert client.messages[-1] == {"role": "assistant", "content": "# Hello"}


class FakeSession:
    def __init__(self, *inputs):
        self.inputs = iter(inputs)

    def prompt(self, *_args, **_kwargs):
        value = next(self.inputs)
        if isinstance(value, BaseException):
            raise value
        return value


def test_prompt_layout_has_three_row_input_configuration():
    session = create_prompt_session()

    windows = list(
        window
        for window in session.layout.find_all_windows()
        if getattr(window.content, "buffer", None) is session.default_buffer
    )

    assert "class:frame" in session.layout.container.style
    assert session.color_depth.name == "DEPTH_24_BIT"
    assert session.placeholder == [("class:placeholder", INPUT_PLACEHOLDER)]
    assert windows
    assert session.layout.container.height.min == 3
    assert session.layout.container.height.max == 3
    assert session.layout.container.height.preferred == 3
    assert all(
        window.height.min == 1
        and window.height.max == 1
        and window.height.preferred == 1
        for window in windows
    )
    assert all(isinstance(window.dont_extend_height, Always) for window in windows)


def test_repl_prints_placeholder_response_and_exits():
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)

    run_repl(FakeSession("", "hello", "/quit"), console)

    assert output.getvalue().count(AGENT_NOT_CONNECTED) == 1
