import io

from rich.console import Console

from joy_agent.config import Settings
from joy_agent.llm import LLMClient, model_name
from joy_agent.ui import run_repl


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

    def prompt(self, _prompt):
        value = next(self.inputs)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeLLM:
    def __init__(self):
        self.inputs = []

    def send(self, user_input):
        self.inputs.append(user_input)
        return "**response**"


def test_repl_ignores_blank_input_and_exits():
    llm = FakeLLM()
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)

    run_repl(llm, FakeSession("", "hello", "quit"), console)

    assert llm.inputs == ["hello"]
    assert "response" in output.getvalue()
    assert "Goodbye!" in output.getvalue()
