import asyncio
import io

from prompt_toolkit.filters import Always
from rich.console import Console

from joy_agent.config import Settings
from joy_agent.llm import LLMClient, model_name
from joy_agent.ui import (
    DEMO_REPLY,
    INPUT_PLACEHOLDER,
    ChatState,
    _complete_submission,
    _submit_buffer,
    create_prompt_session,
    render_assistant_message,
    render_user_message,
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


def test_prompt_layout_has_three_row_input_configuration():
    session = create_prompt_session()

    windows = list(
        window
        for window in session.layout.find_all_windows()
        if getattr(window.content, "buffer", None) is session.default_buffer
    )

    assert "class:frame" in session.joy_input_frame.style
    assert session.color_depth.name == "DEPTH_24_BIT"
    assert session.placeholder == [("class:placeholder", INPUT_PLACEHOLDER)]
    assert windows
    assert session.joy_input_frame.height.min == 3
    assert session.joy_input_frame.height.max == 3
    assert session.joy_input_frame.height.preferred == 3
    assert all(
        window.height.min == 1
        and window.height.max == 1
        and window.height.preferred == 1
        for window in windows
    )
    assert all(isinstance(window.dont_extend_height, Always) for window in windows)


def test_conversation_rendering_is_literal_and_markdown_is_labeled():
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)

    render_user_message("first line\n[not markdown]", console)
    render_assistant_message("**reply**", console)

    rendered = output.getvalue()
    assert "› first line" in rendered
    assert "[not markdown]" in rendered
    assert "Joy" in rendered
    assert "reply" in rendered


class FakeBuffer:
    def __init__(self, text):
        self.text = text

    def reset(self):
        self.text = ""


class FakeApp:
    def __init__(self, console):
        self.joy_console = console
        self.invalidations = 0
        self.tasks = []

    def create_background_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def invalidate(self):
        self.invalidations += 1


class FakeEvent:
    def __init__(self, app, text):
        self.app = app
        self.current_buffer = FakeBuffer(text)


def test_submission_clears_submitted_text_but_preserves_next_draft(monkeypatch):
    async def immediate_terminal_output(function):
        function()

    monkeypatch.setattr("joy_agent.ui.run_in_terminal", immediate_terminal_output)
    monkeypatch.setattr("joy_agent.ui.REPLY_DELAY", 0)

    async def scenario():
        output = io.StringIO()
        console = Console(file=output, force_terminal=False)
        state = ChatState()
        app = FakeApp(console)
        event = FakeEvent(app, "first message")
        _submit_buffer(event, state)
        assert event.current_buffer.text == ""

        draft = "next draft"
        event.current_buffer.text = draft
        await asyncio.gather(*app.tasks)

        assert state.pending is False
        assert event.current_buffer.text == draft
        assert DEMO_REPLY in output.getvalue()

    asyncio.run(scenario())


def test_pending_submission_can_be_cancelled(monkeypatch):
    real_sleep = asyncio.sleep

    async def blocked_sleep(_delay):
        await asyncio.Event().wait()

    async def immediate_terminal_output(function):
        function()

    monkeypatch.setattr("joy_agent.ui.asyncio.sleep", blocked_sleep)
    monkeypatch.setattr("joy_agent.ui.run_in_terminal", immediate_terminal_output)

    async def scenario():
        output = io.StringIO()
        console = Console(file=output, force_terminal=False)
        state = ChatState(pending=True)
        app = FakeApp(console)
        task = asyncio.create_task(_complete_submission(app, state, "message", console))
        state.reply_task = task
        await real_sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert state.pending is False
        assert output.getvalue() == ""

    asyncio.run(scenario())
