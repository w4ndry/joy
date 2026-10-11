import asyncio
import io
import queue
from types import SimpleNamespace

import pytest
from prompt_toolkit.filters import Always
from rich.console import Console

from joy_agent.llm import (
    MODEL,
    SYSTEM_PROMPT,
    MissingCredentialsError,
    StreamHandle,
    StreamingLLM,
)
from joy_agent.ui import (
    INPUT_PLACEHOLDER,
    ChatState,
    _cancel_pending,
    _submit_buffer,
    create_prompt_session,
    render_assistant_message,
    render_user_message,
)


def chunk(content=None, finish_reason=None, refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, refusal=refusal),
                finish_reason=finish_reason,
            )
        ]
    )


class FakeStream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chunks)

    def close(self):
        self.closed = True


class FakeCompletions:
    def __init__(self, streams=None, error=None):
        self.streams = iter(streams or [])
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return next(self.streams)


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)
        self.providers = {}
        self.closed = False

    def close(self):
        self.closed = True


def make_llm(monkeypatch, completions):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(completions)
    llm = StreamingLLM(
        client_factory=lambda **_: client,
        base_url="https://example.test/v1",
    )
    return llm, client


def drain(handle):
    events = []
    while True:
        try:
            events.append(handle.events.get_nowait())
        except queue.Empty:
            return events


def test_missing_credentials_is_clear(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        StreamingLLM()


def test_streamed_chunks_and_history(monkeypatch):
    stream = FakeStream([chunk("Hello "), chunk("world"), chunk(finish_reason="stop")])
    completions = FakeCompletions([stream])
    llm, client = make_llm(monkeypatch, completions)

    handle = llm.start_stream(llm.messages_for("Hi"), 1)
    handle.thread.join(timeout=2)
    events = drain(handle)

    assert [event.content for event in events if event.kind == "text"] == [
        "Hello ",
        "world",
    ]
    assert events[-1].kind == "complete"
    assert completions.calls[0]["model"] == MODEL
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert client is not None

    llm.commit_exchange("Hi", "Hello world")
    assert llm.messages_for("Again")[1:] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello world"},
        {"role": "user", "content": "Again"},
    ]
    llm.close()
    assert client.closed is True


def test_refusal_and_incomplete_streams(monkeypatch):
    refusal_stream = FakeStream(
        [chunk(refusal="I cannot help with that."), chunk(finish_reason="stop")]
    )
    incomplete_stream = FakeStream([chunk("partial")])
    completions = FakeCompletions([refusal_stream, incomplete_stream])
    llm, _ = make_llm(monkeypatch, completions)

    first = llm.start_stream(llm.messages_for("one"), 1)
    first.thread.join(timeout=2)
    first_events = drain(first)
    assert [event.content for event in first_events if event.kind == "refusal"] == [
        "I cannot help with that."
    ]
    assert first_events[-1].kind == "complete"

    second = llm.start_stream(llm.messages_for("two"), 2)
    second.thread.join(timeout=2)
    assert drain(second)[-1].kind == "incomplete"


def test_api_error_is_delivered(monkeypatch):
    llm, _ = make_llm(monkeypatch, FakeCompletions(error=RuntimeError("boom")))
    handle = llm.start_stream(llm.messages_for("Hi"), 4)
    handle.thread.join(timeout=2)
    events = drain(handle)
    assert events[-1].kind == "error"
    assert str(events[-1].error) == "boom"


def test_prompt_layout_has_three_row_input_configuration():
    session = create_prompt_session()
    windows = [
        window
        for window in session.layout.find_all_windows()
        if getattr(window.content, "buffer", None) is session.default_buffer
    ]

    assert "class:frame" in session.joy_input_frame.style
    assert session.color_depth.name == "DEPTH_24_BIT"
    assert session.placeholder == [("class:placeholder", INPUT_PLACEHOLDER)]
    assert session.joy_input_frame.height.min == 3
    assert session.joy_input_frame.height.max == 3
    assert windows
    assert all(
        window.height.min == 1
        and window.height.max == 1
        and window.height.preferred == 1
        for window in windows
    )
    assert all(isinstance(window.dont_extend_height, Always) for window in windows)


def test_conversation_rendering_is_literal_and_indented():
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)

    render_user_message("first line\n[not markdown]", console)
    render_assistant_message("**reply**\nnext", console)

    rendered = output.getvalue()
    assert "› first line" in rendered
    assert "[not markdown]" in rendered
    assert " reply" in rendered
    assert " next" in rendered


class FakeBuffer:
    def __init__(self, text):
        self.text = text

    def reset(self):
        self.text = ""


class FakeApp:
    def __init__(self, console):
        self.joy_console = console
        self.tasks = []

    def create_background_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def invalidate(self):
        pass


class FakeEvent:
    def __init__(self, app, text):
        self.app = app
        self.current_buffer = FakeBuffer(text)


class FakeLLM:
    def __init__(self):
        self.history = []
        self.handle = StreamHandle(1)

    def messages_for(self, text):
        return [{"role": "user", "content": text}]

    def start_stream(self, _messages, _generation_id):
        self.handle.events.put(
            type(
                "Event",
                (),
                {"generation_id": 1, "kind": "text", "content": "reply"},
            )()
        )
        self.handle.events.put(
            type(
                "Event",
                (),
                {"generation_id": 1, "kind": "complete", "content": ""},
            )()
        )
        return self.handle

    def commit_exchange(self, user, assistant):
        self.history.append((user, assistant))


def test_submission_clears_input_and_preserves_next_draft(monkeypatch):
    async def immediate_terminal_output(function):
        function()

    monkeypatch.setattr("joy_agent.ui.run_in_terminal", immediate_terminal_output)

    async def scenario():
        output = io.StringIO()
        state = ChatState(llm=FakeLLM())
        app = FakeApp(Console(file=output, force_terminal=False))
        event = FakeEvent(app, "first message")
        _submit_buffer(event, state)
        assert event.current_buffer.text == ""
        event.current_buffer.text = "next draft"
        await asyncio.gather(*app.tasks)
        assert event.current_buffer.text == "next draft"
        assert state.llm.history == [("first message", "reply")]

    asyncio.run(scenario())


def test_cancellation_preserves_draft_and_discards_partial(monkeypatch):
    async def immediate_terminal_output(function):
        function()

    monkeypatch.setattr("joy_agent.ui.run_in_terminal", immediate_terminal_output)

    async def scenario():
        output = io.StringIO()
        state = ChatState(llm=FakeLLM(), pending=True, preview_text="partial")
        state.stream_handle = StreamHandle(2)
        app = FakeApp(Console(file=output, force_terminal=False))
        event = FakeEvent(app, "next draft")
        state.reply_task = asyncio.create_task(asyncio.sleep(10))
        _cancel_pending(event, state)
        await asyncio.gather(*app.tasks)
        assert event.current_buffer.text == "next draft"
        assert state.pending is False
        assert "partial" in output.getvalue()
        assert "cancelled" in output.getvalue().lower()

    asyncio.run(scenario())
