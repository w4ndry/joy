import asyncio
import io
import queue
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from rich.console import Console

import joy_agent.tools as tool_module
import joy_agent.ui as ui_module
from joy_agent.llm import (
    SYSTEM_PROMPT,
    MissingCredentialsError,
    StreamHandle,
    StreamingLLM,
)
from joy_agent.tools import (
    FileChange,
    ToolError,
    bash,
    edit,
    execution_context,
    read,
    write,
)
from joy_agent.ui import (
    INPUT_PLACEHOLDER,
    ChatState,
    _cancel_pending,
    _normalize_inline_bullets,
    _render_change_lines,
    _submit_buffer,
    _tool_status_text,
    create_prompt_session,
    render_assistant_message,
    render_code_preview,
    render_file_changes,
    render_persistent_summary,
    render_pretty,
    render_syntax,
    render_table,
    render_tool_completion,
    render_tool_elapsed,
    render_user_message,
)


class FakeResponseStream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.events)

    def close(self):
        self.closed = True


class FakeResponses:
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
    def __init__(self, responses):
        self.responses = responses
        self.closed = False

    def close(self):
        self.closed = True


def completed(output):
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(output=output),
    )


def text_delta(value):
    return SimpleNamespace(type="response.output_text.delta", delta=value)


def refusal_delta(value):
    return SimpleNamespace(type="response.refusal.delta", delta=value)


def make_llm(monkeypatch, streams=None, error=None):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses = FakeResponses(streams, error)
    client = FakeClient(responses)
    llm = StreamingLLM(
        client_factory=lambda **_: client,
        base_url="https://example.test/v1",
    )
    return llm, client, responses


def drain(handle):
    events = []
    while True:
        try:
            events.append(handle.events.get_nowait())
        except queue.Empty:
            return events


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_module, "_workspace_root", tmp_path.resolve())
    return tmp_path


def test_missing_credentials_is_clear(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JOY_AGENT_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        StreamingLLM()


def test_system_prompt_requires_real_markdown_bullets():
    assert "each item on its own line as a '-' bullet" in SYSTEM_PROMPT
    assert "Never put multiple bullets in one paragraph" in SYSTEM_PROMPT


def test_tool_file_operations_and_read_window(workspace):
    assert "wrote" in write("nested/example.txt", "one\ntwo\nthree\n")
    assert read("nested/example.txt", offset=2, limit=2) == "2: two\n3: three"
    diff = edit("nested/example.txt", "two", "changed")
    assert "-two" in diff
    assert "changed" in workspace.joinpath("nested/example.txt").read_text()


def test_file_mutations_capture_text_changes_for_the_ui(workspace):
    cancel = threading.Event()
    with execution_context(cancel) as context:
        write("new.txt", "first\nsecond\n")
    assert context.changes == [FileChange("new.txt", None, "first\nsecond\n")]

    workspace.joinpath("existing.txt").write_text("before\n")
    with execution_context(cancel) as context:
        write("existing.txt", "after\n")
    assert context.changes == [FileChange("existing.txt", "before\n", "after\n")]


def test_tool_limits_and_path_restrictions(workspace):
    with pytest.raises(ToolError, match="2,000"):
        read("missing", limit=2_001)
    with pytest.raises(ToolError, match="escapes"):
        read("../../outside")
    outside = workspace.parent / "joy-outside.txt"
    outside.write_text("secret")
    workspace.joinpath("link").symlink_to(outside)
    with pytest.raises(ToolError, match="escapes"):
        read("link")


def test_edit_fuzzy_match_preserves_bom_line_endings_and_unrelated_lines(workspace):
    path = workspace / "fuzzy.txt"
    path.write_bytes(
        "\ufeffbefore — keep\r\nconst x = ‘one’  \r\nafter — keep\r\n".encode()
    )
    edit("fuzzy.txt", "const x = 'one'\n", "const x = 'two'\n")
    result = path.read_bytes().decode()
    assert result.startswith("\ufeff")
    assert "before — keep\r\n" in result
    assert "const x = 'two'\r\n" in result
    assert "after — keep\r\n" in result


def test_edit_rejects_ambiguous_missing_and_noop(workspace):
    write("ambiguous.txt", "same\nsame\n")
    with pytest.raises(ToolError, match="ambiguous"):
        edit("ambiguous.txt", "same", "new")
    with pytest.raises(ToolError, match="whitespace"):
        edit("ambiguous.txt", "  \n", "new")
    with pytest.raises(ToolError, match="could not find"):
        edit("ambiguous.txt", "missing", "new")
    with pytest.raises(ToolError, match="no change"):
        edit("ambiguous.txt", "same\nsame", "same\nsame")


def test_edit_detects_external_change(workspace, monkeypatch):
    path = workspace / "race.txt"
    path.write_text("old")
    original_read = tool_module._read_bytes
    calls = 0

    def changing_read(candidate):
        nonlocal calls
        calls += 1
        value = original_read(candidate)
        if calls == 1:
            candidate.write_text("changed")
        return value

    monkeypatch.setattr(tool_module, "_read_bytes", changing_read)
    with pytest.raises(ToolError, match="changed while"):
        edit("race.txt", "old", "new")


def test_bash_returns_output_exit_code_and_runs_in_workspace(workspace):
    result = bash("printf '%s' \"$PWD\"; printf '\\nerr' >&2; exit 3")
    assert str(workspace) in result
    assert "err" in result
    assert "exit_code=3" in result


def test_bash_timeout_and_validation(workspace):
    result = bash("sleep 2", timeout=1)
    assert "timeout after 1s" in result
    with pytest.raises(ToolError, match="between 1 and 300"):
        bash("true", timeout=301)


def test_bash_cancellation_terminates_process_group(workspace):
    cancel = threading.Event()
    result = []

    def run_command():
        with tool_module.execution_context(cancel):
            result.append(bash("sleep 10", timeout=30))

    thread = threading.Thread(target=run_command)
    thread.start()
    time.sleep(0.2)
    cancel.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert "cancelled" in result[0]


def test_responses_streaming_and_reasoning_replay(monkeypatch):
    first = FakeResponseStream(
        [
            text_delta("Reading "),
            text_delta("file"),
            completed(
                [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "encrypted",
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read",
                        "arguments": '{"path":"README.md"}',
                    },
                ]
            ),
        ]
    )
    second = FakeResponseStream(
        [text_delta("Done"), completed([{"type": "message", "role": "assistant"}])]
    )
    llm, client, responses = make_llm(monkeypatch, [first, second])
    handle = llm.start_stream(llm.messages_for("inspect"), 1)
    handle.thread.join(timeout=3)
    events = drain(handle)
    assert "Reading fileDone" == "".join(
        event.content for event in events if event.kind == "text"
    )
    started = next(event for event in events if event.kind == "tool_started")
    assert started.tool_action == "Reading"
    assert started.tool_target == "README.md"
    assert any(event.kind == "tool_finished" for event in events)
    assert events[-1].kind == "complete"
    assert responses.calls[0]["model"] == "gpt-5.6-luna"
    assert responses.calls[0]["reasoning"] == {"effort": "medium"}
    assert responses.calls[0]["store"] is False
    assert responses.calls[0]["include"] == ["reasoning.encrypted_content"]
    second_input = responses.calls[1]["input"]
    assert second_input[1]["type"] == "reasoning"
    assert second_input[-1]["type"] == "function_call_output"
    assert client.closed is False


def test_refusal_incomplete_and_api_error(monkeypatch):
    refusal = FakeResponseStream([refusal_delta("no"), completed([])])
    incomplete = FakeResponseStream([text_delta("partial")])
    llm, _, _ = make_llm(monkeypatch, [refusal, incomplete])
    first = llm.start_stream(llm.messages_for("one"), 1)
    first.thread.join(timeout=2)
    assert [event.content for event in drain(first) if event.kind == "refusal"] == [
        "no"
    ]
    second = llm.start_stream(llm.messages_for("two"), 2)
    second.thread.join(timeout=2)
    events = drain(second)
    assert events[-1].kind == "incomplete"
    llm, _, _ = make_llm(monkeypatch, error=RuntimeError("boom"))
    failed = llm.start_stream(llm.messages_for("three"), 3)
    failed.thread.join(timeout=2)
    assert str(drain(failed)[-1].error) == "boom"


def test_incomplete_response_does_not_execute_tools(monkeypatch, workspace):
    stream = FakeResponseStream(
        [
            completed(
                [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "write",
                        "arguments": '{"path":"bad","content":"x"}',
                    }
                ]
            ),
            SimpleNamespace(type="response.incomplete"),
        ]
    )
    llm, _, _ = make_llm(monkeypatch, [stream])
    handle = llm.start_stream(llm.messages_for("write"), 1)
    handle.thread.join(timeout=2)
    assert not workspace.joinpath("bad").exists()


def test_prompt_layout_and_rendering():
    session = create_prompt_session()
    assert session.placeholder == [("class:placeholder", INPUT_PLACEHOLDER)]
    assert session.joy_input_frame.height.min == 3
    assert session.joy_input_frame.height.max == 3
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    render_user_message("first line\n[not markdown]", console)
    render_assistant_message("**reply**\nnext", console)
    rendered = output.getvalue()
    assert "› first line" in rendered
    assert "[not markdown]" in rendered
    assert "reply" in rendered
    assert "next" in rendered


def test_tool_status_uses_compact_centered_dot_format():
    state = ChatState(
        tool_statuses=[("bash", "running")],
        tool_started_at=10.0,
    )
    original_monotonic = ui_module.time.monotonic
    ui_module.time.monotonic = lambda: 12.4
    try:
        assert _tool_status_text(state) == "  Working (2s)"
    finally:
        ui_module.time.monotonic = original_monotonic

    state = ChatState(
        tool_statuses=[("bash", "finished")],
        tool_elapsed=4.2,
        tool_finished_at=datetime(2026, 10, 16, 20, 7),
    )
    assert _tool_status_text(state) == "  Worked for 4s · 20:07"


def test_tool_elapsed_is_persistent_below_completion_summary():
    output = io.StringIO()
    console = Console(file=output, record=True, force_terminal=False)
    render_persistent_summary("Done. Changed the file.", console, leading_blank=True)
    render_tool_elapsed(16.2, datetime(2026, 10, 16, 21, 28), console=console)

    lines = console.export_text(clear=False).splitlines()
    assert lines == [
        "",
        "• Done. Changed the file.",
        "",
        "  Worked for 16s · 21:28",
        "",
    ]


def test_assistant_markdown_preserves_headings_code_and_ordered_lists():
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    render_assistant_message(
        "## Project structure\n\n"
        "```text\nprovider:model\n```\n\n"
        "1. First item\n2. Second item",
        console,
    )
    rendered = output.getvalue()
    assert "Project structure" in rendered
    assert "provider:model" in rendered
    assert "1 First item" in rendered
    assert "2 Second item" in rendered


def test_rich_syntax_and_code_preview_render_line_numbers_and_source():
    output = io.StringIO()
    console = Console(file=output, record=True, force_terminal=False)
    render_syntax("def greet():\n    return 'hello'", "python", console)
    render_code_preview('{"ok": true}', "result.json", console)

    rendered = console.export_text(clear=False)
    assert "1 def greet():" in rendered
    assert "2     return 'hello'" in rendered
    assert '1 {"ok": true}' in rendered


def test_rich_pretty_and_table_render_structured_values():
    output = io.StringIO()
    console = Console(file=output, record=True, force_terminal=False)
    render_pretty({"files": ["one.py", "two.py"], "count": 2}, console)
    render_table(
        ["File", "Status"],
        [["one.py", "added"], ["two.py", "edited"]],
        title="Changes",
        console=console,
    )

    rendered = console.export_text(clear=False)
    assert "files" in rendered
    assert "one.py" in rendered
    assert "Changes" in rendered
    assert "Status" in rendered
    assert "edited" in rendered


def test_rich_table_rejects_rows_with_wrong_width():
    with pytest.raises(ValueError, match="row width"):
        render_table(["File", "Status"], [["one.py"]])


def test_file_change_rendering_highlights_lines_and_changed_text():
    change = FileChange(
        "demo.py",
        "keep\nprefix old suffix\nremoved\nend\n",
        "keep\nprefix new suffix\nend\nadded\n",
    )
    rows = _render_change_lines(change, 40)
    assert [row.plain for row in rows] == [
        "2 - prefix old suffix" + " " * 19,
        "2 + prefix new suffix" + " " * 19,
        "3 - removed" + " " * 29,
        "4 + added" + " " * 31,
    ]
    assert len(rows[0].plain) == 40
    assert str(rows[0].spans[0].style) == "#858b98 on #452c30"
    assert str(rows[0].spans[1].style) == "#d9a0a5 on #452c30"
    assert str(rows[1].spans[0].style) == "#858b98 on #263d30"
    assert any(
        str(span.style) == "#9cb9d8 on #263d30" for span in rows[1].spans[1:]
    )

    output = io.StringIO()
    console = Console(file=output, record=True, force_terminal=False, width=40)
    render_file_changes([change], console)
    plain = console.export_text(clear=False)
    styled = console.export_text(clear=False, styles=True)
    assert "demo.py" in plain
    assert "new" in plain
    assert "removed" in plain
    assert "added" in plain
    assert "• Edited demo.py (+2, -2)" in plain
    assert "\x1b[38;2;217;160;165;48;2;69;44;48m-" in styled
    assert "\x1b[38;2;217;160;165;48;2;69;44;48m" in styled
    assert "\x1b[38;2;163;201;168;48;2;38;61;48m+" in styled
    assert "\x1b[38;2;163;201;168;48;2;38;61;48madded" in styled


def test_persistent_summaries_and_failure_lines_are_readable():
    output = io.StringIO()
    console = Console(file=output, record=True, force_terminal=False)
    render_persistent_summary("I’ll edit test.md.", console)
    render_tool_completion("Editing", "test.md", console=console, success=False)
    plain = console.export_text(clear=False)
    styled = console.export_text(clear=False, styles=True)
    assert "• I’ll edit test.md." in plain
    assert "• Failed — editing test.md" in plain
    assert "• I’ll edit test.md.\n\n" in plain
    assert "\x1b[38;2;133;139;152;48;2;30;30;36m•" in styled
    assert "\x1b[38;2;197;203;211;48;2;30;30;36m" in styled


def test_tool_activity_spacing_places_completion_directly_after_diff():
    output = io.StringIO()
    console = Console(file=output, record=True, force_terminal=False, width=32)
    change = FileChange("test.md", "old\n", "new\n")

    render_persistent_summary("I’ll edit test.md.", console)
    render_file_changes([change], console)
    render_persistent_summary("Edited test.md.", console, leading_blank=True)

    plain = console.export_text(clear=False)
    lines = plain.splitlines()
    assert lines[0] == "• I’ll edit test.md."
    assert lines[1] == ""
    assert lines[2] == "• Edited test.md (+1, -1)"
    assert lines[4].startswith("1 + new")
    assert lines[5] == ""
    assert lines[6] == "• Edited test.md."
    assert lines[7] == ""


def test_inline_capability_items_render_as_markdown_bullets():
    formatted = _normalize_inline_bullets(
        "I can: - Read files - Write files - Run tests\n"
        "I can also inspect code."
    )
    assert formatted == (
        "I can:\n\n- Read files\n- Write files\n- Run tests\n\n"
        "I can also inspect code."
    )


def test_inline_capability_items_render_as_separate_rich_bullets():
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    render_assistant_message(
        "I can: - Read files - Write files - Run tests\n"
        "I can also inspect code.",
        console,
    )
    rendered = output.getvalue()
    assert "• Read files" in rendered
    assert "• Write files" in rendered
    assert "• Run tests" in rendered
    assert "• Run tests. I can also" not in rendered


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
            SimpleNamespace(generation_id=1, kind="text", content="reply")
        )
        self.handle.events.put(
            SimpleNamespace(generation_id=1, kind="complete", content="")
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


def test_cancellation_after_tool_use_records_interruption(monkeypatch):
    async def immediate_terminal_output(function):
        function()

    monkeypatch.setattr("joy_agent.ui.run_in_terminal", immediate_terminal_output)

    async def scenario():
        output = io.StringIO()
        llm = FakeLLM()
        state = ChatState(
            llm=llm,
            pending=True,
            preview_text="partial",
            pending_user_text="change it",
            tool_used=True,
        )
        state.stream_handle = StreamHandle(2)
        app = FakeApp(Console(file=output, force_terminal=False))
        event = FakeEvent(app, "next draft")
        state.reply_task = asyncio.create_task(asyncio.sleep(10))
        _cancel_pending(event, state)
        await asyncio.gather(*app.tasks)
        assert state.pending is False
        assert "cancelled" in output.getvalue().lower()

    asyncio.run(scenario())
