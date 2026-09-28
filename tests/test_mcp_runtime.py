"""The shared MCP runtime, driven through a fake connector — no server, no
network. What matters: sessions are reused, a dead one is replaced once, and
an answer from the server (MCPError) is never retried."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import ErrorData

from app.providers.mcp_runtime import McpRuntime


class FakeSession:
    def __init__(self, number: int, script: list) -> None:
        self.number = number
        self._script = script
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        outcome = self._script.pop(0) if self._script else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "slow":
            await asyncio.sleep(10)
        return f"{outcome} from session {self.number}"


class FakeConnector:
    """Hands out numbered FakeSessions and records opens and closes."""

    def __init__(self, script: list | None = None) -> None:
        self.script = script or []
        self.opened: list[tuple[str, dict]] = []
        self.closed = 0
        self.sessions: list[FakeSession] = []

    @asynccontextmanager
    async def __call__(self, url, headers):
        self.opened.append((url, headers))
        session = FakeSession(len(self.opened), self.script)
        self.sessions.append(session)
        try:
            yield session
        finally:
            self.closed += 1


@pytest.fixture
def connector():
    return FakeConnector()


@pytest.fixture
def runtime(connector):
    rt = McpRuntime(connector=connector, call_timeout=0.5)
    yield rt
    rt.close()


def call(rt, url="https://mcp.example/mcp", headers=None, tool="search", arguments=None):
    return rt.run(rt.call_tool(url, headers or {"X-Key": "k"}, tool, arguments or {}))


def test_calls_to_the_same_server_reuse_one_session(runtime, connector):
    assert call(runtime) == "ok from session 1"
    assert call(runtime) == "ok from session 1"
    assert call(runtime, arguments={"q": "x"}) == "ok from session 1"

    assert len(connector.opened) == 1
    assert len(connector.sessions[0].calls) == 3


def test_different_servers_or_keys_get_their_own_sessions(runtime, connector):
    call(runtime, url="https://a.example/mcp")
    call(runtime, url="https://b.example/mcp")
    call(runtime, url="https://a.example/mcp", headers={"X-Key": "other"})

    assert len(connector.opened) == 3


def test_a_failed_connection_is_replaced_and_the_call_retried_once(runtime, connector):
    connector.script[:] = [ConnectionError("session expired"), "ok"]

    assert call(runtime) == "ok from session 2"
    assert len(connector.opened) == 2
    assert connector.closed == 1  # the dead one was closed, not leaked


def test_two_failures_in_a_row_are_raised(runtime, connector):
    connector.script[:] = [ConnectionError("down"), ConnectionError("still down")]

    with pytest.raises(ConnectionError, match="still down"):
        call(runtime)


def test_an_answer_from_the_server_is_not_retried(runtime, connector):
    refused = MCPError.from_error_data(ErrorData(code=-32603, message="refused"))
    connector.script[:] = [refused]

    with pytest.raises(MCPError):
        call(runtime)

    assert len(connector.opened) == 1
    # The session is still good for the next call.
    assert call(runtime) == "ok from session 1"


def test_a_call_that_hangs_times_out_and_reconnects(runtime, connector):
    connector.script[:] = ["slow", "ok"]

    assert call(runtime) == "ok from session 2"


def test_a_connector_that_cannot_connect_raises(connector):
    @asynccontextmanager
    async def refuse(url, headers):
        raise ConnectionRefusedError("no route to host")
        yield

    rt = McpRuntime(connector=refuse)
    try:
        with pytest.raises(ConnectionRefusedError):
            call(rt)
    finally:
        rt.close()


def test_calls_can_run_at_the_same_time_on_one_session(runtime, connector):
    async def three():
        return await asyncio.gather(
            *(runtime.call_tool("https://mcp.example/mcp", {}, "t", {"i": i}) for i in range(3))
        )

    assert runtime.run(three()) == ["ok from session 1"] * 3
    assert len(connector.opened) == 1


def test_close_closes_every_session_and_refuses_new_calls(connector):
    rt = McpRuntime(connector=connector)
    call(rt, url="https://a.example/mcp")
    call(rt, url="https://b.example/mcp")

    rt.close()

    assert connector.closed == 2
    with pytest.raises(RuntimeError, match="shut down"):
        call(rt)
