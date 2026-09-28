"""Long-lived MCP client sessions, reused across calls.

Before this, every MCP call opened its own connection and repeated the
initialize handshake inside a fresh `asyncio.run`. Measured on Google's Maps
Grounding Lite server: a call on a fresh connection took 2.6–3.7 s, the same
call on a reused session 1.5–2.2 s, and connecting plus initialize alone
0.48 s. Three calls at once on one session took 3.3 s.

How it's built:

- One event loop runs in a daemon thread and owns every session. The
  planner's code is synchronous (LangGraph nodes run in worker threads), so
  it hands coroutines to that loop with `run()` and blocks for the result.
- One session per (server URL, headers), opened on first use and kept.
- Each session lives in its own task, because the MCP SDK's streamable-HTTP
  client holds anyio task groups that must be entered and exited in the same
  task. Closing a session means telling that task to finish, never cancelling
  it from another task.
- A call that fails with anything other than an answer from the server (a
  dropped connection, an expired session) closes that session and is retried
  once on a new one. An `MCPError` means the server did answer, so it isn't
  retried: a new connection wouldn't change its mind.
"""

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult

logger = logging.getLogger(__name__)

# Opens one initialized session for (url, headers). Swappable so tests can
# exercise the connection lifecycle without a server.
Connector = Callable[[str, dict[str, str]], AbstractAsyncContextManager[ClientSession]]

CALL_TIMEOUT_SECONDS = 90


@asynccontextmanager
async def streamable_http_session(url: str, headers: dict[str, str]):
    async with httpx2.AsyncClient(headers=headers, timeout=CALL_TIMEOUT_SECONDS) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class _Connection:
    """One open session, held open by its own task until told to stop."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.error: BaseException | None = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.task: asyncio.Task | None = None

    async def hold(self, connector: Connector, url: str, headers: dict[str, str]) -> None:
        try:
            async with connector(url, headers) as session:
                self.session = session
                self.ready.set()
                await self.stop.wait()
        except Exception as exc:
            self.error = exc
        finally:
            self.session = None
            self.ready.set()

    @property
    def alive(self) -> bool:
        return (
            self.session is not None
            and self.task is not None
            and not self.task.done()
            and not self.stop.is_set()
        )


class McpRuntime:
    def __init__(
        self,
        connector: Connector = streamable_http_session,
        call_timeout: float = CALL_TIMEOUT_SECONDS,
    ) -> None:
        self._connector = connector
        self._call_timeout = call_timeout
        self._connections: dict[tuple, _Connection] = {}
        self._lock: asyncio.Lock | None = None
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-runtime", daemon=True
        )
        self._thread.start()

    # --- for synchronous callers -------------------------------------------

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Run `coro` on the runtime's loop and wait for its result."""
        if self._closed:
            coro.close()
            raise RuntimeError("the MCP runtime has been shut down")
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError("run() called from the runtime's own loop; await instead")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.run(self._close_all())
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    # --- for coroutines running on the runtime's loop ----------------------

    async def call_tool(
        self, url: str, headers: dict[str, str], tool: str, arguments: dict
    ) -> CallToolResult:
        key = (url, tuple(sorted(headers.items())))
        for attempt in (1, 2):
            connection = await self._connection(key, url, headers)
            assert connection.session is not None
            try:
                return await asyncio.wait_for(
                    connection.session.call_tool(tool, arguments), self._call_timeout
                )
            except MCPError:
                raise
            except Exception as exc:
                await self._drop(key, connection)
                if attempt == 2:
                    raise
                logger.info("MCP session to %s failed (%s); reconnecting once", url, exc)
        raise AssertionError("unreachable")

    async def _connection(self, key: tuple, url: str, headers: dict[str, str]) -> _Connection:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            existing = self._connections.get(key)
            if existing is not None and existing.alive:
                return existing
            if existing is not None:
                await self._drop(key, existing)

            connection = _Connection()
            connection.task = asyncio.create_task(connection.hold(self._connector, url, headers))
            await connection.ready.wait()
            if connection.error is not None or connection.session is None:
                raise connection.error or RuntimeError(f"could not open an MCP session to {url}")
            self._connections[key] = connection
            logger.info("opened MCP session to %s", url)
            return connection

    async def _drop(self, key: tuple, connection: _Connection) -> None:
        if self._connections.get(key) is connection:
            del self._connections[key]
        connection.stop.set()
        if connection.task is not None:
            await asyncio.wait({connection.task}, timeout=5)

    async def _close_all(self) -> None:
        for key, connection in list(self._connections.items()):
            await self._drop(key, connection)


_runtime: McpRuntime | None = None
_runtime_lock = threading.Lock()


def get_mcp_runtime() -> McpRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = McpRuntime()
        return _runtime


def shutdown_mcp_runtime() -> None:
    """Close every session. Called when the app shuts down."""
    global _runtime
    with _runtime_lock:
        runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.close()
