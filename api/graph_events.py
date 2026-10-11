"""
api/graph_events.py

A tiny in-process pub/sub hub that fans out knowledge-graph *delta* events
(newly created nodes and links) to every connected WebSocket client so the
frontend D3 view can animate ingestion live instead of polling /graph.

Design notes
------------
* No external broker (Redis, etc.) — this is a single-process FastAPI app, so a
  module-level set of connected sockets is sufficient and keeps the free-tier
  footprint zero. If the app is ever scaled to multiple workers, swap this for a
  Redis pub/sub without changing the call sites.
* Broadcasting is best-effort and never raises into the ingestion path: a slow
  or dead client must not break a MERGE. Failed sends simply drop that client.
* The merger emits deltas synchronously from worker threads (Neo4j driver is
  sync), so `publish_threadsafe` schedules the coroutine onto the stored event
  loop. The async `publish` is available for coroutine call sites.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketState

logger = logging.getLogger(__name__)


class GraphEventHub:
    """Fan-out hub for live graph delta events over WebSocket."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        # Captured at startup so sync (threaded) producers can schedule sends.
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the running event loop for threadsafe publishing."""
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info("Graph WS client connected (%d total)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("Graph WS client disconnected (%d remaining)", len(self._clients))

    async def publish(self, event: dict[str, Any]) -> None:
        """Broadcast *event* to all connected clients (coroutine call sites)."""
        if not self._clients:
            return
        # Snapshot under lock, then send outside it so a slow client can't block
        # the registry.
        async with self._lock:
            targets = list(self._clients)

        dead: list[WebSocket] = []
        for ws in targets:
            try:
                if ws.application_state != WebSocketState.CONNECTED:
                    dead.append(ws)
                    continue
                await ws.send_json(event)
            except Exception as exc:  # noqa: BLE001 - best-effort fan-out
                logger.debug("Dropping graph WS client on send failure: %s", exc)
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def publish_threadsafe(self, event: dict[str, Any]) -> None:
        """Schedule a broadcast from a non-async context (worker thread).

        Safe to call from the sync Neo4j merge path. Never raises.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.publish(event), loop)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to schedule graph WS broadcast: %s", exc)

    @property
    def client_count(self) -> int:
        return len(self._clients)


# Module-level singleton shared by the merger (producer) and the WS route
# (consumer registry).
hub = GraphEventHub()
