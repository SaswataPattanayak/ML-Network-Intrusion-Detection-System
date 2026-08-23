"""
server.py
---------
FastAPI backend for the IDS dashboard.

Historical ML/model-diagnostic sections are served from the static
 dashboard_data.json file. Dynamic synthetic monitoring is persisted in
SQLite for counters/history and streamed to connected browsers through a
WebSocket.

Important dashboard behavior:
- A browser refresh does NOT replay old SQLite alerts into the live feed.
- Persistent counters are loaded from /api/alerts/stats.
- Only newly generated alerts are pushed over WebSocket.
- Clear Events removes SQLite monitoring history and clears queued alerts
  waiting to be broadcast.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import alerts_db
from config import (
    DASHBOARD_HISTORICAL_JSON,
    INFERENCE_QUEUE_MAXSIZE,
    SERVER_HOST,
    SERVER_PORT,
    STATIC_DIR,
    WS_BROADCAST_QUEUE_MAXSIZE,
)
from inference_engine import InferenceEngine
from synthetic_flow_generator import SyntheticFlowGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nids.server")


# ------------------------------------------------------------------------
# WebSocket connection manager
# ------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self._connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info(
            "WebSocket client connected (%d total).",
            len(self._connections),
        )

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info(
            "WebSocket client disconnected (%d total).",
            len(self._connections),
        )

    async def broadcast(self, message: Dict) -> None:
        async with self._lock:
            targets = list(self._connections)

        dead: List[WebSocket] = []
        payload = json.dumps(message)
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)


manager = ConnectionManager()

# Bridges the inference engine's background thread into the asyncio loop.
_bridge_queue: "queue.Queue[Dict]" = queue.Queue(
    maxsize=WS_BROADCAST_QUEUE_MAXSIZE
)

_synthetic_generator = None
_inference_engine: Optional[InferenceEngine] = None
_bridge_task: Optional[asyncio.Task] = None


def _on_alert_from_inference_thread(payload: Dict) -> None:
    """Queue an alert for WebSocket delivery without blocking inference."""
    try:
        _bridge_queue.put_nowait(payload)
    except queue.Full:
        # Drop the oldest queued message rather than blocking the inference
        # thread when a browser is slow/disconnected.
        try:
            _bridge_queue.get_nowait()
            _bridge_queue.put_nowait(payload)
        except queue.Empty:
            pass


def _drain_bridge_queue() -> int:
    """Remove alerts already waiting to be broadcast after Clear Events."""
    removed = 0
    while True:
        try:
            _bridge_queue.get_nowait()
            removed += 1
        except queue.Empty:
            return removed


async def _bridge_loop():
    """Forward thread-produced alerts to connected WebSocket clients."""
    loop = asyncio.get_running_loop()
    while True:
        payload = await loop.run_in_executor(None, _bridge_queue.get)
        await manager.broadcast({"type": "alert", "data": payload})


# ------------------------------------------------------------------------
# App lifecycle
# ------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _synthetic_generator, _inference_engine, _bridge_task

    alerts_db.init_db()
    _bridge_task = asyncio.create_task(_bridge_loop())

    flow_queue: "queue.Queue[Dict]" = queue.Queue(
        maxsize=INFERENCE_QUEUE_MAXSIZE
    )
    _inference_engine = InferenceEngine(
        flow_queue,
        on_alert=_on_alert_from_inference_thread,
    )
    _synthetic_generator = SyntheticFlowGenerator(flow_queue)

    _inference_engine.start()
    _synthetic_generator.start()
    logger.info(
        "Synthetic flow -> ML inference pipeline started (no packet capture)."
    )

    yield

    logger.info("Shutting down synthetic flow pipeline ...")
    if _synthetic_generator is not None:
        _synthetic_generator.stop()
    if _inference_engine is not None:
        _inference_engine.stop()
    if _bridge_task is not None:
        _bridge_task.cancel()
        try:
            await _bridge_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="NIDS Synthetic Flow Dashboard", lifespan=lifespan)


# ------------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------------
@app.get("/api/alerts/recent")
def get_recent_alerts(limit: int = 50, attacks_only: bool = False):
    try:
        return JSONResponse(
            alerts_db.fetch_recent_alerts(
                limit=limit,
                attacks_only=attacks_only,
            )
        )
    except Exception:
        logger.exception("Failed to fetch recent alerts")
        return JSONResponse(
            {"error": "failed to fetch alerts"},
            status_code=500,
        )


@app.get("/api/alerts/stats")
def get_alert_stats():
    try:
        return JSONResponse(alerts_db.fetch_stats())
    except Exception:
        logger.exception("Failed to fetch alert stats")
        return JSONResponse(
            {"error": "failed to fetch stats"},
            status_code=500,
        )


@app.get("/api/health")
def health():
    generator_status = (
        _synthetic_generator.status
        if _synthetic_generator is not None
        else "stopped"
    )
    return {
        "status": "ok",
        "synthetic_flow_active": generator_status == "running",
        "synthetic_flow_status": generator_status,
        "connected_dashboards": len(manager._connections),
    }


@app.post("/api/synthetic/start")
def start_synthetic_feed():
    if _synthetic_generator is None:
        return JSONResponse(
            {"error": "synthetic generator is not initialized"},
            status_code=503,
        )
    _synthetic_generator.start()
    return {"status": _synthetic_generator.status}


@app.post("/api/synthetic/pause")
def pause_synthetic_feed():
    if _synthetic_generator is None:
        return JSONResponse(
            {"error": "synthetic generator is not initialized"},
            status_code=503,
        )
    _synthetic_generator.pause()
    return {"status": _synthetic_generator.status}


@app.post("/api/synthetic/resume")
def resume_synthetic_feed():
    if _synthetic_generator is None:
        return JSONResponse(
            {"error": "synthetic generator is not initialized"},
            status_code=503,
        )
    _synthetic_generator.resume()
    return {"status": _synthetic_generator.status}


@app.post("/api/synthetic/stop")
def stop_synthetic_feed():
    if _synthetic_generator is None:
        return JSONResponse(
            {"error": "synthetic generator is not initialized"},
            status_code=503,
        )
    _synthetic_generator.stop()
    return {"status": _synthetic_generator.status}


@app.post("/api/alerts/clear")
def clear_alert_history():
    try:
        deleted = alerts_db.clear_alerts()

        # Do not let alerts that were generated just before the clear action
        # reappear in the browser after the database has been emptied.
        queued = _drain_bridge_queue()

        return {
            "status": "cleared",
            "deleted": deleted,
            "queued_discarded": queued,
        }
    except Exception:
        logger.exception("Failed to clear alert history")
        return JSONResponse(
            {"error": "failed to clear alerts"},
            status_code=500,
        )


# ------------------------------------------------------------------------
# WebSocket endpoint — real-time channel
# ------------------------------------------------------------------------
@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # IMPORTANT: Do not replay SQLite history here. A browser refresh
        # should show a clean live feed. Persistent totals are loaded through
        # /api/alerts/stats instead.
        await websocket.send_text(
            json.dumps({"type": "backlog", "data": []})
        )

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)


# ------------------------------------------------------------------------
# Static dashboard
# ------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(
        description="IDS dashboard using synthetic network-flow demonstration data."
    )
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    if not DASHBOARD_HISTORICAL_JSON.exists():
        logger.warning(
            "%s not found — run train_and_export.py first so the historical "
            "dashboard sections have data.",
            DASHBOARD_HISTORICAL_JSON,
        )

    app.state.cli_args = args
    logger.info(
        "Serving IDS dashboard at http://%s:%d",
        args.host,
        args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
