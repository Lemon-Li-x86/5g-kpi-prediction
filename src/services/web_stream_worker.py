"""One-shot local WebSocket console for a W4 background job.

Usage is internal: spawned by src.ui.web_workflow. Existing jobs are unchanged.
Streaming is event driven; the local log is written every 2 seconds or 64 KiB.
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import os
import sys
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve

from src.services.model_registry import ROOT

JOBS = ROOT / "resources" / "jobs"
LOG = JOBS / "current.log"
STATE = JOBS / "current.json"
HISTORY_CHARS = 200_000
FLUSH_BYTES = 64 * 1024
FLUSH_SECONDS = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StreamHub:
    def __init__(self, token: str):
        self.token = token
        self.history: deque[str] = deque()
        self.history_size = 0
        self.clients: set[asyncio.Queue] = set()
        self.final_status: dict | None = None

    def publish(self, text: str) -> None:
        if not text:
            return
        self.history.append(text)
        self.history_size += len(text)
        while self.history_size > HISTORY_CHARS and self.history:
            old = self.history.popleft()
            self.history_size -= len(old)
        self._push({"type": "text", "data": text})

    def finish(self, status: str, code: int) -> None:
        self.final_status = {"type": "status", "status": status, "exit_code": code}
        self._push(self.final_status)

    def _push(self, payload: dict) -> None:
        for queue in tuple(self.clients):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(payload)

    async def handle(self, socket) -> None:
        # The random token protects the otherwise-local WebSocket endpoint from
        # unrelated browser pages reading private log content.
        url = urlsplit(socket.request.path)
        if parse_qs(url.query).get("token") != [self.token]:
            await socket.close(code=1008, reason="invalid token")
            return

        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        snapshot = "".join(self.history)
        self.clients.add(queue)
        try:
            await socket.send(json.dumps({"type": "reset", "data": snapshot}, ensure_ascii=False))
            if self.final_status is not None:
                await socket.send(json.dumps(self.final_status))
            while True:
                payload = await queue.get()
                await socket.send(json.dumps(payload, ensure_ascii=False))
        except Exception:
            # Closing the browser must never interrupt training.
            pass
        finally:
            self.clients.discard(queue)


async def _execute(task: str, model: str, hub: StreamHub) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u", "-m", "src.services.web_jobs", task]
    if model:
        cmd.append(model)

    JOBS.mkdir(parents=True, exist_ok=True)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = bytearray()
    finished = False

    # Only this supervisor writes current.log. Workers send bytes over a pipe.
    with LOG.open("wb") as logfile:
        def flush_pending() -> None:
            if pending:
                logfile.write(pending)
                logfile.flush()
                pending.clear()

        async def periodic_flush() -> None:
            while not finished:
                await asyncio.sleep(FLUSH_SECONDS)
                flush_pending()

        flush_task = asyncio.create_task(periodic_flush())
        return_code = 1
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(ROOT),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
            )
            assert process.stdout is not None
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                pending.extend(chunk)
                hub.publish(decoder.decode(chunk))
                if len(pending) >= FLUSH_BYTES:
                    flush_pending()
            hub.publish(decoder.decode(b"", final=True))
            return_code = await process.wait()
        except Exception:
            diagnostic = traceback.format_exc()
            pending.extend(diagnostic.encode("utf-8", errors="replace"))
            hub.publish(diagnostic)
            try:
                STATE.write_text(
                    json.dumps({"task": task, "model": model, "status": "failed", "error": diagnostic, "finished": _now()}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass
        finally:
            finished = True
            flush_task.cancel()
            try:
                await flush_task
            except asyncio.CancelledError:
                pass
            flush_pending()

    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        final_status = state.get("status", "failed" if return_code else "completed")
    except (OSError, ValueError):
        final_status = "failed" if return_code else "completed"
    if return_code != 0:
        final_status = "failed"
    hub.finish(final_status, return_code)


async def _main(port: int, token: str, task: str, model: str) -> None:
    hub = StreamHub(token)
    # The server runs for the whole job and a short grace period afterwards.
    async with serve(hub.handle, "127.0.0.1", port, max_size=1024 * 1024):
        await _execute(task, model, hub)
        await asyncio.sleep(90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["download", "preprocess", "features", "train_ml", "train_dl", "train_anomaly", "diagnostics"])
    parser.add_argument("model", nargs="?", default="")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    try:
        asyncio.run(_main(args.port, args.token, args.task, args.model))
    except Exception:
        error = traceback.format_exc()
        JOBS.mkdir(parents=True, exist_ok=True)
        with LOG.open("ab") as f:
            f.write(error.encode("utf-8", errors="replace"))
        STATE.write_text(json.dumps({"status": "failed", "task": args.task, "model": args.model, "error": error, "finished": _now()}, ensure_ascii=False), encoding="utf-8")
        raise
