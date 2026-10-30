from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

import compatible_native as native
import compatible_state as state
import isolated_observability as base


def core_worker(configured, directory):
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    scheduler = SimpleNamespace(running=[1], waiting=[1, 2], skipped_waiting=[1],
                                get_kv_cache_usage=lambda: 0.25,
                                _get_prefill_cache_pressure_stats=lambda: (2, 99, 1))
    engine = SimpleNamespace(engine_index=0, scheduler=scheduler, is_running=lambda: True)
    iteration = 0
    while not stopping.is_set():
        iteration += 1
        scheduler.running = list(range(iteration))
        output = SimpleNamespace(new_token_ids=[123], events=[])
        base.observe_core(engine, {0: SimpleNamespace(outputs=[output])})
        stopping.wait(0.05)


def api_worker(configured, directory, port):
    registry = CollectorRegistry()
    for name in state.GAUGES:
        Gauge(name, "Original gauge", ["model_name", "engine"], registry=registry).labels(configured["model"], "0").set(0)
    waiting = Gauge(state.WAITING, "Waiting reasons", ["model_name", "engine", "reason"], registry=registry)
    for reason in ("capacity", "deferred"):
        waiting.labels(configured["model"], "0", reason).set(0)
    for name in ("vllm:generation_tokens", "vllm:num_preemptions"):
        Counter(name, "Original counter", ["model_name", "engine"], registry=registry).labels(configured["model"], "0").inc(10)
    Histogram("vllm:time_to_first_token_seconds", "Original TTFT", registry=registry).observe(0.1)
    server = SimpleNamespace(started=True, should_exit=False)
    engine = SimpleNamespace(errored=False)

    async def startup(app):
        app["pulse"] = base.start_api_pulse(server, engine)
        threading.Thread(target=native.export_loop, args=(server, directory, configured, registry), daemon=True).start()

    async def stop(app):
        server.should_exit = True
        app["pulse"].cancel()
        await asyncio.gather(app["pulse"], return_exceptions=True)

    async def stall(request):
        Path(directory, "stall-entered").touch()
        time.sleep(2)
        return web.Response(text="finished")

    async def health(request):
        return web.Response()

    app = web.Application()
    app.on_startup.append(startup)
    app.on_cleanup.append(stop)
    app.router.add_post("/stall", stall)
    app.router.add_get("/health", health)
    web.run_app(app, host="127.0.0.1", port=port, print=None, access_log=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", choices=["core", "api"])
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    directory = os.environ["VLLM_OBSERVABILITY_DIRECTORY"]
    configured = json.loads(Path(directory, "config.json").read_text())
    if args.worker == "core":
        core_worker(configured, directory)
    else:
        api_worker(configured, directory, args.port)


if __name__ == "__main__":
    main()
