"""Opt-in, same-port health/metrics isolation for the sealed GLM DP image."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from multidict import CIMultiDict
from yarl import URL

if __package__:
    from . import compatible_state as state
    from . import bounded_observability as bounded
    from . import isolated_observability as base
else:
    import compatible_state as state
    import bounded_observability as bounded
    import isolated_observability as base


LOG = logging.getLogger(__name__)
CHUNK = 65536
NATIVE_AGE = 3.0
POLL_INTERVAL = 0.25
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "proxy-connection"}


class NativeMetrics:
    def __init__(self, directory, config):
        self.directory = directory
        self.config = config
        self.cached = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def start(self):
        self.thread.start()

    def poll(self):
        while not self.stopped.is_set():
            try:
                self.refresh()
            except Exception:
                self.cached = None
                LOG.exception("Native metric export unavailable")
            self.stopped.wait(0.5)

    def refresh(self):
        frames = []
        for path in Path(self.directory).glob("native-*.prom"):
            with path.open("rb") as source:
                content = source.read(state.MAX_METRICS + base.MAX_BYTES + 1)
            if len(content) > state.MAX_METRICS + base.MAX_BYTES:
                continue
            encoded, separator, payload = content.partition(b"\n")
            if not separator or len(encoded) > base.MAX_BYTES or len(payload) > state.MAX_METRICS:
                continue
            header = json.loads(encoded)
            age = time.monotonic() - header["sampled_at"]
            if (header["epoch"] != self.config["epoch"] or header["run_id"] != self.config["run_id"]
                    or path.name != f'native-{header["pid"]}.prom' or not header["start"]
                    or base.process_identity(header["pid"]) != header["start"]
                    or not state.finite_age(age, maximum=float("inf"))):
                continue
            frames.append((header["sampled_at"], payload))
        self.cached = None
        if frames:
            sampled_at, payload = max(frames, key=lambda frame: frame[0])
            self.cached = (payload, time.monotonic() + NATIVE_AGE, sampled_at)


def local_server(directory, host, port):
    observation = state.LocalState(directory)
    config = observation.config
    native = NativeMetrics(directory, config)
    if config["api_count"]:
        native.start()

    class Handler(base.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/native-metrics":
                cached = native.cached
                remaining = cached[1] - time.monotonic() if cached else 0.0
                payload = cached[0] if remaining > 0 else b"Native metrics unavailable\n"
                status = 200 if remaining > 0 else 503
                content_type = "text/plain; version=0.0.4; charset=utf-8"
                self.send_response(status)
                self.send_header("X-VLLM-Native-Valid-For", str(max(0.0, remaining)))
                self.send_header("X-VLLM-Native-Snapshot-Age", str(time.monotonic() - cached[2] if cached else -1))
            elif self.path in {"/health", "/readyz", "/load"}:
                report = observation.report()
                condition = {"/health": "health_ok", "/readyz": "api_responsive", "/load": "load_fresh"}[self.path]
                payload = json.dumps(report, allow_nan=False).encode()
                status = 200 if report[condition] else 503
                content_type = "application/json"
                self.send_response(status)
            else:
                self.send_error(404)
                return
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    with base.BoundedHTTPServer((host, port), Handler) as server:
        base.atomic_json(Path(directory) / "observer.json", {"pid": os.getpid(), "port": server.server_port})
        try:
            server.serve_forever(poll_interval=0.1)
        finally:
            native.stopped.set()


async def bounded_read(response, maximum):
    chunks, total = [], 0
    async for chunk in response.content.iter_chunked(CHUNK):
        total += len(chunk)
        if total > maximum:
            raise ValueError("Peer response exceeds the bounded payload size")
        chunks.append(chunk)
    return b"".join(chunks)


def clean_headers(headers):
    forbidden = set(HOP_HEADERS)
    for value in headers.getall("Connection", []):
        forbidden.update(token.strip().lower() for token in value.split(","))
    return CIMultiDict((name, value) for name, value in headers.items() if name.lower() not in forbidden)


class Frontend:
    def __init__(self, config):
        self.config = config
        self.group = state.GroupState(config)
        self.native = None
        self.rendered = None
        self.inflight = 0
        self.task = None
        self.probes = None
        self.inference = None

    async def start(self, app):
        self.probes = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1), trust_env=False,
                                           auto_decompress=False, connector=aiohttp.TCPConnector(limit=16))
        self.inference = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=5, sock_read=None),
            connector=aiohttp.TCPConnector(limit=self.config["max_inflight"]), trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(), auto_decompress=False,
            skip_auto_headers={"User-Agent", "Accept-Encoding", "Content-Type"},
        )
        self.probes._retry_connection = False
        self.inference._retry_connection = False
        self.task = asyncio.create_task(self.poll())

    async def stop(self, app):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        await self.probes.close()
        await self.inference.close()

    async def peer(self, index, peer):
        started = time.monotonic()
        try:
            async with self.probes.get(peer["url"].rstrip("/") + "/load", allow_redirects=False) as response:
                if response.status not in {200, 503}:
                    raise ValueError("Unexpected observer response")
                report = json.loads(await bounded_read(response, state.MAX_REPORT))
                self.group.accept(index, report, started)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            pass
        except (ValueError, TypeError):
            self.group.samples.pop(index, None)

    async def native_metrics(self):
        started = time.monotonic()
        try:
            async with self.probes.get(self.config["local_origin"] + "/native-metrics", allow_redirects=False) as response:
                if response.status != 200:
                    self.native = None
                    return
                lifetime = float(response.headers["X-VLLM-Native-Valid-For"])
                if not state.finite_age(lifetime, NATIVE_AGE):
                    raise ValueError("Invalid native metric deadline")
                payload = await bounded_read(response, state.MAX_METRICS)
                age = float(response.headers["X-VLLM-Native-Snapshot-Age"])
                if not state.finite_age(age, maximum=float("inf")):
                    raise ValueError("Invalid native source age")
                self.native = (payload, started + lifetime, started - age)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            pass
        except (ValueError, KeyError):
            self.native = None

    async def poll(self):
        while True:
            await asyncio.gather(self.native_metrics(), *(self.peer(index, peer)
                                for index, peer in enumerate(self.config["peers"])))
            view = self.group.view()
            try:
                if self.native is None or self.native[1] <= time.monotonic():
                    raise ValueError("Native metric cache expired")
                native_age = time.monotonic() - self.native[2]
                payload = await asyncio.to_thread(state.merge_metrics, self.native[0], view,
                                                  self.config["served_model_name"], native_age)
                self.rendered = (payload, min(view["deadline"], self.native[1]), self.native[2])
            except Exception:
                self.rendered = None
            await asyncio.sleep(POLL_INTERVAL)

    async def health(self, request):
        view = self.group.view()
        return web.Response(status=200 if view["health_ok"] else 503, body=b"", headers={
            "X-VLLM-Health-State": view["health_state"],
            "X-VLLM-API-Responsive": str(int(view["api_responsive"])),
            "Cache-Control": "no-store",
        })

    async def metrics(self, request):
        view = self.group.view()
        cached = self.rendered
        now = time.monotonic()
        if not view["load_fresh"] or cached is None or min(cached[1], view["deadline"]) <= now:
            return web.Response(status=503, text="Metrics unavailable; no synthetic idle ranks\n")
        if request.query:
            return web.Response(status=400, text="This sealed collector contract uses unfiltered /metrics\n")
        return web.Response(body=cached[0], headers={
            "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
            "Cache-Control": "no-store", "X-VLLM-Load-Valid-For": str(min(cached[1], view["deadline"]) - now),
            "X-VLLM-Health-State": view["health_state"], "X-VLLM-API-Responsive": str(int(view["api_responsive"])),
            "X-VLLM-API-Valid-For": str(max(0.0, view["api_deadline"] - now)),
            "X-VLLM-Health-Valid-For": str(max(0.0, min(view["health_deadline"], cached[1], view["deadline"]) - now)),
            "X-VLLM-Native-Snapshot-Age": str(now - cached[2]),
            "X-VLLM-Model": self.config["model"], "X-VLLM-Role": self.config["role"],
            "X-VLLM-Epoch": self.config["epoch"],
        })

    async def proxy(self, request):
        if request.headers.get("Upgrade"):
            return web.Response(status=501, text="WebSocket upgrade is outside this GLM HTTP candidate\n")
        if self.inflight >= self.config["max_inflight"]:
            return web.Response(status=503, text="Inference connection budget exhausted\n")
        self.inflight += 1
        outgoing = None
        try:
            target = URL(self.config["backend"] + request.raw_path, encoded=True)
            body = request.content.iter_chunked(CHUNK) if request.can_read_body else None
            async with self.inference.request(request.method, target, headers=clean_headers(request.headers),
                                              data=body, allow_redirects=False) as response:
                outgoing = web.StreamResponse(status=response.status, reason=response.reason,
                                              headers=clean_headers(response.headers))
                await outgoing.prepare(request)
                async for chunk in response.content.iter_any():
                    await outgoing.write(chunk)
                await outgoing.write_eof()
                return outgoing
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
            if outgoing is not None and outgoing.prepared:
                if request.transport is not None:
                    request.transport.close()
                raise
            return web.Response(status=504 if isinstance(error, asyncio.TimeoutError) else 502,
                                text="Inference upstream unavailable; request was not retried\n")
        finally:
            self.inflight -= 1

    def application(self):
        app = web.Application(client_max_size=0)
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.stop)
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/metrics/", self.metrics)
        app.router.add_route("*", "/{tail:.*}", self.proxy)
        return app


async def run_frontend(config, directory):
    frontend = Frontend(config)
    runner = web.AppRunner(frontend.application(), handler_cancellation=True, auto_decompress=False,
                           shutdown_timeout=3, access_log=None)
    await runner.setup()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)
    try:
        site = web.TCPSite(runner, config["public_host"], config["public_port"])
        await site.start()
        base.atomic_json(Path(directory) / "frontend.json", {"pid": os.getpid(), "port": config["public_port"]})
        await stopped.wait()
    finally:
        await runner.cleanup()


def session_config(args, directory):
    config = base.session_config(args)
    if (getattr(args, "uds", None) or getattr(args, "ssl_keyfile", None)
            or getattr(args, "ssl_certfile", None) or getattr(args, "root_path", None)):
        raise ValueError("TLS, UDS and nonempty root_path require a separately validated compatibility entrypoint")
    peers = json.loads(os.environ.get("VLLM_COMPAT_OBSERVER_PEERS", "null"))
    state.validate_peers(peers, args.data_parallel_size, config["ranks"], config["api_count"])
    observer_port = int(os.environ.get("VLLM_OBSERVABILITY_PORT", "18081"))
    backend_port = int(os.environ.get("VLLM_COMPAT_BACKEND_PORT", "18000"))
    public_port = args.port
    if any(type(port) is not int or not 1 <= port <= 65535 for port in (observer_port, backend_port, public_port)):
        raise ValueError("Invalid port")
    if len({observer_port, backend_port, public_port}) != 3:
        raise ValueError("Public, observer and private inference ports must differ")
    served = getattr(args, "served_model_name", None) or args.model
    if isinstance(served, list):
        served = served[0]
    if not isinstance(served, str) or not served:
        raise ValueError("A stable served model label is required")
    config.update(
        peers=peers, served_model_name=served, public_host=args.host or "0.0.0.0", public_port=public_port,
        observer_host=os.environ.get("VLLM_OBSERVABILITY_HOST", "127.0.0.1"), observer_port=observer_port,
        local_origin=f"http://127.0.0.1:{observer_port}", backend=f"http://127.0.0.1:{backend_port}",
        backend_port=backend_port, max_inflight=1024,
    )
    if config["observer_host"] not in {"0.0.0.0", "127.0.0.1"}:
        raise ValueError("Bind the observer to 0.0.0.0 (private fabric) or loopback")
    if len(peers) > 1 and config["observer_host"] != "0.0.0.0":
        raise ValueError("Cross-Pod aggregation requires the private observer listener")
    return config


def wait_process(process, marker):
    deadline = time.monotonic() + 10
    while not marker.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("Compatibility process failed to bind before engine startup")
        time.sleep(0.05)


@contextlib.contextmanager
def observation_session(args):
    with bounded.observation_session(args):
        yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["local", "frontend"])
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    config = base.read_json(Path(args.directory) / "config.json")
    if args.command == "local":
        local_server(args.directory, config["observer_host"], config["observer_port"])
    else:
        asyncio.run(run_frontend(config, args.directory))


if __name__ == "__main__":
    main()
