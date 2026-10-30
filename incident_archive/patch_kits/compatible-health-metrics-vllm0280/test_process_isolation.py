from __future__ import annotations

import concurrent.futures
import contextlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vllm.entrypoints.serve.instrumentator import compatible_runtime as runtime
from vllm.entrypoints.serve.instrumentator import isolated_observability as base

from test_compatibility import EPOCH, MODEL


ROOT = Path(__file__).resolve().parent
SITE = Path(importlib.metadata.distribution("vllm").locate_file(""))
MODULES = SITE / "vllm/entrypoints/serve/instrumentator"


def free_ports(count):
    sockets = []
    try:
        for _index in range(count):
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            sockets.append(listener)
        return [listener.getsockname()[1] for listener in sockets]
    finally:
        for listener in sockets:
            listener.close()


def fetch(port, path, timeout=1, data=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers, response.read()


def wait_until(function, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if function():
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for CPU test state")


class ProcessIsolationTests(unittest.TestCase):
    def test_independent_executables_do_not_import_torch_or_native_libraries(self):
        script = (f"import sys; sys.path.insert(0, {str(MODULES)!r}); import compatible_runtime; "
                  "assert not any(name in sys.modules for name in ('torch', 'deep_ep', 'deep_gemm', 'vllm')); print('lightweight')")
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lightweight", result.stdout)

    def test_real_api_loop_stall_does_not_cut_off_live_core_health_or_load(self):
        public_port, observer_port, backend_port = free_ports(3)
        args = SimpleNamespace(host="127.0.0.1", port=public_port, model=MODEL, served_model_name=[MODEL],
                               api_server_count=1, disable_log_stats=False, data_parallel_size=1,
                               data_parallel_size_local=1, data_parallel_start_rank=None,
                               data_parallel_rank=None, data_parallel_multi_port_external_lb=False)
        peers = [{"url": f"http://127.0.0.1:{observer_port}", "ranks": [0], "api_count": 1}]
        environment = {"VLLM_COMPAT_OBSERVABILITY": "1", "VLLM_ISOLATED_OBSERVABILITY": "1",
                       "VLLM_OBSERVABILITY_EPOCH": EPOCH, "VLLM_OBSERVABILITY_MODEL": MODEL,
                       "VLLM_OBSERVABILITY_ROLE": "decode", "VLLM_OBSERVABILITY_RANKS": "0",
                       "VLLM_OBSERVABILITY_HOST": "127.0.0.1", "VLLM_OBSERVABILITY_PORT": str(observer_port),
                       "VLLM_OBSERVABILITY_API_MAX_AGE": "0.5", "VLLM_OBSERVABILITY_CORE_MAX_AGE": "2",
                       "VLLM_COMPAT_BACKEND_PORT": str(backend_port), "VLLM_COMPAT_OBSERVER_PEERS": json.dumps(peers)}
        processes = []
        logs = []
        with mock.patch.dict(os.environ, environment):
            with runtime.observation_session(args):
                self.assertEqual(args.port, backend_port)
                directory = Path(os.environ["VLLM_OBSERVABILITY_DIRECTORY"])
                children = [base.read_json(directory / name)["pid"] for name in ("observer.json", "frontend.json")]
                try:
                    for worker in ("core", "api"):
                        code = (f"import sys,runpy;sys.path.insert(0,{str(MODULES)!r});"
                                f"sys.argv={[str(ROOT / 'fault_workers.py'), worker, '--port', str(backend_port)]!r};"
                                f"runpy.run_path({str(ROOT / 'fault_workers.py')!r},run_name='__main__')")
                        log = (directory / (worker + ".log")).open("wb")
                        logs.append(log)
                        processes.append(subprocess.Popen([sys.executable, "-B", "-c", code], stdout=log, stderr=log))
                    wait_until(lambda: fetch(public_port, "/health")[1].get("X-VLLM-Health-State") == "healthy")
                    wait_until(lambda: fetch(public_port, "/metrics")[0] == 200)
                    first = json.loads(fetch(observer_port, "/load")[2])["cores"]["0"]["metrics"]["running"]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        blocked = executor.submit(fetch, public_port, "/stall", 5, b"")
                        wait_until(lambda: (directory / "stall-entered").exists())
                        wait_until(lambda: fetch(public_port, "/health")[1].get("X-VLLM-Health-State") == "degraded")
                        started = time.monotonic()
                        status, headers, payload = fetch(public_port, "/health")
                        self.assertLess(time.monotonic() - started, 0.5)
                        self.assertEqual((status, payload), (200, b""))
                        self.assertEqual(headers["X-VLLM-API-Responsive"], "0")
                        metric_status, _, body = fetch(public_port, "/metrics")
                        self.assertEqual(metric_status, 200)
                        self.assertIn(b"vllm:time_to_first_token_seconds_bucket", body)
                        current = json.loads(fetch(observer_port, "/load")[2])["cores"]["0"]["metrics"]["running"]
                        self.assertGreater(current, first)
                        with self.assertRaises((TimeoutError, urllib.error.URLError)):
                            fetch(backend_port, "/health", timeout=0.15)
                        self.assertEqual(blocked.result(timeout=5)[0], 200)
                    wait_until(lambda: fetch(public_port, "/health")[1].get("X-VLLM-Health-State") == "healthy")
                    processes[0].terminate()
                    processes[0].wait(timeout=3)
                    wait_until(lambda: fetch(public_port, "/health")[0] == 503)
                    self.assertEqual(fetch(public_port, "/metrics")[0], 503)
                except Exception:
                    for log in logs:
                        log.flush()
                        print(Path(log.name).read_text(), file=sys.stderr)
                    raise
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                        process.wait(timeout=5)
                    for log in logs:
                        log.close()
            self.assertEqual((args.host, args.port), ("127.0.0.1", public_port))
            self.assertNotIn("VLLM_OBSERVABILITY_DIRECTORY", os.environ)
            self.assertTrue(all(base.process_identity(pid) is None for pid in children))


if __name__ == "__main__":
    unittest.main()
