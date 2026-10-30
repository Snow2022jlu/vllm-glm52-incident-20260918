from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import collector
import install
import observer


ROOT = Path(__file__).resolve().parent


def metrics():
    return dict(running=3, waiting=3, waiting_capacity=2, waiting_deferred=1,
                kv_usage=0.25, prefill_miss_requests=2, prefill_miss_tokens=99,
                prefill_pending_requests=1, generation_tokens=10, preemptions=0)


def config(api_count=0, ranks=None):
    return dict(protocol=observer.PROTOCOL, epoch="test-epoch", run_id="test-run",
                model="GLM-5.2-FP8", role="decode", ranks=[0] if ranks is None else ranks,
                api_count=api_count, api_max_age=0.5, core_max_age=2.0,
                parent_pid=os.getpid(), parent_start=observer.process_identity(os.getpid()))


def core_sample(rank=0, **changes):
    result = dict(protocol=observer.PROTOCOL, epoch="test-epoch", run_id="test-run",
                  pid=os.getpid(), start=observer.process_identity(os.getpid()),
                  sampled_at=time.monotonic(), rank=rank, ready=True, metrics=metrics())
    result.update(changes)
    return result


def core_worker():
    stopping = False

    def stop(*_args):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    scheduler = SimpleNamespace(running=[1, 2, 3], waiting=[1, 2], skipped_waiting=[1],
                                get_kv_cache_usage=lambda: 0.25,
                                _get_prefill_cache_pressure_stats=lambda: (2, 99, 1))
    engine = SimpleNamespace(engine_index=0, scheduler=scheduler, is_running=lambda: True)
    while not stopping:
        observer.observe_core(engine)
        time.sleep(0.05)


def get(port, path):
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=1.0)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read()
        return response.status, body, time.monotonic() - started


@contextlib.contextmanager
def running_observer(configuration, core=False):
    with tempfile.TemporaryDirectory() as directory:
        observer.atomic_json(Path(directory) / "config.json", configuration)
        environment = dict(os.environ, VLLM_ISOLATED_OBSERVABILITY="1",
                           VLLM_OBSERVABILITY_DIRECTORY=directory, PYTHONDONTWRITEBYTECODE="1")
        processes = []
        try:
            processes.append(subprocess.Popen([sys.executable, str(ROOT / "observer.py"),
                                               "serve", "--directory", directory, "--port", "0"],
                                              env=environment))
            if core:
                processes.append(subprocess.Popen([sys.executable, str(__file__), "--core-worker"],
                                                  env=environment))
            deadline = time.monotonic() + 5
            while not (Path(directory) / "observer.json").exists():
                if processes[0].poll() is not None or time.monotonic() > deadline:
                    raise AssertionError("Observer did not start")
                time.sleep(0.02)
            port = observer.read_json(Path(directory) / "observer.json")["port"]
            if core:
                while not (Path(directory) / "core-0.json").exists():
                    if time.monotonic() > deadline:
                        raise AssertionError("Core publisher did not start")
                    time.sleep(0.02)
            yield Path(directory), port, processes, environment
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        observer.atomic_json(self.directory / "config.json", config())
        self.state = observer.ObservationState(self.directory)

    def write(self, **changes):
        observer.atomic_json(self.directory / "core-0.json", core_sample(**changes))

    def test_missing_rank_is_unknown_not_zero(self):
        report = self.state.report()
        self.assertFalse(report["load_fresh"])
        self.assertEqual(report["cores"], {})
        self.assertNotIn(b"vllm:num_requests_running", observer.prometheus(report))

    def test_fresh_idle_snapshot_is_healthy_without_generated_tokens(self):
        self.write(metrics=dict(metrics(), running=0, generation_tokens=0))
        report = self.state.report()
        self.assertTrue(report["ready"])
        self.assertTrue(report["load_fresh"])

    def test_old_future_wrong_epoch_wrong_process_rejected(self):
        cases = [dict(sampled_at=time.monotonic() - 20),
                 dict(sampled_at=time.monotonic() + 20),
                 dict(epoch="other-session"), dict(run_id="old-run"),
                 dict(pid=2147483647), dict(start="different-start")]
        for changes in cases:
            with self.subTest(changes=changes):
                self.write(**changes)
                self.assertFalse(self.state.report()["load_fresh"])

    def test_invalid_metrics_rejected(self):
        for changes in ({"kv_usage": 1.01}, {"running": -1}, {"waiting": 99}, {"running": True}):
            self.write(metrics=dict(metrics(), **changes))
            self.assertFalse(self.state.report()["load_fresh"])
        self.assertFalse(observer.valid_metrics(dict(metrics(), kv_usage=math.nan)))

    def test_partial_group_rejected(self):
        self.state.config["ranks"] = [0, 1]
        self.write()
        self.assertFalse(self.state.report()["load_fresh"])

    def test_stopped_core_is_not_ready_but_load_is_visible(self):
        self.write(ready=False)
        report = self.state.report()
        self.assertFalse(report["ready"])
        self.assertTrue(report["load_fresh"])

    def test_all_api_processes_must_have_fresh_pulses(self):
        self.state.config["api_count"] = 2
        self.write()
        second_process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])

        def cleanup():
            second_process.terminate()
            second_process.wait(timeout=3)

        self.addCleanup(cleanup)
        first = self.directory / f"api-{os.getpid()}.json"
        second = self.directory / f"api-{second_process.pid}.json"
        observer.atomic_json(first, core_sample())
        self.assertFalse(self.state.report()["ready"])
        second_sample = core_sample(pid=second_process.pid, start=observer.process_identity(second_process.pid))
        observer.atomic_json(second, second_sample)
        self.assertTrue(self.state.report()["ready"])
        observer.atomic_json(second, dict(second_sample, sampled_at=time.monotonic() - 10))
        self.assertFalse(self.state.report()["ready"])

    def test_duplicate_api_identity_and_unexpected_core_are_rejected(self):
        self.state.config["api_count"] = 2
        self.write()
        observer.atomic_json(self.directory / f"api-{os.getpid()}.json", core_sample())
        observer.atomic_json(self.directory / "api-duplicate.json", core_sample())
        self.assertFalse(self.state.report()["ready"])
        observer.atomic_json(self.directory / "core-1.json", core_sample(rank=1))
        self.assertFalse(self.state.report()["load_fresh"])

    def test_truncated_snapshot_fails_closed(self):
        (self.directory / "core-0.json").write_text('{"partial":')
        self.assertFalse(self.state.report()["load_fresh"])

    def test_prometheus_waiting_preserves_capacity_and_deferred(self):
        self.write()
        text = observer.prometheus(self.state.report()).decode()
        self.assertIn('vllm:num_requests_waiting{engine="0"} 3', text)
        self.assertIn('reason="capacity"} 2', text)
        self.assertIn('reason="deferred"} 1', text)

    def test_writer_queue_is_bounded(self):
        environment = dict(VLLM_OBSERVABILITY_DIRECTORY=str(self.directory))
        with mock.patch.dict(os.environ, environment), mock.patch.object(observer.threading.Thread, "start"):
            writer = observer.SnapshotWriter("core", 0)
            for index in range(1000):
                writer.submit({"sequence": index})
        self.assertEqual(writer.pending.qsize(), 1)
        self.assertEqual(writer.pending.get_nowait()["sequence"], 999)

    def test_core_counters_do_not_drain_scheduler_stats(self):
        scheduler = SimpleNamespace(running=[], waiting=[], skipped_waiting=[],
                                    get_kv_cache_usage=lambda: 0.25,
                                    _get_prefill_cache_pressure_stats=lambda: (0, 0, 0),
                                    make_stats=mock.Mock(side_effect=AssertionError("must not drain")))
        engine = SimpleNamespace(engine_index=0, scheduler=scheduler, is_running=lambda: True)
        with mock.patch.object(observer, "SnapshotWriter"):
            source = observer.CoreObserver(engine)
        output = SimpleNamespace(new_token_ids=[1, 2], events=[SimpleNamespace(type=3)])
        source.observe(engine, {0: SimpleNamespace(outputs=[output])})
        source.observe(engine, {0: SimpleNamespace(outputs=[output])})
        self.assertEqual(source.generation_tokens, 4)
        self.assertEqual(source.preemptions, 2)
        scheduler.make_stats.assert_not_called()


class ProcessIsolationTests(unittest.TestCase):
    def test_blocked_api_still_serves_fresh_load_and_fast_degraded_health(self):
        with running_observer(config(api_count=1), core=True) as (directory, port, processes, environment):
            async def exercise():
                with mock.patch.dict(os.environ, environment):
                    server = SimpleNamespace(started=True, should_exit=False)
                    task = observer.start_api_pulse(server, SimpleNamespace(errored=False))
                    try:
                        await asyncio.sleep(0.35)
                        self.assertEqual(get(port, "/health")[0], 200)
                        time.sleep(0.8)
                        status, body, latency = get(port, "/health")
                        self.assertEqual(status, 503)
                        self.assertIn("api_missing_stale_or_unready", json.loads(body)["reasons"])
                        self.assertLess(latency, 0.5)
                        status, body, latency = get(port, "/load")
                        self.assertEqual(status, 200)
                        self.assertEqual(json.loads(body)["cores"]["0"]["metrics"]["running"], 3)
                        self.assertLess(latency, 0.5)
                        self.assertEqual(get(port, "/metrics")[0], 200)
                        self.assertEqual(get(port, "/livez")[0], 200)
                        await asyncio.sleep(0.35)
                        self.assertEqual(get(port, "/health")[0], 200)
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            asyncio.run(exercise())

    def test_core_exit_does_not_leave_healthy_or_fresh_snapshot(self):
        with running_observer(config(), core=True) as (directory, port, processes, environment):
            self.assertEqual(get(port, "/load")[0], 200)
            processes[1].terminate()
            processes[1].wait(timeout=3)
            self.assertEqual(get(port, "/load")[0], 503)
            self.assertEqual(get(port, "/health")[0], 503)
            self.assertEqual(get(port, "/livez")[0], 200)

    def test_empty_observer_never_advertises_capacity(self):
        with running_observer(config()) as (_, port, _, _):
            self.assertEqual(get(port, "/load")[0], 503)
            self.assertEqual(get(port, "/health")[0], 503)
            self.assertEqual(get(port, "/livez")[0], 200)
            self.assertEqual(get(port, "/not-found")[0], 404)

    def test_slow_http_client_does_not_serialize_other_probes(self):
        with running_observer(config()) as (_, port, _, _):
            with socket.create_connection(("127.0.0.1", port), timeout=1) as slow:
                slow.sendall(b"GET /health HTTP/1.1\r\n")
                time.sleep(0.03)
                status, _, latency = get(port, "/livez")
                self.assertEqual(status, 200)
                self.assertLess(latency, 0.2)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(epoch="test-epoch", model_id="GLM-5.2-FP8", ranks=1,
                                    max_sample_age=10.0, job="test-job", worker_url="http://proxy",
                                    gateway_url="http://gateway")

    def result(self, role="decode", ready=True):
        sample = core_sample(age_seconds=0.01)
        payload = dict(protocol=observer.PROTOCOL, scope="local_pod", epoch="test-epoch",
                       model="GLM-5.2-FP8", role=role, load_fresh=True, live=True,
                       expected_ranks=[0], cores={"0": sample}, expected_api_count=1,
                       core_max_age_seconds=10, ready_valid_for_seconds=1,
                       ready=ready)
        return payload, time.monotonic(), 0.01

    def groups(self, ready=True):
        return {role: collector.group_state([self.result(role, ready)], self.args, role)
                for role in ("prefill", "decode")}

    def test_unready_health_does_not_suppress_load(self):
        body = collector.heartbeat(0, self.groups(), False)
        self.assertFalse(body["ready"])
        self.assertEqual(body["decode_running_requests"], 3)
        self.assertEqual(body["prefill_cache_miss_tokens"], 99)

    def test_api_unready_load_is_still_reported(self):
        body = collector.heartbeat(0, self.groups(False), True)
        self.assertFalse(body["ready"])
        self.assertIn("decode_kv_cache_usage_ratio", body)

    def test_expired_source_is_not_refreshed_by_a_new_heartbeat(self):
        body = collector.heartbeat(0, self.groups(), True, now=time.monotonic() + 20)
        self.assertEqual(body, {"ready": False})

    def test_wrong_model_epoch_duplicate_and_missing_rank_rejected(self):
        for change in ({"model": "GLM-5.3"}, {"epoch": "old-epoch"}, {"expected_ranks": [0, 1]}):
            payload, completed, elapsed = self.result()
            payload.update(change)
            self.assertFalse(collector.group_state([(payload, completed, elapsed)], self.args, "decode")["samples"])
        result = self.result()
        self.assertFalse(collector.group_state([result, result], self.args, "decode")["samples"])

    def test_missing_decode_does_not_erase_fresh_prefill_load(self):
        groups = self.groups()
        groups["decode"] = {"samples": {}, "ready_until": -math.inf}
        body = collector.heartbeat(0, groups, True)
        self.assertFalse(body["ready"])
        self.assertIn("prefill_cache_miss_tokens", body)
        self.assertNotIn("decode_running_requests", body)

    def test_unhealthy_worker_not_registered_but_existing_worker_gets_heartbeat(self):
        expected = collector.expected_worker(self.args, 0)
        for existing, expected_posts in (([], 0), ([expected], 1)):
            calls = []

            def request(method, url, api_key=None, body=None, **_kwargs):
                calls.append((method, url, body))
                return {"workers": existing} if method == "GET" else {}

            with mock.patch.object(collector, "collect", return_value=(self.groups(), False)), \
                    mock.patch.object(collector, "request_json", side_effect=request):
                collector.reconcile(self.args, "test-only-key")
            posts = [call for call in calls if call[0] == "POST"]
            self.assertEqual(len(posts), expected_posts)
            if posts:
                self.assertFalse(posts[0][2]["ready"])
                self.assertIn("decode_running_requests", posts[0][2])

    def test_identity_drift_never_replaces_worker(self):
        existing = collector.expected_worker(self.args, 0)
        existing["url"] = "http://different-proxy"
        with mock.patch.object(collector, "collect", return_value=(self.groups(), True)), \
                mock.patch.object(collector, "request_json", return_value={"workers": [existing]}) as request:
            with self.assertRaises(ValueError):
                collector.reconcile(self.args, "test-only-key")
        self.assertEqual(request.call_count, 1)

    def test_two_pods_must_cover_all_sixteen_ranks(self):
        self.args.ranks = 16
        results = []
        for offset in (0, 8):
            payload, completed, elapsed = self.result()
            payload["expected_ranks"] = list(range(offset, offset + 8))
            payload["expected_api_count"] = 1 if offset == 0 else 0
            payload["cores"] = {str(rank): core_sample(rank, age_seconds=0.01)
                                for rank in payload["expected_ranks"]}
            results.append((payload, completed, elapsed))
        group = collector.group_state(results, self.args, "decode")
        self.assertEqual(set(group["samples"]), set(range(16)))
        self.assertGreater(group["ready_until"], time.monotonic())
        self.assertEqual(collector.group_state(results[:1], self.args, "decode")["samples"], {})

    def test_health_failure_does_not_skip_any_observer_fetch(self):
        self.args.prefill_observer_url = ["http://prefill"]
        self.args.decode_observer_url = ["http://decode"]

        def fetch(url):
            return self.result("prefill" if "prefill" in url else "decode")

        with mock.patch.object(collector, "fetch_observer", side_effect=fetch) as fetched, \
                mock.patch.object(collector, "backend_ready", return_value=False):
            groups, ready = collector.collect(self.args)
        self.assertFalse(ready)
        self.assertEqual(fetched.call_count, 2)
        self.assertEqual(set(groups["prefill"]["samples"]), {0})
        self.assertEqual(set(groups["decode"]["samples"]), {0})


class ConfigurationTests(unittest.TestCase):
    def test_disabled_session_does_not_start_a_process(self):
        with mock.patch.dict(os.environ, {"VLLM_ISOLATED_OBSERVABILITY": "0"}), \
                mock.patch.object(observer.subprocess, "Popen") as start:
            with observer.observation_session(None):
                pass
            start.assert_not_called()

    def test_rank_contract_and_unsupported_modes_fail_closed(self):
        args = SimpleNamespace(data_parallel_size=16, data_parallel_size_local=8,
                               data_parallel_start_rank=None, data_parallel_rank=8,
                               disable_log_stats=False, data_parallel_multi_port_external_lb=False,
                               enable_elastic_ep=False, api_server_count=0)
        environment = dict(VLLM_OBSERVABILITY_EPOCH="test-epoch", VLLM_OBSERVABILITY_MODEL="GLM-5.2-FP8",
                           VLLM_OBSERVABILITY_ROLE="decode", VLLM_OBSERVABILITY_RANKS="8,9,10,11,12,13,14,15")
        with mock.patch.dict(os.environ, environment):
            self.assertEqual(observer.session_config(args)["ranks"], list(range(8, 16)))
            args.data_parallel_rank = 0
            with self.assertRaises(ValueError):
                observer.session_config(args)
            args.data_parallel_rank = 8
            args.disable_log_stats = True
            with self.assertRaises(ValueError):
                observer.session_config(args)

    def test_patch_anchors_are_exact(self):
        with self.assertRaises(ValueError):
            install.replace_once("duplicate duplicate", "duplicate", "new")
        with self.assertRaises(ValueError):
            install.replace_once("missing", "anchor", "new")


if __name__ == "__main__":
    if sys.argv[1:] == ["--core-worker"]:
        core_worker()
    else:
        unittest.main()
