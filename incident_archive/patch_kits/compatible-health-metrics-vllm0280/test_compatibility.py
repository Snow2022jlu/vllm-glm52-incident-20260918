from __future__ import annotations

import asyncio
import contextlib
import copy
import gzip
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import aiohttp
from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.parser import text_string_to_metric_families
from yarl import URL

import static_dp_registrar as registrar
from vllm.entrypoints.serve.instrumentator import compatible_native as native
from vllm.entrypoints.serve.instrumentator import compatible_runtime as runtime
from vllm.entrypoints.serve.instrumentator import compatible_state as state
from vllm.entrypoints.serve.instrumentator import isolated_observability as base


MODEL = "GLM-5.2-FP8"
EPOCH = "cpu-compat-test"


def metrics():
    return dict(running=3, waiting=3, waiting_capacity=2, waiting_deferred=1, kv_usage=0.25,
                prefill_miss_requests=2, prefill_miss_tokens=99, prefill_pending_requests=1,
                generation_tokens=10, preemptions=0)


def config(ranks=None, api_count=1):
    return dict(protocol=base.PROTOCOL, epoch=EPOCH, run_id=uuid.uuid4().hex,
                model=MODEL, role="decode", ranks=list(range(8)) if ranks is None else ranks,
                api_count=api_count, api_max_age=0.5, core_max_age=2.0,
                parent_pid=os.getpid(), parent_start=base.process_identity(os.getpid()))


def sample(configured, **changes):
    value = dict(protocol=base.PROTOCOL, epoch=configured["epoch"], run_id=configured["run_id"],
                 pid=os.getpid(), start=base.process_identity(os.getpid()),
                 sampled_at=time.monotonic(), ready=True)
    value.update(changes)
    return value


def populate(directory, configured, api_age=0.0):
    base.atomic_json(Path(directory) / "config.json", configured)
    for rank in configured["ranks"]:
        base.atomic_json(Path(directory) / f"core-{rank}.json", sample(configured, rank=rank, metrics=metrics()))
    if configured["api_count"]:
        base.atomic_json(Path(directory) / f"api-{os.getpid()}.json",
                         sample(configured, sampled_at=time.monotonic() - api_age))


def report(ranks, api_count, degraded=False, run_id=None):
    configured = config(ranks, api_count)
    if run_id is not None:
        configured["run_id"] = run_id
    return dict(protocol=base.PROTOCOL, compatibility_protocol=state.PROTOCOL,
                scope="local_pod", epoch=EPOCH, model=MODEL, role="decode", run_id=configured["run_id"],
                expected_ranks=ranks, expected_api_count=api_count, live=True, load_fresh=True,
                api_alive=True, health_ok=True, api_responsive=not degraded,
                health_state="degraded" if degraded else "healthy", core_max_age_seconds=10.0,
                ready_valid_for_seconds=0.0 if degraded else 2.0,
                load_valid_for_seconds=9.0,
                cores={str(rank): sample(configured, rank=rank, metrics=metrics(), age_seconds=1.0)
                       for rank in ranks})


def group_config():
    return dict(epoch=EPOCH, model=MODEL, role="decode", served_model_name=MODEL,
                peers=[dict(url="http://127.0.0.1:18181", ranks=list(range(8)), api_count=1),
                       dict(url="http://127.0.0.1:18182", ranks=list(range(8, 16)), api_count=0)],
                local_origin="http://127.0.0.1:18181", max_inflight=8, backend="http://127.0.0.1:18000")


def populate_group(group, degraded=False):
    started = time.monotonic()
    for index, peer in enumerate(group.config["peers"]):
        group.accept(index, report(peer["ranks"], peer["api_count"], degraded and index == 0), started)


def registry(world=16):
    result = CollectorRegistry()
    for name in state.GAUGES:
        gauge = Gauge(name, "Original gauge documentation", ["model_name", "engine"], registry=result)
        for rank in range(world):
            gauge.labels(MODEL, str(rank)).set(0)
    waiting = Gauge(state.WAITING, "Original reason help", ["model_name", "engine", "reason"], registry=result)
    for rank in range(world):
        for reason in ("capacity", "deferred"):
            waiting.labels(MODEL, str(rank), reason).set(0)
    for name, amount in (("vllm:generation_tokens", 123), ("vllm:num_preemptions", 2),
                         ("vllm:spec_decode_num_drafts", 7)):
        counter = Counter(name, "Original counter documentation", ["model_name", "engine"], registry=result)
        for rank in range(world):
            counter.labels(MODEL, str(rank)).inc(amount)
    latency = Histogram("vllm:time_to_first_token_seconds", "Original latency help",
                        ["model_name", "engine"], registry=result, buckets=[0.1, 0.5, 1.0])
    for rank in range(world):
        latency.labels(MODEL, str(rank)).observe(0.25)
    Gauge("process_original_metric", "API process collector retained", registry=result).set(99)
    return result


def parsed(payload):
    return {(sample.name, tuple(sorted(sample.labels.items()))): sample.value
            for family in text_string_to_metric_families(payload.decode()) for sample in family.samples}


class HealthPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.configured = config()
        populate(self.directory, self.configured)
        self.local = state.LocalState(self.directory)

    def test_healthy(self):
        self.assertTrue(self.local.report()["health_ok"])
        self.assertTrue(self.local.report()["api_responsive"])

    def test_api_stale_but_engine_fresh_is_degraded_not_dead(self):
        populate(self.directory, self.configured, api_age=60)
        result = self.local.report()
        self.assertTrue(result["health_ok"])
        self.assertEqual(result["health_state"], "degraded")
        self.assertFalse(result["api_responsive"])
        self.assertTrue(result["load_fresh"])

    def test_idle_engines_do_not_need_generated_tokens(self):
        populate(self.directory, self.configured, api_age=60)
        for rank in self.configured["ranks"]:
            path = self.directory / f"core-{rank}.json"
            content = base.read_json(path)
            content["metrics"].update(running=0, waiting=0, waiting_capacity=0, waiting_deferred=0)
            base.atomic_json(path, content)
        self.assertTrue(self.local.report()["health_ok"])

    def test_explicit_api_error_is_not_hidden_by_engine_progress(self):
        base.atomic_json(self.directory / f"api-{os.getpid()}.json", sample(self.configured, ready=False))
        self.assertFalse(self.local.report()["health_ok"])
        self.assertTrue(self.local.report()["load_fresh"])

    def test_dead_api_identity_is_not_soft_degraded(self):
        base.atomic_json(self.directory / f"api-{os.getpid()}.json", sample(self.configured, start="wrong"))
        self.assertFalse(self.local.report()["health_ok"])

    def test_no_initial_api_pulse_is_not_ready(self):
        (self.directory / f"api-{os.getpid()}.json").unlink()
        self.assertFalse(self.local.report()["health_ok"])

    def test_missing_core_is_unavailable_not_idle(self):
        (self.directory / "core-7.json").unlink()
        result = self.local.report()
        self.assertFalse(result["health_ok"])
        self.assertFalse(result["load_fresh"])

    def test_stale_core_is_not_reported_healthy(self):
        base.atomic_json(self.directory / "core-0.json", sample(self.configured, rank=0, metrics=metrics(),
                                                             sampled_at=time.monotonic() - 5))
        self.assertFalse(self.local.report()["health_ok"])

    def test_headless_member_does_not_require_api(self):
        configured = config(list(range(8, 16)), 0)
        with tempfile.TemporaryDirectory() as directory:
            populate(directory, configured)
            self.assertTrue(state.LocalState(directory).report()["health_ok"])


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.group = state.GroupState(group_config())

    def test_complete_sixteen_rank_union(self):
        populate_group(self.group)
        self.assertEqual(set(self.group.view()["cores"]), {str(rank) for rank in range(16)})
        self.assertTrue(self.group.view()["health_ok"])

    def test_complete_degraded_union_stays_healthy(self):
        populate_group(self.group, degraded=True)
        self.assertTrue(self.group.view()["health_ok"])
        self.assertFalse(self.group.view()["api_responsive"])

    def test_one_pod_is_not_complete_engine_health(self):
        self.group.accept(0, report(list(range(8)), 1), time.monotonic())
        self.assertFalse(self.group.view()["health_ok"])
        self.assertEqual(self.group.view()["cores"], {})

    def test_expired_source_is_not_refreshed_by_http_polling(self):
        populate_group(self.group)
        self.assertFalse(self.group.view(now=time.monotonic() + 20)["load_fresh"])

    def test_api_responsiveness_expires_before_core_activity(self):
        populate_group(self.group)
        result = self.group.view(now=time.monotonic() + 3)
        self.assertTrue(result["load_fresh"])
        self.assertTrue(result["health_ok"])
        self.assertEqual(result["health_state"], "degraded")
        self.assertFalse(result["api_responsive"])

    def test_restarted_member_is_latched_out_of_old_epoch(self):
        first = report(list(range(8)), 1)
        self.group.accept(0, first, time.monotonic())
        with self.assertRaises(ValueError):
            self.group.accept(0, report(list(range(8)), 1), time.monotonic())
        with self.assertRaises(ValueError):
            self.group.accept(0, first, time.monotonic())

    def test_model_role_epoch_and_rank_mismatches_rejected(self):
        for changes in ({"model": "GLM-5.3"}, {"role": "prefill"}, {"epoch": "old"},
                        {"expected_ranks": [0]}, {"api_alive": None}, {"health_ok": "true"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                value = report(list(range(8)), 1)
                value.update(changes)
                self.group.accept(0, value, time.monotonic())

    def test_bad_health_cannot_override_dead_core(self):
        value = report(list(range(8)), 1)
        value["cores"]["0"]["ready"] = False
        with self.assertRaises(ValueError):
            self.group.accept(0, value, time.monotonic())

    def test_roster_rejects_missing_duplicate_and_extra_ranks(self):
        configured = group_config()
        state.validate_peers(configured["peers"], 16, list(range(8)), 1)
        for ranks in ([8, 9], list(range(7, 16)), list(range(8, 17))):
            peers = copy.deepcopy(configured["peers"])
            peers[1]["ranks"] = ranks
            with self.assertRaises(ValueError):
                state.validate_peers(peers, 16, list(range(8)), 1)


class MetricCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.group = state.GroupState(group_config())
        populate_group(self.group, degraded=True)
        self.original = generate_latest(registry())

    def test_all_original_families_and_labels_preserved(self):
        before = parsed(self.original)
        after = parsed(state.merge_metrics(self.original, self.group.view(), MODEL))
        self.assertTrue(before.keys() <= after.keys())
        for key, value in before.items():
            if key[0] not in state.GAUGES and key[0] != state.WAITING:
                self.assertEqual(after[key], value, key)

    def test_live_gauges_override_stale_api_values_for_all_ranks(self):
        payload = state.merge_metrics(self.original, self.group.view(), MODEL)
        self.assertEqual(registrar.rank_metric(payload.decode(), "vllm:num_requests_running", 16, True),
                         {rank: 3 for rank in range(16)})
        self.assertEqual(registrar.rank_metric(payload.decode(), state.WAITING, 16, True, "capacity"),
                         {rank: 2 for rank in range(16)})

    def test_counter_semantics_are_not_replaced_with_engine_output_counts(self):
        payload = state.merge_metrics(self.original, self.group.view(), MODEL)
        self.assertEqual(registrar.rank_metric(payload.decode(), "vllm:generation_tokens_total", 16, True)[0], 123)

    def test_native_source_age_is_explicit(self):
        payload = state.merge_metrics(self.original, self.group.view(), MODEL, native_age=12)
        self.assertEqual(parsed(payload)[("vllm_compat_native_snapshot_age_seconds", ())], 12)

    def test_missing_world_and_native_labels_rejected(self):
        with self.assertRaises(ValueError):
            state.merge_metrics(generate_latest(registry(8)), self.group.view(), MODEL)
        with self.assertRaises(ValueError):
            state.merge_metrics(self.original.replace(MODEL.encode(), b"wrong-model"), self.group.view(), MODEL)

    def test_native_single_process_registry_snapshot_preserves_custom_collectors(self):
        with tempfile.TemporaryDirectory() as directory:
            configured = config()
            native.write_native(directory, configured, registry())
            reader = runtime.NativeMetrics(directory, configured)
            reader.refresh()
            self.assertEqual(parsed(reader.cached[0])[("process_original_metric", ())], 99)

    def test_native_old_epoch_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            configured = config()
            native.write_native(directory, configured, registry())
            reader = runtime.NativeMetrics(directory, dict(configured, epoch="new"))
            reader.refresh()
            self.assertIsNone(reader.cached)


async def start_server(app, **kwargs):
    runner = web.AppRunner(app, handler_cancellation=True, auto_decompress=False, access_log=None, **kwargs)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


class HTTPCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = 0
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.received_first = asyncio.Event()

        async def upstream(request):
            self.calls += 1
            if request.path == "/stream":
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                await response.write(b"data: first\n\n")
                try:
                    await self.release.wait()
                    await response.write(b"data: second\n\n")
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
                return response
            if request.path == "/fail":
                request.transport.close()
                return web.Response()
            if request.path == "/redirect":
                return web.Response(status=307, headers={"Location": "/v1/chat/completions"})
            if request.path == "/load":
                return web.json_response({"server_load": 19})
            if request.path == "/echo":
                return web.Response(body=await request.read(), headers={"Content-Encoding": request.headers.get("Content-Encoding", "identity")})
            if request.path == "/upload":
                first = await request.content.readexactly(4)
                self.received_first.set()
                return web.Response(body=first + await request.content.read())
            return web.json_response({"path": request.raw_path, "body": (await request.read()).decode(),
                                      "authorization": request.headers.get("Authorization"),
                                      "rank": request.headers.get("x-data-parallel-rank"),
                                      "host": request.headers.get("Host")}, headers={"X-Request-Id": "test-id"})

        backend = web.Application()
        backend.router.add_route("*", "/{tail:.*}", upstream)
        self.backend_runner, backend_url = await start_server(backend)
        configured = dict(group_config(), backend=backend_url)
        self.frontend = runtime.Frontend(configured)

        async def no_poll():
            await asyncio.Event().wait()

        self.frontend.poll = no_poll
        populate_group(self.frontend.group, degraded=True)
        payload = state.merge_metrics(generate_latest(registry()), self.frontend.group.view(), MODEL)
        self.frontend.rendered = (payload, time.monotonic() + 5, time.monotonic())
        self.runner, self.url = await start_server(self.frontend.application())
        self.client = aiohttp.ClientSession(auto_decompress=False)

    async def asyncTearDown(self):
        self.release.set()
        await self.client.close()
        await self.runner.cleanup()
        await self.backend_runner.cleanup()

    async def test_same_port_health_degraded_is_200_empty_body(self):
        async with self.client.get(self.url + "/health") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.read(), b"")
            self.assertEqual(response.headers["X-VLLM-Health-State"], "degraded")
        self.assertEqual(self.calls, 0)

    async def test_same_metrics_path_still_serves_full_sixteen_rank_prometheus(self):
        async with self.client.get(self.url + "/metrics") as response:
            payload = await response.read()
            self.assertEqual(response.status, 200)
            self.assertIn(b"vllm:time_to_first_token_seconds_bucket", payload)
            self.assertIn(b'engine="15"', payload)
            self.assertEqual(response.headers["X-VLLM-Model"], MODEL)

    async def test_legacy_url_registrar_parses_real_compatible_http_response(self):
        args = SimpleNamespace(model_id=MODEL, ranks=16)
        result = await asyncio.to_thread(registrar.fetch_metrics, self.url + "/metrics", args, "decode")
        self.assertEqual(result["fields"]["decode_running_requests"][15], 3)
        self.assertEqual(result["fields"]["decode_waiting_requests"][15], 2)
        self.assertEqual(result["fields"]["decode_generation_tokens_total"][15], 123)
        self.assertFalse(result["api_responsive"])

    async def test_missing_rank_fails_health_not_fake_healthy(self):
        self.frontend.group.samples.pop(1)
        async with self.client.get(self.url + "/health") as response:
            self.assertEqual(response.status, 503)
        async with self.client.get(self.url + "/metrics") as response:
            self.assertEqual(response.status, 503)

    async def test_preserves_auth_rank_body_raw_query_and_response_headers(self):
        async with self.client.post(URL(self.url + "/v1/chat/completions?path=%2F&x=1&x=2", encoded=True), data=b'{"stream":true}',
                                    headers={"Authorization": "Bearer synthetic", "x-data-parallel-rank": "13",
                                             "Host": "original-host:8000"}) as response:
            body = await response.json()
            self.assertEqual(body["authorization"], "Bearer synthetic")
            self.assertEqual(body["rank"], "13")
            self.assertEqual(body["body"], '{"stream":true}')
            self.assertEqual(body["host"], "original-host:8000")
            self.assertEqual(body["path"], "/v1/chat/completions?path=%2F&x=1&x=2")
            self.assertEqual(response.headers["X-Request-Id"], "test-id")

    async def test_stream_delivers_first_chunk_without_waiting_for_completion(self):
        async with self.client.post(self.url + "/stream") as response:
            first = await asyncio.wait_for(response.content.readuntil(b"\n\n"), 1)
            self.assertEqual(first, b"data: first\n\n")
            self.release.set()
            self.assertIn(b"second", await response.read())

    async def test_client_disconnect_cancels_upstream_stream(self):
        response = await self.client.post(self.url + "/stream")
        await response.content.readuntil(b"\n\n")
        response.close()
        await asyncio.wait_for(self.cancelled.wait(), 2)
        self.assertEqual(self.calls, 1)

    async def test_upstream_disconnect_is_not_retried(self):
        async with self.client.post(self.url + "/fail", data=b"request") as response:
            self.assertEqual(response.status, 502)
        self.assertEqual(self.calls, 1)

    async def test_get_upstream_disconnect_is_not_retried_either(self):
        async with self.client.get(self.url + "/fail") as response:
            self.assertEqual(response.status, 502)
        self.assertEqual(self.calls, 1)

    async def test_request_upload_is_streamed_not_buffered(self):
        async def pieces():
            yield b"part"
            await asyncio.wait_for(self.received_first.wait(), 1)
            yield b"-two"

        async with self.client.post(self.url + "/upload", data=pieces()) as response:
            self.assertEqual(await response.read(), b"part-two")

    async def test_redirect_is_not_followed(self):
        async with self.client.post(self.url + "/redirect", allow_redirects=False) as response:
            self.assertEqual(response.status, 307)
            self.assertEqual(response.headers["Location"], "/v1/chat/completions")
        self.assertEqual(self.calls, 1)

    async def test_compressed_request_body_is_not_silently_decoded(self):
        encoded = gzip.compress(b"payload" * 100)
        async with self.client.post(self.url + "/echo", data=encoded,
                                    headers={"Content-Encoding": "gzip"}) as response:
            self.assertEqual(await response.read(), encoded)
            self.assertEqual(response.headers["Content-Encoding"], "gzip")

    async def test_native_load_route_is_not_replaced_by_observer_schema(self):
        async with self.client.get(self.url + "/load") as response:
            self.assertEqual(await response.json(), {"server_load": 19})

    async def test_health_has_capacity_while_inference_slots_are_full(self):
        self.frontend.config["max_inflight"] = 1
        response = await self.client.post(self.url + "/stream")
        await response.content.readuntil(b"\n\n")
        async with self.client.get(self.url + "/health") as health:
            self.assertEqual(health.status, 200)
        async with self.client.post(self.url + "/v1/chat/completions") as overloaded:
            self.assertEqual(overloaded.status, 503)
        response.close()


class RegistrarTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(gateway_url="http://gateway", worker_url="http://proxy", model_id=MODEL,
                                    management_api_key_file="/not-a-real-secret",
                                    job="candidate", ranks=16, prefill_metrics_url="http://p:8000/metrics",
                                    decode_metrics_url="http://d:8000/metrics")
        self.samples = {role: {"deadline": time.monotonic() + 10, "epoch": EPOCH, "healthy": True,
                              "api_deadline": time.monotonic() + 2,
                              "api_responsive": True, "native_deadline": time.monotonic() + 3,
                              "fields": {"prefill_cache_miss_tokens" if role == "prefill" else "decode_running_requests":
                                         {rank: 3 for rank in range(16)}}}
                        for role in ("prefill", "decode")}

    def test_health_failure_still_collects_both_old_metric_urls(self):
        with mock.patch.object(registrar, "backend_ready", return_value=False), \
                mock.patch.object(registrar, "fetch_metrics", side_effect=lambda url, args, role: self.samples[role]) as fetch:
            samples, ready = registrar.collect(self.args)
        self.assertFalse(ready)
        self.assertEqual(set(samples), {"prefill", "decode"})
        self.assertEqual({call.args[0] for call in fetch.call_args_list},
                         {self.args.prefill_metrics_url, self.args.decode_metrics_url})

    def test_unhealthy_workers_get_false_heartbeat_and_fresh_load(self):
        workers = [registrar.expected_worker(self.args, rank) for rank in range(16)]
        with mock.patch.object(registrar, "collect", return_value=(self.samples, False)), \
                mock.patch.object(registrar, "request_json", return_value={"workers": workers}) as request:
            registrar.reconcile(self.args, "synthetic")
        heartbeats = [call.args[3] for call in request.call_args_list if call.args[0] == "POST"]
        self.assertEqual(len(heartbeats), 16)
        self.assertTrue(all(not body["ready"] and body["decode_running_requests"] == 3 for body in heartbeats))

    def test_degraded_existing_worker_is_not_cut_off(self):
        for source in self.samples.values():
            source.update(healthy=False, api_responsive=False)
        self.assertTrue(registrar.heartbeat(0, self.samples, True)["ready"])

    def test_degraded_state_does_not_publish_new_capacity(self):
        self.samples["decode"].update(healthy=False, api_responsive=False)
        with mock.patch.object(registrar, "collect", return_value=(self.samples, True)), \
                mock.patch.object(registrar, "request_json", return_value={"workers": []}) as request:
            registrar.reconcile(self.args, "synthetic")
        self.assertEqual(request.call_count, 1)

    def test_expired_api_evidence_does_not_publish_new_capacity(self):
        self.samples["decode"]["api_deadline"] = time.monotonic() - 1
        with mock.patch.object(registrar, "collect", return_value=(self.samples, True)), \
                mock.patch.object(registrar, "request_json", return_value={"workers": []}) as request:
            registrar.reconcile(self.args, "synthetic")
        self.assertEqual(request.call_count, 1)

    def test_old_samples_are_not_renewed_by_heartbeats(self):
        body = registrar.heartbeat(0, self.samples, True, now=time.monotonic() + 20)
        self.assertEqual(body, {"ready": False})

    def test_stale_native_counters_do_not_hide_fresh_load(self):
        self.samples["decode"]["fields"]["decode_generation_tokens_total"] = {rank: 99 for rank in range(16)}
        self.samples["decode"]["native_deadline"] = time.monotonic() - 1
        body = registrar.heartbeat(0, self.samples, True)
        self.assertEqual(body["decode_running_requests"], 3)
        self.assertNotIn("decode_generation_tokens_total", body)

    def test_mixed_pd_epochs_are_rejected(self):
        self.samples["decode"]["epoch"] = "wrong"
        with mock.patch.object(registrar, "backend_ready", return_value=True), \
                mock.patch.object(registrar, "fetch_metrics", side_effect=lambda url, args, role: self.samples[role]):
            samples, ready = registrar.collect(self.args)
        self.assertEqual(samples, {})

    def test_missing_duplicate_or_out_of_world_samples_not_zero_filled(self):
        good = '\n'.join(f'load{{engine="{rank}"}} 1' for rank in range(16))
        for text in (good.split('\n', 1)[1], good + '\nload{engine="0"} 2', good + '\nload{engine="16"} 1'):
            with self.assertRaises(ValueError):
                registrar.rank_metric(text, "load", 16, True)

    def test_legacy_cli_flags_still_parse(self):
        arguments = ["registrar"]
        for name, value in vars(self.args).items():
            arguments += ["--" + name.replace("_", "-"), str(value)]
        with mock.patch.object(sys, "argv", arguments):
            parsed_args = registrar.parse_args()
        self.assertEqual(parsed_args.decode_metrics_url, "http://d:8000/metrics")


class InstalledContractTests(unittest.TestCase):
    def test_installed_payloads_match_offline_manifest(self):
        root = Path("/opt/compatible-health-metrics")
        manifest = json.loads((root / "installed-manifest.json").read_text())
        site = Path(importlib.metadata.distribution("vllm").locate_file(""))
        for entry in manifest["runtime_files"]:
            self.assertEqual(hashlib.sha256((site / entry["path"]).read_bytes()).hexdigest(), entry["after_sha256"])
        for name in ("compatible_state.py", "compatible_runtime.py", "compatible_native.py"):
            self.assertEqual((root / name).read_bytes(), (site / "vllm/entrypoints/serve/instrumentator" / name).read_bytes())

    def test_disabled_compatibility_does_not_rewrite_port(self):
        args = SimpleNamespace(host="0.0.0.0", port=8000)
        with mock.patch.dict(os.environ, {"VLLM_COMPAT_OBSERVABILITY": "0", "VLLM_ISOLATED_OBSERVABILITY": "0"}):
            with runtime.observation_session(args):
                self.assertEqual((args.host, args.port), ("0.0.0.0", 8000))

    def test_single_api_process_and_full_group_roster_supported(self):
        args = SimpleNamespace(host="0.0.0.0", port=8000, model=MODEL, served_model_name=[MODEL],
                               api_server_count=1, disable_log_stats=False, data_parallel_size=16,
                               data_parallel_size_local=8, data_parallel_start_rank=None, data_parallel_rank=None,
                               data_parallel_multi_port_external_lb=False)
        environment = {"VLLM_OBSERVABILITY_EPOCH": EPOCH, "VLLM_OBSERVABILITY_MODEL": MODEL,
                       "VLLM_OBSERVABILITY_ROLE": "decode", "VLLM_OBSERVABILITY_RANKS": "0,1,2,3,4,5,6,7",
                       "VLLM_OBSERVABILITY_HOST": "0.0.0.0", "VLLM_COMPAT_OBSERVER_PEERS": json.dumps(group_config()["peers"])}
        with mock.patch.dict(os.environ, environment):
            configured = runtime.session_config(args, "/tmp/test")
        self.assertEqual(configured["api_count"], 1)
        self.assertEqual(configured["public_port"], 8000)
        self.assertEqual(configured["backend"], "http://127.0.0.1:18000")


if __name__ == "__main__":
    unittest.main()
