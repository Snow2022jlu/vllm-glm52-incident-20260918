from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, "/opt/bounded-health-metrics")

import test_compatibility as legacy
import test_process_isolation as process_legacy
from prometheus_client import generate_latest
from vllm.entrypoints.serve.instrumentator import bounded_observability as bounded


base, state, runtime, registrar = legacy.base, legacy.state, legacy.runtime, legacy.registrar
original_report = legacy.report
original_registrar_setup = legacy.RegistrarTests.setUp


def bounded_report(*args, **kwargs):
    report = original_report(*args, **kwargs)
    report.update(api_health_max_age_seconds=1800.0,
                  health_valid_for_seconds=report["load_valid_for_seconds"] if report["health_ok"] else 0.0)
    return report


def bounded_registrar_setup(test):
    original_registrar_setup(test)
    for sample in test.samples.values():
        sample.update(health_ok=True, health_deadline=sample["deadline"])


legacy.report = bounded_report
legacy.RegistrarTests.setUp = bounded_registrar_setup


def installed_manifest(test):
    root = Path("/opt/bounded-health-metrics")
    site = Path(legacy.importlib.metadata.distribution("vllm").locate_file(""))
    entries = {}
    for directory in (Path("/opt/compatible-health-metrics"), root):
        manifest = json.loads((directory / "installed-manifest.json").read_text())
        entries.update({entry["path"]: entry for entry in manifest["runtime_files"]})
    for path, entry in entries.items():
        test.assertEqual(hashlib.sha256((site / path).read_bytes()).hexdigest(), entry["after_sha256"])
    publisher = manifest["publisher"]
    test.assertEqual(hashlib.sha256(Path(publisher["path"]).read_bytes()).hexdigest(), publisher["after_sha256"])


legacy.InstalledContractTests.test_installed_payloads_match_offline_manifest = installed_manifest


class BoundedPolicyTests(unittest.TestCase):
    def test_exact_thirty_minute_boundary(self):
        for age, healthy in ((1799.999, True), (1800.0, False), (1800.001, False), (3600.0, False)):
            with self.subTest(age=age), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(base.time, "monotonic", return_value=10000.0):
                    configured = legacy.config([0])
                    legacy.populate(directory, configured, api_age=age)
                    report = state.LocalState(directory).report()
                self.assertEqual(report["health_ok"], healthy)
                self.assertEqual(report["api_health_max_age_seconds"], 1800.0)
                self.assertTrue(report["load_fresh"])
                if not healthy:
                    self.assertIn("api_health_grace_expired", report["reasons"])

    def test_core_activity_and_repeated_reads_do_not_renew_api_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(base.time, "monotonic", return_value=10000.0) as clock:
                configured = legacy.config([0])
                legacy.populate(directory, configured)
                local = state.LocalState(directory)
                for elapsed in (0, 1799, 1800, 3600):
                    clock.return_value = 10000.0 + elapsed
                    base.atomic_json(Path(directory) / "core-0.json",
                                     legacy.sample(configured, rank=0, metrics=legacy.metrics()))
                    for _attempt in range(3):
                        self.assertEqual(local.report()["health_ok"], elapsed < 1800)

    def test_new_api_pulse_recovers_expired_health(self):
        with tempfile.TemporaryDirectory() as directory:
            configured = legacy.config([0])
            legacy.populate(directory, configured, api_age=3600)
            local = state.LocalState(directory)
            self.assertFalse(local.report()["health_ok"])
            base.atomic_json(Path(directory) / f"api-{os.getpid()}.json", legacy.sample(configured))
            self.assertEqual(local.report()["health_state"], "healthy")

    def test_headless_fresh_core_does_not_need_own_api_pulse(self):
        with tempfile.TemporaryDirectory() as directory:
            configured = legacy.config([0], 0)
            legacy.populate(directory, configured)
            self.assertTrue(state.LocalState(directory).report()["health_ok"])

    def test_missing_or_old_protocol_health_contract_is_rejected(self):
        for change in ({"compatibility_protocol": "vllm-compatible-observability-v1"},
                       {"health_valid_for_seconds": None}, {"api_health_max_age_seconds": 3600}):
            report = legacy.report(list(range(8)), 1)
            report.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                state.GroupState(legacy.group_config()).accept(0, report, time.monotonic())

    def test_group_expires_health_before_load_without_another_poll(self):
        group = state.GroupState(legacy.group_config())
        started = time.monotonic()
        for index, ranks in enumerate((list(range(8)), list(range(8, 16)))):
            report = legacy.report(ranks, int(index == 0), degraded=True)
            report["health_valid_for_seconds"] = 0.2 if index == 0 else 9.0
            group.accept(index, report, started)
        self.assertTrue(group.view(started + 0.1)["health_ok"])
        view = group.view(started + 0.3)
        self.assertFalse(view["health_ok"])
        self.assertTrue(view["load_fresh"])
        self.assertEqual(view["health_state"], "unavailable")

    def test_auxiliary_failure_revokes_health_without_destroying_load(self):
        with tempfile.TemporaryDirectory() as directory:
            configured = legacy.config([0])
            legacy.populate(directory, configured)
            base.atomic_json(Path(directory) / bounded.FAILURE_FILE, {"errors": ["frontend exited"]})
            report = state.LocalState(directory).report()
            self.assertFalse(report["health_ok"])
            self.assertFalse(report["ready"])
            self.assertTrue(report["load_fresh"])
            self.assertIn("auxiliary_process_failed", report["reasons"])


class PublisherTests(unittest.TestCase):
    def samples(self, health="healthy", remaining=2.0):
        samples = {}
        args = SimpleNamespace(model_id=legacy.MODEL, ranks=16)
        for role in ("prefill", "decode"):
            response = io.BytesIO(generate_latest(legacy.registry()))
            response.headers = {
                "X-VLLM-Model": legacy.MODEL, "X-VLLM-Role": role,
                "X-VLLM-Epoch": legacy.EPOCH, "X-VLLM-Load-Valid-For": "3",
                "X-VLLM-API-Valid-For": "2" if health == "healthy" else "0",
                "X-VLLM-Native-Snapshot-Age": "0.1", "X-VLLM-Health-State": health,
                "X-VLLM-API-Responsive": "1" if health == "healthy" else "0",
                "X-VLLM-Health-Valid-For": str(remaining),
            }
            with mock.patch.object(registrar.urllib.request, "urlopen", return_value=response):
                samples[role] = registrar.fetch_metrics("http://unused/metrics", args, role)
        return samples

    def test_unavailable_source_vetoes_positive_proxy_probe(self):
        samples = self.samples("unavailable", 0.0)
        body = registrar.heartbeat(0, samples, True)
        self.assertFalse(body["ready"])
        self.assertIn("decode_running_requests", body)
        self.assertIn("prefill_cache_miss_tokens", body)

    def test_either_pd_source_can_veto_readiness(self):
        for role in ("prefill", "decode"):
            samples = self.samples()
            samples[role] = self.samples("unavailable", 0.0)[role]
            self.assertFalse(registrar.heartbeat(0, samples, True)["ready"])

    def test_degraded_is_retained_only_until_source_deadline(self):
        samples = self.samples("degraded", 0.2)
        now = time.monotonic()
        self.assertTrue(registrar.heartbeat(0, samples, True, now)["ready"])
        body = registrar.heartbeat(0, samples, True, now + 0.4)
        self.assertFalse(body["ready"])
        self.assertIn("decode_running_requests", body)

    def test_unknown_state_or_excessive_deadline_is_rejected(self):
        for health, remaining in (("unknown", 1.0), ("healthy", 0.0), ("healthy", 3.1), ("degraded", float("nan"))):
            with self.subTest(health=health, remaining=remaining), self.assertRaises(ValueError):
                self.samples(health, remaining)

    def test_h02_remains_explicitly_unfixed(self):
        samples = self.samples()
        samples["decode"]["native_deadline"] = time.monotonic() - 1
        body = registrar.heartbeat(0, samples, True)
        self.assertIn("decode_running_requests", body)
        self.assertNotIn("decode_generation_tokens_total", body)
        manifest = json.loads(Path("/opt/bounded-health-metrics/installed-manifest.json").read_text())
        self.assertIs(manifest["h02_fixed"], False)


def session_inputs():
    public, observer, backend = process_legacy.free_ports(3)
    args = SimpleNamespace(host="127.0.0.1", port=public, model=legacy.MODEL,
                           served_model_name=[legacy.MODEL], api_server_count=1, disable_log_stats=False,
                           data_parallel_size=1, data_parallel_size_local=1, data_parallel_start_rank=None,
                           data_parallel_rank=None, data_parallel_multi_port_external_lb=False)
    environment = {
        "VLLM_COMPAT_OBSERVABILITY": "1", "VLLM_ISOLATED_OBSERVABILITY": "1",
        "VLLM_OBSERVABILITY_EPOCH": "bounded-test", "VLLM_OBSERVABILITY_MODEL": legacy.MODEL,
        "VLLM_OBSERVABILITY_ROLE": "decode", "VLLM_OBSERVABILITY_RANKS": "0",
        "VLLM_OBSERVABILITY_HOST": "127.0.0.1", "VLLM_OBSERVABILITY_PORT": str(observer),
        "VLLM_COMPAT_BACKEND_PORT": str(backend),
        "VLLM_COMPAT_OBSERVER_PEERS": json.dumps([
            {"url": f"http://127.0.0.1:{observer}", "ranks": [0], "api_count": 1}]),
    }
    return args, environment


class SupervisorTests(unittest.TestCase):
    def run_case(self, action):
        args, environment = session_inputs()
        children, directories = [], []
        original_popen = subprocess.Popen

        def launch(*arguments, **keywords):
            child = original_popen(*arguments, **keywords)
            children.append(child)
            directories.append(Path(arguments[0][-1]))
            return child

        try:
            with mock.patch.dict(os.environ, environment):
                with mock.patch.object(bounded.subprocess, "Popen", side_effect=launch):
                    action(args, children)
            self.assertNotIn("VLLM_OBSERVABILITY_DIRECTORY", os.environ)
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    child.send_signal(signal.SIGCONT)
                    child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            for directory in set(directories):
                if directory.exists():
                    shutil.rmtree(directory)
            self.assertTrue(all(child.poll() is not None for child in children))

    def test_stopped_frontend_does_not_skip_observer_cleanup(self):
        def action(args, children):
            original = args.host, args.port
            with self.assertRaisesRegex(RuntimeError, "Auxiliary cleanup incomplete"):
                with runtime.observation_session(args):
                    directory = Path(os.environ["VLLM_OBSERVABILITY_DIRECTORY"])
                    children[1].send_signal(signal.SIGSTOP)
            self.assertIsNotNone(children[0].poll())
            self.assertIsNone(children[1].poll())
            self.assertTrue((directory / bounded.FAILURE_FILE).exists())
            self.assertEqual((args.host, args.port), original)

        self.run_case(action)

    def test_frontend_exit_is_latched_without_killing_launcher_or_observer(self):
        def action(args, children):
            with runtime.observation_session(args):
                directory = Path(os.environ["VLLM_OBSERVABILITY_DIRECTORY"])
                configured = base.read_json(directory / "config.json")
                legacy.populate(directory, configured)
                self.assertTrue(state.LocalState(directory).report()["health_ok"])
                children[1].terminate()
                children[1].wait(timeout=5)
                process_legacy.wait_until(lambda: (directory / bounded.FAILURE_FILE).exists())
                self.assertFalse(state.LocalState(directory).report()["health_ok"])
                self.assertIsNone(children[0].poll())
                self.assertEqual(base.process_identity(os.getpid()), configured["parent_start"])

        self.run_case(action)

    def test_partial_start_failure_cleans_children_and_restores_arguments(self):
        def action(args, children):
            original = args.host, args.port
            with mock.patch.object(runtime, "wait_process", side_effect=RuntimeError("bind failed")):
                with self.assertRaisesRegex(RuntimeError, "bind failed"):
                    with runtime.observation_session(args):
                        self.fail("Failed startup must not enter the model body")
            self.assertTrue(all(child.poll() is not None for child in children))
            self.assertEqual((args.host, args.port), original)

        self.run_case(action)

    def test_termination_failure_does_not_skip_other_children(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = bounded.AuxiliarySupervisor(directory)
            first, second = mock.Mock(pid=10001), mock.Mock(pid=10002)
            first.poll.return_value = second.poll.return_value = None
            second.terminate.side_effect = OSError("test signal failure")
            supervisor.children = [("first", first), ("second", second)]
            with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
                supervisor.close(timeout=0.01)
            first.terminate.assert_called_once()
            first.wait.assert_called_once()
            second.wait.assert_called_once()


def load_tests(loader, tests, pattern):
    tests.addTests(loader.loadTestsFromModule(legacy))
    tests.addTests(loader.loadTestsFromModule(process_legacy))
    return tests


if __name__ == "__main__":
    unittest.main()
