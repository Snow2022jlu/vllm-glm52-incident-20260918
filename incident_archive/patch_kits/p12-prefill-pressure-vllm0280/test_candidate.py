from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import apply_p12 as installer
import build_candidate
from test_support import METRICS, RequestFixture, prometheus_text, reconcile_fixture, scheduler_fixture


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def make_installation(root, version="0.28.0"):
    site = root / "site-packages"
    shutil.copytree(ROOT / "sources/vllm-0.28.0/vllm", site / "vllm")
    metadata = site / f"vllm-{version}.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_text(f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n")
    collector = root / "static_dp_registrar.py"
    shutil.copyfile(ROOT / "sources/collectors/r106.py", collector)
    return site, collector


def snapshots(site, collector):
    return {path: path.read_bytes() for path in [*(site / relative for relative in installer.RUNTIME_PATHS), collector]}


def p12_target(node):
    return any(isinstance(child, (ast.Name, ast.Attribute)) and getattr(child, "id", getattr(child, "attr", "")).startswith(("prefill_cache", "gauge_prefill_cache", "report_prefill_cache")) for child in ast.walk(node))


class RemoveP12(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "_get_prefill_cache_pressure_stats":
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        return None if any(p12_target(target) for target in node.targets) else self.generic_visit(node)

    def visit_AnnAssign(self, node):
        return None if p12_target(node.target) else self.generic_visit(node)

    def visit_Expr(self, node):
        return None if p12_target(node) else self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [keyword for keyword in node.keywords if not (keyword.arg or "").startswith("prefill_cache_")]
        return self.generic_visit(node)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="p12-installer-")
        self.root = Path(self.temporary.name)
        self.site, self.collector = make_installation(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def apply(self, **options):
        return installer.apply(self.site, self.collector, **options)

    def test_build_is_deterministic_and_payloads_preserve_upstream(self):
        self.assertEqual(build_candidate.build(), MANIFEST)
        for record in MANIFEST["runtime_files"]:
            original = ast.parse((ROOT / "sources/vllm-0.28.0" / record["path"]).read_text())
            payload = ast.parse((ROOT / record["payload"]).read_text())
            stripped = RemoveP12().visit(copy.deepcopy(payload))
            self.assertEqual(ast.dump(original), ast.dump(stripped), record["path"])
        before_collector = ast.parse((ROOT / "sources/collectors/r106.py").read_text())
        after_collector = ast.parse((ROOT / MANIFEST["collector"]["payload"]).read_text())
        for old, new in zip(before_collector.body, after_collector.body):
            if isinstance(old, ast.FunctionDef) and old.name == "reconcile":
                continue
            self.assertEqual(ast.dump(old), ast.dump(new))

    def test_default_dry_run_and_cli_default(self):
        before = snapshots(self.site, self.collector)
        self.assertTrue(all(record["status"] == "would_apply" for record in self.apply()["files"]))
        completed = subprocess.run([sys.executable, str(ROOT / "apply_p12.py"), "--site-packages-root", str(self.site), "--collector-file", str(self.collector)], check=True, text=True, capture_output=True)
        self.assertFalse(json.loads(completed.stdout)["offline_write_requested"])
        self.assertEqual(before, snapshots(self.site, self.collector))

    def test_apply_repeat_verify_and_permissions(self):
        os.chmod(self.collector, 0o750)
        result = self.apply(apply_offline=True)
        self.assertEqual(len(result["files"]), 5)
        self.assertTrue(all(record["status"] == "applied" for record in result["files"]))
        self.assertTrue(all(record["status"] == "already_applied" for record in self.apply(apply_offline=True)["files"]))
        self.assertTrue(all(record["status"] == "verified" for record in self.apply(verify_installed=True)["files"]))
        self.assertEqual(stat.S_IMODE(self.collector.stat().st_mode), 0o750)
        self.assertFalse(list(self.root.rglob("*.p12-*")))

    def test_verify_rejects_unpatched_and_partial(self):
        with self.assertRaisesRegex(RuntimeError, "not completely patched"):
            self.apply(verify_installed=True)
        record = MANIFEST["runtime_files"][0]
        (self.site / record["path"]).write_bytes((ROOT / record["payload"]).read_bytes())
        with self.assertRaisesRegex(RuntimeError, "not completely patched"):
            self.apply(verify_installed=True)
        self.apply(apply_offline=True)
        self.apply(verify_installed=True)

    def test_all_32_partial_installation_states_resume(self):
        for state in range(32):
            site, collector = make_installation(self.root / f"state-{state}")
            records = [*(dict(record, target=site / record["path"]) for record in MANIFEST["runtime_files"]), dict(MANIFEST["collector"], target=collector)]
            for position, record in enumerate(records):
                if state & (1 << position):
                    record["target"].write_bytes((ROOT / record["payload"]).read_bytes())
            installer.apply(site, collector, apply_offline=True)
            installer.apply(site, collector, verify_installed=True)

    def test_unknown_last_runtime_or_collector_has_no_partial_writes(self):
        for target in (self.site / installer.RUNTIME_PATHS[-1], self.collector):
            original = target.read_bytes()
            target.write_bytes(original + b"\n")
            before = snapshots(self.site, self.collector)
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                self.apply(apply_offline=True)
            self.assertEqual(before, snapshots(self.site, self.collector))
            target.write_bytes(original)

    def test_payload_corruption_and_manifest_scope_refused(self):
        bundle = self.root / "bundle"
        shutil.copytree(ROOT, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        payload = bundle / MANIFEST["collector"]["payload"]
        original = payload.read_bytes()
        payload.write_bytes(original + b"\n")
        before = snapshots(self.site, self.collector)
        with self.assertRaisesRegex(RuntimeError, "corrupt payload"):
            self.apply(apply_offline=True, bundle_root=bundle)
        payload.write_bytes(original)
        manifest = copy.deepcopy(MANIFEST)
        manifest["runtime_files"][-1]["path"] = "../outside.py"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "write scope"):
            self.apply(apply_offline=True, bundle_root=bundle)
        self.assertEqual(before, snapshots(self.site, self.collector))

    def test_wrong_missing_and_ambiguous_versions_refused(self):
        metadata = next(self.site.glob("vllm-*.dist-info/METADATA"))
        original = metadata.read_bytes()
        for version in ("0.26.0", "0.28.1", "0.28.0.dev1"):
            metadata.write_text(f"Name: vllm\nVersion: {version}\n")
            with self.assertRaisesRegex(RuntimeError, "requires vLLM"):
                self.apply(apply_offline=True)
        metadata.write_text("Name: vllm\nVersion: 0.28.0+cu130\n")
        self.assertEqual(self.apply()["vllm_version"], "0.28.0+cu130")
        duplicate = self.site / "vllm-duplicate.dist-info/METADATA"
        duplicate.parent.mkdir()
        duplicate.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            self.apply()
        duplicate.unlink()
        metadata.unlink()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            self.apply()

    def test_explicit_components_and_invalid_options(self):
        with self.assertRaises(ValueError):
            installer.apply(self.site)
        with self.assertRaises(ValueError):
            self.apply(apply_offline=True, verify_installed=True)
        original_collector = self.collector.read_bytes()
        installer.apply(site_packages_root=self.site, component="vllm", apply_offline=True)
        self.assertEqual(self.collector.read_bytes(), original_collector)
        installer.apply(collector_file=self.collector, component="collector", apply_offline=True)
        self.apply(verify_installed=True)

    def test_symlink_files_roots_and_payloads_refused(self):
        original = self.collector.read_bytes()
        external = self.root / "external.py"
        external.write_bytes(original)
        self.collector.unlink()
        self.collector.symlink_to(external)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.apply(apply_offline=True)
        self.collector.unlink()
        self.collector.write_bytes(original)
        alias = self.root / "site-alias"
        alias.symlink_to(self.site, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            installer.apply(alias, self.collector)
        bundle = self.root / "bundle"
        shutil.copytree(ROOT, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        payload = bundle / MANIFEST["collector"]["payload"]
        payload.unlink()
        payload.symlink_to(ROOT / MANIFEST["collector"]["payload"])
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.apply(bundle_root=bundle)

    def test_write_failure_restores_all_written_files(self):
        before = snapshots(self.site, self.collector)
        replace = installer.atomic_replace
        failed = False
        def fail_once(target, data):
            nonlocal failed
            if target == self.site / installer.RUNTIME_PATHS[2] and not failed:
                failed = True
                raise OSError("synthetic write failure")
            replace(target, data)
        with patch.object(installer, "atomic_replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "synthetic"):
                self.apply(apply_offline=True)
        self.assertEqual(before, snapshots(self.site, self.collector))

    def test_error_after_replace_restores_files(self):
        before = snapshots(self.site, self.collector)
        replace = installer.atomic_replace
        failed = False
        def fail_once(target, data):
            nonlocal failed
            replace(target, data)
            if target == self.collector and not failed:
                failed = True
                raise OSError("synthetic post-replace error")
        with patch.object(installer, "atomic_replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "post-replace"):
                self.apply(apply_offline=True)
        self.assertEqual(before, snapshots(self.site, self.collector))

    def test_target_change_after_preflight_is_not_overwritten(self):
        prepare = installer.prepare
        altered = self.collector.read_bytes() + b"\n"
        before = snapshots(self.site, self.collector)
        def change_target(*args, **kwargs):
            result = prepare(*args, **kwargs)
            self.collector.write_bytes(altered)
            return result
        with patch.object(installer, "prepare", side_effect=change_target):
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                self.apply(apply_offline=True)
        before[self.collector] = altered
        self.assertEqual(before, snapshots(self.site, self.collector))

    def test_metadata_change_after_preflight_blocks_writes(self):
        prepare = installer.prepare
        before = snapshots(self.site, self.collector)
        metadata = next(self.site.glob("vllm-*.dist-info/METADATA"))
        def change_metadata(*args, **kwargs):
            result = prepare(*args, **kwargs)
            metadata.write_text("Name: vllm\nVersion: 0.29.0\n")
            return result
        with patch.object(installer, "prepare", side_effect=change_metadata):
            with self.assertRaisesRegex(RuntimeError, "metadata changed"):
                self.apply(apply_offline=True)
        self.assertEqual(before, snapshots(self.site, self.collector))

    def test_rollback_failure_is_reported_not_silenced(self):
        before = snapshots(self.site, self.collector)
        replace = installer.atomic_replace
        first = self.site / installer.RUNTIME_PATHS[0]
        second = self.site / installer.RUNTIME_PATHS[1]
        def fail_write_and_restore(target, data):
            if target == second:
                raise OSError("synthetic second-file failure")
            if target == first and data == before[first]:
                raise OSError("synthetic restore failure")
            replace(target, data)
        with patch.object(installer, "atomic_replace", side_effect=fail_write_and_restore):
            with self.assertRaisesRegex(RuntimeError, "rollback incomplete"):
                self.apply(apply_offline=True)
        self.assertNotEqual(first.read_bytes(), before[first])
        self.assertEqual(second.read_bytes(), before[second])
        self.apply(apply_offline=True)
        self.apply(verify_installed=True)

    def test_source_patch_files_apply_exactly(self):
        repo = self.root / "source-tree"
        shutil.copytree(ROOT / "sources/vllm-0.28.0", repo)
        shutil.copyfile(ROOT / "sources/collectors/r106.py", repo / "static_dp_registrar.py")
        for filename in ("v028-p12.patch", "collector-p12.patch"):
            subprocess.run(["git", "apply", "--check", str(ROOT / filename)], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "apply", str(ROOT / filename)], cwd=repo, check=True, capture_output=True)
        for record in MANIFEST["runtime_files"]:
            self.assertEqual((repo / record["path"]).read_bytes(), (ROOT / record["payload"]).read_bytes())
        self.assertEqual((repo / "static_dp_registrar.py").read_bytes(), (ROOT / MANIFEST["collector"]["payload"]).read_bytes())


class BehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="p12-installed-behavior-")
        cls.site, cls.collector = make_installation(Path(cls.temporary.name))
        installer.apply(cls.site, cls.collector, apply_offline=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_normal_pressure_and_read_only_scan(self):
        cases = [
            ({}, (0, 0, 1)),
            ({"prefill_cache_lookup_complete": True}, (1, 1024, 0)),
            ({"prefill_cache_lookup_complete": True, "prefill_cached_tokens": 768}, (1, 256, 0)),
            ({"prefill_cache_lookup_complete": True, "num_computed_tokens": 1024, "num_in_flight_tokens": 256}, (1, 256, 0)),
            ({"prefill_cache_lookup_complete": True, "num_computed_tokens": 1024}, (0, 0, 0)),
            ({"prefill_stats": None}, (0, 0, 0)),
        ]
        for values, expected in cases:
            scheduler, status_type = scheduler_fixture(self.site)
            request = RequestFixture(status_type, **values)
            scheduler.requests[request.request_id] = request
            before = request.__dict__.copy()
            self.assertEqual(scheduler._get_prefill_cache_pressure_stats(), expected)
            self.assertEqual(before, request.__dict__)
            scheduler.report_prefill_cache_pressure = False
            self.assertEqual(scheduler._get_prefill_cache_pressure_stats(), (0, 0, 0))

    def test_all_finished_statuses_are_excluded(self):
        scheduler, status_type = scheduler_fixture(self.site)
        for status in status_type:
            if not status_type.is_finished(status):
                continue
            for lookup in (False, True):
                request = RequestFixture(status_type, status=status, prefill_cache_lookup_complete=lookup, num_computed_tokens=768)
                scheduler.requests = {request.request_id: request}
                self.assertEqual(scheduler._get_prefill_cache_pressure_stats(), (0, 0, 0))

    def test_actual_abort_delay_lifecycle(self):
        for delayed in (False, True):
            scheduler, status_type = scheduler_fixture(self.site)
            request = RequestFixture(status_type, status=status_type.WAITING_FOR_REMOTE_KVS if delayed else status_type.WAITING, prefill_cache_lookup_complete=True, num_computed_tokens=768)
            scheduler.requests[request.request_id] = request
            scheduler.waiting.append(request)
            self.assertEqual(scheduler._get_prefill_cache_pressure_stats(), (1, 256, 0))
            self.assertEqual(scheduler.finish_requests(request.request_id, status_type.FINISHED_ABORTED), [request])
            self.assertEqual(request.request_id in scheduler.requests, delayed)
            self.assertEqual(scheduler._get_prefill_cache_pressure_stats(), (0, 0, 0))
            if delayed:
                scheduler._free_blocks(request)
                self.assertFalse(scheduler.requests)

    def test_preemption_resets_lookup_and_preserves_stale_state(self):
        scheduler, status_type = scheduler_fixture(self.site)
        request = RequestFixture(status_type, status=status_type.RUNNING, prefill_cache_lookup_complete=True, prefill_cached_tokens=512, num_computed_tokens=768, num_in_flight_tokens=128, spec_token_ids=[1, 2])
        scheduler.requests[request.request_id] = request
        scheduler._preempt_request(request, 1.0)
        self.assertFalse(request.prefill_cache_lookup_complete)
        self.assertEqual(request.prefill_cached_tokens, 0)
        self.assertEqual(request.num_stale_output_tokens, 128)
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertEqual(request.spec_token_ids, [])
        self.assertEqual(scheduler._get_prefill_cache_pressure_stats(), (0, 0, 1))

    def test_inflight_tokens_remain_pressure_after_scheduling(self):
        scheduler, status_type = scheduler_fixture(self.site)
        request = RequestFixture(status_type, status=status_type.RUNNING, prefill_cache_lookup_complete=True, num_computed_tokens=768)
        scheduler.requests[request.request_id] = request
        before = scheduler._get_prefill_cache_pressure_stats()
        scheduler._update_after_schedule(type("Output", (), {"num_scheduled_tokens": {request.request_id: 256}, "has_structured_output_requests": False})())
        self.assertEqual(request.num_computed_tokens, 1024)
        self.assertEqual(request.num_in_flight_tokens, 256)
        self.assertEqual(scheduler._get_prefill_cache_pressure_stats(), before)

    def test_scheduler_stats_prometheus_collector_roundtrip_16_ranks(self):
        stats_by_engine = {}
        expected = []
        for rank in range(16):
            scheduler, status_type = scheduler_fixture(self.site)
            request = RequestFixture(status_type, prefill_cache_lookup_complete=rank % 2 == 0, prefill_cached_tokens=rank * 32)
            scheduler.requests[request.request_id] = request
            scheduler.log_stats = True
            stats = scheduler.make_stats()
            stats_by_engine[rank] = stats
            expected.append({metric: getattr(stats, metric) for metric in METRICS})
        text = prometheus_text(self.site, stats_by_engine)
        heartbeats = reconcile_fixture(self.collector, text, ranks=16)
        self.assertEqual(len(heartbeats), 16)
        for heartbeat, pressure in zip(heartbeats, expected):
            self.assertEqual({metric: heartbeat[metric] for metric in METRICS}, pressure)
            self.assertTrue(heartbeat["ready"])
            self.assertIn("decode_kv_cache_usage_ratio", heartbeat)

    def test_zero_metrics_and_disabled_stats(self):
        scheduler, status_type = scheduler_fixture(self.site)
        self.assertIsNone(scheduler.make_stats())
        scheduler.log_stats = True
        stats = scheduler.make_stats()
        text = prometheus_text(self.site, {0: stats, 1: stats})
        for heartbeat in reconcile_fixture(self.collector, text):
            self.assertEqual([heartbeat[metric] for metric in METRICS], [0, 0, 0])

    def test_missing_family_or_rank_is_unknown_not_zero(self):
        samples = "\n".join(f'vllm:{metric}{{engine="{rank}"}} 0' for metric in METRICS for rank in range(2))
        variants = ["unrelated 9\n", samples.replace('vllm:prefill_cache_miss_tokens{engine="1"} 0', ""), samples.replace('vllm:prefill_cache_pending_requests{engine="0"} 0', ""), None]
        for text in variants:
            heartbeats = reconcile_fixture(self.collector, text)
            self.assertEqual(len(heartbeats), 2)
            for heartbeat in heartbeats:
                self.assertTrue(all(metric not in heartbeat for metric in METRICS))
                self.assertIn("decode_kv_cache_usage_ratio", heartbeat)

    def test_partial_invalid_samples_do_not_abort_heartbeats(self):
        valid = "\n".join(f'vllm:{metric}{{engine="{rank}"}} 0' for metric in METRICS for rank in range(2))
        for metric in METRICS:
            for bad in ("NaN", "Inf", "-1", "0.5"):
                text = valid.replace(f'vllm:{metric}{{engine="0"}} 0', f'vllm:{metric}{{engine="0"}} {bad}')
                heartbeats = reconcile_fixture(self.collector, text)
                self.assertEqual(len(heartbeats), 2)
                self.assertTrue(all(all(name not in heartbeat for name in METRICS) for heartbeat in heartbeats))

    def test_original_collector_reproduces_counterexamples(self):
        original = ROOT / "sources/collectors/r106.py"
        heartbeats = reconcile_fixture(original, "unrelated 9\n")
        self.assertTrue(all(heartbeat["prefill_cache_miss_tokens"] == 0 for heartbeat in heartbeats))
        with self.assertRaises(TypeError):
            reconcile_fixture(original, 'vllm:prefill_cache_miss_requests{engine="0"} 1\nvllm:prefill_cache_miss_tokens{engine="0"} NaN\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for record in source_manifest["files"]:
        if hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"source checksum mismatch: {record['path']}")
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "successful": result.wasSuccessful(), "source_records_verified": len(source_manifest["files"]), "behavior_uses_installed_candidate_files": True, "prometheus_roundtrip_engines": 16, "gpu_or_live_network_executed": False}
    (arguments.output_dir / "test-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
