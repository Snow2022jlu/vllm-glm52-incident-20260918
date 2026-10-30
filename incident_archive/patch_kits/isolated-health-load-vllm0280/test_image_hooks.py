from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import logging
import queue
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


MANIFEST = Path("/opt/isolated-health-load/installed-manifest.json")


@unittest.skipUnless(MANIFEST.exists(), "Run these tests inside the candidate image")
class InstalledHookTests(unittest.TestCase):
    def setUp(self):
        self.site = Path(importlib.metadata.distribution("vllm").locate_file(""))
        self.core_source = (self.site / "vllm/v1/engine/core.py").read_text()

    def method(self, name, enabled, observations):
        tree = ast.parse(self.core_source)
        core = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "EngineCoreProc")
        method = next(node for node in core.body if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = dict(queue=queue, time=time, logger=logging.getLogger("hook-test"), DEBUG=logging.DEBUG,
                         observation_enabled=lambda: enabled,
                         observe_core=lambda *values: observations.append(values))
        exec(compile(ast.Module(body=[method], type_ignores=[]), "installed_core_hook", "exec"), namespace)
        return namespace[name]

    def test_idle_polling_preserves_disabled_and_nonblocking_behavior(self):
        for enabled, block in ((False, True), (True, True), (True, False)):
            with self.subTest(enabled=enabled, block=block):
                observations = []
                calls = []
                engine = SimpleNamespace(active=True)

                class InputQueue(queue.Queue):
                    def get(self, block=True, timeout=None):
                        calls.append((block, timeout))
                        engine.active = False
                        raise queue.Empty

                engine.input_queue = InputQueue()
                engine.aborts_queue = queue.Queue()
                engine.process_input_queue_block = block
                engine.has_work = lambda: False
                engine.is_running = lambda: engine.active
                engine._notify_idle_state_callbacks = lambda: None
                engine._handle_client_request = lambda *_args: None
                self.method("_process_input_queue", enabled, observations)(engine)
                self.assertEqual(calls, [(block, 0.25 if enabled and block else None)])
                self.assertGreaterEqual(len(observations), 1)

    def test_output_and_post_step_order_preserved(self):
        observations = []
        batch = object()
        order = []
        engine = SimpleNamespace(step_fn=lambda: ({3: batch}, True), output_queue=queue.Queue(),
                                 post_step=lambda executed: order.append(executed))
        executed = self.method("_process_engine_step", True, observations)(engine)
        self.assertTrue(executed)
        self.assertEqual(engine.output_queue.get_nowait(), (3, batch))
        self.assertEqual(order, [True])
        self.assertEqual(observations, [(engine, {3: batch})])

    def test_installed_manifest_matches_every_changed_file(self):
        manifest = json.loads(MANIFEST.read_text())
        for entry in manifest["runtime_files"]:
            actual = hashlib.sha256((self.site / entry["path"]).read_bytes()).hexdigest()
            self.assertEqual(actual, entry["after_sha256"])

    def test_single_observer_wraps_headless_single_and_multi_api_paths(self):
        source = (self.site / "vllm/entrypoints/cli/serve.py").read_text()
        tree = ast.parse(source)
        sessions = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                    and any(isinstance(item.context_expr, ast.Call)
                            and isinstance(item.context_expr.func, ast.Name)
                            and item.context_expr.func.id == "observation_session"
                            for item in node.items)]
        self.assertEqual(len(sessions), 1)
        wrapped = ast.unparse(sessions[0])
        self.assertIn("run_headless(args)", wrapped)
        self.assertIn("run_multi_api_server(args)", wrapped)
        self.assertIn("run_server(args)", wrapped)


if __name__ == "__main__":
    unittest.main()
