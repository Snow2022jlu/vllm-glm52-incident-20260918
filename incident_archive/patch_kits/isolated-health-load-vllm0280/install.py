"""Apply exact-hash, offline-only hooks to the immutable P12 image baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE = "vllm.entrypoints.serve.instrumentator.isolated_observability"
MODULE_PATH = MODULE.replace(".", "/") + ".py"
BEFORE = {
    "vllm/v1/engine/core.py": "f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad",
    "vllm/entrypoints/launcher.py": "94566e08afbe40aef653184aa49bb1cfc6bdc8e9bf10881ae05cab65475bcd2e",
    "vllm/entrypoints/cli/serve.py": "1becda2116e54732ae3f049d4660088bf8fb103f6dcc13ccb207813c8eab6bf9",
}


def checksum(content):
    return hashlib.sha256(content).hexdigest()


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError(f"Expected exactly one anchor: {anchor[:90]!r}")
    return source.replace(anchor, replacement, 1)


def transform(path, source):
    if path.endswith("engine/core.py"):
        source = replace_once(source, "import queue\n", "import queue\n" +
                              f"from {MODULE} import enabled as observation_enabled, observe_core\n")
        anchor = '        """Exits when an engine step needs to be performed."""\n'
        source = replace_once(source, anchor, anchor + "        observe_core(self)\n")
        anchor = "        while not self.has_work() and self.is_running():\n"
        source = replace_once(source, anchor, anchor + "            observe_core(self)\n")
        source = replace_once(source, "                req = self.input_queue.get(block=block)\n",
                              "                req = self.input_queue.get(\n"
                              "                    block=block,\n"
                              "                    timeout=0.25 if block and observation_enabled() else None,\n"
                              "                )\n")
        anchor = "                self._handle_client_request(*req)\n            except queue.Empty:\n                break\n"
        source = replace_once(source, anchor,
                              "                self._handle_client_request(*req)\n"
                              "            except queue.Empty:\n"
                              "                if block and observation_enabled():\n"
                              "                    continue\n"
                              "                break\n")
        anchor = "        outputs, model_executed = self.step_fn()\n"
        source = replace_once(source, anchor, anchor + "        observe_core(self, outputs)\n")
    elif path.endswith("entrypoints/launcher.py"):
        source = replace_once(source, "import asyncio\n", "import asyncio\n" +
                              f"from {MODULE} import start_api_pulse\n")
        anchor = "    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))\n"
        source = replace_once(source, anchor,
                              "    observation_task = start_api_pulse(server, app.state.engine_client)\n" + anchor)
        anchor = "        shutdown_task.cancel()\n        watchdog_task.cancel()\n"
        source = replace_once(source, anchor, anchor +
                              "        if observation_task is not None:\n"
                              "            observation_task.cancel()\n"
                              "            await asyncio.gather(observation_task, return_exceptions=True)\n")
    elif path.endswith("cli/serve.py"):
        source = replace_once(source, "import argparse\n", "import argparse\n" +
                              f"from {MODULE} import enabled as observation_enabled, observation_session\n")
        anchor = '        if getattr(args, "grpc", False):\n'
        source = replace_once(source, anchor,
                              '        if observation_enabled() and getattr(args, "grpc", False):\n'
                              '            raise ValueError("Isolated observation requires the Python HTTP frontend")\n' + anchor)
        anchor = """        if is_multi_port:
            run_dp_supervisor(args)
        elif args.api_server_count < 1:
            run_headless(args)
        elif args.api_server_count > 1 or rust_frontend_path:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            args.api_server_count = None
            uvloop.run(run_server(args))
"""
        source = replace_once(source, anchor, "        with observation_session(args):\n" +
                              "".join("    " + line for line in anchor.splitlines(keepends=True)))
    else:
        raise ValueError(f"Unapproved patch target: {path}")
    compile(source, path, "exec")
    return source


def install(site, output):
    if importlib.metadata.version("vllm").split("+")[0] != "0.28.0":
        raise ValueError("Only the pinned vLLM 0.28.0 baseline is supported")
    prepared = {}
    for relative, expected in BEFORE.items():
        content = (site / relative).read_bytes()
        if checksum(content) != expected:
            raise ValueError(f"Baseline hash mismatch: {relative}")
        prepared[relative] = transform(relative, content.decode()).encode()
    if (site / MODULE_PATH).exists():
        raise ValueError("Refusing to overwrite an existing observer module")
    prepared[MODULE_PATH] = (ROOT / "observer.py").read_bytes()
    manifest = {"base_commit": "2cf0a6915ce544dc493a0990f2ea38d81601128a",
                "base_image": "workspacegenaiacr.azurecr.io/lidong1/vllm-glm52-pd@sha256:4e241f7dd1af5ec766e21387e9f2dc991fe2753fa640cff904681260a843d76b",
                "runtime_files": []}
    for relative, content in prepared.items():
        manifest["runtime_files"].append({"path": relative, "before_sha256": BEFORE.get(relative),
                                          "after_sha256": checksum(content)})
    for relative, content in prepared.items():
        (site / relative).write_bytes(content)
    output.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-offline", action="store_true", required=True)
    args = parser.parse_args()
    site = Path(importlib.metadata.distribution("vllm").locate_file(""))
    install(site, ROOT / "installed-manifest.json")


if __name__ == "__main__":
    main()
