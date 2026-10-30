"""Offline exact-hash extension of the sealed obs-r2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREFIX = "vllm/entrypoints/serve/instrumentator/"
MODULES = ("compatible_state.py", "compatible_runtime.py", "compatible_native.py")
BEFORE = {
    "vllm/entrypoints/cli/serve.py": "62058475083a09bec205d55b6351e0239e76b3120c16057318ef7853395bd5b7",
    "vllm/entrypoints/launcher.py": "557b7b608644193ad0f04018bf16d2b2814596df50b82724b0c32478f46d324c",
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def transform(path, source):
    if path.endswith("cli/serve.py"):
        anchor = "from vllm.entrypoints.serve.instrumentator.isolated_observability import enabled as observation_enabled, observation_session\n"
        replacement = ("from vllm.entrypoints.serve.instrumentator.isolated_observability import enabled as observation_enabled\n"
                       "from vllm.entrypoints.serve.instrumentator.compatible_runtime import observation_session\n")
    elif path.endswith("launcher.py"):
        anchor = "from vllm.entrypoints.serve.instrumentator.isolated_observability import start_api_pulse\n"
        replacement = "from vllm.entrypoints.serve.instrumentator.compatible_native import start_api_pulse\n"
    else:
        raise ValueError("Unapproved hook")
    if source.count(anchor) != 1:
        raise ValueError("Pinned hook anchor changed")
    source = source.replace(anchor, replacement, 1)
    compile(source, path, "exec")
    return source


def install():
    expected_versions = {"vllm": "0.28.0", "aiohttp": "3.14.3", "prometheus-client": "0.24.1"}
    if any(importlib.metadata.version(name) != version for name, version in expected_versions.items()):
        raise ValueError("The pinned vLLM / HTTP / metric dependencies changed")
    site = Path(importlib.metadata.distribution("vllm").locate_file(""))
    prepared = {}
    for path, expected in BEFORE.items():
        content = (site / path).read_bytes()
        if digest(content) != expected:
            raise ValueError(f"Pinned obs-r2 source mismatch: {path}")
        prepared[path] = transform(path, content.decode()).encode()
    for name in MODULES:
        path = PREFIX + name
        if (site / path).exists():
            raise ValueError("Refusing to overwrite another compatibility candidate")
        content = (ROOT / name).read_bytes()
        compile(content, path, "exec")
        prepared[path] = content
    manifest = {"parent_local_image_config_id": "sha256:ab4bdeb8ed7ff802706cbe6ea0fcd64621eb7d20d7e2beedb578a56d6f8ffb9d",
                "runtime_files": [{"path": path, "before_sha256": BEFORE.get(path), "after_sha256": digest(content)}
                                  for path, content in prepared.items()]}
    for path, content in prepared.items():
        (site / path).write_bytes(content)
    (ROOT / "installed-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-offline", action="store_true", required=True)
    parser.parse_args()
    install()
