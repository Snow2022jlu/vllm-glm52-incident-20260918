"""Cumulative exact allowlist; original Docker native libraries remain protected."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path("/opt/clean-pd")
PATCH_ROOT = Path("/opt/bounded-health-metrics")
PREFIX = "vllm/entrypoints/serve/instrumentator/"


def verify(runtime=False):
    spec = importlib.util.spec_from_file_location("p12_guard", ROOT / "baseline_image_guard.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    paths = [Path("/opt/isolated-health-load/installed-manifest.json"),
             Path("/opt/compatible-health-metrics/installed-manifest.json"),
             PATCH_ROOT / "installed-manifest.json"]
    manifests = [json.loads(path.read_text()) for path in paths]
    expected_paths = [
        {"vllm/v1/engine/core.py", "vllm/entrypoints/launcher.py", "vllm/entrypoints/cli/serve.py",
         PREFIX + "isolated_observability.py"},
        {"vllm/entrypoints/launcher.py", "vllm/entrypoints/cli/serve.py"} | {
            PREFIX + name for name in ("compatible_state.py", "compatible_runtime.py", "compatible_native.py")},
        {PREFIX + name for name in ("compatible_state.py", "compatible_runtime.py", "bounded_observability.py")},
    ]
    entries, additions = {}, set()
    for manifest, expected in zip(manifests, expected_paths):
        if {entry["path"] for entry in manifest["runtime_files"]} != expected:
            raise RuntimeError("Observation patch allowlist changed")
        for entry in manifest["runtime_files"]:
            entries[entry["path"]] = entry
            if entry["before_sha256"] is None:
                additions.add(str((baseline.site_root() / entry["path"]).resolve()))
    extra = {str((baseline.site_root() / path).resolve()): entry["after_sha256"]
             for path, entry in entries.items()}
    original_changes = baseline.expected_changes
    baseline.expected_changes = lambda p12, site: original_changes(p12, site) | extra

    def verify_files(before, after, allowed):
        if set(after) != set(before) | additions:
            raise RuntimeError("Protected file inventory differs from the exact allowlist")
        if {path: value for path, value in after.items() if before.get(path) != value} != allowed:
            raise RuntimeError("Unapproved protected file change")

    baseline.verify_files = verify_files
    publisher = manifests[-1]["publisher"]
    if publisher["path"] != "/opt/compatible-health-metrics/static_dp_registrar.py":
        raise RuntimeError("Publisher path changed")
    for path in (Path(publisher["path"]), PATCH_ROOT / "static_dp_registrar.py"):
        if hashlib.sha256(path.read_bytes()).hexdigest() != publisher["after_sha256"]:
            raise RuntimeError("Publisher differs from the bounded candidate manifest")
    report = baseline.verify(runtime)
    report.update(observer_patch=manifests[0], compatibility_patch=manifests[1], bounded_patch=manifests[2])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["verify", "runtime"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(args.operation == "runtime"), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result)
