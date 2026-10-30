"""Preserve the official native inventory and exact cumulative Python allowlist."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path("/opt/clean-pd")
PATCH_ROOT = Path("/opt/compatible-health-metrics")


def verify(runtime=False):
    spec = importlib.util.spec_from_file_location("p12_guard", ROOT / "baseline_image_guard.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    previous = json.loads(Path("/opt/isolated-health-load/installed-manifest.json").read_text())
    manifest = json.loads((PATCH_ROOT / "installed-manifest.json").read_text())
    old_paths = {"vllm/v1/engine/core.py", "vllm/entrypoints/launcher.py", "vllm/entrypoints/cli/serve.py",
                 "vllm/entrypoints/serve/instrumentator/isolated_observability.py"}
    new_paths = {"vllm/entrypoints/launcher.py", "vllm/entrypoints/cli/serve.py"} | {
        "vllm/entrypoints/serve/instrumentator/" + name
        for name in ("compatible_state.py", "compatible_runtime.py", "compatible_native.py")
    }
    if ({entry["path"] for entry in previous["runtime_files"]} != old_paths
            or {entry["path"] for entry in manifest["runtime_files"]} != new_paths):
        raise RuntimeError("Compatibility file allowlist changed")
    entries = {entry["path"]: entry for entry in previous["runtime_files"]}
    entries.update({entry["path"]: entry for entry in manifest["runtime_files"]})
    extra = {str((baseline.site_root() / path).resolve()): entry["after_sha256"] for path, entry in entries.items()}
    additions = {str((baseline.site_root() / entry["path"]).resolve())
                 for patch in (previous, manifest) for entry in patch["runtime_files"] if entry["before_sha256"] is None}
    original_changes = baseline.expected_changes
    baseline.expected_changes = lambda p12, site: original_changes(p12, site) | extra

    def verify_files(before, after, allowed):
        if set(after) != set(before) | additions:
            raise RuntimeError("Native/Python file inventory differs from the sealed allowlist")
        if {path: value for path, value in after.items() if before.get(path) != value} != allowed:
            raise RuntimeError("Unapproved file content change")

    baseline.verify_files = verify_files
    report = baseline.verify(runtime)
    report["observer_patch"] = previous
    report["compatibility_patch"] = manifest
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["verify", "runtime"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.operation == "runtime"), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
