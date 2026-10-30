"""Extend the original provenance check with an exact, isolated-observer allowlist."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATCH_ROOT = Path("/opt/isolated-health-load")


def load_baseline_guard():
    spec = importlib.util.spec_from_file_location("baseline_image_guard", ROOT / "baseline_image_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(runtime=False):
    baseline = load_baseline_guard()
    manifest = json.loads((PATCH_ROOT / "installed-manifest.json").read_text())
    expected_paths = {
        "vllm/v1/engine/core.py", "vllm/entrypoints/launcher.py",
        "vllm/entrypoints/cli/serve.py",
        "vllm/entrypoints/serve/instrumentator/isolated_observability.py",
    }
    if {entry["path"] for entry in manifest["runtime_files"]} != expected_paths:
        raise RuntimeError("Isolated-observer file allowlist changed")
    original_changes = baseline.expected_changes
    extra = {str((baseline.site_root() / entry["path"]).resolve()): entry["after_sha256"]
             for entry in manifest["runtime_files"]}
    additions = {str((baseline.site_root() / entry["path"]).resolve())
                 for entry in manifest["runtime_files"] if entry["before_sha256"] is None}

    def expected_changes(p12_manifest, site):
        return original_changes(p12_manifest, site) | extra

    def verify_files(before, after, allowed):
        if set(after) != set(before) | additions:
            raise RuntimeError("Protected package inventory differs from the exact candidate contract")
        changes = {path: value for path, value in after.items() if before.get(path) != value}
        if changes != allowed:
            raise RuntimeError("Protected files differ from P12 plus the exact observer allowlist")

    baseline.expected_changes = expected_changes
    baseline.verify_files = verify_files
    report = baseline.verify(runtime)
    report["observer_patch"] = manifest
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
