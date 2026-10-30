"""Validate a local candidate without registry writes, GPU access or live Pods."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = "workspacegenaiacr.azurecr.io/lidong1/vllm-glm52-pd@sha256:4e241f7dd1af5ec766e21387e9f2dc991fe2753fa640cff904681260a843d76b"
PAYLOADS = ("observer.py", "install.py", "image_guard.py", "collector.py")


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Validation command failed ({result.returncode}): {result.stderr}")
    return result.stdout


def validate(image, output):
    baseline = json.loads(command(["docker", "image", "inspect", BASE]))[0]
    candidate = json.loads(command(["docker", "image", "inspect", image]))[0]
    layers = baseline["RootFS"]["Layers"]
    if candidate["RootFS"]["Layers"][:len(layers)] != layers:
        raise RuntimeError("Candidate does not preserve every baseline layer")
    script = """
python3 -B /opt/clean-pd/image_guard.py verify --output /tmp/guard.json > /tmp/guard-output
python3 -B - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('/opt/isolated-health-load')
guard = json.loads(Path('/tmp/guard.json').read_text())
payloads = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('observer.py', 'install.py', 'image_guard.py', 'collector.py')}
print(json.dumps({'guard': guard, 'payloads': payloads}))
PY
"""
    stdout = command(["docker", "run", "--rm", "--network", "none", "--read-only",
                      "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--cap-drop", "ALL",
                      "--env", "UV_CACHE_DIR=/tmp/uv",
                      "--security-opt", "no-new-privileges", "--cpus", "4", "--memory", "2g",
                      "--entrypoint", "/bin/sh", candidate["Id"], "-ec", script])
    verification = json.loads(stdout)
    expected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PAYLOADS}
    if verification["payloads"] != expected:
        raise RuntimeError("Image payload differs from the local tested candidate")
    added = next(entry for entry in verification["guard"]["observer_patch"]["runtime_files"]
                 if entry["before_sha256"] is None)
    if added["after_sha256"] != expected["observer.py"]:
        raise RuntimeError("Installed observer differs from its validated payload")
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    packages = verification["guard"]["packages"]
    summary = {
        "image_tag": image, "local_image_config_id": candidate["Id"], "base_image": BASE,
        "all_baseline_layers_preserved": True,
        "protected_files": verification["guard"]["protected_files"],
        "allowed_vllm_paths": list(verification["guard"]["allowed_vllm_changes"]),
        "all_image_payloads_match_local_source": True,
        "packages": {name: packages[name] for name in ("vllm", "torch", "mooncake-transfer-engine-cuda13", "nixl", "deep-ep", "deep-gemm") if name in packages},
        "gpu_tested": False, "pushed": False, "live_resources_changed": False,
        "dependency_check": verification["guard"]["dependency_check"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.image, args.output)


if __name__ == "__main__":
    main()
