"""Independently verify a local candidate; no registry writes or GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT_ID = "sha256:ab4bdeb8ed7ff802706cbe6ea0fcd64621eb7d20d7e2beedb578a56d6f8ffb9d"
PAYLOADS = ("compatible_state.py", "compatible_runtime.py", "compatible_native.py",
            "static_dp_registrar.py", "install.py", "image_guard.py")


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Verification command failed: {result.stderr}")
    return result.stdout


def verify(image, output):
    parent = json.loads(command(["docker", "image", "inspect", PARENT_ID]))[0]
    candidate = json.loads(command(["docker", "image", "inspect", image]))[0]
    layers = parent["RootFS"]["Layers"]
    if candidate["RootFS"]["Layers"][:len(layers)] != layers:
        raise RuntimeError("A parent layer was replaced")
    script = """
python3 -B /opt/clean-pd/image_guard.py verify --output /tmp/guard.json > /tmp/guard-stdout
python3 -B - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('/opt/compatible-health-metrics')
names = ('compatible_state.py', 'compatible_runtime.py', 'compatible_native.py',
         'static_dp_registrar.py', 'install.py', 'image_guard.py')
payloads = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
print(json.dumps({'guard': json.loads(Path('/tmp/guard.json').read_text()), 'payloads': payloads}))
PY
"""
    result = command(["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                      "--security-opt", "no-new-privileges", "--cpus", "4", "--memory", "2g",
                      "--env", "UV_CACHE_DIR=/tmp/uv", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                      "--entrypoint", "/bin/sh", candidate["Id"], "-ec", script])
    verification = json.loads(result)
    expected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PAYLOADS}
    if verification["payloads"] != expected:
        raise RuntimeError("Image payloads do not match the local tested source")
    manifest = verification["guard"]["compatibility_patch"]
    for entry in manifest["runtime_files"]:
        if entry["before_sha256"] is None and entry["after_sha256"] != expected[Path(entry["path"]).name]:
            raise RuntimeError("Installed compatibility module differs from its source payload")
    summary = {
        "image_tag": image, "local_image_config_id": candidate["Id"], "parent_local_image_config_id": PARENT_ID,
        "all_parent_layers_preserved": True, "protected_files": verification["guard"]["protected_files"],
        "all_payloads_match_local_source": True, "compatibility_runtime_files": manifest["runtime_files"],
        "packages": verification["guard"]["packages"], "dependency_check": verification["guard"]["dependency_check"],
        "pushed": False, "live_resources_changed": False, "gpu_tested": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"packages", "compatibility_runtime_files"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.image, args.output)
