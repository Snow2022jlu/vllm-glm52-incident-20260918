"""Read-only local image verification, including deterministic source deltas."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT_ID = "sha256:069f0f5021e8bd435b0b022e81021d0678abcc29bb29311f278228cf08104032"
PAYLOADS = ("bounded_observability.py", "install.py", "image_guard.py")


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Local verification failed: {result.stderr}")
    return result.stdout


def verify(image, output):
    parent = json.loads(command(["docker", "image", "inspect", PARENT_ID]))[0]
    candidate = json.loads(command(["docker", "image", "inspect", image]))[0]
    layers = parent["RootFS"]["Layers"]
    if candidate["RootFS"]["Layers"][:len(layers)] != layers:
        raise RuntimeError("Immutable parent layer prefix changed")
    environment = dict(entry.split("=", 1) for entry in candidate["Config"]["Env"])
    if any(environment.get(name) != "0" for name in ("VLLM_COMPAT_OBSERVABILITY", "VLLM_ISOLATED_OBSERVABILITY")):
        raise RuntimeError("Observation must remain opt-in")
    script = """
python3 -B /opt/clean-pd/image_guard.py verify --output /tmp/guard.json > /tmp/guard-stdout
python3 -B - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('/opt/bounded-health-metrics')
names = ('bounded_observability.py', 'install.py', 'image_guard.py')
print(json.dumps({'guard': json.loads(Path('/tmp/guard.json').read_text()),
                  'payloads': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}))
PY
"""
    result = command(["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                      "--security-opt", "no-new-privileges", "--cpus", "4", "--memory", "2g", "--pids-limit", "256",
                      "--env", "UV_CACHE_DIR=/tmp/uv", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                      "--entrypoint", "/bin/sh", candidate["Id"], "-ec", script])
    verification = json.loads(result)
    expected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PAYLOADS}
    if verification["payloads"] != expected:
        raise RuntimeError("Image payloads differ from local source")
    spec = importlib.util.spec_from_file_location("bounded_install", ROOT / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    parent_source = ROOT.parent / "compatible-health-metrics-vllm0280"
    expected_runtime = {installer.MODULE: expected["bounded_observability.py"]}
    for name, transform in (("compatible_state.py", installer.transform_state),
                            ("compatible_runtime.py", installer.transform_runtime)):
        source = (parent_source / name).read_bytes()
        if installer.digest(source) != installer.BEFORE[installer.PREFIX + name]:
            raise RuntimeError("Local immutable source changed")
        expected_runtime[installer.PREFIX + name] = installer.digest(transform(source.decode()).encode())
    publisher = (parent_source / "static_dp_registrar.py").read_bytes()
    if installer.digest(publisher) != installer.PUBLISHER_BEFORE:
        raise RuntimeError("Local immutable publisher changed")
    manifest = verification["guard"]["bounded_patch"]
    if {entry["path"]: entry["after_sha256"] for entry in manifest["runtime_files"]} != expected_runtime:
        raise RuntimeError("Installed delta differs from the deterministic local transform")
    if manifest["publisher"]["after_sha256"] != installer.digest(installer.transform_publisher(publisher.decode()).encode()):
        raise RuntimeError("Installed publisher differs from the local delta")
    summary = {
        "local_image_config_id": candidate["Id"], "parent_local_image_config_id": PARENT_ID,
        "all_parent_layers_preserved": True, "all_payloads_match_local_source": True,
        "all_runtime_deltas_match_local_source": True, "protected_files": verification["guard"]["protected_files"],
        "api_grace_seconds": 1800, "h02_fixed": False, "pushed": False,
        "live_resources_changed": False, "gpu_tested": False, "promotion_allowed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.image, args.output)
