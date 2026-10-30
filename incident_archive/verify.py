"""Verify historical source hashes and optionally replay the four patches."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, expected):
    for relative, checksum in expected.items():
        path = root / relative
        if digest(path) != checksum:
            raise SystemExit(f"Hash mismatch: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=relative, feature_version=(3, 12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "incident_archive/manifest.json").read_text())
    verify(ROOT, manifest["runtime_files"])
    verify(ROOT, manifest["incident_source_files"])
    publisher = manifest["publisher"]
    verify(ROOT, {publisher["path"]: publisher["sha256"]})
    for patch in manifest["patch_series"]:
        verify(ROOT, {patch["patch"]: patch["sha256"]})
    if args.replay:
        with tempfile.TemporaryDirectory(prefix="vllm-incident-replay-") as temp:
            replay = Path(temp)
            for relative in manifest["runtime_files"]:
                result = subprocess.run(
                    ["git", "show", f"upstream-v0.28.0:{relative}"],
                    cwd=ROOT, capture_output=True, check=False,
                )
                if result.returncode == 0:
                    path = replay / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(result.stdout)
            for patch in manifest["patch_series"]:
                subprocess.run(
                    ["git", "apply", str(ROOT / patch["patch"])],
                    cwd=replay, check=True,
                )
            verify(replay, manifest["runtime_files"])
    print(json.dumps({
        "runtime_files": len(manifest["runtime_files"]),
        "incident_source_files": len(manifest["incident_source_files"]),
        "publisher": "verified",
        "patches": len(manifest["patch_series"]),
        "patch_replay": args.replay,
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
