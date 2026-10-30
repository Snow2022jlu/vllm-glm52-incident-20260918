"""Install the archived Python overlay into an offline image build only."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--apply-offline", action="store_true", required=True)
    args = parser.parse_args()
    archive = Path(__file__).resolve().parent
    manifest = json.loads((archive / "manifest.json").read_text())
    if importlib.metadata.version("vllm").split("+")[0] != "0.28.0":
        raise SystemExit("Expected vLLM 0.28.0")
    if importlib.metadata.version("mooncake-transfer-engine-cuda13") != "0.3.13.post1":
        raise SystemExit("Expected Mooncake CUDA13 0.3.13.post1")
    site = Path(importlib.metadata.distribution("vllm").locate_file(""))
    prepared = []
    for relative, expected in manifest["runtime_files"].items():
        source = args.source_root / relative
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise SystemExit(f"Source mismatch: {relative}")
        compile(content, relative, "exec")
        prepared.append((source, site / relative))
    publisher = manifest["publisher"]
    source = args.source_root / publisher["path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != publisher["sha256"]:
        raise SystemExit("Publisher mismatch")
    for prepared_source, target in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prepared_source, target)
    shutil.copyfile(source, archive / "static_dp_registrar.py")
    print("Installed 12 historical vLLM files and the publisher; no services started.")


if __name__ == "__main__":
    main()
