from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parent
RUNTIME_PATHS = (
    "vllm/v1/request.py",
    "vllm/v1/metrics/stats.py",
    "vllm/v1/metrics/loggers.py",
    "vllm/v1/core/sched/scheduler.py",
)


@dataclass(frozen=True)
class PreparedFile:
    component: str
    target: Path
    original: bytes
    payload: bytes


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_path(path: Path, directory: bool = False) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute.resolve(strict=True) != absolute:
        raise RuntimeError(f"refusing symlink path: {absolute}")
    if not (absolute.is_dir() if directory else absolute.is_file()):
        raise RuntimeError(f"not a {'directory' if directory else 'regular file'}: {absolute}")
    return absolute


def installed_version(site_packages_root: Path) -> tuple[str, Path, bytes]:
    matches = []
    for metadata in sorted(site_packages_root.glob("vllm-*.dist-info/METADATA")):
        metadata = checked_path(metadata)
        data = metadata.read_bytes()
        fields = BytesParser().parsebytes(data)
        if fields.get("Name", "").strip().lower() == "vllm":
            matches.append((fields.get("Version", "").strip(), metadata, data))
    if len(matches) != 1:
        raise RuntimeError("expected exactly one vLLM wheel METADATA under site-packages")
    version, metadata, data = matches[0]
    if not re.fullmatch(r"0\.28\.0(?:\+[A-Za-z0-9_.-]+)?", version):
        raise RuntimeError(f"P12 requires vLLM 0.28.0, not {version!r}")
    return version, metadata, data


def read_payload(bundle_root: Path, record: dict) -> bytes:
    relative = Path(record["payload"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("payload",):
        raise RuntimeError("invalid payload path in manifest")
    payload = checked_path(bundle_root / relative).read_bytes()
    if digest(payload) != record["after_sha256"]:
        raise RuntimeError(f"corrupt payload: {relative}")
    ast.parse(payload, filename=str(relative), feature_version=(3, 12))
    return payload


def prepare(
    site_packages_root: Path | None,
    collector_file: Path | None,
    component: str,
    bundle_root: Path = ROOT,
) -> tuple[list[PreparedFile], list[tuple[Path, bytes]], str | None]:
    if component not in ("all", "vllm", "collector"):
        raise ValueError(f"unknown component: {component}")
    wants_vllm = component in ("all", "vllm")
    wants_collector = component in ("all", "collector")
    if wants_vllm != (site_packages_root is not None) or wants_collector != (collector_file is not None):
        raise ValueError("all requires both paths; component-specific mode requires only its corresponding path")
    bundle_root = checked_path(bundle_root, directory=True)
    manifest = json.loads(checked_path(bundle_root / "manifest.json").read_text())
    if manifest["candidate"] != "p12-prefill-pressure-vllm0280" or manifest["base_version"] != "0.28.0":
        raise RuntimeError("unexpected candidate manifest")
    records = manifest["runtime_files"]
    if len(records) != len(RUNTIME_PATHS) or {record["path"] for record in records} != set(RUNTIME_PATHS):
        raise RuntimeError("unexpected runtime write scope")
    selected = []
    guards = []
    version = None
    if wants_vllm:
        root = checked_path(site_packages_root, directory=True)
        version, metadata, data = installed_version(root)
        guards.append((metadata, data))
        by_path = {record["path"]: record for record in records}
        selected.extend(("vllm", root / relative, by_path[relative]) for relative in RUNTIME_PATHS)
    if wants_collector:
        selected.append(("collector", collector_file, manifest["collector"]))
    prepared = []
    targets = set()
    for selected_component, target, record in selected:
        target = checked_path(target)
        if target in targets:
            raise RuntimeError(f"duplicate write target: {target}")
        targets.add(target)
        payload = read_payload(bundle_root, record)
        current = target.read_bytes()
        if digest(current) not in (record["before_sha256"], record["after_sha256"]):
            raise RuntimeError(f"refusing unknown {selected_component} source: {target} sha256={digest(current)}")
        prepared.append(PreparedFile(selected_component, target, current, payload))
    return prepared, guards, version


def atomic_replace(target: Path, data: bytes) -> None:
    mode = checked_path(target).stat().st_mode & 0o777
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".p12-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply(
    site_packages_root: Path | None = None,
    collector_file: Path | None = None,
    component: str = "all",
    apply_offline: bool = False,
    verify_installed: bool = False,
    bundle_root: Path = ROOT,
) -> dict:
    if apply_offline and verify_installed:
        raise ValueError("verify-installed cannot be combined with apply-offline")
    prepared, guards, version = prepare(site_packages_root, collector_file, component, bundle_root)
    if verify_installed and any(item.original != item.payload for item in prepared):
        raise RuntimeError("selected components are not completely patched")
    updated = []
    results = []
    try:
        for item in prepared:
            if checked_path(item.target).read_bytes() != item.original:
                raise RuntimeError(f"target changed after preflight: {item.target}")
            for guard, original in guards:
                if checked_path(guard).read_bytes() != original:
                    raise RuntimeError(f"metadata changed after preflight: {guard}")
            status = "already_applied" if item.original == item.payload else "would_apply"
            if apply_offline and item.original != item.payload:
                updated.append(item)
                atomic_replace(item.target, item.payload)
                status = "applied"
            if verify_installed:
                status = "verified"
            results.append({"component": item.component, "path": str(item.target), "status": status, "before_sha256": digest(item.original), "after_sha256": digest(item.payload)})
        if apply_offline:
            for item in prepared:
                if checked_path(item.target).read_bytes() != item.payload:
                    raise RuntimeError(f"post-write mismatch: {item.target}")
            for guard, original in guards:
                if checked_path(guard).read_bytes() != original:
                    raise RuntimeError(f"metadata changed after write: {guard}")
    except BaseException as error:
        failures = []
        for item in reversed(updated):
            try:
                current = checked_path(item.target).read_bytes()
                if current not in (item.original, item.payload):
                    raise RuntimeError("unexpected concurrent content; refusing to overwrite")
                if current != item.original:
                    atomic_replace(item.target, item.original)
            except BaseException as restore_error:
                failures.append(f"{item.target}: {restore_error}")
        if failures:
            raise RuntimeError("offline rollback incomplete; rebuild or inspect: " + "; ".join(failures)) from error
        raise
    return {"candidate": "p12-prefill-pressure-vllm0280", "component": component, "vllm_version": version, "offline_write_requested": apply_offline, "files": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="P12 offline candidate; dry-run by default. Never patch running workers or shared ConfigMaps.")
    parser.add_argument("--component", choices=("all", "vllm", "collector"), default="all")
    parser.add_argument("--site-packages-root", type=Path)
    parser.add_argument("--collector-file", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply-offline", action="store_true")
    mode.add_argument("--verify-installed", action="store_true")
    arguments = parser.parse_args()
    try:
        result = apply(**vars(arguments))
    except (ValueError, RuntimeError, OSError, SyntaxError) as error:
        parser.exit(1, f"P12: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
