"""Bounded health evidence and non-destructive auxiliary process supervision."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

if __package__:
    from . import isolated_observability as base
else:
    import isolated_observability as base


API_GRACE_SECONDS = 1800.0
FAILURE_FILE = "auxiliary-failure.json"
LOG = logging.getLogger(__name__)


def local_report(observer, protocol):
    report = base.ObservationState.report(observer)
    paths = list(observer.directory.glob("api-*.json"))
    samples = [observer.snapshot(path, math.inf) for path in paths]
    api_alive = len(paths) == observer.config["api_count"] and all(
        sample is not None and sample.get("ready") is True
        and path.name == f'api-{sample["pid"]}.json'
        for path, sample in zip(paths, samples)
    )
    remaining = min((API_GRACE_SECONDS - sample["age_seconds"] for sample in samples
                     if sample is not None), default=API_GRACE_SECONDS) if api_alive else 0.0
    auxiliary_failed = (observer.directory / FAILURE_FILE).exists()
    core_active = report["load_fresh"] and all(
        sample.get("ready") is True for sample in report["cores"].values()
    )
    healthy = bool(core_active and api_alive and remaining > 0 and not auxiliary_failed)
    responsive = bool(report["ready"] and healthy)
    if api_alive and remaining <= 0:
        report["reasons"].append("api_health_grace_expired")
    if auxiliary_failed:
        report["reasons"].append("auxiliary_process_failed")
    lifetime = min((observer.config["core_max_age"] - sample["age_seconds"]
                    for sample in report["cores"].values()), default=0.0)
    report.update(
        compatibility_protocol=protocol, health_ok=healthy, api_alive=bool(api_alive),
        api_responsive=responsive, ready=responsive,
        ready_valid_for_seconds=report["ready_valid_for_seconds"] if responsive else 0.0,
        health_state=("healthy" if responsive else "degraded") if healthy else "unavailable",
        load_valid_for_seconds=max(0.0, lifetime) if report["load_fresh"] else 0.0,
        api_health_max_age_seconds=API_GRACE_SECONDS,
        health_valid_for_seconds=max(0.0, min(remaining, lifetime)) if healthy else 0.0,
        auxiliary_failed=auxiliary_failed,
    )
    return report


class AuxiliarySupervisor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.children = []
        self.stopped = threading.Event()
        self.thread = None

    def record_failure(self, errors):
        LOG.error("Auxiliary observation failure; hold resources and use complete-session control: %s", errors)
        try:
            base.atomic_json(self.directory / FAILURE_FILE,
                             {"errors": errors, "recorded_at": time.monotonic()})
        except OSError:
            LOG.exception("Unable to persist auxiliary failure evidence")

    def check(self):
        errors = []
        for name, child in self.children:
            status = child.poll()
            if status is not None:
                errors.append(f"{name} pid={child.pid} exited with status={status}")
        if errors:
            self.record_failure(errors)
        return errors

    def monitor(self):
        while not self.stopped.wait(0.25):
            try:
                if self.check():
                    return
            except Exception as error:
                self.record_failure([f"auxiliary monitor failed: {error}"])
                return

    def start(self):
        if self.check():
            raise RuntimeError("Auxiliary process exited before model startup")
        self.thread = threading.Thread(target=self.monitor, daemon=True,
                                       name="vllm-observation-supervisor")
        self.thread.start()

    def close(self, timeout=5.0):
        self.stopped.set()
        errors = []
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                errors.append("auxiliary monitor did not stop")
        for name, child in reversed(self.children):
            try:
                if child.poll() is None:
                    child.terminate()
            except OSError as error:
                errors.append(f"{name} pid={child.pid} terminate failed: {error}")
        deadline = time.monotonic() + timeout
        for name, child in reversed(self.children):
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(f"{name} pid={child.pid} cleanup incomplete: {error}")
        if errors:
            self.record_failure(errors)
            raise RuntimeError(f"Auxiliary cleanup incomplete; retained evidence at {self.directory}: {errors}")


@contextlib.contextmanager
def observation_session(args):
    if __package__:
        from . import compatible_runtime as runtime
    else:
        import compatible_runtime as runtime

    flag = os.environ.get("VLLM_COMPAT_OBSERVABILITY", "0")
    if flag not in {"0", "1"}:
        raise ValueError("VLLM_COMPAT_OBSERVABILITY must be 0 or 1")
    if flag == "0":
        with base.observation_session(args):
            yield
        return
    if not base.enabled():
        raise ValueError("Compatibility requires the isolated EngineCore observation hooks")
    if os.environ.get("VLLM_OBSERVABILITY_DIRECTORY"):
        raise ValueError("Refusing to reuse another run's observation directory")
    original_host, original_port = args.host, args.port
    directory = tempfile.mkdtemp(prefix="vllm-bounded-", dir="/tmp")
    supervisor = AuxiliarySupervisor(directory)
    try:
        config = runtime.session_config(args, directory)
        base.atomic_json(Path(directory) / "config.json", config)
        os.environ["VLLM_OBSERVABILITY_DIRECTORY"] = directory
        for command, marker in (("local", "observer.json"), ("frontend", "frontend.json")):
            if command == "frontend" and not config["api_count"]:
                continue
            process = subprocess.Popen([sys.executable, str(Path(runtime.__file__).resolve()), command,
                                        "--directory", directory])
            supervisor.children.append((command, process))
            runtime.wait_process(process, Path(directory) / marker)
        supervisor.start()
        if config["api_count"]:
            args.host, args.port = "127.0.0.1", config["backend_port"]
        yield
    finally:
        args.host, args.port = original_host, original_port
        os.environ.pop("VLLM_OBSERVABILITY_DIRECTORY", None)
        supervisor.close()
        shutil.rmtree(directory)
