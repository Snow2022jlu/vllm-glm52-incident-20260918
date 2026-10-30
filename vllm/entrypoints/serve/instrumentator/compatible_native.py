"""Export the API's original registry off its asyncio loop, preserving collectors."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from prometheus_client import generate_latest

if __package__:
    from . import isolated_observability as base
else:
    import isolated_observability as base


MAX_METRICS = 8 * 1024 * 1024
LOG = logging.getLogger(__name__)


def write_native(directory, config, registry):
    pid = os.getpid()
    header = {"epoch": config["epoch"], "run_id": config["run_id"], "pid": pid,
              "start": base.process_identity(pid), "sampled_at": time.monotonic()}
    payload = generate_latest(registry)
    if len(payload) > MAX_METRICS:
        raise ValueError("Native registry exceeds the bounded snapshot size")
    content = json.dumps(header, allow_nan=False).encode() + b"\n" + payload
    target = Path(directory) / f"native-{pid}.prom"
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, target)


def export_loop(server, directory, config, registry):
    while not server.should_exit:
        try:
            candidates = [int(path.stem.split("-")[1]) for path in Path(directory).glob("api-*.json")]
            live = [pid for pid in candidates if base.process_identity(pid) is not None]
            if not live or os.getpid() == min(live):
                write_native(directory, config, registry)
        except Exception:
            LOG.exception("Native registry snapshot failed; source age will remain visible")
        time.sleep(0.5)


def start_api_pulse(server, engine):
    pulse = base.start_api_pulse(server, engine)
    if os.environ.get("VLLM_COMPAT_OBSERVABILITY", "0") == "1":
        from vllm.v1.metrics.prometheus import get_prometheus_registry

        directory = os.environ["VLLM_OBSERVABILITY_DIRECTORY"]
        config = base.read_json(Path(directory) / "config.json")
        threading.Thread(target=export_loop, args=(server, directory, config, get_prometheus_registry()),
                         daemon=True, name="vllm-native-metric-snapshot").start()
    return pulse
