"""Offline exact-hash delta on the immutable compat-r3 Python payload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREFIX = "vllm/entrypoints/serve/instrumentator/"
MODULE = PREFIX + "bounded_observability.py"
PUBLISHER = Path("/opt/compatible-health-metrics/static_dp_registrar.py")
BEFORE = {
    PREFIX + "compatible_state.py": "c483ac6f7ab7e6bd98385662f447f9a657e9cd6d70fe93c97c97247e5e9a728b",
    PREFIX + "compatible_runtime.py": "24dd7cb20f874b8e6a7223593cf9d29843f974e04bad4d49f621d1efd5400621",
}
PUBLISHER_BEFORE = "75ced3e9544379e39981b80f1162bd710e753ad72bcd6c3eb4d3beddcae1005b"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Pinned source anchor changed: {old[:80]!r}")
    return source.replace(old, new, 1)


def transform_state(source):
    source = replace(source, "    from . import isolated_observability as base\n",
                     "    from . import isolated_observability as base\n    from . import bounded_observability as bounded\n")
    source = replace(source, "    import isolated_observability as base\n",
                     "    import isolated_observability as base\n    import bounded_observability as bounded\n")
    source = replace(source, 'PROTOCOL = "vllm-compatible-observability-v1"',
                     'PROTOCOL = "vllm-compatible-observability-v2"')
    start = source.index("class LocalState(base.ObservationState):\n")
    end = source.index("\ndef validate_peers(", start)
    source = source[:start] + ("class LocalState(base.ObservationState):\n"
                              "    def report(self):\n"
                              "        return bounded.local_report(self, PROTOCOL)\n\n") + source[end:]
    source = replace(source, '            health = report.get("health_ok") is True\n',
                     '            health = report.get("health_ok") is True\n'
                     '            health_remaining = report.get("health_valid_for_seconds")\n'
                     '            if (report.get("api_health_max_age_seconds") != bounded.API_GRACE_SECONDS\n'
                     '                    or not finite_age(health_remaining, bounded.API_GRACE_SECONDS)\n'
                     '                    or health_remaining > remaining\n'
                     '                    or health and health_remaining <= 0):\n'
                     '                raise ValueError("Missing or invalid bounded health deadline")\n')
    source = replace(source, '            self.samples[index] = (report, deadline, min(deadline, started + api_remaining))',
                     '            self.samples[index] = (report, deadline, min(deadline, started + api_remaining),\n'
                     '                                   min(deadline, started + health_remaining))')
    source = replace(source, '"api_deadline": now, "cores": {}}',
                     '"api_deadline": now, "health_deadline": now, "cores": {}}')
    source = replace(source, '        healthy = all(report["health_ok"] for report in reports)',
                     '        healthy = all(item[0]["health_ok"] and item[3] > now for item in current)')
    source = replace(source, '        responsive = all(item[0]["api_responsive"] and item[2] > now for item in current)',
                     '        responsive = healthy and all(item[0]["api_responsive"] and item[2] > now for item in current)')
    source = replace(source, '                "api_deadline": min(item[2] for item in current) if responsive else now,',
                     '                "api_deadline": min(item[2] for item in current) if responsive else now,\n'
                     '                "health_deadline": min(item[3] for item in current) if healthy else now,')
    return source


def transform_runtime(source):
    source = replace(source, "    from . import compatible_state as state\n",
                     "    from . import compatible_state as state\n    from . import bounded_observability as bounded\n")
    source = replace(source, "    import compatible_state as state\n",
                     "    import compatible_state as state\n    import bounded_observability as bounded\n")
    source = replace(source, '            "X-VLLM-API-Valid-For": str(max(0.0, view["api_deadline"] - now)),',
                     '            "X-VLLM-API-Valid-For": str(max(0.0, view["api_deadline"] - now)),\n'
                     '            "X-VLLM-Health-Valid-For": str(max(0.0, min(view["health_deadline"], cached[1], view["deadline"]) - now)),')
    start = source.index("@contextlib.contextmanager\ndef observation_session(args):\n")
    end = source.index("\ndef main():", start)
    return source[:start] + ("@contextlib.contextmanager\ndef observation_session(args):\n"
                             "    with bounded.observation_session(args):\n"
                             "        yield\n\n") + source[end:]


def transform_publisher(source):
    source = replace(source, '    native_age = float(headers["X-VLLM-Native-Snapshot-Age"])\n',
                     '    native_age = float(headers["X-VLLM-Native-Snapshot-Age"])\n'
                     '    health_state = headers.get("X-VLLM-Health-State")\n'
                     '    health_lifetime = float(headers["X-VLLM-Health-Valid-For"])\n'
                     '    if (health_state not in {"healthy", "degraded", "unavailable"}\n'
                     '            or not math.isfinite(health_lifetime) or not 0 <= health_lifetime <= 300\n'
                     '            or health_lifetime > lifetime\n'
                     '            or health_state != "unavailable" and health_lifetime <= 0):\n'
                     '        raise ValueError("Invalid bounded health evidence")\n')
    source = replace(source, '    return {"epoch": epoch, "deadline": deadline, "fields": fields,',
                     '    return {"epoch": epoch, "deadline": deadline, "fields": fields,\n'
                     '            "health_ok": health_state in {"healthy", "degraded"},\n'
                     '            "health_deadline": started + health_lifetime,')
    source = replace(source, '    body = {"ready": ready and set(fresh) == {"prefill", "decode"}}',
                     '    body = {"ready": bool(ready and set(fresh) == {"prefill", "decode"}\n'
                     '                          and all(sample.get("health_ok") is True\n'
                     '                                  and sample.get("health_deadline", 0) > now\n'
                     '                                  for sample in fresh.values()))}')
    return source


def install():
    expected = {"vllm": "0.28.0", "aiohttp": "3.14.3", "prometheus-client": "0.24.1"}
    if any(importlib.metadata.version(name) != version for name, version in expected.items()):
        raise ValueError("Pinned runtime dependencies changed")
    site = Path(importlib.metadata.distribution("vllm").locate_file(""))
    transforms = {PREFIX + "compatible_state.py": transform_state,
                  PREFIX + "compatible_runtime.py": transform_runtime}
    prepared = {}
    for relative, before in BEFORE.items():
        content = (site / relative).read_bytes()
        if digest(content) != before:
            raise ValueError(f"Expected sealed compat-r3 source: {relative}")
        prepared[relative] = transforms[relative](content.decode()).encode()
    if (site / MODULE).exists():
        raise ValueError("Refusing to overwrite another bounded candidate")
    prepared[MODULE] = (ROOT / "bounded_observability.py").read_bytes()
    publisher = PUBLISHER.read_bytes()
    if digest(publisher) != PUBLISHER_BEFORE:
        raise ValueError("Expected sealed compat-r3 publisher")
    publisher = transform_publisher(publisher.decode()).encode()
    for path, content in prepared.items():
        compile(content, path, "exec")
    compile(publisher, str(PUBLISHER), "exec")
    manifest = {
        "parent_local_image_config_id": "sha256:069f0f5021e8bd435b0b022e81021d0678abcc29bb29311f278228cf08104032",
        "api_grace_seconds": 1800, "h02_fixed": False,
        "runtime_files": [{"path": path, "before_sha256": BEFORE.get(path), "after_sha256": digest(content)}
                          for path, content in prepared.items()],
        "publisher": {"path": str(PUBLISHER), "before_sha256": PUBLISHER_BEFORE,
                      "after_sha256": digest(publisher)},
    }
    for path, content in prepared.items():
        (site / path).write_bytes(content)
    PUBLISHER.write_bytes(publisher)
    (ROOT / "static_dp_registrar.py").write_bytes(publisher)
    (ROOT / "installed-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-offline", action="store_true", required=True)
    parser.parse_args()
    install()
