"""Compatibility health policy and exact-world metric aggregation."""

from __future__ import annotations

import math
import re
import time
from urllib.parse import urlsplit

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.parser import text_string_to_metric_families

if __package__:
    from . import isolated_observability as base
else:
    import isolated_observability as base


PROTOCOL = "vllm-compatible-observability-v1"
MAX_REPORT = 256 * 1024
MAX_METRICS = 8 * 1024 * 1024
GAUGES = {value: key for key, value in base.METRICS.items()
          if key not in {"generation_tokens", "preemptions"}}
WAITING = "vllm:num_requests_waiting_by_reason"


class LocalState(base.ObservationState):
    def report(self):
        report = super().report()
        paths = list(self.directory.glob("api-*.json"))
        samples = [self.snapshot(path, math.inf) for path in paths]
        api_alive = len(paths) == self.config["api_count"] and all(
            sample is not None and sample.get("ready") is True
            and path.name == f'api-{sample["pid"]}.json'
            for path, sample in zip(paths, samples)
        )
        core_active = report["load_fresh"] and all(
            sample.get("ready") is True for sample in report["cores"].values()
        )
        healthy = core_active and api_alive
        lifetime = min((self.config["core_max_age"] - sample["age_seconds"]
                        for sample in report["cores"].values()), default=0.0)
        report.update(
            compatibility_protocol=PROTOCOL,
            health_ok=bool(healthy),
            api_alive=bool(api_alive),
            api_responsive=bool(report["ready"]),
            health_state=("healthy" if report["ready"] else "degraded") if healthy else "unavailable",
            load_valid_for_seconds=max(0.0, lifetime) if report["load_fresh"] else 0.0,
        )
        return report


def validate_peers(peers, world, local_ranks, local_api_count):
    if type(world) is not int or not 1 <= world <= 64:
        raise ValueError("A fixed DP world of 1..64 is required")
    if not isinstance(peers, list) or not 1 <= len(peers) <= 8:
        raise ValueError("Provide the entire fixed engine's observer roster")
    seen_ranks, seen_urls = set(), set()
    local_count = 0
    for peer in peers:
        if not isinstance(peer, dict) or set(peer) != {"url", "ranks", "api_count"}:
            raise ValueError("Peer fields must be url, ranks, api_count")
        address = urlsplit(peer["url"])
        if (address.scheme != "http" or not address.hostname or not address.port
                or address.username or address.password or address.query or address.fragment
                or address.path not in {"", "/"}):
            raise ValueError("Use explicit private HTTP observer origins without credentials")
        ranks = peer["ranks"]
        if (not isinstance(ranks, list) or not ranks
                or any(type(rank) is not int or not 0 <= rank < world for rank in ranks)
                or ranks != sorted(set(ranks)) or seen_ranks.intersection(ranks)):
            raise ValueError("Missing, duplicate or out-of-world ranks")
        if type(peer["api_count"]) is not int or not 0 <= peer["api_count"] <= 64:
            raise ValueError("Invalid expected API process count")
        origin = peer["url"].rstrip("/")
        if origin in seen_urls:
            raise ValueError("Duplicate observer origin")
        seen_urls.add(origin)
        seen_ranks.update(ranks)
        if ranks == local_ranks:
            local_count += 1
            if peer["api_count"] != local_api_count:
                raise ValueError("Local API count does not match the sealed roster")
    if seen_ranks != set(range(world)) or local_count != 1:
        raise ValueError("Roster must cover the entire engine and exactly one local member")
    if sum(peer["api_count"] > 0 for peer in peers) != 1:
        raise ValueError("This candidate requires exactly one API head per engine")
    return peers


def finite_age(value, maximum=300):
    return type(value) in {float, int} and math.isfinite(value) and 0 <= value <= maximum


class GroupState:
    def __init__(self, config):
        self.config = config
        self.samples = {}
        self.identities = {}
        self.rejected = set()

    def accept(self, index, report, started, received=None):
        received = time.monotonic() if received is None else received
        peer = self.config["peers"][index]
        try:
            if not isinstance(report, dict):
                raise ValueError("Invalid report")
            if any(type(report.get(name)) is not bool for name in
                   ("live", "load_fresh", "health_ok", "api_alive", "api_responsive")):
                raise ValueError("Missing or non-boolean observation state")
            if any(report.get(key) != self.config[key] for key in ("epoch", "model", "role")):
                raise ValueError("Model, role or epoch mismatch")
            if (report.get("compatibility_protocol") != PROTOCOL
                    or report.get("scope") != "local_pod"
                    or report.get("expected_ranks") != peer["ranks"]
                    or report.get("expected_api_count") != peer["api_count"]
                    or not re.fullmatch(r"[a-f0-9]{32}", report.get("run_id", ""))):
                raise ValueError("Observer identity does not match the roster")
            identity = report["run_id"]
            if index in self.rejected or self.identities.get(index, identity) != identity:
                self.rejected.add(index)
                raise ValueError("A member restarted inside the same sealed epoch")
            self.identities[index] = identity
            if report.get("load_fresh") is not True:
                self.samples.pop(index, None)
                return
            cores = report.get("cores", {})
            maximum = report.get("core_max_age_seconds")
            remaining = report.get("load_valid_for_seconds")
            if (not finite_age(maximum) or maximum <= 0 or not finite_age(remaining)
                    or remaining > maximum or set(cores) != {str(rank) for rank in peer["ranks"]}):
                raise ValueError("Invalid core coverage or source deadline")
            for rank, sample in cores.items():
                if (sample.get("rank") != int(rank) or not base.valid_metrics(sample.get("metrics"))
                        or not finite_age(sample.get("age_seconds"))
                        or sample["age_seconds"] > maximum
                        or sample.get("epoch") != self.config["epoch"]
                        or sample.get("run_id") != identity):
                    raise ValueError("Invalid core sample")
                remaining = min(remaining, maximum - sample["age_seconds"])
            health = report.get("health_ok") is True
            if health and (report.get("live") is not True or report.get("api_alive") is not True
                           or any(sample.get("ready") is not True for sample in cores.values())):
                raise ValueError("Inconsistent positive health")
            if report.get("api_responsive") is True and not health:
                raise ValueError("Inconsistent API responsiveness")
            api_remaining = report.get("ready_valid_for_seconds", 0.0)
            if not finite_age(api_remaining) or report["api_responsive"] and api_remaining <= 0:
                raise ValueError("Missing API freshness deadline")
            deadline = started + remaining
            if not math.isfinite(started) or started > received or deadline <= received:
                raise ValueError("Expired source sample")
            self.samples[index] = (report, deadline, min(deadline, started + api_remaining))
        except (ValueError, TypeError, KeyError):
            self.samples.pop(index, None)
            raise

    def view(self, now=None):
        now = time.monotonic() if now is None else now
        current = [self.samples.get(index) for index in range(len(self.config["peers"]))]
        complete = all(item is not None and item[1] > now for item in current)
        if not complete:
            return {"load_fresh": False, "health_ok": False, "api_responsive": False,
                    "health_state": "unavailable", "deadline": now, "api_deadline": now, "cores": {}}
        reports = [item[0] for item in current]
        healthy = all(report["health_ok"] for report in reports)
        responsive = all(item[0]["api_responsive"] and item[2] > now for item in current)
        return {"load_fresh": True, "health_ok": healthy, "api_responsive": responsive,
                "health_state": ("healthy" if responsive else "degraded") if healthy else "unavailable",
                "deadline": min(item[1] for item in current),
                "api_deadline": min(item[2] for item in current) if responsive else now,
                "cores": {rank: sample for report in reports for rank, sample in report["cores"].items()}}


def merge_metrics(native, view, model_name, native_age=0.0):
    if not view["load_fresh"]:
        raise ValueError("Never substitute missing ranks with idle values")
    if len(native) > MAX_METRICS:
        raise ValueError("Native metric payload is too large")
    families = list(text_string_to_metric_families(native.decode("utf-8")))
    world = set(view["cores"])
    seen = {name: set() for name in GAUGES}
    saw_reasons = set()
    for family in families:
        for index, sample in enumerate(family.samples):
            if sample.name not in GAUGES and sample.name != WAITING:
                continue
            labels = sample.labels
            expected = {"model_name", "engine"} | ({"reason"} if sample.name == WAITING else set())
            if (set(labels) != expected or labels["model_name"] != model_name
                    or labels["engine"] not in world):
                raise ValueError("Native model/rank metric labels changed")
            rank = labels["engine"]
            if sample.name == WAITING:
                reason = labels["reason"]
                if reason not in {"capacity", "deferred"} or (rank, reason) in saw_reasons:
                    raise ValueError("Unknown or duplicate waiting reason")
                saw_reasons.add((rank, reason))
                value = view["cores"][rank]["metrics"][f"waiting_{reason}"]
            else:
                if rank in seen[sample.name]:
                    raise ValueError("Duplicate native rank gauge")
                seen[sample.name].add(rank)
                value = view["cores"][rank]["metrics"][GAUGES[sample.name]]
            family.samples[index] = sample._replace(value=value)
    if any(ranks != world for ranks in seen.values()):
        raise ValueError("Native gauges have not initialized the complete engine")
    if saw_reasons != {(rank, reason) for rank in world for reason in ("capacity", "deferred")}:
        raise ValueError("Native waiting-by-reason metrics are incomplete")
    for name, value in (("api_responsive", view["api_responsive"]), ("engine_healthy", view["health_ok"])):
        families.append(GaugeMetricFamily(f"vllm_compat_{name}", "Independent observation state", value=int(value)))
    families.append(GaugeMetricFamily("vllm_compat_native_snapshot_age_seconds",
                                     "Age of the original registry snapshot, not EngineCore load", value=native_age))
    registry = CollectorRegistry()

    class SnapshotCollector:
        def collect(self):
            return families

    registry.register(SnapshotCollector())
    payload = generate_latest(registry)
    if len(payload) > MAX_METRICS:
        raise ValueError("Merged metric payload is too large")
    return payload
