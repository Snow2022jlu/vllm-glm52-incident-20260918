"""Companion P16+D16 registrar: report fresh load even when readiness fails."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from observer import PROTOCOL, valid_metrics


LOG = logging.getLogger("isolated_load_registrar")
MAX_BODY = 1024 * 1024


def request_json(method, url, api_key=None, body=None, timeout=3.0):
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, method=method, headers=headers, data=encoded)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read(MAX_BODY + 1)
    if len(content) > MAX_BODY:
        raise ValueError("Oversized observer or gateway response")
    return json.loads(content) if content else {}


def backend_ready(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3.0) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def fetch_observer(url):
    started = time.monotonic()
    payload = request_json("GET", url.rstrip("/") + "/load")
    completed = time.monotonic()
    return payload, completed, completed - started


def group_state(results, args, role):
    samples = {}
    readiness_deadlines = []
    api_count = 0
    try:
        for payload, completed, elapsed in results:
            if (payload["protocol"] != PROTOCOL or payload["scope"] != "local_pod"
                    or payload["epoch"] != args.epoch or payload["model"] != args.model_id
                    or payload["role"] != role or payload["load_fresh"] is not True
                    or payload["live"] is not True):
                raise ValueError("Observer identity, epoch or freshness mismatch")
            ranks = payload["expected_ranks"]
            if (not isinstance(ranks, list) or not ranks
                    or any(type(rank) is not int or not 0 <= rank < args.ranks for rank in ranks)
                    or len(set(ranks)) != len(ranks)
                    or set(payload["cores"]) != {str(rank) for rank in ranks}):
                raise ValueError("Observer local-rank set is invalid")
            core_age = float(payload["core_max_age_seconds"])
            ready_for = float(payload["ready_valid_for_seconds"])
            if any(not math.isfinite(value) or not 0 <= value <= 300 for value in (core_age, ready_for)):
                raise ValueError("Invalid observer freshness window")
            count = payload["expected_api_count"]
            if type(count) is not int or not 0 <= count <= 64:
                raise ValueError("Invalid API process count")
            api_count += count
            readiness_deadlines.append(completed + ready_for - elapsed
                                       if payload["ready"] is True else -math.inf)
            for rank in ranks:
                sample = payload["cores"][str(rank)]
                age = float(sample["age_seconds"])
                if (rank in samples or sample["rank"] != rank
                        or sample["epoch"] != args.epoch or not valid_metrics(sample["metrics"])
                        or not math.isfinite(age) or age < 0):
                    raise ValueError("Duplicate rank or invalid source sample")
                deadline = completed + min(core_age, args.max_sample_age) - age - elapsed
                samples[rank] = {"metrics": sample["metrics"], "deadline": deadline}
        if set(samples) != set(range(args.ranks)) or api_count == 0:
            raise ValueError("Incomplete communication group or missing API head")
        return {"samples": samples, "ready_until": min(readiness_deadlines)}
    except (KeyError, TypeError, ValueError, OverflowError):
        LOG.warning("%s observations incomplete, stale or incompatible; reporting not ready", role)
        return {"samples": {}, "ready_until": -math.inf}


def collect(args):
    urls = [("prefill", url) for url in args.prefill_observer_url]
    urls += [("decode", url) for url in args.decode_observer_url]
    results = {"prefill": [], "decode": []}
    failed = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(urls) + 1)) as executor:
        health = executor.submit(backend_ready, args.worker_url)
        futures = {executor.submit(fetch_observer, url): role for role, url in urls}
        for future in concurrent.futures.as_completed(futures):
            role = futures[future]
            try:
                results[role].append(future.result())
            except Exception:
                failed.add(role)
                LOG.warning("%s observer fetch failed", role)
        ready = health.result()
    return {role: group_state([] if role in failed else results[role], args, role)
            for role in results}, ready


def heartbeat(rank, groups, proxy_ready, now=None):
    now = time.monotonic() if now is None else now
    ready = proxy_ready and all(
        group["ready_until"] > now
        and group["samples"]
        and all(sample["deadline"] > now for sample in group["samples"].values())
        for group in groups.values()
    )
    body = {"ready": bool(ready)}
    mappings = {
        "prefill": {
            "prefill_miss_requests": "prefill_cache_miss_requests",
            "prefill_miss_tokens": "prefill_cache_miss_tokens",
            "prefill_pending_requests": "prefill_cache_pending_requests",
        },
        "decode": {
            "kv_usage": "decode_kv_cache_usage_ratio",
            "running": "decode_running_requests",
            "waiting_capacity": "decode_waiting_requests",
            "generation_tokens": "decode_generation_tokens_total",
            "preemptions": "decode_preemptions_total",
        },
    }
    for role, fields in mappings.items():
        sample = groups[role]["samples"].get(rank)
        if sample is not None and sample["deadline"] > now:
            body.update({target: sample["metrics"][source] for source, target in fields.items()})
    return body


def expected_worker(args, rank):
    return {
        "id": f"{args.job}:prefill-dp{rank}", "url": args.worker_url.rstrip("/"),
        "model_id": args.model_id, "node_id": args.job + ":prefill", "dp_rank": rank,
        "labels": {"job": args.job, "role": "prefill", "topology": "vllm-2p2d",
                   "gateway_target_kind": "dp_rank", "decode_selection": "session-fixed-dp-rank"},
        "initial_ready": False,
    }


def reconcile(args, api_key):
    groups, ready = collect(args)
    base = args.gateway_url.rstrip("/")
    registry = request_json("GET", base + "/workers", api_key)
    if not isinstance(registry.get("workers"), list):
        raise ValueError("Malformed Gateway worker registry")
    existing = {worker["id"]: worker for worker in registry["workers"]}

    def send_rank(rank):
        expected = expected_worker(args, rank)
        actual = existing.get(expected["id"])
        if actual is None:
            if not heartbeat(rank, groups, ready)["ready"]:
                return
            request_json("POST", base + "/workers", api_key, expected)
        elif any(actual.get(key) != expected[key]
                 for key in ("id", "url", "model_id", "node_id", "dp_rank", "labels")):
            raise ValueError("Refusing to replace a different worker identity")
        encoded = urllib.parse.quote(expected["id"], safe="._:-")
        request_json("POST", f"{base}/workers/{encoded}/heartbeat", api_key,
                     heartbeat(rank, groups, ready))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, args.ranks)) as executor:
        for result in executor.map(send_rank, range(args.ranks)):
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    for name in ("gateway-url", "management-api-key-file", "worker-url", "model-id", "job", "epoch"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prefill-observer-url", action="append", required=True)
    parser.add_argument("--decode-observer-url", action="append", required=True)
    parser.add_argument("--ranks", type=int, default=16, choices=[16])
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-sample-age", type=float, default=10.0)
    args = parser.parse_args()
    if (not math.isfinite(args.interval) or args.interval < 0.25
            or not math.isfinite(args.max_sample_age) or not 0 < args.max_sample_age <= 300):
        raise ValueError("Invalid sampling interval or maximum sample age")
    if len(args.prefill_observer_url) != 2 or len(args.decode_observer_url) != 2:
        raise ValueError("This candidate requires the two complete P16 and D16 Pod pairs")
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    api_key = Path(args.management_api_key_file).read_text().strip()
    if not api_key:
        raise ValueError("Empty management API key")
    while True:
        try:
            reconcile(args, api_key)
        except Exception:
            LOG.exception("Observation reconciliation failed")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
