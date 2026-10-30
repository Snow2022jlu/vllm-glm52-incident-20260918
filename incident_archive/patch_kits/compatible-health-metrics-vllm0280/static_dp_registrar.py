"""Same legacy CLI and heartbeat fields; health no longer gates metric collection."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import re
import time
import urllib.parse
import urllib.request


LOG = logging.getLogger("static_dp_registrar")
ENGINE_LABEL = re.compile(r'(?:^|,)\s*engine="([0-9]+)"(?:,|$)')
MAX_METRICS = 8 * 1024 * 1024


def request_json(method, url, api_key=None, body=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = None
    if body is not None:
        payload = json.dumps(body, allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=payload, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=2) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Gateway response exceeds the bounded size")
    return json.loads(raw) if raw else {}


def backend_ready(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def rank_metric(text, metric_name, ranks, integer=False, reason=None):
    values = {}
    for raw in text.splitlines():
        if not raw.startswith(metric_name + "{"):
            continue
        labels, separator, tail = raw[len(metric_name) + 1:].partition("}")
        if not separator or (reason is not None and f'reason="{reason}"' not in labels):
            continue
        match = ENGINE_LABEL.search(labels)
        if match is None:
            raise ValueError("Missing engine label")
        rank, value = int(match[1]), float(tail.strip().split()[0])
        if (rank not in range(ranks) or rank in values or not math.isfinite(value)
                or value < 0 or integer and not value.is_integer()):
            raise ValueError("Duplicate, unknown or invalid rank sample")
        values[rank] = int(value) if integer else value
    if set(values) != set(range(ranks)):
        raise ValueError(f"Incomplete {metric_name}; missing ranks are not idle")
    return values


def fetch_metrics(url, args, role):
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=1) as response:
        headers = response.headers
        raw = response.read(MAX_METRICS + 1)
    if len(raw) > MAX_METRICS:
        raise ValueError("Metric payload exceeds limit")
    if headers.get("X-VLLM-Model") != args.model_id or headers.get("X-VLLM-Role") != role:
        raise ValueError("Metric source belongs to another model/role")
    lifetime = float(headers["X-VLLM-Load-Valid-For"])
    api_lifetime = float(headers["X-VLLM-API-Valid-For"])
    native_age = float(headers["X-VLLM-Native-Snapshot-Age"])
    if (not math.isfinite(lifetime) or not 0 < lifetime <= 300
            or not math.isfinite(api_lifetime) or not 0 <= api_lifetime <= 300
            or not math.isfinite(native_age) or native_age < 0):
        raise ValueError("Invalid source validity")
    deadline = started + lifetime
    epoch = headers.get("X-VLLM-Epoch", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", epoch) or deadline <= time.monotonic():
        raise ValueError("Expired or unbound source")
    mappings = ({
        "prefill_cache_miss_requests": ("vllm:prefill_cache_miss_requests", True, None),
        "prefill_cache_miss_tokens": ("vllm:prefill_cache_miss_tokens", True, None),
        "prefill_cache_pending_requests": ("vllm:prefill_cache_pending_requests", True, None),
    } if role == "prefill" else {
        "decode_kv_cache_usage_ratio": ("vllm:kv_cache_usage_perc", False, None),
        "decode_running_requests": ("vllm:num_requests_running", True, None),
        "decode_waiting_requests": ("vllm:num_requests_waiting_by_reason", True, "capacity"),
    })
    if role == "decode" and native_age < 3:
        mappings.update({
            "decode_generation_tokens_total": ("vllm:generation_tokens_total", True, None),
            "decode_preemptions_total": ("vllm:num_preemptions_total", True, None),
        })
    text = raw.decode("utf-8")
    fields = {name: rank_metric(text, metric, args.ranks, integer, reason)
              for name, (metric, integer, reason) in mappings.items()}
    if "decode_kv_cache_usage_ratio" in fields and any(value > 1 for value in fields["decode_kv_cache_usage_ratio"].values()):
        raise ValueError("KV utilization exceeds 1")
    return {"epoch": epoch, "deadline": deadline, "fields": fields,
            "api_deadline": started + min(api_lifetime, lifetime),
            "native_deadline": started + max(0.0, 3 - native_age),
            "healthy": headers.get("X-VLLM-Health-State") == "healthy",
            "api_responsive": headers.get("X-VLLM-API-Responsive") == "1"}


def collect(args):
    samples = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        health = executor.submit(backend_ready, args.worker_url)
        futures = {executor.submit(fetch_metrics, getattr(args, f"{role}_metrics_url"), args, role): role
                   for role in ("prefill", "decode")}
        for future in concurrent.futures.as_completed(futures):
            role = futures[future]
            try:
                samples[role] = future.result()
            except Exception:
                LOG.exception("%s metrics unavailable", role)
        ready = health.result()
    if len({sample["epoch"] for sample in samples.values()}) > 1:
        LOG.error("P/D epochs differ; refusing mixed-session load")
        samples = {}
    return samples, ready


def expected_worker(args, rank):
    return {"id": f"{args.job}:prefill-dp{rank}", "url": args.worker_url.rstrip("/"),
            "model_id": args.model_id, "node_id": args.job + ":prefill", "dp_rank": rank,
            "labels": {"job": args.job, "role": "prefill", "topology": "vllm-2p2d",
                       "gateway_target_kind": "dp_rank", "decode_selection": "session-fixed-dp-rank"},
            "initial_ready": True}


def heartbeat(rank, samples, ready, now=None):
    now = time.monotonic() if now is None else now
    fresh = {role: sample for role, sample in samples.items() if sample["deadline"] > now}
    body = {"ready": ready and set(fresh) == {"prefill", "decode"}}
    for sample in fresh.values():
        for name, values in sample["fields"].items():
            if name in {"decode_generation_tokens_total", "decode_preemptions_total"} and sample["native_deadline"] <= now:
                continue
            body[name] = values[rank]
    return body


def reconcile(args, api_key):
    samples, ready = collect(args)
    base_url = args.gateway_url.rstrip("/")
    registry = request_json("GET", base_url + "/workers", api_key)
    if not isinstance(registry.get("workers"), list):
        raise ValueError("Malformed worker registry")
    existing = {worker["id"]: worker for worker in registry["workers"]}

    def send(rank):
        expected = expected_worker(args, rank)
        actual = existing.get(expected["id"])
        if actual is None:
            if not heartbeat(rank, samples, ready)["ready"] or not all(
                sample["healthy"] and sample["api_responsive"] and sample["api_deadline"] > time.monotonic()
                for sample in samples.values()
            ):
                return
            request_json("POST", base_url + "/workers", api_key, expected)
        elif any(actual.get(key) != expected[key] for key in ("id", "url", "model_id", "node_id", "dp_rank", "labels")):
            raise ValueError("Refusing to delete or replace a worker with different identity")
        encoded = urllib.parse.quote(expected["id"], safe="._:-")
        request_json("POST", f"{base_url}/workers/{encoded}/heartbeat", api_key,
                     heartbeat(rank, samples, ready))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(send, range(args.ranks)))


def parse_args():
    parser = argparse.ArgumentParser()
    for name in ("gateway-url", "management-api-key-file", "worker-url", "prefill-metrics-url",
                 "decode-metrics-url", "model-id", "job"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.ranks != 16 or not math.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("This publisher candidate is sealed for P16+D16 and a positive interval")
    return args


def main():
    args = parse_args()
    with open(args.management_api_key_file, encoding="utf-8") as source:
        api_key = source.read().strip()
    if not api_key:
        raise ValueError("Management key is empty")
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            reconcile(args, api_key)
        except Exception:
            LOG.exception("Reconciliation failed")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
