#!/usr/bin/env python3
"""Keep 16 logical Prefill DP-rank workers registered in the session relay."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger("static_dp_registrar")
ENGINE_LABEL = re.compile(r'(?:^|,)\s*engine="([0-9]+)"(?:,|$)')


def request_json(method, url, api_key=None, body=None, timeout=10.0):
    payload = None
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=payload, headers=headers, method=method
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def backend_ready(worker_url):
    try:
        with urllib.request.urlopen(
            worker_url.rstrip("/") + "/health", timeout=3.0
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def fetch_text(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="strict")


def rank_metric(
    text,
    metric_name,
    ranks,
    integer=False,
    required_label=None,
    fill_missing_zero=False,
):
    prefix = metric_name + "{"
    values = {}
    for raw in text.splitlines():
        if not raw.startswith(prefix):
            continue
        labels, separator, tail = raw[len(prefix):].partition("}")
        if not separator:
            continue
        match = ENGINE_LABEL.search(labels)
        if match is None:
            continue
        if required_label is not None:
            label_name, label_value = required_label
            label_pattern = re.compile(
                rf'(?:^|,)\s*{re.escape(label_name)}="{re.escape(label_value)}"(?:,|$)'
            )
            if label_pattern.search(labels) is None:
                continue
        try:
            value = float(tail.strip().split()[0])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"invalid {metric_name} sample") from exc
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"non-finite or negative {metric_name} sample")
        if integer and not value.is_integer():
            raise RuntimeError(f"fractional {metric_name} sample")
        rank = int(match.group(1))
        if rank < ranks:
            values[rank] = int(value) if integer else value
    missing = sorted(set(range(ranks)) - set(values))
    if missing and fill_missing_zero:
        values.update({rank: 0 for rank in missing})
    elif missing:
        raise RuntimeError(f"{metric_name} missing ranks: {missing}")
    return values


def worker_id(job, rank):
    return f"{job}:prefill-dp{rank}"


def expected_worker(args, rank):
    return {
        "id": worker_id(args.job, rank),
        "url": args.worker_url.rstrip("/"),
        "model_id": args.model_id,
        "node_id": args.job + ":prefill",
        "dp_rank": rank,
        "labels": {
            "job": args.job,
            "role": "prefill",
            "topology": "vllm-2p2d",
            "gateway_target_kind": "dp_rank",
            "decode_selection": "session-fixed-dp-rank",
        },
        "initial_ready": True,
    }


def same_identity(actual, expected):
    return all(
        actual.get(key) == expected.get(key)
        for key in ("id", "url", "model_id", "node_id", "dp_rank", "labels")
    )


def reconcile(args, api_key):
    base = args.gateway_url.rstrip("/")
    registry = request_json("GET", base + "/workers", api_key)
    workers = registry.get("workers")
    if not isinstance(workers, list):
        raise RuntimeError("Gateway /workers response is malformed")
    existing = {
        worker.get("id"): worker
        for worker in workers
        if isinstance(worker, dict) and isinstance(worker.get("id"), str)
    }

    for rank in range(args.ranks):
        expected = expected_worker(args, rank)
        actual = existing.get(expected["id"])
        if actual is None:
            request_json("POST", base + "/workers", api_key, expected)
            LOG.info("registered worker=%s rank=%d", expected["id"], rank)
        elif not same_identity(actual, expected):
            if actual.get("total_sessions", 0) or actual.get("request_inflight", 0):
                raise RuntimeError(
                    f"refusing to replace active worker identity {expected['id']}"
                )
            request_json("POST", base + "/workers", api_key, expected)
            LOG.info("replaced drifted empty worker=%s rank=%d", expected["id"], rank)

    miss_requests = miss_tokens = pending_requests = None
    decode_kv = decode_running = decode_waiting = None
    decode_generation = decode_preemptions = None
    try:
        prefill_metrics = fetch_text(args.prefill_metrics_url)
        miss_requests = rank_metric(
            prefill_metrics,
            "vllm:prefill_cache_miss_requests",
            args.ranks,
            integer=True,
            fill_missing_zero=True,
        )
        miss_tokens = rank_metric(
            prefill_metrics,
            "vllm:prefill_cache_miss_tokens",
            args.ranks,
            integer=True,
            fill_missing_zero=True,
        )
        pending_requests = rank_metric(
            prefill_metrics,
            "vllm:prefill_cache_pending_requests",
            args.ranks,
            integer=True,
            fill_missing_zero=True,
        )
    except Exception:
        LOG.exception("Prefill metrics scrape failed")
    try:
        decode_metrics = fetch_text(args.decode_metrics_url)
        decode_kv = rank_metric(
            decode_metrics,
            "vllm:kv_cache_usage_perc",
            args.ranks,
            fill_missing_zero=True,
        )
        if any(value > 1.0 for value in decode_kv.values()):
            raise RuntimeError("vllm:kv_cache_usage_perc is outside [0,1]")
        decode_running = rank_metric(
            decode_metrics,
            "vllm:num_requests_running",
            args.ranks,
            integer=True,
            fill_missing_zero=True,
        )
        try:
            decode_waiting = rank_metric(
                decode_metrics,
                "vllm:num_requests_waiting_by_reason",
                args.ranks,
                integer=True,
                required_label=("reason", "capacity"),
                fill_missing_zero=True,
            )
        except RuntimeError:
            decode_waiting = rank_metric(
                decode_metrics,
                "vllm:num_requests_waiting",
                args.ranks,
                integer=True,
                fill_missing_zero=True,
            )
            LOG.warning(
                "Decode capacity-waiting metric unavailable; using total waiting"
            )
        decode_generation = rank_metric(
            decode_metrics,
            "vllm:generation_tokens_total",
            args.ranks,
            integer=True,
        )
        decode_preemptions = rank_metric(
            decode_metrics,
            "vllm:num_preemptions_total",
            args.ranks,
            integer=True,
        )
    except Exception:
        decode_kv = decode_running = decode_waiting = None
        decode_generation = decode_preemptions = None
        LOG.exception("Decode metrics scrape failed")
    for rank in range(args.ranks):
        heartbeat = {"ready": True}
        if miss_requests is not None:
            heartbeat.update({
                "prefill_cache_miss_requests": miss_requests[rank],
                "prefill_cache_miss_tokens": miss_tokens[rank],
                "prefill_cache_pending_requests": pending_requests[rank],
            })
        if decode_kv is not None:
            heartbeat["decode_kv_cache_usage_ratio"] = decode_kv[rank]
            heartbeat["decode_running_requests"] = decode_running[rank]
            heartbeat["decode_waiting_requests"] = decode_waiting[rank]
            heartbeat["decode_generation_tokens_total"] = decode_generation[rank]
            heartbeat["decode_preemptions_total"] = decode_preemptions[rank]
        encoded = urllib.parse.quote(worker_id(args.job, rank), safe="._:-")
        request_json(
            "POST",
            f"{base}/workers/{encoded}/heartbeat",
            api_key,
            heartbeat,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--management-api-key-file", required=True)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--prefill-metrics-url", required=True)
    parser.add_argument("--decode-metrics-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ranks != 16:
        raise ValueError("this manifest is sealed for exactly 16 Prefill DP ranks")
    with open(args.management_api_key_file, encoding="utf-8") as handle:
        api_key = handle.read().strip()
    if not api_key:
        raise ValueError("management API key file is empty")
    while True:
        try:
            if backend_ready(args.worker_url):
                reconcile(args, api_key)
            else:
                LOG.info("backend not ready; registration deferred")
        except Exception:
            LOG.exception("static DP-rank reconciliation failed")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
