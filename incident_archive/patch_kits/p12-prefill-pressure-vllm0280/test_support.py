from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import enum
from functools import partial
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import urllib.parse

if any(os.environ.get(name) for name in ("PROMETHEUS_MULTIPROC_DIR", "prometheus_multiproc_dir")):
    raise RuntimeError("unset Prometheus multiprocess directory variables in this test process")

from prometheus_client import CollectorRegistry, Gauge, generate_latest


ROOT = Path(__file__).resolve().parent
METRICS = (
    "prefill_cache_miss_requests",
    "prefill_cache_miss_tokens",
    "prefill_cache_pending_requests",
)


def execute(nodes, namespace, filename="extracted-source"):
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *copy.deepcopy(nodes)],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)


def source_class(path, name):
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name)


class RequestFixture:
    def __init__(self, status_type, **values):
        self.request_id = "fixture"
        self.client_index = 0
        self.status = status_type.WAITING
        self.status_type = status_type
        self.prefill_stats = object()
        self.prefill_cache_lookup_complete = False
        self.prefill_cached_tokens = 0
        self.num_prompt_tokens = 1024
        self.num_tokens = 1024
        self.num_computed_tokens = 0
        self.num_in_flight_tokens = 0
        self.num_stale_output_tokens = 0
        self.drop_stale_output = False
        self.num_output_placeholders = 0
        self.num_preemptions = 0
        self.spec_token_ids = []
        self.use_structured_output = False
        self.__dict__.update(values)

    def is_finished(self):
        return self.status_type.is_finished(self.status)


class QueueFixture(list):
    def remove_requests(self, requests):
        self[:] = [request for request in self if request not in requests]

    def prepend_request(self, request):
        self.insert(0, request)


def scheduler_fixture(runtime_root):
    request_path = runtime_root / "vllm/v1/request.py"
    stats_path = runtime_root / "vllm/v1/metrics/stats.py"
    scheduler_path = runtime_root / "vllm/v1/core/sched/scheduler.py"
    namespace = {"enum": enum, "dataclass": dataclass, "field": field, "remove_all": lambda requests, removed: [request for request in requests if request not in removed]}
    execute([source_class(request_path, "RequestStatus")], namespace)
    execute([source_class(stats_path, name) for name in ("BaseCacheStats", "PrefixCacheStats", "SchedulerStats")], namespace)
    method_names = {"finish_requests", "_free_request", "_free_blocks", "_get_prefill_cache_pressure_stats", "make_stats", "_preempt_request", "_update_after_schedule"}
    methods = [node for node in source_class(scheduler_path, "Scheduler").body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    if len(methods) != len(method_names):
        raise AssertionError("missing scheduler method")
    execute(methods, namespace)
    scheduler_type = type("SchedulerFixture", (), {name: namespace[name] for name in method_names})
    scheduler = scheduler_type()
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = QueueFixture()
    scheduler.skipped_waiting = QueueFixture()
    scheduler._inflight_prefills = set()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.reset_preempted_req_ids = set()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda request: None)
    scheduler.ec_connector = None
    scheduler._connector_finished = lambda request: (False, None)
    scheduler._free_request_blocks = lambda request: None
    scheduler.report_prefill_cache_pressure = True
    scheduler.log_stats = False
    scheduler.defer_block_free = False
    scheduler.enable_return_routed_experts = False
    scheduler.kv_cache_manager = SimpleNamespace(make_prefix_cache_stats=namespace["PrefixCacheStats"], usage=0.25)
    scheduler.connector_prefix_cache_stats = None
    scheduler.kv_metrics_collector = None
    return scheduler, namespace["RequestStatus"]


def prometheus_text(runtime_root, stats_by_engine):
    logger_class = source_class(runtime_root / "vllm/v1/metrics/loggers.py", "PrometheusStatLogger")
    constructor = next(node for node in logger_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    init_nodes = [node for node in constructor.body if isinstance(node, ast.Assign) and any(isinstance(target, (ast.Name, ast.Attribute)) and getattr(target, "id", getattr(target, "attr", "")).startswith("gauge_prefill_cache_") for target in node.targets)]
    record = next(node for node in logger_class.body if isinstance(node, ast.FunctionDef) and node.name == "record")
    record_nodes = sorted([node for node in ast.walk(record) if isinstance(node, ast.Expr) and any(isinstance(child, ast.Attribute) and child.attr.startswith("gauge_prefill_cache_") for child in ast.walk(node))], key=lambda node: node.lineno)
    if len(init_nodes) != 6 or len(record_nodes) != 3:
        raise AssertionError("unexpected metric constructor/record statements")
    utils_path = ROOT / "sources/vllm-0.28.0/vllm/v1/metrics/utils.py"
    utility = next(node for node in ast.parse(utils_path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "create_metric_per_engine")
    registry = CollectorRegistry()
    namespace = {
        "self": SimpleNamespace(_gauge_cls=partial(Gauge, registry=registry)),
        "labelnames": ["model_name", "engine"],
        "per_engine_labelvalues": {engine: ["fixture", str(engine)] for engine in stats_by_engine},
    }
    execute([utility, *init_nodes], namespace)
    for engine, stats in stats_by_engine.items():
        namespace.update(engine_idx=engine, scheduler_stats=stats)
        execute(record_nodes, namespace)
    return generate_latest(registry).decode()


def reconcile_fixture(collector_file, prefill_text, ranks=2):
    names = {"rank_metric", "worker_id", "expected_worker", "same_identity", "reconcile"}
    module = ast.parse(collector_file.read_text())
    nodes = [node for node in module.body if (isinstance(node, ast.FunctionDef) and node.name in names) or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ENGINE_LABEL" for target in node.targets))]
    namespace = {"math": math, "re": re, "urllib": urllib, "LOG": SimpleNamespace(info=lambda *args: None, warning=lambda *args: None, exception=lambda *args: None)}
    execute(nodes, namespace)
    args = SimpleNamespace(gateway_url="http://fixture.invalid", job="fixture", ranks=ranks, worker_url="http://worker.invalid", model_id="fixture", prefill_metrics_url="prefill-fixture", decode_metrics_url="decode-fixture")
    registry = {"workers": [namespace["expected_worker"](args, rank) for rank in range(ranks)]}
    heartbeats = []
    def request_json(method, url, api_key=None, body=None):
        if method == "GET":
            if url != args.gateway_url + "/workers":
                raise AssertionError(url)
            return registry
        if method != "POST" or not url.endswith("/heartbeat"):
            raise AssertionError(url)
        heartbeats.append(body)
        return {}
    def fetch_text(url):
        if url == args.decode_metrics_url:
            return "\n".join(f'vllm:{metric}{{engine="{rank}"}} 0' for metric in ("kv_cache_usage_perc", "num_requests_running", "num_requests_waiting_by_reason", "generation_tokens_total", "num_preemptions_total") for rank in range(ranks))
        if url != args.prefill_metrics_url:
            raise AssertionError(url)
        if prefill_text is None:
            raise OSError("synthetic scrape failure")
        return prefill_text
    namespace.update(request_json=request_json, fetch_text=fetch_text)
    namespace["reconcile"](args, None)
    return heartbeats
