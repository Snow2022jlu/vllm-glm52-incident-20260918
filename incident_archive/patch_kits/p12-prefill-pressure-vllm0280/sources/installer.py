#!/usr/bin/env python3
"""Apply the GLM P-rank Prefill cache-pressure metrics backport."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

FILES = {
    "v1/request.py": (
        "92124fbad28cda49bd06fa12c2c4fd5f53fc9381ddb4dc35f275c5ccfbd27378",
        "85179d9006b3be8b32b183c1f0cea31e3eea73e5128629bf3899adc12aea9fce",
        [
            (
                """        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt
""",
                """        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        # Cache-pressure metrics need to distinguish requests whose prefix-cache
        # lookup is still pending from requests with a known amount of Prefill
        # work. This is scheduler-owned state and is never exposed as a label.
        self.prefill_cache_lookup_complete = False
        self.prefill_cached_tokens = 0
        self.cache_salt: str | None = cache_salt
""",
            ),
        ],
    ),
    "v1/metrics/stats.py": (
        "7682d3086bf5395f163f423677485432ee2c035aada2f370c0f82b291e53ca47",
        "7944c6b82eb02a673b524a2b863b6d38bf6c75357a308c5b98d922a03755edd2",
        [
            (
                """    num_waiting_reqs: int = 0  # length of the "waiting" request queue
    num_skipped_waiting_reqs: int = 0  # length of the "skipped waiting" queue

    # These are used for internal DP load-balancing.
""",
                """    num_waiting_reqs: int = 0  # length of the "waiting" request queue
    num_skipped_waiting_reqs: int = 0  # length of the "skipped waiting" queue

    # Current Prefill cache-pressure on a KV producer. A consumer reports 0.
    prefill_cache_miss_requests: int = 0
    prefill_cache_miss_tokens: int = 0
    prefill_cache_pending_requests: int = 0

    # These are used for internal DP load-balancing.
""",
            ),
        ],
    ),
    "v1/core/sched/scheduler.py": (
        "6dbe3d244d4d192e0a38d93b4f185c81758a0fadedd369ebe2fd3354414e3e47",
        "f1cca1192c670c67c3a842f7a75fafa361a28cc8dde0a9b42b95d984616475b1",
        [
            (
                """        self.defer_block_free = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
""",
                """        self.defer_block_free = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        self.report_prefill_cache_pressure = bool(
            kv_transfer_config is not None and kv_transfer_config.is_kv_producer
        )
        if kv_transfer_config is not None:
""",
            ),
            (
                """                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
""",
                """                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # A connector may need multiple scheduler passes to resolve
                    # an asynchronous cache lookup. Until this pass completes,
                    # expose the request as pending rather than as a cache miss.
                    request.prefill_cache_lookup_complete = False
                    request.prefill_cached_tokens = 0
                    # Get locally-cached tokens.
""",
            ),
            (
                """                    assert num_computed_tokens <= request.num_tokens

                    # Skip request with pending mm encoding prefetches
""",
                """                    assert num_computed_tokens <= request.num_tokens
                    request.prefill_cache_lookup_complete = True
                    request.prefill_cached_tokens = num_computed_tokens

                    # Skip request with pending mm encoding prefetches
""",
            ),
            (
                """        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
""",
                """        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.prefill_cache_lookup_complete = False
        request.prefill_cached_tokens = 0
        if request.spec_token_ids:
""",
            ),
            (
                """        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
""",
                """        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        (
            prefill_cache_miss_requests,
            prefill_cache_miss_tokens,
            prefill_cache_pending_requests,
        ) = self._get_prefill_cache_pressure_stats()
        return SchedulerStats(
""",
            ),
            (
                """            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
""",
                """            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            prefill_cache_miss_requests=prefill_cache_miss_requests,
            prefill_cache_miss_tokens=prefill_cache_miss_tokens,
            prefill_cache_pending_requests=prefill_cache_pending_requests,
            kv_cache_usage=self.kv_cache_manager.usage,
""",
            ),
            (
                """    def make_spec_decoding_stats(
""",
                """    def _get_prefill_cache_pressure_stats(self) -> tuple[int, int, int]:
        \"\"\"Return current uncached Prefill pressure for this DP-rank.

        The scan is intentionally side-effect free: it never starts a local or
        external cache lookup. ``num_computed_tokens`` advances optimistically
        at scheduling time, so in-flight tokens are subtracted to keep them in
        the remaining-work total until the model step completes.
        \"\"\"
        if not self.report_prefill_cache_pressure:
            return 0, 0, 0

        miss_requests = 0
        miss_tokens = 0
        pending_requests = 0
        for request in self.requests.values():
            if request.prefill_stats is None:
                continue
            if not request.prefill_cache_lookup_complete:
                pending_requests += 1
                continue

            settled_tokens = max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            )
            # Before admission, cache hits are known but have not yet been
            # copied into request.num_computed_tokens.
            if (
                settled_tokens == 0
                and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            ):
                settled_tokens = request.prefill_cached_tokens

            remaining_tokens = max(0, request.num_prompt_tokens - settled_tokens)
            if remaining_tokens:
                miss_requests += 1
                miss_tokens += remaining_tokens

        return miss_requests, miss_tokens, pending_requests

    def make_spec_decoding_stats(
""",
            ),
        ],
    ),
    "v1/metrics/loggers.py": (
        "d5a91626cf79118f98ae6438c842cb30fc44ca14463c7e2af5871e3da8e51bcf",
        "9301e7d45775764885f7ce0abba1ee1c47da47bd995366f6c05915a41c882441",
        [
            (
                """        self.gauge_scheduler_waiting = create_metric_per_engine(
            gauge_scheduler_waiting, per_engine_labelvalues
        )

        gauge_waiting_by_reason = self._gauge_cls(
""",
                """        self.gauge_scheduler_waiting = create_metric_per_engine(
            gauge_scheduler_waiting, per_engine_labelvalues
        )

        gauge_prefill_cache_miss_requests = self._gauge_cls(
            name="vllm:prefill_cache_miss_requests",
            documentation=(
                "Number of active requests with a completed cache lookup and "
                "remaining uncached Prefill work on this KV producer."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_prefill_cache_miss_requests = create_metric_per_engine(
            gauge_prefill_cache_miss_requests, per_engine_labelvalues
        )

        gauge_prefill_cache_miss_tokens = self._gauge_cls(
            name="vllm:prefill_cache_miss_tokens",
            documentation=(
                "Remaining uncached Prefill tokens, including in-flight work, "
                "on this KV producer."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_prefill_cache_miss_tokens = create_metric_per_engine(
            gauge_prefill_cache_miss_tokens, per_engine_labelvalues
        )

        gauge_prefill_cache_pending_requests = self._gauge_cls(
            name="vllm:prefill_cache_pending_requests",
            documentation=(
                "Number of active Prefill requests whose cache lookup has not "
                "completed on this KV producer."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_prefill_cache_pending_requests = create_metric_per_engine(
            gauge_prefill_cache_pending_requests, per_engine_labelvalues
        )

        gauge_waiting_by_reason = self._gauge_cls(
""",
            ),
            (
                """            self.gauge_scheduler_waiting[engine_idx].set(total_waiting)
            self.gauge_waiting_by_reason[WAITING_REASON_CAPACITY][engine_idx].set(
""",
                """            self.gauge_scheduler_waiting[engine_idx].set(total_waiting)
            self.gauge_prefill_cache_miss_requests[engine_idx].set(
                scheduler_stats.prefill_cache_miss_requests
            )
            self.gauge_prefill_cache_miss_tokens[engine_idx].set(
                scheduler_stats.prefill_cache_miss_tokens
            )
            self.gauge_prefill_cache_pending_requests[engine_idx].set(
                scheduler_stats.prefill_cache_pending_requests
            )
            self.gauge_waiting_by_reason[WAITING_REASON_CAPACITY][engine_idx].set(
""",
            ),
        ],
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _patch_one(relative: str, dry_run: bool) -> tuple[str, str]:
    before_sha256, after_sha256, replacements = FILES[relative]
    target = ROOT / relative
    source = target.read_bytes()
    digest = _sha256(source)
    if digest == after_sha256:
        compile(source, str(target), "exec")
        print(f"PREFILL_CACHE_PRESSURE already_applied target={target} sha256={digest}")
        return digest, digest
    if digest != before_sha256:
        raise RuntimeError(f"refusing unexpected vLLM source: {target} sha256={digest}")

    text = source.decode("utf-8")
    for index, (old, new) in enumerate(replacements, 1):
        if text.count(old) != 1 or new in text:
            raise RuntimeError(
                f"replacement mismatch: target={target} replacement={index}"
            )
        text = text.replace(old, new, 1)

    patched = text.encode("utf-8")
    compile(patched, str(target), "exec")
    patched_digest = _sha256(patched)
    if after_sha256.startswith("__"):
        if not dry_run:
            raise RuntimeError(f"unset after hash for {target}: {patched_digest}")
    elif patched_digest != after_sha256:
        raise RuntimeError(
            f"patched hash mismatch: target={target} expected={after_sha256} "
            f"actual={patched_digest}"
        )

    if not dry_run:
        fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(patched)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o644)
            os.replace(temporary_name, target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
    print(
        f"PREFILL_CACHE_PRESSURE {'dry_run' if dry_run else 'applied'} "
        f"target={target} before={digest} after={patched_digest}"
    )
    return digest, patched_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for relative in FILES:
        _patch_one(relative, args.dry_run)


if __name__ == "__main__":
    main()
