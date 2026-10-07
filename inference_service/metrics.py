"""Prometheus metrics for the inference service.

All metrics here are updated from real measurements taken in app.py /
batching.py — there is no synthetic/fake data. Scraped at GET /metrics in
the standard Prometheus text exposition format via `prometheus_client`.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

registry = CollectorRegistry()

REQUEST_LATENCY = Histogram(
    "rtml_request_latency_seconds",
    "End-to-end latency of /infer requests (enqueue -> response), seconds",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=registry,
)

REQUESTS_TOTAL = Counter(
    "rtml_requests_total",
    "Total inference requests received",
    ["outcome"],  # ok | backpressure | timeout | error
    registry=registry,
)

CACHE_EVENTS_TOTAL = Counter(
    "rtml_cache_events_total",
    "Cache hit/miss events",
    ["result"],  # hit | miss
    registry=registry,
)

BATCH_SIZE = Histogram(
    "rtml_batch_size",
    "Number of requests grouped into a single forward pass",
    buckets=(1, 2, 4, 8, 16, 32, 64),
    registry=registry,
)

QUEUE_DEPTH = Gauge(
    "rtml_queue_depth",
    "Current number of in-flight (queued + processing) requests",
    registry=registry,
)

FORWARD_PASSES_TOTAL = Counter(
    "rtml_forward_passes_total",
    "Total model forward passes executed (lower than requests_total implies batching worked)",
    registry=registry,
)


def render_latest() -> bytes:
    return generate_latest(registry)
