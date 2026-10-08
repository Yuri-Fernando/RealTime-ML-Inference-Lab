"""Admission control, kept separate and explicit from dynamic batching.

Why a separate module: the MVP originally decided accept/reject for a new
request *inside* `DynamicBatcher.submit()`, mixed together with queueing and
batching concerns. That made the admission decision itself untestable in
isolation and conflated two different responsibilities:

    - Admission control: "given current load, should this NEW request be
      allowed in at all?" (capacity? -> queue/execute : reject/defer)
    - Dynamic batching: "given requests that WERE admitted, how do we group
      them into efficient forward passes?"

This module implements only the first responsibility, following the
"03_RealTime-ML-Inference-Lab.md" section 11 model:

    request
      |
      v
    capacity?
      |-- yes -> queue/execute (ADMIT)
      `-- no  -> reject/defer (REJECT)

It is deliberately simple for the MVP: a single bounded-queue capacity check
(no priority lanes, no deadline-awareness, no per-tenant quota — those are
explicitly out of scope, see master spec section 4.3 vs. this MVP's
boundaries). What IS real here: the decision is a pure function of
`(current_in_flight, max_queue_depth)`, fully unit-testable without asyncio,
without a model, without a running event loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AdmissionOutcome(str, Enum):
    ADMIT = "admit"
    REJECT = "reject"


@dataclass(frozen=True)
class AdmissionDecision:
    outcome: AdmissionOutcome
    in_flight: int
    max_queue_depth: int

    @property
    def admitted(self) -> bool:
        return self.outcome is AdmissionOutcome.ADMIT

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.outcome.value} (in_flight={self.in_flight}, "
            f"max_queue_depth={self.max_queue_depth})"
        )


class AdmissionController:
    """Stateless decision function over externally-tracked load.

    This controller does not own the in-flight counter itself — the caller
    (DynamicBatcher) owns and mutates that counter under its own lock,
    because incrementing it atomically with the decision must happen inside
    the same critical section. The controller's job is only the decision
    rule itself, isolated so it can be tested/changed independently of
    queueing/locking mechanics.
    """

    def __init__(self, max_queue_depth: int) -> None:
        self.max_queue_depth = max_queue_depth

    def decide(self, current_in_flight: int) -> AdmissionDecision:
        if current_in_flight >= self.max_queue_depth:
            return AdmissionDecision(
                outcome=AdmissionOutcome.REJECT,
                in_flight=current_in_flight,
                max_queue_depth=self.max_queue_depth,
            )
        return AdmissionDecision(
            outcome=AdmissionOutcome.ADMIT,
            in_flight=current_in_flight,
            max_queue_depth=self.max_queue_depth,
        )
