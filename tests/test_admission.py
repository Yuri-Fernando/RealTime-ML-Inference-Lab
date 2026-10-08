"""Tests for inference_service.admission: the admission-control decision
rule, isolated from queueing/batching/asyncio entirely.

This is a pure unit test — no event loop, no model, no DynamicBatcher —
proving the "capacity? -> queue/execute : reject/defer" decision
(03_RealTime-ML-Inference-Lab.md section 11) is correct on its own, which
was impossible when the same check was inlined inside DynamicBatcher.submit().
"""
from inference_service.admission import AdmissionController, AdmissionOutcome


def test_admits_when_below_capacity():
    controller = AdmissionController(max_queue_depth=4)
    decision = controller.decide(current_in_flight=0)
    assert decision.outcome is AdmissionOutcome.ADMIT
    assert decision.admitted is True


def test_admits_at_one_below_limit():
    controller = AdmissionController(max_queue_depth=4)
    decision = controller.decide(current_in_flight=3)
    assert decision.admitted is True


def test_rejects_at_exact_capacity():
    controller = AdmissionController(max_queue_depth=4)
    decision = controller.decide(current_in_flight=4)
    assert decision.outcome is AdmissionOutcome.REJECT
    assert decision.admitted is False


def test_rejects_above_capacity():
    controller = AdmissionController(max_queue_depth=4)
    decision = controller.decide(current_in_flight=999)
    assert decision.admitted is False


def test_zero_capacity_rejects_everything():
    controller = AdmissionController(max_queue_depth=0)
    decision = controller.decide(current_in_flight=0)
    assert decision.admitted is False


def test_decision_carries_the_inputs_for_observability():
    controller = AdmissionController(max_queue_depth=10)
    decision = controller.decide(current_in_flight=7)
    assert decision.in_flight == 7
    assert decision.max_queue_depth == 10


def test_mutating_max_queue_depth_changes_future_decisions():
    controller = AdmissionController(max_queue_depth=2)
    assert controller.decide(current_in_flight=2).admitted is False
    controller.max_queue_depth = 5
    assert controller.decide(current_in_flight=2).admitted is True
