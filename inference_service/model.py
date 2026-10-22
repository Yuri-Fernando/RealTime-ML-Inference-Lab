"""Small, real PyTorch model used as the serving backend stand-in.

This is intentionally a lightweight CNN (not a production model). The point
of this repo is the *serving infrastructure* (batching, caching,
backpressure, metrics, load testing) — not model accuracy. Weights are
randomly initialized with a fixed seed (not trained) so results are
deterministic across runs and therefore cacheable/testable.

Device selection is real: if a CUDA GPU is available (verified with
`torch.cuda.is_available()`), the model runs on it; otherwise it falls back
to CPU. See README.md "Status" section for what was actually verified in
the development environment used to build this repo.
"""
from __future__ import annotations

import threading

import torch
import torch.nn as nn

from inference_service.config import settings


class TinyCNN(nn.Module):
    """A small CNN over CIFAR-like 3x32x32 inputs -> NUM_CLASSES logits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ModelBackend:
    """Wraps the model, device placement, and a forward-pass counter.

    The forward-pass counter is what batching tests assert against: it lets
    tests prove that N concurrent logical requests resulted in fewer than N
    actual `model(...)` calls, i.e. that batching really happened.
    """

    def __init__(self) -> None:
        torch.manual_seed(settings.MODEL_SEED)
        self.device = torch.device(
            "cpu" if settings.FORCE_CPU or not torch.cuda.is_available() else "cuda"
        )
        self.model = TinyCNN(num_classes=settings.NUM_CLASSES).to(self.device)
        self.model.eval()
        self._lock = threading.Lock()
        self.forward_pass_count = 0
        self.total_items_processed = 0

    @torch.inference_mode()
    def infer_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Run one real forward pass for a batch of shape (B, 3, 32, 32).

        Returns softmax probabilities of shape (B, NUM_CLASSES) on CPU.
        """
        with self._lock:
            self.forward_pass_count += 1
            self.total_items_processed += batch.shape[0]
        batch = batch.to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        return probs.to("cpu")

    def reset_counters(self) -> None:
        with self._lock:
            self.forward_pass_count = 0
            self.total_items_processed = 0

    # V0.3: identificação para a interface plugável (backends/base.py).
    name = "tinycnn"
    kind = "tensor"

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "kind": self.kind,
            "device": str(self.device),
            "num_classes": settings.NUM_CLASSES,
        }


# Nome da V0.3 para o mesmo backend: o TinyCNN é um backend entre vários.
TinyCNNBackend = ModelBackend


# Module-level singleton used by the FastAPI app (one model instance per process).
model_backend = ModelBackend()
