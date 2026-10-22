"""HuggingFaceBackend: geração de texto com um SLM local, em 5 precisões.

Configurações suportadas (``QUANT_MODES``):

| modo | pesos          | como                                           |
|------|----------------|------------------------------------------------|
| fp32 | float32        | ``dtype=torch.float32``                        |
| fp16 | float16        | ``dtype=torch.float16``                        |
| bf16 | bfloat16       | ``dtype=torch.bfloat16``                       |
| int8 | LLM.int8()     | bitsandbytes ``load_in_8bit=True``             |
| nf4  | NF4 4-bit      | bitsandbytes ``load_in_4bit`` + ``nf4``        |

int8 e nf4 exigem GPU CUDA (bitsandbytes). Nada é simulado: o modelo é
carregado de verdade e a memória reportada vem de ``get_memory_footprint()`` e
``torch.cuda.max_memory_allocated()``.

Batching de geração: o tokenizer usa padding à ESQUERDA (obrigatório para
decoder-only em lote) e cada sequência tem seu próprio orçamento de tokens e
seu próprio ``threading.Event`` de cancelamento, ambos avaliados por um
``StoppingCriteria`` por sequência a cada passo de decodificação. Uma sequência
cancelada deixa de gerar tokens imediatamente (o slot do lote continua
existindo, mas o lote inteiro termina assim que todas as sequências param).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from inference_service.backends.base import (
    BackendConfigError,
    GenerationRequest,
    GenerationResult,
)
from inference_service.backends.quality_set import GREEDY_PROMPTS, PERPLEXITY_TEXTS
from inference_service.config import settings


@dataclass(frozen=True)
class QuantSpec:
    mode: str
    dtype_name: str  # nome do dtype dos pesos não quantizados
    bits: int  # bits efetivos por peso nas camadas lineares
    needs_cuda: bool
    bnb: Optional[str]  # None | "int8" | "nf4"


QUANT_MODES: dict[str, QuantSpec] = {
    "fp32": QuantSpec("fp32", "float32", 32, False, None),
    "fp16": QuantSpec("fp16", "float16", 16, True, None),
    "bf16": QuantSpec("bf16", "bfloat16", 16, False, None),
    "int8": QuantSpec("int8", "float16", 8, True, "int8"),
    "nf4": QuantSpec("nf4", "float16", 4, True, "nf4"),
}


def resolve_quant(mode: str, cuda_available: bool) -> QuantSpec:
    """Valida o modo contra o ambiente. Função pura: não carrega nada."""
    key = (mode or "").strip().lower()
    if key not in QUANT_MODES:
        raise BackendConfigError(
            f"modo de quantização desconhecido: {mode!r} (válidos: {sorted(QUANT_MODES)})"
        )
    spec = QUANT_MODES[key]
    if spec.needs_cuda and not cuda_available:
        raise BackendConfigError(
            f"modo {key!r} exige GPU CUDA neste backend (bitsandbytes/fp16); "
            "sem GPU use fp32 ou bf16"
        )
    return spec


def build_load_kwargs(spec: QuantSpec) -> dict:
    """kwargs para ``from_pretrained`` (importa torch/transformers sob demanda)."""
    import torch

    dtype = getattr(torch, spec.dtype_name)
    kwargs: dict = {"dtype": dtype}
    if spec.bnb == "int8":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif spec.bnb == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    return kwargs


class HuggingFaceBackend:
    """Backend de texto sobre ``transformers`` (execução local, em processo)."""

    kind = "text"
    supports_batching = True

    def __init__(
        self,
        model_id: Optional[str] = None,
        quant: Optional[str] = None,
        force_cpu: Optional[bool] = None,
        local_files_only: Optional[bool] = None,
        use_chat_template: Optional[bool] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id or settings.HF_MODEL
        force_cpu = settings.FORCE_CPU if force_cpu is None else force_cpu
        cuda = torch.cuda.is_available() and not force_cpu
        self.spec = resolve_quant(quant or settings.HF_QUANT, cuda_available=cuda)
        self.name = f"hf-{self.spec.mode}"
        self.device = torch.device("cuda" if cuda else "cpu")
        local_only = settings.HF_LOCAL_ONLY if local_files_only is None else local_files_only
        self.use_chat_template = (
            settings.HF_CHAT_TEMPLATE if use_chat_template is None else use_chat_template
        )

        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left", local_files_only=local_only
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = build_load_kwargs(self.spec)
        t0 = time.perf_counter()
        if self.spec.bnb is not None:
            # bitsandbytes posiciona os pesos direto na GPU durante o load.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map={"": 0}, local_files_only=local_only, **load_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, local_files_only=local_only, **load_kwargs
            ).to(self.device)
        self.load_seconds = time.perf_counter() - t0
        self.model.eval()

        gen_cfg = getattr(self.model, "generation_config", None)
        eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        eos_ids = eos if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else [])
        stop = set(int(x) for x in eos_ids)
        if self.tokenizer.eos_token_id is not None:
            stop.add(int(self.tokenizer.eos_token_id))
        if self.tokenizer.pad_token_id is not None:
            stop.add(int(self.tokenizer.pad_token_id))
        self._stop_ids = stop

        self._lock = threading.Lock()  # contadores
        self._gen_lock = threading.Lock()  # uma geração por vez na GPU
        self.forward_pass_count = 0  # = lotes de geração executados
        self.total_items_processed = 0
        self.tokens_generated_total = 0

    # ------------------------------------------------------------------ util
    def _prompt_text(self, prompt: str) -> str:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def reset_counters(self) -> None:
        with self._lock:
            self.forward_pass_count = 0
            self.total_items_processed = 0
            self.tokens_generated_total = 0

    def describe(self) -> dict:
        torch = self._torch
        info = {
            "backend": self.name,
            "kind": self.kind,
            "model": self.model_id,
            "quant": self.spec.mode,
            "bits": self.spec.bits,
            "dtype": self.spec.dtype_name,
            "device": str(self.device),
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
            "memory_footprint_bytes": int(self.model.get_memory_footprint()),
            "load_seconds": round(self.load_seconds, 3),
        }
        if torch.cuda.is_available() and self.device.type == "cuda":
            info["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        return info

    # ------------------------------------------------------------- geração
    def generate_batch(
        self,
        requests: Sequence[GenerationRequest],
        cancel_events: Optional[Sequence[threading.Event]] = None,
    ) -> list[GenerationResult]:
        """Uma geração em lote (bloqueante). Uma "forward pass" do contador é
        um lote inteiro de geração."""
        if not requests:
            return []
        torch = self._torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria
        from transformers import StoppingCriteriaList

        n = len(requests)
        events = list(cancel_events) if cancel_events is not None else [threading.Event() for _ in requests]
        budgets = [max(1, int(r.max_new_tokens)) for r in requests]

        class _FirstToken(LogitsProcessor):
            """Marca o instante em que o primeiro token é decidido (fim do prefill)."""

            def __init__(self) -> None:
                self.t_first: Optional[float] = None

            def __call__(self, input_ids, scores):
                if self.t_first is None:
                    self.t_first = time.perf_counter()
                return scores

        prompt_len_holder: dict = {}

        class _PerSequenceStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                gen_len = input_ids.shape[1] - prompt_len_holder["len"]
                done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                for i in range(input_ids.shape[0]):
                    if gen_len >= budgets[i] or events[i].is_set():
                        done[i] = True
                return done

        with self._gen_lock:
            with self._lock:
                self.forward_pass_count += 1
                self.total_items_processed += n

            texts = [self._prompt_text(r.prompt) for r in requests]
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not (
                    self.use_chat_template and getattr(self.tokenizer, "chat_template", None)
                ),
            ).to(self.device)
            prompt_len = int(enc["input_ids"].shape[1])
            prompt_len_holder["len"] = prompt_len
            prompt_tokens = [int(x) for x in enc["attention_mask"].sum(dim=1).tolist()]

            first = _FirstToken()
            t_start = time.perf_counter()
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max(budgets),
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    logits_processor=LogitsProcessorList([first]),
                    stopping_criteria=StoppingCriteriaList([_PerSequenceStop()]),
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ttft_ms = (
                (first.t_first - t_start) * 1000.0 if first.t_first is not None else None
            )

        generated = out[:, prompt_len:].tolist()
        results: list[GenerationResult] = []
        total_tokens = 0
        for i in range(n):
            # Só os primeiros `budgets[i]` tokens pertencem a esta sequência:
            # depois do orçamento (ou do cancelamento) o generate() completa
            # com pad_token, que NÃO deve ser lido como um EOS genuíno.
            ids = generated[i][: budgets[i]]
            count = len(ids)
            hit_stop = False
            for j, tok_id in enumerate(ids):
                if tok_id in self._stop_ids:
                    count = j
                    hit_stop = True
                    break
            cancelled = events[i].is_set()
            if cancelled:
                finish = "cancelled"
            elif hit_stop:
                finish = "stop"
            else:
                finish = "length"
            text = self.tokenizer.decode(ids[:count], skip_special_tokens=True)
            total_tokens += count
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens[i],
                    completion_tokens=count,
                    ttft_ms=ttft_ms,
                    finish_reason=finish,
                    cancelled=cancelled,
                )
            )
        with self._lock:
            self.tokens_generated_total += total_tokens
        return results

    async def agenerate(self, request: GenerationRequest) -> GenerationResult:  # pragma: no cover
        import asyncio

        return (await asyncio.to_thread(self.generate_batch, [request]))[0]

    # ------------------------------------------------------------ qualidade
    def evaluate_quality(self, max_new_tokens: int = 24) -> dict:
        """Perplexidade + saídas gregas sobre o conjunto fixo (quality_set.py).

        Não altera os contadores de serving.
        """
        torch = self._torch
        nll = 0.0
        count = 0
        with self._gen_lock, torch.inference_mode():
            for text in PERPLEXITY_TEXTS:
                enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                n_tokens = int(enc["input_ids"].shape[1])
                if n_tokens < 2:
                    continue
                out = self.model(**enc, labels=enc["input_ids"])
                nll += float(out.loss.item()) * (n_tokens - 1)
                count += n_tokens - 1
        perplexity = math.exp(nll / count) if count else float("nan")

        saved = (self.forward_pass_count, self.total_items_processed, self.tokens_generated_total)
        results = self.generate_batch(
            [GenerationRequest(p, max_new_tokens) for p in GREEDY_PROMPTS]
        )
        with self._lock:
            (
                self.forward_pass_count,
                self.total_items_processed,
                self.tokens_generated_total,
            ) = saved
        return {
            "model": self.model_id,
            "quant": self.spec.mode,
            "perplexity": perplexity,
            "perplexity_tokens": count,
            "greedy_prompts": list(GREEDY_PROMPTS),
            "greedy_outputs": [r.text for r in results],
        }
