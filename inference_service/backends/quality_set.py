"""Conjunto fixo e pequeno para medir degradação por quantização.

Duas medidas, ambas calculadas pelo próprio backend sob teste:

1. Perplexidade sobre ``PERPLEXITY_TEXTS`` (teacher forcing, sem geração).
2. Saídas greedy (decodificação determinística) para ``GREEDY_PROMPTS``, comparadas depois contra
   a configuração de referência (FP32) pelo benchmark.

É um sinal barato e reprodutível de que a quantização mexeu (ou não) na
distribuição do modelo — NÃO é um benchmark de qualidade de tarefa e não
substitui avaliação com dataset e juízes. Os textos são escritos aqui, sem
dados externos.
"""
from __future__ import annotations

PERPLEXITY_TEXTS: tuple[str, ...] = (
    "O processamento em lote dinâmico agrupa requisições que chegam em uma "
    "janela curta de tempo para executar uma única passagem pelo modelo.",
    "Controle de admissão rejeita novas requisições quando a fila atinge a "
    "capacidade, devolvendo contrapressão em vez de aumentar a latência de todos.",
    "Dynamic batching groups requests that arrive within a short window so that "
    "a single forward pass serves all of them on the GPU.",
    "Quantization reduces the memory footprint of a model by storing weights "
    "with fewer bits, trading a small amount of accuracy for lower VRAM usage.",
    "A latência de cauda, medida pelos percentis 95 e 99, mostra o que os piores "
    "casos enfrentam e costuma importar mais do que a média.",
    "The capital of France is Paris, and the capital of Brazil is Brasília.",
)

GREEDY_PROMPTS: tuple[str, ...] = (
    "Explique em uma frase o que é batching dinâmico.",
    "What is 2 + 2? Answer with just the number.",
    "Liste três cores primárias.",
    "Translate to English: 'O gato dorme no sofá.'",
    "Qual é a capital do Brasil?",
    "Write a one-line Python function that returns the square of x.",
    "Resuma em uma frase: servir modelos com baixa latência exige fila, lote e métricas.",
    "Name two benefits of quantizing a language model.",
)
