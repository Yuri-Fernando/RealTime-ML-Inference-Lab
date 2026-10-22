# Rodada preservada, mas NÃO usar para comparar latência/vazão

Esta é a primeira rodada completa do benchmark de quantização (`quant-20261021-150837`).
Ela foi **preservada** por transparência, mas a comparação de latência e vazão entre os modos
está **contaminada**: a GPU (RTX 3060 Ti, 8 GB) é compartilhada com outros processos da máquina e,
durante a rodada, esses processos estavam usando a GPU.

Evidências no próprio relatório:

- baseline ocioso antes do servidor: FP16 com 94% de uso e 6210 MiB; BF16 com 89,5% e 7012 MiB
  (versus ~1600 MiB e <10% quando a GPU está livre);
- "VRAM processo (nvidia-smi)" negativa no BF16 (outro processo liberou memória no meio da rodada);
- FP16 a concorrência 4: P95 de 342 985 ms (343 s), claramente um estancamento por disputa de GPU,
  enquanto BF16 (mesmo custo computacional) fez P95 de 11,9 s.

O que continua válido nesta rodada (por-processo, independe da disputa): memória dos pesos,
pico de VRAM alocada pelo PyTorch, perplexidade e concordância das saídas greedy.

A rodada substituta usa `--idle-wait-s` (espera a GPU ficar ociosa) e marca cada modo com
"Medição limpa?". Os números do README vêm dela.
