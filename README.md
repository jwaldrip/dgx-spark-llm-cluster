# DGX Spark LLM cluster field notes

Working launchers and measured numbers for serving large open-weight models across a
three-node NVIDIA DGX Spark cluster (GB10, SM121, arm64, 128 GB unified memory per node,
direct ConnectX-7 ring, no switch).

Everything here was measured on real hardware. Where a number is inferred rather than
measured, it says so. Where a claim comes from someone else's report, it is labelled and
linked. Vendor benchmark tables are not treated as evidence that a model works on this
hardware.

## What is running

| Nodes | Model | Serving | Role |
|---|---|---|---|
| 1 | Qwen3.8-Flash-Next NVFP4 | vLLM, single node | bounded implementation work |
| 2 | GLM-5.3-Flash NVFP4 | vLLM, tensor parallel 2 | long context, higher capacity |

The third node stays independent on purpose. Three separate schedulers, three KV pools and
three failure domains beat one large sharded model for concurrent agent traffic.

## Headline measurements

Per-node hardware, verified on all three:

```text
GPU        NVIDIA GB10        driver 580.173.02
arch       aarch64            Ubuntu 24.04.3
memory     121 GiB visible of 128 GB unified
disk       3.7 TB NVMe
docker     29.2.1
```

Interconnect, measured with `ib_write_bw` at 1 MiB messages:

```text
single rail                     109.12 Gb/s
single rail, RoCE MTU 4096      109.27 Gb/s
single rail, 8 queue pairs      111.86 Gb/s
both rails concurrently          98.28 Gb/s each  =  196.55 Gb/s
```

**About 110 Gb/s is a hard per-link ceiling, roughly 55% of the 200 Gb/s nominal link
speed.** Neither jumbo frames nor queue-pair count move it. Both rails together nearly
double it. See [docs/interconnect.md](docs/interconnect.md).

GLM-5.3-Flash at tensor parallel 2, live:

```text
weights per rank      89.2 GiB
available KV per rank  4.27 GiB
GPU KV cache size    332,475 tokens   (1.27 full 262K contexts)
model load                392 s
MoE backend            MARLIN
KV dtype               BF16, unquantized
```

Qwen3.8-Flash-Next on one node, live:

```text
resident weights      79.42 GiB   (47.7 GiB PLE table memory-mapped from NVMe)
GPU KV cache size    519,943 to 537,322 tokens
single stream          20.9 tok/s
four-way aggregate     42.3 tok/s
184K cold prefill        103.6 s
184K warm prefix hit       1.82 s
prefix cache hit rate     87.7%   over 22.4M queries
```

## Contents

- [docs/interconnect.md](docs/interconnect.md) - the fabric ceiling, why bonding matters
  for prefill and not decode, and the PCIe topology that decides which NCCL knob applies.
- [docs/checkpoint-selection.md](docs/checkpoint-selection.md) - why the official FP8
  GLM release does not fit two nodes, and why one NVFP4 quantization of it is unusable on
  vLLM today.
- [docs/gotchas.md](docs/gotchas.md) - the traps that cost a launch each.
- [docs/parallelism.md](docs/parallelism.md) - tensor versus pipeline versus expert
  parallel on a fabric 11x slower than local memory, with the per-token byte math.
- [launchers/](launchers/) - the actual scripts, with every departure from the upstream
  recipe annotated with its reason.

## What is not proven here

- Speculative decoding. The published 46.9 tok/s for GLM on two Sparks depends on a
  DFlash2 drafter. These launchers run without it. Reported throughput without speculative
  decoding is 14.3 tok/s, which is slower than the single-node Qwen worker.
- Rail bonding. The hardware sustains 196.55 Gb/s across both rails, but nothing here
  proves NCCL engages both on a two-rank job.
- Any quality comparison between the two models. That work is separate and is not in this
  repo yet.

## Credit

The two-node GLM path stands on
[tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark),
which did the hard day-zero debugging, published a container image, and documented seven
distinct GB10 bugs. The launchers here differ from theirs in five specific ways, each
annotated in place. Where we disagree with that recipe, the reason is stated and measured.

## License

MIT. See [LICENSE](LICENSE).
