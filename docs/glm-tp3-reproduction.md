# Reproducing GLM-5.3-Flash at TP=3 on three DGX Sparks

Someone published a working three-node config for this exact model on this exact
hardware. It beats every number measured in this repo, and it is the only
configuration found that serves the real main lane. This document records what
they published, the one place their recipe cannot be followed verbatim, and the
plan that follows from that.

Source: <https://github.com/FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks>, author
Christopher Kapic, 4 commits, no releases, source only, nothing pinned.

## Why this displaces the data-parallel design in docs/data-parallel.md

The workload measured from session transcripts: main-lane prompt p95 782,561
tokens, max 908,183, mean 122,835, at 4 to 10 concurrent sessions.

    configuration            max_model_len    KV tokens    conc @ mean    serves p95?
    GLM TP=3 "1m"            1,048,576        5,476,368    44.6           yes
    GLM TP=3 "big-pool"      3,763,638        3,763,638    48.3           yes
    GLM TP=2 (measured here) 262,144          332,475      2.71           no
    Qwen DP=3 (this repo)    262,144          1,559,829    12.7           no

Qwen3.8-Flash-Next caps at 262,144. No amount of replication changes that, so
the data-parallel work in this repo cannot serve the lane it was built for. That
is the whole reason to switch targets, and it was known before the DP launcher
was written.

Their per-rank weight figure is independently consistent: 163.1 GiB of NVFP4
expert weight over 3, plus about 7 GiB of non-expert and vision tensors, gives
about 63.1 GiB against their measured 63.64 GiB (`results/t-ep.md`). The same
arithmetic at TP=2 predicts about 90 GiB against the 92.1 GiB measured here. The
KV gain comes from sharding expert *weights*; MLA KV stays a full per-rank copy.

## What they published

Padding, which is what makes TP=3 legal at all. GLM's real
[config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json)
has `num_attention_heads` 64, `num_key_value_heads` 64, `moe_intermediate_size`
2048, `n_routed_experts` 288, `qk_rope_head_dim` 0, `kv_lora_rank` 512.

    field                   stock    padded    why
    num_attention_heads     64       66        66/3 = 22, satisfies divide()
    moe_intermediate_size   2048     2112      2112/3 = 704, AND 704 % 16 == 0
    vocab (154,880)         -        lcm(64,3) padded in-module at runtime
    n_routed_experts        288      288       already 288/3 = 96, untouched

Both pads are exactly 33/32, a 3.125% tax on attention and MoE FLOPs running on
zero weights. Not a memory cost.

Fabric, and this is a different mechanism from ours, not a stricter version of
it. They **disable stock IB entirely** (`NCCL_IB_DISABLE=1`) and route all
tensor-parallel traffic through a third-party NCCL net plugin
(`NCCL_NET=Mesh`, <https://github.com/autoscriptlabs/nccl-mesh-plugin>). Their
`docs/FABRIC.md` explains why, and it matches the topology here:

> Three DGX Sparks with two ConnectX-7 ports each almost never form **one** RoCE
> `/24`. The usual wiring is a **triangle of pairwise links**. There is no single
> HCA that can see all three peers.
>
> Pinning `NCCL_IB_GID_INDEX` makes it worse: the IPv4-mapped RoCEv2 slot is not
> the same index on every port.

So `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, which is the correct lever for a
two-node link and is documented in docs/interconnect.md, is not the lever here.
Under TP=3 across a triangle, no HCA sees all three peers and the plugin owns
routing. Both statements can be true at once; they describe different topologies.

Engine knobs: `--block-size 2304`, `--kv-cache-dtype fp8_e4m3`,
`--gpu-memory-utilization 0.85`, MTP via
`--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`, and a
2200 MHz clock cap applied per boot on the host, outside every script:

```bash
sudo nvidia-smi -i 0 -lgc 0,2200
```

Things they measured as not working, worth not rediscovering: `--block-size` 64
and 128 fail to serve on SM121 because fp8 DeepGEMM `block_kv` must be 64-aligned
as a kernel size and a user block-size conflicts with it at warmup;
`flashinfer_cutlass` as the MoE backend fails to compile on their image
(`nvrtc.h: No such file or directory`).

## The one place their recipe cannot be followed

Their `scripts/start.sh` defaults `--moe-backend marlin`, and per their own
`results/t7-moe-backend.md` marlin is not a preference, it is the only backend
that boots on their image. Their weights are
[LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4),
ModelOpt NVFP4, fetched from `main` with no `--revision`.

That combination is the exact code path of
[vllm-project/vllm#54150](https://github.com/vllm-project/vllm/issues/54150),
open and unfixed, no PR: `ModelOptNvFp4FusedMoE`'s marlin repack in
`modelopt.py` takes only the gate global scale for the fused w13 GEMM and
applies it to the up shard too, mis-scaling up-projection output by
`ws2_gate/ws2_up` per expert. Ratio mean 1.15, p90 1.29, **max 10.0**, on 69.1%
of expert pairs.

The issue's own measured output corruption is 86 U+FFFD across 6 runs on the
LibertAIDAI checkpoint against 0/6 on RedHatAI's, which matches the corruption
measured independently here. A corrupted token inside a tool-call block desyncs
the parser into a repetition lock, so this is disqualifying for agent serving
regardless of how good the throughput is.

None of their four bind-mounted overlays
(`vocab_parallel_embedding.py`, `weight_utils.py`, `parameter.py`, `model.py`)
touch `modelopt.py` or carry either candidate fix from the issue comments.

RedHatAI's checkpoint is structurally immune because its gate and up scales are
shared by construction, not merely equal in practice.

Also worth recording: their overlay notes cite vLLM commit `g487ecf187` as the
tree their base image was built from. That is the same otherwise-unresolvable
commit this repo's GLM image reports, which corroborates that both descend from
the same day-0 GLM-5.3-Flash fork build.

## A second defect in their padding script

`scripts/pad-tp3-config.py` computes `moe_intermediate_size` by naive
ceil-to-multiple-of-3, which yields **2049**, not the 2112 the recipe actually
serves. 2112 is the only value that satisfies both the TP divisibility check and
NVFP4's 16-element quant-group alignment (`verify_group_size_divides_partition`).
2049 fails the second one.

The correct value lives only as a hardcoded literal in `start.sh`'s
`HF_OVERRIDES`. So running their generic script against a fresh checkout writes
an illegal MoE width, and because native MTP's `SpeculativeConfig` reads
`config.json` directly rather than the `--hf-overrides`, the two disagree. Pad to
2112 explicitly; do not trust the script.

## Plan

Reproduce their topology, not their checkpoint.

1. Build the mesh plugin and the TP=3 image from their Dockerfile and four
   overlays. Pin the base image and the weights by digest, since they pin
   nothing.
2. Pad `config.json` to 66 / 2112 explicitly, not via their script.
3. Boot TP=3 on **RedHatAI compressed-tensors** (rev `36c184c6`), which is what
   is already served here and is immune to #54150.
4. Gate on tool-call fidelity before any OMP role points at it. The acceptance
   suite in this repo already covers this: repository repair, cross-file repair,
   and a U+FFFD scan. A green `/health` proves nothing.
5. Only then measure the 1m variant against a real 782K prompt.

The honest risk on step 3: nobody has run RedHatAI's checkpoint at TP=3.
RedHatAI's own vendor command uses TP=4, where 64/4 divides evenly and no head
padding is needed, so the padding and overlay machinery has never been exercised
against compressed-tensors' loader. It is W4A4 mixed-precision with the MTP layer
at FP8 rather than BF16. The substitution may simply not load.

If it does not, the fallback is not to accept corrupted tool calls. It is to run
LibertAIDAI plus marlin with one of the two candidate #54150 diffs applied as a
fifth overlay. The root cause is known and the patch is small, which makes that a
real option rather than a hope.

## What this repo got wrong

The DP work in docs/data-parallel.md was built after a scout had already reported
this recipe, and after the note "do not cut over to Qwen DP=3 yet, test GLM TP=3
big-pool first" was written. Building the invented path anyway cost three node
wedges. The published config was the right target from the moment it was found.
