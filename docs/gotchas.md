# Traps

Each of these cost a failed launch or a wrong conclusion. Ordered roughly by how much time
they burn before you understand them.

## A clean boot proves nothing on GB10

`--moe-backend` is not a tuning knob on this hardware. It decides between correct output and
confident garbage, and the wrong choice still returns HTTP 200 from `/health`.

Reported across two independent rigs, checkpoint held fixed, backend varied:

```text
flashinfer_trtllm    refuses to load: kernel does not support current device
flashinfer_cutedsl   refuses to load: same
flashinfer_cutlass   loads, 6/6 runs collapse into "locklocklock..."
cutlass              loads, 6/6 runs produce no content at all
emulation            loads, collapses into repeated </arg_value>
marlin               correct
```

Confirmed on GB10 specifically: the NVFP4 flashinfer-cutlass MoE kernels *run* and emit
garbage.

**Therefore your acceptance gate must read generated text.** A container that is up, a
`/health` that returns 200, and a `/v1/models` that lists the model are all compatible with
an engine that produces nothing but repetition. Also: probe `/health`, never `/v1/models`,
for liveness. The latter can answer 200 from config alone with a dead engine behind it.

## Hugging Face cache snapshots are symlink farms

Bind-mounting the snapshot directory into a container gives you a directory of dangling
symlinks, because every file points into `../../blobs/`. The failure surfaces as something
unrelated-sounding:

```text
ValueError: The supplied chat template string (/models/.../chat_template.jinja)
appears path-like, but doesn't exist!
```

Mount the **repo root** and point the model path at the snapshot inside it:

```bash
-v "$HOME/.cache/huggingface/hub/models--ORG--NAME:/models/repo:ro"
# then serve /models/repo/snapshots/<revision>
```

## `enable_thinking: false` leaks deliberation into content

Published recipes set `--default-chat-template-kwargs '{"enable_thinking":false}'`. With
that setting, GLM-5.3-Flash's internal deliberation lands in `message.content`:

```text
CONTENT: 'The user is asking a simple factual question... They want it in one sentence.
          The capital of Germany is Berlin. Why is Berlin significant? ...'
```

With `true`, content is clean and tool calls carry no prose at all:

```text
plain request     CONTENT: 'The capital of Germany is Berlin.'
tool call         tool_calls: [...]   content: None   finish_reason: tool_calls
tool result       CONTENT: 'Build status for **spark**: Status: green, Revision: ...'
```

Note that `reasoning_content` stays `None` either way, so the `glm45` reasoning parser is
not populating that field on this path. It does not matter once thinking is enabled, because
the visible content is correct, but do not expect a separated reasoning channel.

Tool and reasoning parser names for this model are `glm47` and `glm45` respectively. Do not
use `--tool-call-parser glm`; output gets swallowed.

## Reasoning tokens are charged against `max_tokens`

With thinking enabled, GLM spends a large and variable reasoning budget before it emits
anything. If `max_tokens` is too small the request returns `finish_reason: length` with
**empty content** and a 200 status.

Measured on the same prompt, a three-sentence answer in Korean:

```text
max_tokens=220    content_len=0    completion=220   finish=length
max_tokens=600    content_len=0    completion=600   finish=length
max_tokens=1500   content_len=213  completion=907   finish=stop
max_tokens=4000   content_len=234  completion=851   finish=stop
```

About 700 reasoning tokens for a trivial question. It is task dependent rather than fixed:
an arithmetic word problem spent 103 tokens at low effort, 112 at high and 198 at max.

Two consequences. Any probe or role with a tight token cap will silently return nothing, and
an empty string passes naive content checks. And cheap, high-frequency roles are the wrong
place for this model: at roughly 14 to 20 tok/s decode, 700 reasoning tokens is most of a
minute for a one-line answer. Put those roles on a smaller model.

## Temperature 0 is not reproducible on this endpoint

Measured serially against an idle engine, no concurrent traffic, identical request four
times:

```text
sha1=dd2e1893c8cf996b  len=142  completion=380
sha1=7e8b289a6f4a2ba4  len=186  completion=303
sha1=dd2e1893c8cf996b  len=142  completion=219
sha1=dd2e1893c8cf996b  len=142  completion=369
```

Final content is usually stable and occasionally is not. Reasoning length is never stable.
Suspected mechanisms are cross-rank reduction ordering under tensor parallelism, MoE routing
sensitivity to batch numerics, and marlin's emulated FP4 path. None of them are reachable
from configuration.

Practical rule: **never build a one-shot A/B comparison against this endpoint.** Sample and
compare distributions, or you will attribute your own noise to whatever you changed.

## Never pin the KV cache size

Published launchers carry `--kv-cache-memory` / `--kv-cache-memory-bytes` pins. The same
repository's own README explains why that is wrong: with the pool pinned, vLLM never
subtracts the measured activation peak, `--gpu-memory-utilization` becomes dead, and the
engine dies on the first long prompt.

Let the profiler size it, then read the `Available KV cache memory` line on **every rank**.
The pool is built from the minimum across ranks, and the head rank's log line is not
representative.

## Prefix caching does not work on this architecture

Independently reported at 0 hits out of 22,825 queries on GLM-5.3-Flash. Do not carry
`--enable-prefix-caching` into the launch expecting it to help.

For contrast, on the Qwen3.8-Flash-Next single-node deployment in this cluster, prefix
caching runs at 87.7% hit rate over 22.4 million queries and is worth 103.6 s versus 1.82 s
on a 184K-token prefill. Same cluster, same engine family, completely different value.

## Swap must be on, with swappiness zero

Both extremes fail, for different reasons. Default swappiness pages the vLLM worker out
mid-load into a UVM driver livelock, one thread pinned at 100% CPU with shard loading frozen
and no self-recovery. Fully disabling swap kills the worker during the MoE repack with no
valve.

```bash
printf 'vm.swappiness = 0\n' | sudo tee /etc/sysctl.d/90-gb10-vllm.conf
sudo sysctl --system && sysctl vm.swappiness
```

The runtime value does not survive a reboot without that file. Verify after reboot rather
than assuming.

Also drop caches on both nodes before each launch, because unified memory means the page
cache is competing with your model:

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## Launch order matters, and so does image parity

Worker rank first, roughly 25 seconds of head start, then the head rank. Tear both ranks
down fully before relaunching either. Confirm the image digest matches on both nodes before
every launch: a silent image mismatch between ranks is a real and reproduced failure.

## `--help` needs a GPU

Building vLLM's argument parser infers a device, so `docker run --rm <image> --help` without
`--gpus all` dies with `Failed to infer device type`. Use `--help=all` with the GPU attached
to enumerate the real flag surface, and check spellings rather than trusting a recipe. On
this image it is `--kv-cache-memory-bytes`, not `--kv-cache-memory`.

## `pkill -f` matches the script that calls it

Not GB10 specific, but it cost a download. A script containing
`pkill -f 'hf download SOMEREPO'` matches its own parent shell, whose command line contains
that string, and kills itself partway through. Kill by recorded PID instead.

## Binding the wrong RDMA device reads like a cable fault

On a ring, the device that owns an address is not always the one you expect. Dialing an
address bound to `enp1s0f1np1` while binding `rocep1s0f0` produces:

```text
Failed to modify QP 297 to RTR
Unable to Connect the HCA's through the link
```

Resolve the mapping first with `ip -br -4 addr` and
`ls /sys/class/net/<iface>/device/infiniband`.

## An engine that is up is not an engine that is fast

Reported GLM-5.3-Flash throughput on two Sparks is 14.3 tok/s without speculative decoding
and 46.9 tok/s with a DFlash2 drafter at 74.1% draft acceptance. The single-node
Qwen3.8-Flash-Next worker in this cluster measures 20.9 tok/s single stream. Bringing up the
bigger model on two nodes can therefore make generation *slower* than the one-node model it
was meant to surpass, unless the speculative path works.
