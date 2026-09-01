# Tensor, pipeline or expert parallel on a slow fabric

The interesting question on a two-node Spark pair is not which parallelism scheme is
theoretically best. It is whether a fabric 11x to 20x slower than local memory changes the
usual answer. It does not, and the byte math shows why.

Architecture constants, from GLM-5.3-Flash `config.json`:

```text
hidden_size            4096
num_hidden_layers        45      (34 KDA linear attention + 11 sparse MLA)
first_k_dense_replace     3      so 42 of 45 layers are MoE
n_routed_experts        288
num_experts_per_tok       8
activations            bf16      2 bytes per element
```

Measured fabric: 13.7 GB/s on one rail, 24.6 GB/s bonded. Local memory: about 273 GB/s.

## Per token, per direction, during decode

| scheme | collectives per token | payload per token | at 13.7 GB/s | at 24.6 GB/s | per 8192-token prefill chunk |
|---|---:|---:|---:|---:|---:|
| **TP2** | 90 all-reduce | 0.70 MiB | 54 us | 30 us | 6.04 GB |
| PP2 | 1 send | 8 KiB | 0.6 us | 0.3 us | 67 MB |
| EP2 | 84 all-to-all | 2.69 MB | 196 us | 110 us | 22.0 GB |

Tensor parallel does two all-reduces per layer, one after attention and one after the MLP,
each carrying `hidden_size` per token. Pipeline parallel sends one activation hop per token
across the stage boundary. Expert parallel dispatches each token's hidden vector to whichever
rank holds each of its top-8 experts and combines the results back.

## Tensor parallel wins anyway

Against an observed 21.3 ms per token, TP2's wire time is **0.14% to 0.25% of decode**.
Bandwidth is not the constraint. The real cost is 90 collective *launches* per token, each
only 8 KiB, so the exposure is latency rather than throughput. Settling that number needs
`all_reduce_perf -b 8K -e 8K` on the real fabric, which is not measured here. The nearest
available anchor is a torch profile on an on-node NVLink TP2 setup showing all-reduce at 11%
of decode time, which is an optimistic floor for a cross-node RoCE pair.

Memory is a wash across all three schemes. TP shards every layer's weights, PP splits the
layer stack roughly 22/23, EP splits the routed experts. All three land about 90 GiB per
node. **No scheme buys headroom over another**, so memory cannot break the tie.

## Pipeline parallel is 90x cheaper on wire and still loses

PP2 moves 8 KiB per token instead of 0.70 MiB, and its prefill chunk is 67 MB instead of
6.04 GB. On a slow fabric that looks like the obvious answer, and vLLM's own multi-node
documentation recommends pipeline over tensor parallel for links without NVLink.

The catch is that guidance targets aggregate throughput at meaningful concurrency. At a
single agent stream there is exactly one microbatch in flight, so one node idles while the
other computes its half. Expected single-stream throughput roughly halves and per-token
latency roughly doubles. That is standard pipeline-bubble reasoning rather than a
measurement, and nobody appears to have run PP for this model on two Sparks. Engine support
for pipeline parallel on `glm5_next` is also unverified.

If your workload is many concurrent requests rather than one interactive agent, revisit this.
The wire math genuinely favors it.

## Expert parallel is measured, and it loses

EP2 needs 3.7x more wire than TP2 across 84 small all-to-alls, which has a worse latency
profile than a two-rank ring all-reduce, and buys no memory. It is also directly measured as
a regression on far better-connected hardware: `--enable-expert-parallel` produced 1756 tok/s
against a 1863 tok/s plain baseline at 64 concurrency on an eight-GPU on-node setup, a 5.7%
loss. A second independent reporter lists it under switches that did nothing, and notes
exactly 0% gain at TP2 even after retuning for TP2 shapes.

Expert parallel's payoff mechanism is expert locality and hidden all-to-all latency under
high concurrency with data-parallel attention. None of that exists at single-stream
operation.

## Where the fabric actually costs you

Prefill, not decode. At `--max-num-batched-tokens 8192`, a 262,144-token prompt is 32 chunks:

```text
exposed wire, single rail    14.1 s
exposed wire, bonded         7.9 s
```

That is wire time if unhidden by compute overlap, so treat it as an upper bound. Cross-checked
against a reported two-node TTFT of about 6.6 s at 16K tokens warm, wire accounts for roughly
13% of TTFT single rail and 7% bonded.

**So rail bonding is a prefill optimization.** If your prompts are long, it is worth the
verification effort. If your bottleneck is generation speed, it is not.
