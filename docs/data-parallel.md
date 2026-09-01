# One endpoint across the cluster

The instinct that a three-node cluster should present a single endpoint is right. The
instinct that this means sharding the model is wrong, at least for a model that fits one
node.

## Tensor parallel is not the answer here, and for this checkpoint is not even legal

Qwen3.8-Flash-Next NVFP4 fits one Spark at 79.42 GiB resident, with 47.7 GiB of PLE table
memory-mapped from NVMe. Sharding it would buy no memory and cost cross-node collectives on
a fabric measured at 24.6 GB/s per neighbour against roughly 273 GB/s of local memory.

Even if you wanted to, the arithmetic forbids TP3:

```text
num_attention_heads   24     /3 = 8       ok
num_key_value_heads    2     /3           ILLEGAL
hidden_size         2560     /3           not an integer
num_experts          512     /3           not an integer
```

With 2 KV heads, tensor parallelism is capped at 2 for the attention path regardless of
anything else.

## Data parallel is the answer

```text
--data-parallel-size 3 --data-parallel-size-local 1
--data-parallel-rank <n> --data-parallel-address <rank-0 ring IP>
--data-parallel-rpc-port 29551 --data-parallel-external-lb
```

Each replica is a full copy on its own node. One logical endpoint, three schedulers, three
KV pools, three failure domains.

```text                     one node        DP=3
KV pool                   519,943      1,559,829 tokens
concurrent @ 122,835      4.23             12.7 requests
generation, single         20.9            ~62.7 tok/s
generation, 4-way          42.3           ~126   tok/s
```

The concurrency figure is the one that matters for agent work, and it is derived rather than
guessed: KV pool tokens divided by the observed mean prompt length. Setting an admission cap
any other way is how you end up in preemption thrash, which is documented in
[gotchas.md](gotchas.md).

## The part that will bite you: caches are per replica

**vLLM's DP load balancer routes by queue depth.** That is correct for a stateless service.
This one is not stateless: each replica keeps its own prefix cache, and on this workload the
cache is worth more than everything else combined.

```text
prefix cache hit rate, single node    87.7%   over 22.4M queries
prompt : generation                  181 : 1
184K prefill, cold                   103.6 s
184K prefill, warm                     1.82 s
```

Route a session's turns by load and each hop is a cold prefill. Three replicas with
load-based routing tends toward a 1-in-3 chance of landing warm, so you would spend two
extra nodes to make prefill roughly three times worse.

That is why `--data-parallel-external-lb` is set and
[`router/prefix-affinity-router.ts`](../router/prefix-affinity-router.ts) sits in front. The
router owns placement, vLLM owns execution.

## How the router decides

It hashes the **stable prefix** of a request, not the body: the system message, the tool
schemas, and the first user turn. That is exactly the span a growing conversation keeps
re-sending unchanged, so the key does not move as turns accumulate. Keying on the whole
conversation would change the key every turn, which is identical to having no affinity.

Two choices that read as backwards and are not:

**Load vetoes, it never leads.** A busy-but-under-cap replica keeps its session. When every
replica is saturated the request queues on the preferred one rather than spraying, because
waiting for a warm cache beats a cold prefill elsewhere when prefill is 99% of the cost.

**Rendezvous hashing, not modulo.** Removing or adding a replica rehomes only that
replica's share of sessions. Modulo would reshuffle everything and cold-start the whole
fleet's cache on any topology change.

Displacement is deterministic: a session pushed off its first choice always lands on the
same second choice, so that replica's cache stays useful too.

## Verify placement, do not assume it

```bash
./launchers/measure-cache-affinity.sh http://127.0.0.1:18010 \
    http://10.100.138.2:8000 http://10.100.138.1:8000 http://10.100.140.1:8000
```

It drives one growing session and reads each replica's real `prefix_cache_*` counters,
because the router believing it was sticky is not evidence the engine got a hit. **The
verdict is the replicas that show no traffic at all.** A scattered session shows queries on
every one of them.

Two things to know about the numbers. The counters are per engine, not per session, so run
against an idle fleet or read the rate as fleet-wide. And a fresh session scores well below
the lifetime rate: measured on one live node, an 8-turn session with a 9,057-token stable
prefix reached 61.2%, because turn 0 is entirely cold.

## Router tests

```text
bun test router/
13 pass, 0 fail
```

Including a real-HTTP integration test that spawns the router against three mock replicas and
confirms a growing conversation lands on exactly one of them, `affinityRate 1.0`.

The suite was checked against a mutant: keying on every user turn instead of the first breaks
`a session stays pinned to one replica`, expected 1 replica, received 3. A test that cannot
fail is not a test.
