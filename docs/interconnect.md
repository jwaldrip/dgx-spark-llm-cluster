# The fabric

Three DGX Sparks wired as a direct ring with ConnectX-7, no switch. Each adjacent pair has
two parallel rails on separate /24s. Twelve links total, all reporting 200000 Mb/s.

Node names below are generic. Substitute your own.

```text
node-a  <-> node-b     10.100.140.x , 10.100.141.x
node-b  <-> node-c     10.100.138.x , 10.100.139.x
node-c  <-> node-a     10.100.136.x , 10.100.137.x
```

## The ceiling is about half of nominal

Measured with `ib_write_bw`, 1 MiB messages, 5000 iterations, between two nodes on one rail:

| configuration | network MTU | RoCE MTU | queue pairs | BW average |
|---|---:|---:|---:|---:|
| baseline | 1500 | 1024 | 1 | 109.12 Gb/s |
| jumbo frames | 9000 | 4096 | 1 | 109.27 Gb/s |
| multi-QP | 9000 | 4096 | 8 | 111.86 Gb/s |

**Raising MTU bought 0.14%. Eight queue pairs bought 2.5%.** About 110 Gb/s, or 13.7 GB/s,
is the real per-link ceiling on GB10, roughly 55% of the 200 Gb/s the driver reports.

Do not spend effort on jumbo frames expecting throughput. The one unmeasured reason to set
them anyway is small-message latency: a RoCE MTU of 4096 carries an 8 KiB collective in two
packets instead of eight. That is unproven here and would need `all_reduce_perf -b 8K`.

Both rails driven at once, one `ib_write_bw` pair per rail, simultaneously:

```text
rail 1    98.28 Gb/s
rail 2    98.27 Gb/s
          -----------
total    196.55 Gb/s   (24.6 GB/s)
```

So bonding nearly doubles cross-node bandwidth at roughly a 10% per-rail cost. Against
GB10's approximately 273 GB/s local memory bandwidth, the fabric is about 20x slower on one
rail and 11x slower bonded.

## Which NCCL knob applies depends on PCIe topology

This decides whether `NCCL_IB_MERGE_NICS` can do anything at all, and it is worth checking
on your own hardware rather than assuming.

```bash
for d in /sys/class/infiniband/*; do
  echo -n "$(basename $d) -> "; readlink $d/device | sed 's|.*/||'
done
```

On these Sparks:

```text
rocep1s0f0    0000:01:00.0    card A port 0
rocep1s0f1    0000:01:00.1    card A port 1
roceP2p1s0f0  0002:01:00.0    card B port 0
roceP2p1s0f1  0002:01:00.1    card B port 1
```

Two physically separate cards in different PCIe domains, each dual port. The two rails
between any adjacent pair land on **different cards**, one port from card A and one from
card B.

That matters because NCCL's documentation scopes `NCCL_IB_MERGE_NICS` to merging dual-port
ports of the *same* card. It cannot bond these rails. Merging across separate cards is
governed by `NCCL_NET_MERGE_LEVEL` and `NCCL_NET_MERGE_POLICY`, whose distance semantics
are not documented on the NCCL 2.31.2 environment page.

One more documentation detail that invalidates a line carried in several published recipes:
`NCCL_IB_ADDR_RANGE` is defined as the range NCCL uses *when `NCCL_IB_GID_INDEX` is left
unset*. Every working Spark recipe pins `NCCL_IB_GID_INDEX=3`, so their
`NCCL_IB_ADDR_RANGE` lines do nothing. Device selection is controlled entirely by
`NCCL_IB_HCA`.

## Bonding is a prefill lever, not a decode lever

For a tensor-parallel-2 decode of GLM-5.3-Flash, hidden size 4096 in bf16 gives 8192 bytes
per all-reduce, two all-reduces per layer, 45 layers:

```text
90 collectives per token, 0.70 MiB per token per direction

wire time at 13.7 GB/s    54 us per token
wire time at 24.6 GB/s    30 us per token
observed decode time      21.3 ms per token  (at a reported 46.9 tok/s)
```

Wire is 0.14% to 0.25% of decode time. **Bandwidth is categorically not the decode
constraint.** What could matter is the *latency* of 90 tiny collective launches per token,
which needs `all_reduce_perf -b 8K -e 8K` to settle and is not measured here.

Prefill is different. A chunked prefill at `--max-num-batched-tokens 8192` moves
8192 x 8192 x 90 bytes, about 6.04 GB per chunk:

```text
per chunk at 13.7 GB/s    441 ms
per chunk at 24.6 GB/s    246 ms
```

So bonding is worth real time on long prompts and essentially nothing on generation. Ship
single rail first, because that is the configuration with field evidence behind it, and
only spend the bonding verification effort if long-prompt TTFT is your actual pain point.

## Verifying that bonding engaged, if you try it

Config alone does not tell you. During a real decode, poll every candidate device on both
nodes:

```bash
cat /sys/class/infiniband/<dev>/ports/1/counters/port_xmit_data
```

If only one device's counter advances, bonding did not engage regardless of what you set.
`/sys/class/net/*/statistics` stays near zero even when RDMA is working correctly, so do
not read that and conclude the fabric is idle.

## Reproducing the measurements

Server on one node, client on its peer. The device must match the interface holding the
address you dial, which is easy to get wrong on a ring:

```bash
# node-c, the address 10.100.138.1 lives on enp1s0f1np1 -> rocep1s0f1
ib_write_bw -d rocep1s0f1 -F --report_gbits -s 1048576 -n 5000

# node-b, dialing it from enp1s0f0np0 -> rocep1s0f0
ib_write_bw -d rocep1s0f0 -F --report_gbits -s 1048576 -n 5000 10.100.138.1
```

Binding the wrong device produces `Failed to modify QP to RTR / Unable to Connect the HCA's
through the link`, which reads like a cabling fault and is not one.
