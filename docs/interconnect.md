# The fabric

Three DGX Sparks wired as a direct ring with ConnectX-7, no switch. **Three cables total**,
one per adjacent pair, which is what NVIDIA documents for a three-device ring: each device
connects to the other two.

The confusing part, and the source of a wrong conclusion in the first version of this
document, is that each node shows **four** netdevs for **two** physical ports.

```bash
for i in /sys/class/net/en*; do
  echo "$(basename $i)  port=$(cat $i/phys_port_name)  switch=$(cat $i/phys_switch_id)"
done
```

```text
enp1s0f0np0     port p0    switch 0d3a2e000347bb4c
enP2p1s0f0np0   port p0    switch 0d3a2e000347bb4c
enp1s0f1np1     port p1    switch 0d3a2e000347bb4c
enP2p1s0f1np1   port p1    switch 0d3a2e000347bb4c
```

One ConnectX-7 (`phys_switch_id` identical across all four), two physical QSFP ports, and
**each port is presented as two netdevs across two PCIe domains**. So a single cable carries
two /24s, one per PCIe function.

```text
node-a p0  <->  node-b p1     10.100.140.x , 10.100.141.x    ONE cable
node-b p0  <->  node-c p1     10.100.138.x , 10.100.139.x    ONE cable
node-c p0  <->  node-a p1     10.100.136.x , 10.100.137.x    ONE cable
```

Do not read those subnet pairs as two rails. `readlink /sys/class/infiniband/<dev>/device`
shows two PCIe domains and invites exactly that mistake. `phys_port_name` is the field that
settles it.

## A single netdev reaches half the line rate. The cable reaches all of it.

Measured with `ib_write_bw`, 1 MiB messages, 5000 iterations, between two adjacent nodes:

| configuration | network MTU | RoCE MTU | queue pairs | BW average |
|---|---:|---:|---:|---:|
| one netdev | 1500 | 1024 | 1 | 109.12 Gb/s |
| one netdev, jumbo | 9000 | 4096 | 1 | 109.27 Gb/s |
| one netdev, 8 QPs | 9000 | 4096 | 8 | 111.86 Gb/s |
| **both netdevs of the same port** | 9000 | 4096 | 1 each | **196.55 Gb/s** |

Neither MTU (0.14%) nor queue-pair count (2.5%) moves a single netdev off roughly
110 Gb/s. That is a **per-PCIe-function** ceiling, not a link ceiling. Drive both functions
of the same physical port and the cable delivers **196.55 Gb/s, about 98% of its 200 Gb/s
line rate**.

```text
per neighbour, one netdev     13.7 GB/s
per neighbour, both netdevs   24.6 GB/s
GB10 local memory            ~273 GB/s
```

So the fabric is about 11x slower than local memory when used properly, and 20x slower if
you pin a single interface. Jumbo frames still buy nothing for bandwidth; the reason to set
them is small-message latency, since a RoCE MTU of 4096 carries an 8 KiB collective in two
packets instead of eight. That remains unmeasured here and would need `all_reduce_perf -b 8K`.

**Practical consequence.** Every published multi-node recipe, including NVIDIA's own vLLM
playbook, pins one `MN_IF_NAME` / `NCCL_SOCKET_IFNAME`. On this hardware that uses half of
each cable. To get line rate, both functions of the port must be in play.

## `NCCL_IB_MERGE_NICS` is the right lever, and PCIe BDFs will tell you otherwise

```bash
for d in /sys/class/infiniband/*; do
  echo -n "$(basename $d) -> "; readlink $d/device | sed 's|.*/||'
done
```

```text
rocep1s0f0    0000:01:00.0
rocep1s0f1    0000:01:00.1
roceP2p1s0f0  0002:01:00.0
roceP2p1s0f1  0002:01:00.1
```

Two PCIe domains, which reads like two separate cards. **It is not.** `phys_switch_id` is
identical across all four and `phys_port_name` shows only `p0` and `p1`, so these are two
PCIe function views of one dual-port card.

That inverts the conclusion. NCCL scopes `NCCL_IB_MERGE_NICS` to merging the ports of a
dual-port NIC, which is exactly this case, so it is the correct mechanism for getting both
functions of a port working on one cable. `NCCL_NET_MERGE_LEVEL` and
`NCCL_NET_MERGE_POLICY`, which govern merging across genuinely distinct cards, are not the
relevant knobs here.

Trust `phys_port_name` and `phys_switch_id` over BDFs on this hardware. The BDF layout is
what produced the wrong answer the first time.

One documentation detail that invalidates a line carried in several published recipes:
`NCCL_IB_ADDR_RANGE` is defined as the range NCCL uses *when `NCCL_IB_GID_INDEX` is left
unset*. Every working Spark recipe pins `NCCL_IB_GID_INDEX=3`, so their
`NCCL_IB_ADDR_RANGE` lines do nothing. Device selection is controlled entirely by
`NCCL_IB_HCA`.


## Using both port functions is a prefill lever, not a decode lever

For a tensor-parallel-2 decode of GLM-5.3-Flash, hidden size 4096 in bf16 gives 8192 bytes
per all-reduce, two all-reduces per layer, 45 layers:

```text
90 collectives per token, 0.70 MiB per token per direction

wire time, one netdev     54 us per token
wire time, both          30 us per token
observed decode time     21.3 ms per token  (at a reported 46.9 tok/s)
```

Wire is 0.14% to 0.25% of decode time. **Bandwidth is categorically not the decode
constraint.** What could matter is the *latency* of 90 tiny collective launches per token,
which needs `all_reduce_perf -b 8K -e 8K` to settle and is not measured here.

Prefill is different. A chunked prefill at `--max-num-batched-tokens 8192` moves
8192 x 8192 x 90 bytes, about 6.04 GB per chunk:

```text
per chunk, one netdev     441 ms
per chunk, both           246 ms
```

So driving both functions is worth real time on long prompts and essentially nothing on
generation. It is also nearly free, since it is one cable either way: the only cost is
configuring NCCL to use both functions instead of the single pinned interface every
published recipe assumes.

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
