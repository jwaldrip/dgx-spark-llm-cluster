#!/usr/bin/env bash
#
# Qwen3.8-Flash-Next NVFP4 as a data-parallel replica set across N DGX Spark nodes,
# presented as ONE OpenAI-compatible endpoint.
#
# Why data parallel and not tensor parallel. The model fits one node at 79.42 GiB resident,
# so sharding it buys nothing and costs cross-node collectives on a fabric measured at
# 24.6 GB/s per neighbour against roughly 273 GB/s of local memory. And TP3 is not even legal
# for this checkpoint:
#
#     num_attention_heads   24     /3 = 8       ok
#     num_key_value_heads    2     /3           ILLEGAL
#     hidden_size         2560     /3           not an integer
#
# DP gives one logical endpoint, N schedulers, N KV pools and N failure domains. Each
# replica holds a FULL copy, so aggregate KV is N x 496,770 tokens.
#
# IMPORTANT: run this behind router/gateway.ts, not bare. vLLM's own DP balancer routes by
# queue depth, and every replica keeps its own prefix cache. Scattering a session's turns
# across replicas turns a 1.82 s warm prefill back into 103.6 s cold, on a workload that is
# 181 prompt tokens per generated token. That is why --data-parallel-external-lb is set: the
# router owns placement, vLLM owns execution.
#
# THREE THINGS THAT EACH COST A LAUNCH, all learned the hard way:
#
#  1. NO `serve` SUBCOMMAND. The image ENTRYPOINT is already ["vllm","serve"], so passing
#     `serve <path>` yields `vllm serve serve <path>`, which eats "serve" as the model name:
#       vllm: error: unrecognized arguments: /hf/hub/...
#
#  2. NO --headless WITH EXTERNAL LB. vLLM rejects it outright:
#       RuntimeError: Remote engine 1 must not use --headless in external or hybrid dp lb mode
#     Under external LB every rank runs its own API server, which is what the gateway wants.
#
#  3. VERIFY MEMORY IS BACK BEFORE LOADING. See the preflight below. Iterating on flags
#     against a live 79 GiB load wedged two nodes badly enough to need a power cycle.
#
# Usage: DP_ADDR=<rank0 LAN ip> ./launch-qwen-dp.sh <rank>     rank 0 last
set -euo pipefail

RANK="${1:?usage: DP_ADDR=<rank0-lan-ip> launch-qwen-dp.sh <rank>}"

# ---------------------------------------------------------------- configuration
IMAGE="${QWEN_IMAGE:-qwen38-flash-dgx:209646c}"
NAME="qwen38-dp${RANK}"
REV="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
REPO_HOST="$HOME/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
MODEL_PATH="/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$REV"

DP_SIZE="${DP_SIZE:-3}"
# Coordination runs over the SHARED LAN, not the QSFP triangle. DP replicas exchange no
# tensors, only light coordination, so there is no reason to put it on RoCE. That also
# sidesteps two things the field reports call fatal on a three-node pairwise triangle:
#   * NCCL_IB_GID_INDEX=3 is correct for a two-node single-subnet link and fatal on a
#     triangle, per FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks docs/FABRIC.md.
#   * A pairwise triangle is not one RoCE /24, so cross-rank IB needs a custom NCCL mesh
#     plugin built from source. Unnecessary here.
# Consequence worth knowing: under DP the QSFP fabric is IDLE. It earns its keep only for
# tensor/pipeline parallel or for KV pooling.
# NO APOSTROPHES in a ${VAR:?message}. An apostrophe here opens a quote context that
# swallows following lines until the next one, silently leaving later assignments unset.
DP_ADDR="${DP_ADDR:?set DP_ADDR to the LAN address of rank 0}"
DP_PORT="${DP_PORT:-29561}"
PORT="${PORT:-8100}"
LAN_IF="${LAN_IF:-enP7s7}"
HOST_IP="$(ip -4 -o addr show "$LAN_IF" | awk '{print $4}' | cut -d/ -f1)"
# -------------------------------------------------------------------------------

test -f "$REPO_HOST/snapshots/$REV/config.json" || { echo "weights missing" >&2; exit 1; }
docker rm -f "$NAME" 2>/dev/null || true

# GB10 unified memory: page cache and any previous model share ONE 121 GiB pool, and with
# swappiness 0 plus memlock there is no relief valve. Loading 79 GiB on top of a previous
# allocation that has not been released wedges the node hard enough to need a power cycle.
# Ask me how I know. Refuse to start until memory is genuinely back.
# Best-effort: the load-bearing check is the free-memory wait below, which needs no
# privileges. Skip the cache drop rather than fail when sudo is unavailable.
sync
if ! echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
  echo "note: could not drop caches (no passwordless sudo); relying on the memory check" >&2
fi
avail=0
for _ in $(seq 1 60); do
  avail="$(free -g | awk '/Mem:/{print $7}')"
  [ "$avail" -ge "${MIN_AVAIL_GIB:-100}" ] && break
  sleep 5
done
if [ "$avail" -lt "${MIN_AVAIL_GIB:-100}" ]; then
  echo "only ${avail} GiB available, need ${MIN_AVAIL_GIB:-100}; refusing to load" >&2
  exit 1
fi
echo "preflight ok: ${avail} GiB available"

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$REPO_HOST:/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_QSA_EXACT_TOPK=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME="$LAN_IF" -e GLOO_SOCKET_IFNAME="$LAN_IF" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" \
    --load-format safetensors \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --no-enable-flashinfer-autotune \
    --kv-cache-dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --data-parallel-size "$DP_SIZE" \
    --data-parallel-size-local 1 \
    --data-parallel-rank "$RANK" \
    --data-parallel-address "$DP_ADDR" \
    --data-parallel-rpc-port "$DP_PORT" \
    --data-parallel-external-lb

echo "launched $NAME rank=$RANK/$DP_SIZE host=$HOST_IP lan=$LAN_IF port=$PORT"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2
  exit 1
}

cat <<'NOTE'

Each replica serves on its own :8000. Put the affinity router in front of all of them:

  ROUTER_UPSTREAMS=http://10.100.138.2:8000,http://10.100.138.1:8000,http://10.100.140.1:8000 \
  ROUTER_MAX_INFLIGHT=4 bun run router/prefix-affinity-router.ts

Then verify placement is actually sticky before trusting it:

  ./measure-cache-affinity.sh http://127.0.0.1:18010

A green /health proves nothing on this hardware. Read generated text, and read
/router/stats affinityRate.
NOTE
