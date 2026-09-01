#!/usr/bin/env bash
#
# Qwen3.8-Flash-Next NVFP4 as a data-parallel replica set across N DGX Spark nodes,
# presented as ONE OpenAI-compatible endpoint.
#
# Why data parallel and not tensor parallel. The model fits one node at 79.42 GiB resident,
# so sharding it buys nothing and costs cross-node collectives on a fabric measured at
# 13.7 GB/s per rail against roughly 273 GB/s of local memory. And TP3 is not even legal
# for this checkpoint:
#
#     num_attention_heads   24     /3 = 8       ok
#     num_key_value_heads    2     /3           ILLEGAL
#     hidden_size         2560     /3           not an integer
#
# DP gives one logical endpoint, N schedulers, N KV pools and N failure domains. Each
# replica holds a FULL copy, so aggregate KV is N x 519,943 tokens.
#
# IMPORTANT: run this behind router/prefix-affinity-router.ts, not bare. vLLM's own DP
# balancer routes by queue depth, and every replica keeps its own prefix cache. Scattering a
# session's turns across replicas turns a 1.82 s warm prefill back into 103.6 s cold, on a
# workload that is 181 prompt tokens per generated token. That is why --data-parallel-external-lb
# is set: the router owns placement, vLLM owns execution.
#
# Usage: ./launch-qwen-dp.sh <rank>     rank 0 last, non-zero ranks first
set -euo pipefail

RANK="${1:?usage: launch-qwen-dp.sh <rank>}"

# ---------------------------------------------------------------- configuration
IMAGE="${QWEN_IMAGE:-qwen38-flash-dgx:209646c}"
NAME="qwen38-dp"
REV="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
REPO_HOST="$HOME/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
MODEL_PATH="/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$REV"

DP_SIZE="${DP_SIZE:-3}"
DP_ADDR="${DP_ADDR:-10.100.138.2}"   # rank 0's ring address
DP_PORT="${DP_PORT:-29551}"
PORT="${PORT:-8000}"

# Ring identity per rank. The RDMA device must own the address: verify with
#   ip -br -4 addr   and   ls /sys/class/net/<iface>/device/infiniband
case "$RANK" in
  0) HOST_IP=10.100.138.2; IFACE=enp1s0f0np0; IB=rocep1s0f0 ;;
  1) HOST_IP=10.100.138.1; IFACE=enp1s0f1np1; IB=rocep1s0f1 ;;
  2) HOST_IP=10.100.140.1; IFACE=enp1s0f0np0; IB=rocep1s0f0 ;;
  *) echo "unknown rank $RANK" >&2; exit 2 ;;
esac
# -------------------------------------------------------------------------------

test -f "$REPO_HOST/snapshots/$REV/config.json" || { echo "weights missing" >&2; exit 1; }
docker rm -f "$NAME" 2>/dev/null || true

# GB10 unified memory: the page cache competes with the model. Reclaim before loading.
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$REPO_HOST:/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_QSA_EXACT_TOPK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="=$IB" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  "$IMAGE" \
    serve "$MODEL_PATH" \
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

echo "launched $NAME rank=$RANK host=$HOST_IP hca=$IB dp=$DP_SIZE"
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
