#!/usr/bin/env bash
#
# GLM-5.3-Flash NVFP4 (RedHatAI compressed-tensors) at tensor parallel 2 across two
# DGX Spark nodes (GB10 / SM121), OpenAI-compatible on port 8000 from the head rank.
#
# Day one configuration: NO speculative decoding. Base engine first, DFlash2 second.
#
# Five deliberate departures from the upstream field recipe, each with its reason:
#
#   1. RedHatAI compressed-tensors checkpoint, not a ModelOpt NVFP4 build.
#      vllm#54150: vLLM's w13_weight_scale_2[:,0] pick mis-scales one shard of the fused
#      w13 GEMM by up to 10x on ModelOpt checkpoints, producing intermittent corrupted
#      token IDs. Fix posted, not merged. See docs/checkpoint-selection.md.
#
#   2. BF16 KV (default --kv-cache-dtype), not fp8_e4m3.
#      No GLM checkpoint ships KV scaling factors, so fp8 KV runs at vLLM's unit scale.
#      That silent 1.0 substitution is a quality variable we refuse to carry into a first
#      bring-up. Revisit as a controlled change once the base is trusted.
#
#   3. No --kv-cache-memory-bytes pin.
#      Pinning stops vLLM subtracting the measured activation peak, which makes
#      --gpu-memory-utilization dead and kills the engine on the first long prompt.
#
#   4. No --enable-prefix-caching.
#      Independently measured at 0 hits in 22,825 queries on this architecture.
#
#   5. --default-chat-template-kwargs enable_thinking=true, not false.
#      With false, the model's deliberation lands in message.content. With true, content is
#      clean and tool calls carry no prose. See docs/gotchas.md.
#
# --moe-backend marlin is mandatory rather than tuning: every other backend on GB10 either
# refuses to load or boots cleanly and emits garbage.
#
# Usage: ./launch-glm-tp2.sh <0|1>     rank 1 (worker) FIRST, wait ~25s, then rank 0 (head)
set -euo pipefail

NODE_RANK="${1:?usage: launch-glm-tp2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

# ---------------------------------------------------------------- configuration
IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2"
NAME="vllm_glm53"
REV="36c184c6cda000a481711306df5adde42f63321a"

# HF cache snapshots are symlink farms into ../../blobs, so the repo ROOT must be mounted,
# not just the snapshot directory, or every weight and template is a dangling link.
REPO_HOST="$HOME/.cache/huggingface/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4"
MODEL_PATH="/models/glm-repo/snapshots/$REV"
CACHE_HOST="/var/tmp/glm53-vllm-cache"
PATCH="$HOME/patches/sparse_attn_indexer_kpool.py"

# Fabric. Set these to your own ring addresses and devices. The RDMA device must be the one
# that owns the address: check `ip -br -4 addr` and `ls /sys/class/net/<if>/device/infiniband`.
HEAD_IP="10.100.138.2"     # rank 0
WORKER_IP="10.100.138.1"   # rank 1
HEAD_HCA="rocep1s0f0";   HEAD_IF="enp1s0f0np0"
WORKER_HCA="rocep1s0f1"; WORKER_IF="enp1s0f1np1"

MPORT="29521"
PORT="8000"
# -------------------------------------------------------------------------------

# TP2 across two nodes is a SINGLE-SUBNET link, so NCCL_IB_GID_INDEX=3 below is correct
# here. It is NOT correct on a three-node pairwise triangle, where each pair sits on its
# own /24: FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks docs/FABRIC.md calls pinning it fatal
# there, and a triangle additionally needs a custom NCCL mesh plugin. Do not copy this
# block to three nodes.
#
# Also note each physical QSFP port is TWO PCIe x4 twins of ~100 Gb/s. Pinning one HCA, as
# every published recipe including NVIDIA's does, uses half the cable. Naming both twins
# (NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1) is what reaches ~196 Gb/s. See docs/interconnect.md.
case "$NODE_RANK" in
  0) HOST_IP="$HEAD_IP";   IB_HCA="$HEAD_HCA";   IFACE="$HEAD_IF";   HEADLESS="" ;;
  1) HOST_IP="$WORKER_IP"; IB_HCA="$WORKER_HCA"; IFACE="$WORKER_IF"; HEADLESS="--headless" ;;
esac

test -f "$REPO_HOST/snapshots/$REV/config.json" || { echo "checkpoint missing" >&2; exit 1; }
test -f "$PATCH" || { echo "SM121 indexer top-k patch missing at $PATCH" >&2; exit 1; }
mkdir -p "$CACHE_HOST"
docker rm -f "$NAME" 2>/dev/null || true

# GB10 unified memory has no relief valve: one 121 GiB pool shared by page cache and model,
# swappiness 0, weights pinned via memlock. Loading on top of an unreleased allocation wedges
# the node hard enough to need a power cycle. Wait for memory to actually come back.
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
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
  --device /dev/infiniband:/dev/infiniband \
  -v "$REPO_HOST:/models/glm-repo:ro" \
  -v "$CACHE_HOST:/cache" \
  -v "$PATCH:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="=$IB_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e TP_SOCKET_IFNAME="$IFACE" -e MN_IF_NAME="$IFACE" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-seqs 6 \
    --block-size 2304 \
    --moe-backend marlin \
    --enforce-eager \
    --max-num-batched-tokens 8192 \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --chat-template "$MODEL_PATH/chat_template.jinja" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP hca=$IB_HCA"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2
  exit 1
}

cat <<'NOTE'

Readiness takes several minutes (checkpoint load plus TileLang and FlashInfer warmup).
Watch the head rank:

  docker logs -f vllm_glm53 2>&1 | grep -E 'Loading safetensors|Model loading took|Available KV|GPU KV cache size'

Then probe /health, never /v1/models, and run smoke-test.sh. A container that is up and a
/health that returns 200 are both compatible with an engine that emits pure repetition on
this hardware.
NOTE
