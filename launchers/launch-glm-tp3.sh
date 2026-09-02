#!/usr/bin/env bash
#
# GLM-5.3-Flash NVFP4 (RedHatAI compressed-tensors) at tensor parallel 3 across three
# DGX Spark nodes (GB10 / SM121), OpenAI-compatible on port 8000 from the head rank.
#
# WHY TP=3 AND NOT MORE DATA-PARALLEL QWEN REPLICAS. Qwen3.8-Flash-Next caps at 262,144
# tokens no matter how many DP replicas front it; the main lane's measured prompt p95 is
# 782,561 tokens (max 908,183). No amount of Qwen replication serves that lane. GLM TP=3
# does, at either variant below. See docs/glm-tp3-reproduction.md for the full table.
#
# WHY THIS IS A DIFFERENT CHECKPOINT THAN THE PUBLISHED RECIPE. The source recipe
# (https://github.com/FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks) runs LibertAIDAI's ModelOpt
# NVFP4 build with --moe-backend marlin. That exact combination is the code path of
# vllm-project/vllm#54150 (open, unfixed): the marlin repack applies the gate global scale
# to the up shard of the fused w13 GEMM too, mis-scaling up-projection output up to 10x on
# 69.1% of expert pairs. Their own issue measurement: 86 U+FFFD replacement characters
# across 6 runs on that checkpoint, 0/6 on RedHatAI's. A corrupted token inside a tool-call
# block desyncs the parser into a repetition lock, which disqualifies it for agent serving
# regardless of throughput. So this launcher runs RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-
# tensors mixed-precision, structurally immune because gate and up scales are shared by
# construction) instead. gate-toolcall-fidelity.sh is what proves that decision held.
#
# NOBODY HAS RUN THIS SUBSTITUTION AT TP=3. RedHatAI's own vendor command uses TP=4, where
# 64/4 divides evenly and no head padding is needed. The padding and overlay machinery
# below has only ever been exercised against LibertAIDAI's ModelOpt loader. The first boot
# of this script is the experiment, not a known-good reproduction. If model construction or
# weight loading fails, that is the finding, not a bug in this file: see the "if it does
# not load" fallback in docs/glm-tp3-reproduction.md (LibertAIDAI + marlin + one of the two
# candidate #54150 patches as a fifth overlay).
#
# THREE DEPARTURES FROM launch-glm-tp2.sh, EACH DELIBERATE:
#
#   1. --kv-cache-dtype fp8_e4m3, not tp2's auto/BF16. tp2 chose BF16 specifically because
#      no GLM checkpoint ships KV scaling factors, so fp8 KV runs at vLLM's unit scale, a
#      quality variable tp2 refused to carry into its first bring-up. TP=3 re-accepts that
#      exact risk on purpose: the published recipe's block-size 2304 / fp8 KV pairing is
#      the only proven-serving configuration at this scale, and block-size 64/128 do not
#      serve on SM121 at all (fp8 DeepGEMM block_kv must be 64-aligned as a kernel size and
#      a user block-size conflicts with it at warmup). Revisit once TP=3 base is trusted,
#      same as tp2's own note about revisiting fp8 KV.
#
#   2. --mm-encoder-tp-mode data. GLM-5.3-Flash's vision tower has 16 heads, and 16 is not
#      divisible by 3. Left unset, VllmConfig construction fails outright before any weight
#      loads (FlyCockpit NOTES.md T2c/T2d). tp2 does not need this at TP=2 (16/2 is exact).
#
#   3. No --kv-cache-memory / --kv-cache-memory-bytes pin, even though the published recipe
#      pins one for its "1m" variant. docs/gotchas.md is explicit that pinning is wrong on
#      this hardware: it stops vLLM subtracting the measured activation peak, makes
#      --gpu-memory-utilization dead, and kills the engine on the first long prompt. Let the
#      profiler size the pool, same as tp2, and read "Available KV cache memory" on every
#      rank once it is up (the pool is the minimum across ranks; the head rank's log line is
#      not representative).
#
# ONE ADDITION BEYOND THE ASSIGNED MOUNT LIST: this launcher still bind-mounts the SM121
# sparse-attention indexer top-k patch (see stage-checkpoint.sh) that tp2 requires. That fix
# is about SM121 hardware plus this checkpoint family decoding past ~24K context, not about
# TP degree, and both variants below exceed 24K by a wide margin. Omitting it would not be a
# smaller launcher, it would be a launcher that reliably dies on real traffic.
#
# WHAT IS CONFIRMED VS. STILL IN FLIGHT, as of the time this file was written (batch
# hub reports, all cited by agent id):
#   - Mesh transport (NCCL_NET=Mesh / NCCL_NET_PLUGIN=mesh / NCCL_IB_DISABLE=1, no GID pin)
#     and the plugin mount shape: CONFIRMED by MeshPlugin (built + dlopen-tested).
#   - The four overlay files' in-container destinations and content compatibility with our
#     image: CONFIRMED by OverlayCompat (diffed against files extracted from the actual
#     image, throwaway container import test, exit 0), independently cross-checked here via
#     `docker run --entrypoint python3 <image> -c "import vllm; print(vllm.__file__)"` on
#     spark-74a7, same path.
#   - config.json padding / --hf-overrides shape: ConfigPadding confirmed fields nest under
#     "text_config" (not top-level, unlike the published recipe) and confirmed vocab_size
#     stays at the real 154880 (padding is in-module only). moe_intermediate_size is NOT the
#     published recipe's 2112: ConfigPadding found 2112/3=704 fails the FP8-quantized MTP
#     layer's (layer 45) intermediate_size_per_partition%128==0 check, and is padding to
#     2304 instead (768/rank, divisible by both the NVFP4 group's 16-alignment and the FP8
#     group's 128-alignment). OverlayCompat independently confirmed from the overlay
#     model.py source that this is safe: model.py's own 2048->2112 override only fires when
#     it sees literal stock 2048 on disk, so a config.json that already carries 2304 skips
#     that branch untouched. THE FAILURE MODE TO WATCH: if the mounted config.json is ever
#     a stock-2048 file relying on --hf-overrides alone to carry the pad, model.py's own
#     2112 fallback fires silently and reintroduces the exact FP8 alignment bug this
#     paragraph describes. This script does not hardcode any of that; it reads
#     ~/glm-tp3/config/config.json and ~/glm-tp3/hf_overrides.json as opaque files and only
#     checks that the overrides file is valid JSON. CONFIRM the mounted config.json actually
#     contains 2304, not 2048, before trusting a launch. linear_num_heads is deliberately
#     left UNSET by ConfigPadding pending confirmation that GLM's KDA/linear-attention
#     layers are even head-sharded under TP, unlike the published recipe which sets it.
#   - spark-3a0c (rank 2) is missing all THREE of: the GLM image, the checkpoint, and the
#     SM121 indexer patch, confirmed independently by MeshPlugin and OverlayCompat and by
#     the direct probe this script's author ran. It also has only ~37 GiB free while serving
#     production Qwen. The preflight checks below will refuse to launch rank 2 until every
#     one of those is resolved; that refusal is correct, not a bug. See the final report for
#     the exact staging commands.
#
# Usage: ./launch-glm-tp3.sh <0|1|2>           rank 2, then rank 1, then rank 0 (head) LAST
#        ./launch-glm-tp3.sh --preflight       apply the two per-boot host settings (needs
#                                               an interactive sudo password; run once after
#                                               every reboot, before any rank)
#        VARIANT=fast|1m ./launch-glm-tp3.sh <rank>     default fast (262144); 1m = 1048576
set -euo pipefail

# ---------------------------------------------------------------- boot-time host preflight
# nvidia-smi's clock lock and the enP7s7 MTU do NOT survive a reboot, and both were observed
# reverted after today's reboots. This script cannot apply them as part of a normal launch:
# sudo needs an interactive password this script does not have, and the normal launch path
# may run non-interactively over ssh with no TTY to prompt on. Run this mode by hand, once
# per boot, before rank 0:
#
#   ssh <node> './launch-glm-tp3.sh --preflight'
# Both settings are now PERSISTENT, so this mode is a verifier and a repair path, not a
# per-boot ritual. The clock cap is installed as gb10-clock-cap.service (systemd oneshot,
# enabled on all three nodes, verified ExecMainStatus 0 after boot). The RoCE MTUs come
# back at 9000 from netplan 99-nvidia-sync-cluster.yaml on their own. Run this only if the
# checks below report drift.
if [ "${1:-}" = "--preflight" ]; then
  echo "Verifying per-boot host state on $(hostname). Repair steps need sudo interactively."
  echo
  echo "1. Clock cap (recipe specifies 2200; installed as gb10-clock-cap.service):"
  systemctl is-enabled gb10-clock-cap.service 2>/dev/null || echo "  UNIT MISSING on this node"
  nvidia-smi -q -d CLOCK 2>/dev/null | sed -n '/Clocks Event/,+1p' | sed 's/^/  /'
  echo "  repair: sudo nvidia-smi -i 0 -lgc 0,2200"
  echo "  note: 'Applications Clocks' stays 2418 on GB10 and is a DIFFERENT field. Reading it"
  echo "        to check the lock reports a false failure; the lock reports gpuClkMax 2200."
  echo
  echo "2. RoCE MTUs (the QSFP triangle; 9000 expected, from netplan):"
  for roce in enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1; do
    printf '  %s %s\n' "$roce" "$(cat "/sys/class/net/$roce/mtu" 2>/dev/null || echo absent)"
    echo "    repair: sudo ip link set $roce mtu 9000"
  done
  echo
  echo "3. Control-plane LAN (${LAN_IF:-enP7s7}); 1500 here is CORRECT, do not raise it:"
  printf '  %s %s\n' "${LAN_IF:-enP7s7}" "$(cat "/sys/class/net/${LAN_IF:-enP7s7}/mtu" 2>/dev/null || echo absent)"
  exit 0
fi

RANK="${1:?usage: launch-glm-tp3.sh <0|1|2>  (or --preflight)}"
case "$RANK" in
  0|1|2) ;;
  *) echo "rank must be 0, 1, or 2 (got $RANK)" >&2; exit 2 ;;
esac

VARIANT="${VARIANT:-fast}"
case "$VARIANT" in
  fast) MAX_MODEL_LEN=262144 ;;
  1m)   MAX_MODEL_LEN=1048576 ;;
  *) echo "VARIANT must be fast or 1m (got $VARIANT)" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- configuration
# Same image family as launch-glm-tp2.sh on purpose: its vLLM tree reports commit
# g487ecf187, the exact commit the published recipe's base image was built from. That is
# the whole shortcut this batch exists to verify -- mount the four overlays over this
# image's already-installed vllm package, build nothing.
IMAGE="${GLM_TP3_IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
NAME="vllm_glm53"
# CHECKPOINT selects which of the two staged checkpoints to serve. They are NOT
# interchangeable, and the difference is load-bearing:
#   redhat   compressed-tensors, W4A4 (activations quantized too). Immune to vllm#54150
#            by construction. Loads and serves at TP=3 but emits a repetition lock; needs
#            the routed_experts w2 scale overlay even to load. See docs/tp3-bisect.md.
#   libertai ModelOpt NVFP4, W4A16 weight-only. What the published recipe actually uses,
#            and what its four overlays were written against. Affected by vllm#54150, so
#            it needs the modelopt overlay in overlay/modelopt-vllm54150.diff.
CHECKPOINT="${CHECKPOINT:-libertai}"
case "$CHECKPOINT" in
  redhat)
    HF_REPO="models--RedHatAI--GLM-5.3-Flash-NVFP4"
    REV="36c184c6cda000a481711306df5adde42f63321a" ;;
  libertai)
    HF_REPO="models--LibertAIDAI--GLM-5.3-Flash-NVFP4"
    REV="caca4e6a4ebbd66f159d3d2fc256683fd6e27177" ;;
  *) echo "CHECKPOINT must be redhat or libertai (got $CHECKPOINT)" >&2; exit 2 ;;
esac

# HF cache snapshots are symlink farms into ../../blobs (docs/gotchas.md), so the repo ROOT
# must be mounted, not just the snapshot directory, or every weight and template is a
# dangling link.
REPO_HOST="$HOME/.cache/huggingface/hub/$HF_REPO"
MODEL_PATH="/models/glm-repo/snapshots/$REV"
CACHE_HOST="/var/tmp/glm53-vllm-cache"
# Overridable. This is OUR TP=2-era SM121 indexer patch, not one of FlyCockpit's four
# overlays. Their NOTES.md states the indexer is REPLICATED at TP=3 rather than padded, and
# index_n_heads is 32, which does not divide by 3 -- so a patch that shards it is a live
# suspect for garbage output. Point this at the image's stock file to take it out of play.
PATCH="${PATCH:-$HOME/patches/sparse_attn_indexer_kpool.py}"
VLLM_PKG="/usr/local/lib/python3.12/dist-packages/vllm"
# Overridable so a pad set can be swapped without editing this file. The published recipe
# documents exactly TWO pads (num_attention_heads and moe_intermediate_size); any others in
# a config here were invented locally and must be treated as unvalidated until gated.
CONFIG_HOST="${CONFIG_HOST:-$HOME/glm-tp3/config/config.json}"
HF_OVERRIDES_FILE="${HF_OVERRIDES_FILE:-$HOME/glm-tp3/hf_overrides.json}"
OVERLAY_ROOT="$HOME/glm-tp3/overlay/vllm"
# Overridable so a failing load can be bisected overlay by overlay. Point one at the image's
# own stock file to neutralise just that mount without disturbing the others. This is how the
# e_score_correction_bias mis-routing was isolated to model.py.
OVERLAY_MODEL_PY="${OVERLAY_MODEL_PY:-$OVERLAY_ROOT/models/glm5next/nvidia/model.py}"
OVERLAY_VOCAB_PY="${OVERLAY_VOCAB_PY:-$OVERLAY_ROOT/model_executor/layers/vocab_parallel_embedding.py}"
OVERLAY_WEIGHT_UTILS_PY="${OVERLAY_WEIGHT_UTILS_PY:-$OVERLAY_ROOT/model_executor/model_loader/weight_utils.py}"
# Optional extra `-e NAME=value` arguments, as one pre-quoted string, for one-off
# instrumentation. Expanded unquoted, so no value here may contain a space. Shell
# variables set outside the container are NOT visible to it; they must come through here.
EXTRA_ENV="${EXTRA_ENV:-}"
OVERLAY_PARAMETER_PY="${OVERLAY_PARAMETER_PY:-$OVERLAY_ROOT/model_executor/parameter.py}"
PLUGIN_DIR="$HOME/glm-tp3/plugin"
# Optional extra read-only mounts, as a single pre-quoted string of -v arguments. Used to
# drop an instrumented copy of a vLLM module in for one diagnostic run without editing this
# file. Expanded unquoted on purpose so each -v and its argument become separate argv words,
# which means no path here may contain a space.
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
# ---- end adjust-here block

# Rank map, fixed by the batch contract. Coordination (Gloo/TCPStore/master-addr/ssh) is the
# shared LAN, not the QSFP triangle; the triangle carries only NCCL tensor-parallel traffic,
# routed by the mesh plugin, not by this script.
#   rank 0  spark-74a7  10.4.1.119   head    (master-addr, non-headless, serves the API)
#   rank 1  spark-29d7  10.4.1.233   worker  (headless)
#   rank 2  spark-3a0c  10.4.1.191   worker  (headless) -- currently serving production
#           Qwen (qwen38-flash). The memory guard below will correctly refuse to launch
#           here until that container is stopped and its memory is actually released.
RANK0_LAN="10.4.1.119"
LAN_IF="${LAN_IF:-enP7s7}"
HOST_IP="$(ip -4 -o addr show "$LAN_IF" | awk '{print $4}' | cut -d/ -f1)"
case "$RANK" in
  0) HEADLESS="" ;;
  *) HEADLESS="--headless" ;;
esac

MPORT="${MPORT:-29521}"
PORT="${PORT:-8000}"

# --enforce-eager is NOT free. Independent field reports on this hardware measured
# 20.9 -> 36.7 tok/s from compiling instead of forcing eager, and one of the three ways
# this repo's own GLM TP=2 comparison was crippled was leaving it on. Default it ON for the
# very first boot of a checkpoint nobody has run at TP=3, because a compile failure during
# a 63 GiB-per-rank load is an expensive way to learn something. Then retest with
# EAGER= to claim the throughput back, and record both numbers.
EAGER="${EAGER---enforce-eager}"

# SPEC and KV_DTYPE are the two engine flags TP=3 adds that the proven TP=2 config never
# carried, which makes them the first things to bisect when a load fails. MTP pulls in
# layer 45, and on THIS checkpoint layer 45 is the FP8 block-quantized group rather than
# NVFP4, so it exercises a quantization path the other 42 expert layers never touch.
#   SPEC=           disable speculative decoding entirely
#   KV_DTYPE=auto   BF16 KV, exactly what TP=2 used
# No surrounding single quotes on the JSON: this is expanded unquoted so the shell splits it
# NEVER put a brace-containing default inside ${VAR-default}: bash ends the expansion at the
# first unescaped }, so the JSON's own closing brace terminates it and the remaining } is
# appended as literal text. With SPEC set empty that produced SPEC=} and vLLM died on
# "unrecognized arguments: }". Build the default separately.
SPEC_DEFAULT='--speculative-config {"method":"mtp","num_speculative_tokens":4}'

# Both of these now DEFAULT to the proven TP=2 engine configuration rather than to the
# published recipe's, because TP=2 is the only configuration of this checkpoint that has
# ever actually served on this hardware, and it deliberately carried neither.
#
# fp8_e4m3 KV is opt-in, not default: no GLM checkpoint ships KV scaling factors, so fp8 KV
# runs at vLLM's unit scale of 1.0. That silent substitution is a quality variable, and
# docs/checkpoint-selection.md already refused to carry it into the TP=2 bring-up. The same
# reasoning applies here, and more so, since TP=3 on compressed-tensors is unproven.
#
# MTP is opt-in because it pulls in layer 45, which on THIS checkpoint is the FP8
# block-quantized group rather than NVFP4, exercising a quantization path the other 42
# expert layers never touch. Carrying it into a first boot conflates two unknowns.
#
# Set SPEC="$SPEC_DEFAULT" and KV_DTYPE=fp8_e4m3 to re-add them as controlled changes once
# the base loads and passes the fidelity gate, and record a measurement for each.
SPEC="${SPEC-}"
KV_DTYPE="${KV_DTYPE:-auto}"

# The published recipe uses 0.85. Measured here at 0.85 on the 1m variant, rank 0 settled at
# 1.4 GiB available of 121 GiB, with 120 GiB used. Ranks 1 and 2 sat near 6.5 GiB. Rank 0 is
# tighter because it also runs the API server. That is too close to the edge on hardware
# whose failure mode is a wedge needing a physical power cycle, and which has no relief
# valve: swappiness 0, weights pinned via memlock, one pool shared with page cache. The
# trade is real and small: each 0.01 is roughly 1.2 GiB of KV, about 29K tokens at this
# model's per-token cost. Buy the headroom.
GMU="${GMU:-0.80}"
# -------------------------------------------------------------------------------

# ---------------------------------------------------------------- preflight: everything the
# engine needs must already be ON THIS NODE. Fail fast and specifically, before touching
# memory, rather than mid-load.
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image $IMAGE not pulled on $(hostname). Confirmed missing on spark-3a0c as of this" >&2
  echo "writing (MeshPlugin, OverlayCompat); pull it there before rank 2 can start, and" >&2
  echo "solve the ~37 GiB free headroom problem first -- do not pull a 31 GiB image onto a" >&2
  echo "node this close to the edge without freeing space." >&2
  exit 1
}
test -f "$REPO_HOST/snapshots/$REV/config.json" || { echo "checkpoint missing at $REPO_HOST/snapshots/$REV; not staged on this node" >&2; exit 1; }
test -f "$PATCH" || { echo "SM121 indexer top-k patch missing at $PATCH" >&2; exit 1; }
test -f "$CONFIG_HOST" || { echo "padded config.json missing at $CONFIG_HOST -- ConfigPadding has not staged it here yet" >&2; exit 1; }
test -f "$HF_OVERRIDES_FILE" || { echo "hf-overrides file missing at $HF_OVERRIDES_FILE -- ConfigPadding has not staged it here yet" >&2; exit 1; }
for f in "$OVERLAY_MODEL_PY" "$OVERLAY_VOCAB_PY" "$OVERLAY_WEIGHT_UTILS_PY" "$OVERLAY_PARAMETER_PY"; do
  test -f "$f" || { echo "overlay file missing: $f -- OverlayCompat has not staged it here yet" >&2; exit 1; }
done
test -f "$PLUGIN_DIR/libnccl-net-mesh.so" || { echo "mesh plugin missing at $PLUGIN_DIR/libnccl-net-mesh.so -- MeshPlugin has not staged it here yet" >&2; exit 1; }

HF_OVERRIDES="$(cat "$HF_OVERRIDES_FILE")"
python3 -c 'import json,sys; json.loads(sys.argv[1])' "$HF_OVERRIDES" || {
  echo "hf-overrides file at $HF_OVERRIDES_FILE does not parse as JSON" >&2
  exit 1
}

# Best-effort per-boot state check. Cannot fix these (no sudo password); can at least say so
# loudly instead of letting a slow node read as a mystery. Never blocks the launch on this.
# Check the ROCE interfaces, not the LAN: jumbo frames matter for the QSFP triangle that
# carries tensor-parallel traffic. enP7s7 is the coordination LAN and is correctly 1500;
# warning on that one sends you chasing a non-problem. Verified after the 2026-09-01
# reboots: all four RoCE netdevs came back at 9000 from netplan, only the LAN reset.
for roce in enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1; do
  m="$(cat "/sys/class/net/$roce/mtu" 2>/dev/null || echo absent)"
  [ "$m" = "9000" ] || echo "note: $roce MTU is $m, expected 9000 (netplan 99-nvidia-sync-cluster.yaml)" >&2
done

# Preserve the image's own CUDA library path rather than clobbering it. The plugin's own
# README says prepend, not replace: "export LD_LIBRARY_PATH=/path/to/plugin:${LD_LIBRARY_PATH:-}".
BASE_LD_LIBRARY_PATH="$(docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^LD_LIBRARY_PATH=//p')"
MESH_LD_LIBRARY_PATH="/opt/nccl-mesh${BASE_LD_LIBRARY_PATH:+:$BASE_LD_LIBRARY_PATH}"

docker rm -f "$NAME" 2>/dev/null || true

# GB10 unified memory has no relief valve: one 121 GiB pool shared by page cache and model,
# swappiness 0, weights pinned via memlock. Loading on top of an unreleased allocation wedges
# the node hard enough to need a power cycle -- three power cycles cost in one day on this
# cluster already. THIS IS THE SINGLE MOST IMPORTANT GUARD IN THIS FILE. Refuse to start
# until memory is genuinely back, every time, on every rank, no exceptions.
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
  echo "if this is spark-3a0c: qwen38-flash is still resident. Stop it deliberately first," >&2
  echo "this script will not do it for you." >&2
  exit 1
fi
echo "preflight ok: ${avail} GiB available"

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$REPO_HOST:/models/glm-repo:ro" \
  -v "$CONFIG_HOST:$MODEL_PATH/config.json:ro" \
  -v "$OVERLAY_MODEL_PY:$VLLM_PKG/models/glm5next/nvidia/model.py:ro" \
  -v "$OVERLAY_VOCAB_PY:$VLLM_PKG/model_executor/layers/vocab_parallel_embedding.py:ro" \
  -v "$OVERLAY_WEIGHT_UTILS_PY:$VLLM_PKG/model_executor/model_loader/weight_utils.py:ro" \
  -v "$OVERLAY_PARAMETER_PY:$VLLM_PKG/model_executor/parameter.py:ro" \
  -v "$PATCH:$VLLM_PKG/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  -v "$PLUGIN_DIR:/opt/nccl-mesh:ro" \
  -v "$CACHE_HOST:/cache" \
  $EXTRA_MOUNTS \
  $EXTRA_ENV \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e LD_LIBRARY_PATH="$MESH_LD_LIBRARY_PATH" \
  -e NCCL_NET=Mesh -e NCCL_NET_PLUGIN=mesh -e NCCL_ALGO=Ring \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME="=$LAN_IF" \
  -e GLOO_SOCKET_IFNAME="$LAN_IF" -e TP_SOCKET_IFNAME="$LAN_IF" -e MN_IF_NAME="$LAN_IF" \
  -e NCCL_MESH_DEBUG=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 3 \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 6 \
    --block-size 2304 \
    --moe-backend marlin \
    --kv-cache-dtype "$KV_DTYPE" \
    $SPEC \
    --hf-overrides "$HF_OVERRIDES" \
    --mm-encoder-tp-mode data \
    $EAGER \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --chat-template "$MODEL_PATH/chat_template.jinja" \
    --distributed-executor-backend mp \
    --nnodes 3 --node-rank "$RANK" \
    --master-addr "$RANK0_LAN" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$RANK/3 variant=$VARIANT max_model_len=$MAX_MODEL_LEN host=$HOST_IP"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2
  exit 1
}

cat <<NOTE

Readiness takes several minutes per rank (checkpoint load, TP=3 weight sharding, TileLang
and FlashInfer warmup), and this is the first time this checkpoint has ever loaded at TP=3.
Watch each rank, not just the head:

  docker logs -f vllm_glm53 2>&1 | grep -E 'Loading safetensors|Model loading took|Available KV|GPU KV cache size|Traceback'

A container that is up and a /health that returns 200 are both compatible with an engine
that emits pure repetition on this hardware (docs/gotchas.md). Before pointing any OMP role
at this endpoint, run:

  ./gate-toolcall-fidelity.sh http://<rank0-host>:8000 glm-5.3-flash

Never probe /v1/models for liveness; it can answer 200 from config alone with a dead engine
behind it. Probe /health for liveness only, and gate-toolcall-fidelity.sh for correctness.
NOTE
