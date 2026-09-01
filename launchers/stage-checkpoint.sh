#!/usr/bin/env bash
#
# Stage the GLM-5.3-Flash NVFP4 checkpoint plus the SM121 indexer patch on this node.
# Run on BOTH nodes of the pair: each rank loads from local NVMe.
#
# Roughly 184 GiB per node. Set an HF token first or the hub rate-limits you:
#   printf '%s' "$HF_TOKEN" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
set -uo pipefail

REPO="RedHatAI/GLM-5.3-Flash-NVFP4"
REV="36c184c6cda000a481711306df5adde42f63321a"
EXPECT_BYTES=197881157135

ST="$HOME/.local/state/spark-models"
LOG="$ST/glm-download.log"
PIDF="$ST/glm-download.pid"
mkdir -p "$ST" "$HOME/patches"

# hf CLI in its own venv, so nothing collides with system python
if [ ! -x "$HOME/.venvs/hf/bin/hf" ]; then
  python3 -m venv "$HOME/.venvs/hf"
  "$HOME/.venvs/hf/bin/pip" -q install -U 'huggingface_hub[cli]'
fi

# SM121 sparse-attention indexer top-k fix. Without it the engine dies on any decode past
# roughly 24K context on GB10. The launcher bind-mounts this over the shipped module.
curl -fsSL -o "$HOME/patches/sparse_attn_indexer_kpool.py" \
  'https://raw.githubusercontent.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/main/docker/sparse_attn_indexer_kpool_sm121.py'
echo "patch: $(wc -c < "$HOME/patches/sparse_attn_indexer_kpool.py") bytes  sha256=$(sha256sum "$HOME/patches/sparse_attn_indexer_kpool.py" | cut -c1-16)"
echo "compare that sha across both nodes before launching"

# Note: do not pkill -f on a pattern that appears in this script's own command line. The
# pattern matches the parent shell and kills the script partway through.
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "download already running, pid $(cat "$PIDF")"
  exit 0
fi

nohup "$HOME/.venvs/hf/bin/hf" download "$REPO" --revision "$REV" --max-workers 8 \
  > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") repo=$REPO rev=${REV:0:8}"
echo "follow: tail -f $LOG"

cat <<NOTE

When it finishes, verify rather than assume. Both nodes must agree byte for byte:

  snap=\$HOME/.cache/huggingface/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4/snapshots/$REV
  du -sbL "\$snap"                    # expect $EXPECT_BYTES
  find \$HOME/.cache/huggingface/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4 \\
       -name '*.incomplete' | wc -l   # expect 0

Then set the host state the engine needs (does not survive reboot):

  printf 'vm.swappiness = 0\\n' | sudo tee /etc/sysctl.d/90-gb10-vllm.conf
  sudo sysctl --system && sysctl vm.swappiness
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

Swap stays ON with swappiness 0. Disabling swap entirely fails differently, see
docs/gotchas.md.
NOTE
