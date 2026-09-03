#!/usr/bin/env bash
# Live status board for the GLM TP=3 cluster. Runs on the workstation, polls
# rank 0 through the existing tunnel. No agents installed on the nodes.
#
#   watch -c -n 2 ./kpi/monitor.sh          # or just run it in a loop
#
# Everything shown is a native surface: nvidia-smi on the nodes and vLLM's
# Prometheus /metrics. Nothing extra is installed anywhere.

set -euo pipefail
EP="${EP:-http://127.0.0.1:18002}"

kv=$(curl -fsS --max-time 4 "$EP/metrics" 2>/dev/null | awk '
  /^vllm:kv_cache_usage_perc/            {kv=$NF}
  /^vllm:num_requests_running/           {run=$NF}
  /^vllm:num_requests_waiting/           {wait=$NF}
  /^vllm:prefix_cache_hits_total/        {h=$NF}
  /^vllm:prefix_cache_queries_total/     {q=$NF}
  /^vllm:num_preemptions_total/          {pre=$NF}
  END {printf "run=%s wait=%s kv=%.0f%% hit=%s preempt=%s", run, wait, kv*100,
       (q>0 ? sprintf("%.1f%%", 100*h/q) : "n/a"), pre}')
[ -n "${kv:-}" ] || kv="endpoint unreachable"

echo "spark-glm TP=3  $(date '+%H:%M:%S')"
echo "  vLLM : $kv"
i=0
for n in 10.4.1.119 10.4.1.233 10.4.1.191; do
  i=$((i+1))
  printf "  rank%d: %s\n" "$i" \
    "$(ssh -o ConnectTimeout=4 "jwaldrip@$n" 'nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader' 2>/dev/null || echo unreachable)"
done
