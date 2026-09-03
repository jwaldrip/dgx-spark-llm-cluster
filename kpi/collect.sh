#!/usr/bin/env bash
#
# Snapshot the local cluster's serving state into an append-only JSONL, once per run.
# Intended to run from cron or a LaunchAgent every few minutes for the duration of a
# trial, so that at the end there is DATA rather than recollection.
#
# Design notes worth keeping:
#
# * Every counter vLLM exposes is a CUMULATIVE total since engine start, not a rate. A
#   restart resets them to zero. So the collector records the raw totals plus the engine
#   start time, and the report computes deltas only WITHIN a continuous engine run. Diffing
#   across a restart silently produces negative or nonsense rates, which is the classic way
#   these dashboards end up lying.
# * A failed scrape is recorded as a row with ok=false, never skipped. An absent row and a
#   zero row mean opposite things, and a gap that reads as "quiet" is how an outage gets
#   reported as healthy.
# * Node memory is recorded because the failure mode on this hardware is a unified-memory
#   wedge needing a physical power cycle, and headroom trending down is the early warning.
set -uo pipefail

OUT="${OUT:-$HOME/.omp/kpi/local-serving.jsonl}"
RANK0="${RANK0:-10.4.1.119}"
NODES="${NODES:-10.4.1.119 10.4.1.233 10.4.1.191}"
EP="${EP:-http://$RANK0:8000}"

mkdir -p "$(dirname "$OUT")"
now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- endpoint scrape -------------------------------------------------------------
metrics="$(curl -fsS --max-time 10 "$EP/metrics" 2>/dev/null)"
scrape_ok=true
[ -z "$metrics" ] && scrape_ok=false

m() { # sum a counter across label sets; empty when absent
  printf '%s\n' "$metrics" | awk -v k="$1" '
    index($0, k) == 1 { s += $NF; seen = 1 }
    END { if (seen) printf "%.6f", s }'
}

health="$(curl -fsS --max-time 8 -o /dev/null -w '%{http_code}' "$EP/health" 2>/dev/null || echo 0)"

# --- per-node health -------------------------------------------------------------
nodes_json="["
first=true
for ip in $NODES; do
  raw="$(timeout 25 ssh -o ConnectTimeout=10 -o BatchMode=yes "jwaldrip@$ip" \
    'st=$(docker inspect vllm_glm53 --format "{{.State.Status}}|{{.RestartCount}}|{{.State.StartedAt}}" 2>/dev/null || echo "absent|0|");
     mem=$(free -g | awk "/Mem:/{print \$7}");
     sw=$(free -m | awk "/Swap:/{print \$3}");
     echo "$st|$mem|$sw"' 2>/dev/null)"
  if [ -z "$raw" ]; then
    row="{\"ip\":\"$ip\",\"reachable\":false}"
  else
    IFS='|' read -r cst crestarts cstarted cmem csw <<<"$raw"
    row="{\"ip\":\"$ip\",\"reachable\":true,\"container\":\"${cst:-?}\",\"restarts\":${crestarts:-0},\"started_at\":\"${cstarted:-}\",\"mem_avail_gib\":${cmem:-0},\"swap_used_mib\":${csw:-0}}"
  fi
  $first || nodes_json="$nodes_json,"
  nodes_json="$nodes_json$row"
  first=false
done
nodes_json="$nodes_json]"

# --- emit ------------------------------------------------------------------------
# jq builds the row so a missing metric becomes null rather than a syntax error.
jq -cn \
  --arg ts "$now" \
  --argjson ok "$scrape_ok" \
  --arg health "$health" \
  --argjson nodes "$nodes_json" \
  --arg running        "$(m 'vllm:num_requests_running')" \
  --arg waiting        "$(m 'vllm:num_requests_waiting')" \
  --arg wait_capacity  "$(m 'vllm:num_requests_waiting_by_reason{engine="0",model_name="glm-5.3-flash",reason="capacity"')" \
  --arg preemptions    "$(m 'vllm:num_preemptions_total')" \
  --arg kv_usage       "$(m 'vllm:kv_cache_usage_perc')" \
  --arg pc_queries     "$(m 'vllm:prefix_cache_queries_total')" \
  --arg pc_hits        "$(m 'vllm:prefix_cache_hits_total')" \
  --arg prompt_tok     "$(m 'vllm:prompt_tokens_total')" \
  --arg gen_tok        "$(m 'vllm:generation_tokens_total')" \
  --arg iter_count     "$(m 'vllm:iteration_tokens_total_count')" \
  --arg iter_sum       "$(m 'vllm:iteration_tokens_total_sum')" \
  '{
     ts: $ts, scrape_ok: $ok, health_code: ($health|tonumber),
     nodes: $nodes,
     gauges: {
       requests_running: ($running|if .=="" then null else tonumber end),
       requests_waiting: ($waiting|if .=="" then null else tonumber end),
       waiting_capacity: ($wait_capacity|if .=="" then null else tonumber end),
       kv_cache_usage_perc: ($kv_usage|if .=="" then null else tonumber end)
     },
     counters: {
       preemptions_total: ($preemptions|if .=="" then null else tonumber end),
       prefix_cache_queries_total: ($pc_queries|if .=="" then null else tonumber end),
       prefix_cache_hits_total: ($pc_hits|if .=="" then null else tonumber end),
       prompt_tokens_total: ($prompt_tok|if .=="" then null else tonumber end),
       generation_tokens_total: ($gen_tok|if .=="" then null else tonumber end),
       iteration_tokens_count: ($iter_count|if .=="" then null else tonumber end),
       iteration_tokens_sum: ($iter_sum|if .=="" then null else tonumber end)
     }
   }' >> "$OUT"

echo "appended $now to $OUT"
