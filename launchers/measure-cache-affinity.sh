#!/usr/bin/env bash
#
# Answer one question: does a session's turns keep hitting the same replica's prefix cache?
#
# This is the number that decides whether a data-parallel replica set is a win or a
# regression. Per-replica caches mean scattered routing turns a 1.82 s warm prefill into a
# 103.6 s cold one, on a workload measured at 181 prompt tokens per generated token.
#
# Reads real per-replica counters, not the router's own bookkeeping, because the router
# believing it was sticky is not evidence that the engine got a cache hit.
#
# CAVEAT, read this before trusting a number. vLLM's prefix_cache_* counters are per
# ENGINE, not per session, so any other traffic on a replica lands in the same delta. Run
# this against an otherwise idle fleet, or treat the rate as fleet-wide rather than as this
# session's. The affinity verdict itself is robust either way: it is the replicas showing
# NO traffic that prove placement was sticky.
#
# Also expect a fresh session to score well below the lifetime rate. Measured on one live
# node, an 8-turn session with a ~9k-token stable prefix scored 61.2% because turn 0 is
# entirely cold, against 87.7% lifetime across 22.4M queries on long-lived sessions.
#
# Usage: ./measure-cache-affinity.sh <endpoint> [replica-metrics-url ...]
#   ./measure-cache-affinity.sh http://127.0.0.1:18010 \
#       http://10.100.138.2:8000 http://10.100.138.1:8000 http://10.100.140.1:8000
set -uo pipefail

ENDPOINT="${1:?usage: measure-cache-affinity.sh <endpoint> [replica-url ...]}"
shift || true
REPLICAS=("$@")
MODEL="${MODEL:-qwen3.8-flash-next}"
TURNS="${TURNS:-8}"

snapshot() {
  for r in "${REPLICAS[@]}"; do
    q=$(curl -fsS --max-time 5 "$r/metrics" 2>/dev/null | awk '/^vllm:prefix_cache_queries_total\{/{print $2}')
    h=$(curl -fsS --max-time 5 "$r/metrics" 2>/dev/null | awk '/^vllm:prefix_cache_hits_total\{/{print $2}')
    echo "$r ${q:-0} ${h:-0}"
  done
}

# A realistic agent prompt: a large stable system block, then a conversation that grows.
# The stable block is what a warm cache is supposed to absorb on every turn after the first.
build_body() {
  python3 - "$1" "$MODEL" <<'PY'
import json, sys
turns, model = int(sys.argv[1]), sys.argv[2]
system = "You are a coding agent operating in a repository. " * 900   # ~8k tokens of stable prefix
msgs = [{"role": "system", "content": system},
        {"role": "user", "content": "Begin the task."}]
for i in range(turns):
    msgs.append({"role": "assistant", "content": f"Step {i}: inspected a file."})
    msgs.append({"role": "user", "content": f"Continue with step {i+1}."})
print(json.dumps({"model": model, "messages": msgs, "max_tokens": 24, "temperature": 0}))
PY
}

echo "=== before ==="
BEFORE=$(snapshot); echo "$BEFORE"

echo
echo "=== driving $TURNS turns of one growing session ==="
for i in $(seq 0 $((TURNS - 1))); do
  build_body "$i" > /tmp/affinity-body.json
  hdr=$(curl -sS -D - -o /tmp/affinity-out.json --max-time 600 \
        "$ENDPOINT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @/tmp/affinity-body.json \
        | awk 'tolower($1) ~ /^x-router-(replica|affinity):/ {print $1, $2}' | tr '\n' ' ')
  toks=$(python3 -c "import json;d=json.load(open('/tmp/affinity-out.json'));print(d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null)
  echo "turn $i  prompt_tokens=$toks  $hdr"
done

echo
echo "=== after ==="
AFTER=$(snapshot); echo "$AFTER"

echo
echo "=== per-replica delta ==="
python3 - <<PY
before = """$BEFORE""".strip().splitlines()
after  = """$AFTER""".strip().splitlines()
tot_q = tot_h = 0
for b, a in zip(before, after):
    url, bq, bh = b.split()
    _,   aq, ah = a.split()
    dq, dh = float(aq) - float(bq), float(ah) - float(bh)
    tot_q += dq; tot_h += dh
    rate = f"{100*dh/dq:.1f}%" if dq else "no traffic"
    print(f"{url:32s} queries +{dq:>10,.0f}  hits +{dh:>10,.0f}  {rate}")
if tot_q:
    print(f"\nfleet hit rate: {100*tot_h/tot_q:.1f}%  (single-node baseline measured 87.7%)")
    print("Replicas with 'no traffic' are the proof of affinity: a scattered session")
    print("would show queries on every one of them.")
else:
    print("\nno prefix-cache counters moved: check that the replica URLs expose /metrics")
PY
