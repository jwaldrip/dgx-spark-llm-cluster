#!/usr/bin/env bash
#
# Acceptance gate for a GLM-5.3-Flash endpoint on GB10.
#
# This exists because a clean boot proves nothing on this hardware: with the wrong MoE
# backend the engine loads, answers /health with 200, lists the model on /v1/models, and
# emits pure repetition. Every check below reads generated text.
#
# Usage: ./smoke-test.sh [endpoint]     default http://127.0.0.1:8000
set -uo pipefail

BASE="${1:-http://127.0.0.1:8000}"
E="$BASE/v1/chat/completions"
MODEL="glm-5.3-flash"
fail=0

say() { printf '\n=== %s ===\n' "$1"; }

say "0. liveness"
curl -fsS --max-time 10 "$BASE/health" >/dev/null && echo "health ok" || { echo "HEALTH FAILED"; exit 1; }

say "1. coherence, and no deliberation leaking into content"
curl -sS "$E" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"In one sentence: what is the capital of Germany?\"}],
  \"max_tokens\":200,\"temperature\":0}" | python3 -c '
import sys, json
d = json.load(sys.stdin); m = d["choices"][0]["message"]
c = m.get("content") or ""
print("content:", repr(c)[:200])
bad = any(s in c for s in ("The user is asking", "The user wants", "They want it in"))
print("VERDICT:", "FAIL deliberation leaked into content" if bad
      else ("PASS" if "Berlin" in c else "FAIL wrong answer"))
' || fail=1

say "2. token corruption probe (non-Latin, temperature 0, three runs)"
# 1500, not a few hundred: this prompt measured 907 completion tokens, nearly all of them
# reasoning. At max_tokens 600 the response is finish_reason=length with EMPTY content and
# zero U+FFFD, which a naive check scores as a pass.
for i in 1 2 3; do
curl -sS "$E" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"한국어로 인공지능의 미래에 대해 세 문장으로 설명해 주세요.\"}],
  \"max_tokens\":1500,\"temperature\":0}" | python3 -c '
import sys, json
d = json.load(sys.stdin); ch = d["choices"][0]
t = ch["message"].get("content") or ""
n = t.count(chr(0xFFFD)); h = sum(1 for c in t if 0xAC00 <= ord(c) <= 0xD7A3)
u = d["usage"]["completion_tokens"]; f = ch.get("finish_reason")
verdict = "PASS" if n == 0 and h > 10 else (
    "FAIL empty content, raise max_tokens" if not t else "FAIL")
print(f"len={len(t):5d}  U+FFFD={n}  hangul={h}  completion={u:5d}  finish={f}  {verdict}")
' || fail=1
done

say "3. structured tool call"
curl -sS "$E" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Call get_build_status exactly once for project \\\"spark\\\". Do not answer in prose.\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_build_status\",\"parameters\":{\"type\":\"object\",\"properties\":{\"project\":{\"type\":\"string\"}},\"required\":[\"project\"]}}}],
  \"tool_choice\":\"auto\",\"max_tokens\":300,\"temperature\":0}" > /tmp/gb_tc.json
python3 -c '
import json
d = json.load(open("/tmp/gb_tc.json")); c = d["choices"][0]; m = c["message"]
tc = m.get("tool_calls") or []
ok = (len(tc) == 1 and tc[0]["function"]["name"] == "get_build_status"
      and json.loads(tc[0]["function"]["arguments"]).get("project") == "spark"
      and c.get("finish_reason") == "tool_calls")
print("tool_calls:", json.dumps(tc)[:220])
print("content:", repr(m.get("content"))[:120])
print("VERDICT:", "PASS" if ok else "FAIL")
' || fail=1

say "4. tool result continuation"
python3 - <<'PY' > /tmp/gb_cont.json
import json
d = json.load(open("/tmp/gb_tc.json"))
a = d["choices"][0]["message"]
tc = a["tool_calls"][0]
print(json.dumps({
  "model": "glm-5.3-flash",
  "messages": [
    {"role": "user", "content": 'Call get_build_status exactly once for project "spark". Then report the result.'},
    a,
    {"role": "tool", "tool_call_id": tc["id"],
     "content": json.dumps({"project": "spark", "status": "green", "revision": "abc1234"})},
  ],
  "max_tokens": 300, "temperature": 0,
}))
PY
curl -sS "$E" -H 'Content-Type: application/json' -d @/tmp/gb_cont.json | python3 -c '
import sys, json
c = json.load(sys.stdin)["choices"][0]["message"].get("content") or ""
print("content:", repr(c)[:260])
print("VERDICT:", "PASS" if ("green" in c and "abc1234" in c) else "FAIL did not read tool result")
' || fail=1

say "5. temperature-0 stability sample (four runs)"
# This is a sample, not a determinism assertion. On this TP2 endpoint, temperature 0 is NOT
# reproducible: measured serially against an idle engine, three of four runs produced
# identical content while one differed, and reasoning length varied 219 to 380 completion
# tokens on every run. Suspected causes are cross-rank reduction ordering, MoE routing
# sensitivity to batch numerics, and marlin's emulated FP4 path. Treat varying hashes as
# expected, and never build a one-shot A/B comparison on this endpoint.
#
# max_tokens must be generous: with thinking enabled, reasoning tokens are charged against
# this budget, so a small value returns finish_reason=length and EMPTY content. An empty
# string hashes consistently, which would make a broken engine look perfectly stable.
for i in 1 2 3 4; do
curl -sS "$E" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"List three prime numbers above 100.\"}],
  \"max_tokens\":600,\"temperature\":0}" | python3 -c '
import sys, json, hashlib
d = json.load(sys.stdin); ch = d["choices"][0]
t = ch["message"].get("content") or ""
u = d["usage"]["completion_tokens"]; f = ch.get("finish_reason")
h = hashlib.sha1(t.encode()).hexdigest()[:16]
flag = "  <-- FAIL empty content, raise max_tokens" if not t else ""
print(f"sha1={h}  len={len(t):4d}  completion={u:4d}  finish={f}{flag}")
'
done
echo "(content should be non-empty every run; identical hashes are common but not guaranteed)"

printf '\n'
[ "$fail" -eq 0 ] && echo "smoke test completed, read the VERDICT lines" || echo "one or more checks errored"
