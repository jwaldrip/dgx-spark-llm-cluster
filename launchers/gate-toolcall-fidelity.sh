#!/usr/bin/env bash
#
# Acceptance gate for a GLM-5.3-Flash endpoint. MUST pass BEFORE any OMP role is pointed
# at it.
#
# A green /health proves nothing on this hardware (docs/gotchas.md): the wrong MoE backend
# loads clean, answers /health 200, lists the model on /v1/models, and emits pure
# repetition. And the RedHatAI-at-TP=3 substitution this repo made to dodge vllm#54150
# (docs/glm-tp3-reproduction.md) has never been run before this cluster tried it, so THIS
# script is what decides whether that substitution actually held, not a passing container.
#
# Two signatures of the same bug, both from vllm-project/vllm#54150's own measurement: the
# marlin repack mis-scales one shard of the fused w13 MoE GEMM by up to 10x, producing
# corrupted token IDs -- measured as 86 U+FFFD replacement characters across 6 runs on the
# corrupting ModelOpt checkpoint, 0/6 on the RedHatAI checkpoint this cluster runs. A
# corrupted token landing inside a tool-call block desyncs the parser into a repetition
# lock (docs/gotchas.md's "locklocklock..." collapse), so a runaway-repetition scan on
# generated text, including tool-call JSON, is the second signature of the identical defect.
# Neither check is optional; either one alone can pass while the other fails.
#
# Usage: ./gate-toolcall-fidelity.sh <endpoint-url> <model-name> [runs]     runs default 6,
#        matching the #54150 issue's own 6-run measurement methodology.
#
#   ./gate-toolcall-fidelity.sh http://127.0.0.1:18003 glm-5.3-flash
#   ./gate-toolcall-fidelity.sh http://spark-74a7.local:8000 glm-5.3-flash 10
set -uo pipefail

BASE="${1:?usage: gate-toolcall-fidelity.sh <endpoint-url> <model-name> [runs]}"
MODEL="${2:?usage: gate-toolcall-fidelity.sh <endpoint-url> <model-name> [runs]}"
RUNS="${3:-6}"
BASE="${BASE%/}"

command -v python3 >/dev/null 2>&1 || { echo "python3 required, none on PATH" >&2; exit 2; }

echo "=== 0. liveness ==="
curl -fsS --max-time 10 "$BASE/health" >/dev/null && echo "health ok" || {
  echo "FAIL: $BASE/health unreachable. A live engine is a precondition for this gate, not" >&2
  echo "what it proves -- fix connectivity before reading anything below as a real result." >&2
  exit 1
}

python3 - "$BASE" "$MODEL" "$RUNS" <<'PY'
import json
import re
import sys
import urllib.error
import urllib.request

base, model, runs = sys.argv[1], sys.argv[2], int(sys.argv[3])
endpoint = f"{base}/v1/chat/completions"
TIMEOUT = 300

# The specific corruption signature of vllm-project/vllm#54150.
FFFD = "\ufffd"

# The specific desync signature: a short run repeated back to back many times, the
# "locklocklock..." collapse docs/gotchas.md documents for the wrong MoE backend, and the
# same failure mode a corrupted token inside a tool-call block produces per the issue.
REPETITION_LOCK = re.compile(r"(.{1,24}?)\1{7,}", re.DOTALL)


def repetition_lock(text):
    m = REPETITION_LOCK.search(text or "")
    return (True, m.group(0)[:160]) if m else (False, None)


def post(payload, timeout=TIMEOUT):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


results = {"corruption": [], "toolcall": None, "continuation": None}
fail = False

# ---------------------------------------------------------------------- 1. corruption probe
print(f"\n=== 1. token corruption probe ({runs} runs, non-Latin generation, temperature 0) ===")
# max_tokens must be generous: with thinking enabled, reasoning tokens are charged against
# it first (docs/gotchas.md). A small cap returns finish_reason=length with EMPTY content
# and zero U+FFFD, which a naive check scores as a pass on nothing. 1500 is the value this
# repo's own smoke-test.sh proved sufficient (907 measured completion tokens on this prompt).
PROMPT = "한국어로 인공지능의 미래에 대해 다섯 문장으로 설명해 주세요."
total_fffd = 0
for i in range(1, runs + 1):
    try:
        d = post(
            {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 1500,
                "temperature": 0,
            }
        )
        choice = d["choices"][0]
        text = choice["message"].get("content") or ""
        n_fffd = text.count(FFFD)
        completion = d.get("usage", {}).get("completion_tokens", -1)
        finish = choice.get("finish_reason")
        locked, sample = repetition_lock(text)
        total_fffd += n_fffd
        empty = not text
        run_fail = n_fffd > 0 or locked or empty
        fail = fail or run_fail
        results["corruption"].append(
            {"run": i, "fffd": n_fffd, "completion": completion, "finish": finish,
             "empty": empty, "repetition_lock": locked}
        )
        verdict = "FAIL" if run_fail else "PASS"
        print(
            f"  run {i}: len={len(text):5d}  U+FFFD={n_fffd}  completion={completion:5d}  "
            f"finish={finish}  repetition_lock={locked}  {verdict}"
        )
        if empty:
            print(f"    empty content -- raise max_tokens or check finish_reason={finish}, "
                  f"an empty run proves nothing either way")
        if locked:
            print(f"    repetition sample: {sample!r}")
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        fail = True
        results["corruption"].append({"run": i, "error": str(e)})
        print(f"  run {i}: FAIL request error: {e}")

# --------------------------------------------------------------- 2. structured tool call
print("\n=== 2. structured tool call ===")
tc_msg = None
try:
    d = post(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": 'Call get_build_status exactly once for project "spark". '
                    "Do not answer in prose.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_build_status",
                        "parameters": {
                            "type": "object",
                            "properties": {"project": {"type": "string"}},
                            "required": ["project"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 300,
            "temperature": 0,
        }
    )
    choice = d["choices"][0]
    tc_msg = choice["message"]
    tool_calls = tc_msg.get("tool_calls") or []
    args_ok = False
    args_text = ""
    if len(tool_calls) == 1:
        args_text = tool_calls[0]["function"]["arguments"]
        try:
            args = json.loads(args_text)
            args_ok = args.get("project") == "spark"
        except json.JSONDecodeError:
            args_ok = False
    locked, sample = repetition_lock(json.dumps(tool_calls) + (tc_msg.get("content") or ""))
    ok = (
        len(tool_calls) == 1
        and tool_calls[0]["function"]["name"] == "get_build_status"
        and args_ok
        and choice.get("finish_reason") == "tool_calls"
        and not locked
    )
    results["toolcall"] = {"ok": ok, "tool_calls": tool_calls, "repetition_lock": locked}
    fail = fail or not ok
    print(f"  tool_calls: {json.dumps(tool_calls)[:220]}")
    print(f"  arguments well-formed and correct: {args_ok}")
    print(f"  repetition_lock: {locked}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    if locked:
        print(f"    repetition sample: {sample!r}")
except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
    fail = True
    print(f"  FAIL request error: {e}")

# ------------------------------------------------------------ 3. tool result continuation
print("\n=== 3. tool result continuation (multi-turn) ===")
if tc_msg is not None and (tc_msg.get("tool_calls") or []):
    try:
        tc = tc_msg["tool_calls"][0]
        d = post(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Call get_build_status exactly once for project "spark". '
                        "Then report the result.",
                    },
                    tc_msg,
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(
                            {"project": "spark", "status": "green", "revision": "abc1234"}
                        ),
                    },
                ],
                "max_tokens": 300,
                "temperature": 0,
            }
        )
        content = d["choices"][0]["message"].get("content") or ""
        locked, sample = repetition_lock(content)
        ok = "green" in content and "abc1234" in content and not locked
        results["continuation"] = {"ok": ok, "repetition_lock": locked}
        fail = fail or not ok
        print(f"  content: {content[:260]!r}")
        print(f"  repetition_lock: {locked}")
        print(f"  {'PASS' if ok else 'FAIL'}")
        if locked:
            print(f"    repetition sample: {sample!r}")
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
        fail = True
        print(f"  FAIL request error: {e}")
else:
    fail = True
    print("  SKIPPED (no tool call to continue) -- counts as FAIL, step 2 must pass first")

# --------------------------------------------------------------------------------- summary
print(f"\n=== SUMMARY ===")
print(f"corruption probe: {total_fffd} total U+FFFD across {runs} runs "
      f"(reference: 86/6 corrupting checkpoint, 0/6 immune checkpoint)")
print(f"tool call:        {'PASS' if results['toolcall'] and results['toolcall']['ok'] else 'FAIL'}")
print(f"continuation:     {'PASS' if results['continuation'] and results['continuation']['ok'] else 'FAIL'}")
print(f"repetition locks: "
      f"{sum(1 for r in results['corruption'] if r.get('repetition_lock'))} of {runs} corruption runs, "
      f"toolcall={bool(results['toolcall'] and results['toolcall'].get('repetition_lock'))}, "
      f"continuation={bool(results['continuation'] and results['continuation'].get('repetition_lock'))}")
print(f"\nVERDICT: {'FAIL' if fail else 'PASS'} -- endpoint {base} model {model}")
sys.exit(1 if fail else 0)
PY
exit $?
