#!/usr/bin/env python3
"""Discriminate KV eviction from prefix diversity as the cause of hit-rate decay.

Why this exists: collected data showed the prefix-cache hit rate falling as
concurrency rose (97.5% at 1 session -> 73.9% at 4) while kv_cache_usage_perc
sat at 23.7%. Eviction cannot explain a fall at a quarter-full pool, so two
mechanisms were confounded:

  eviction         - concurrent contexts exceed the KV pool, so prefixes that
                     are still in use get reclaimed. More KV (a fourth node)
                     fixes it.
  prefix diversity - concurrent sessions share no prefix, and each session's
                     first call is necessarily a cold miss. More KV does NOT
                     fix it, and buying hardware for it would be waste.

The earlier observation used ~120K prompts, so 4 sessions occupied only ~18% of
the 2.65M pool -- of course KV was low. This test uses the MEASURED main-lane
mean context of ~351K tokens, at which 9 concurrent sessions demand ~3.15M
tokens against a ~2.65M pool, so eviction is forced rather than hoped for.

Design. Two barriered waves per arm, N sessions each holding one long unique
prefix:

  wave 1 (cold) - every session sends its prefix. All N prefixes are therefore
                  resident simultaneously, which IS the eviction condition.
  wave 2 (warm) - every session re-sends its prefix plus a short new suffix.

The discriminator is wave 2's hit rate in isolation:

  warm hit stays high as N rises -> the pool held every prefix; decay seen
                                    elsewhere is diversity, not capacity.
  warm hit collapses as N rises -> live prefixes were reclaimed between the
                                    waves. That is eviction, and only more KV
                                    fixes it.

Measured from vllm:prefix_cache_{queries,hits}_total differenced across each
wave. Per-request cached_tokens is NOT usable here: this build reports
prompt_tokens_details as null, so an earlier version of this script read a
constant 0.0 warm hit rate and would have "proved" total eviction. Verified
against the endpoint before trusting it.

Because the counters are global, foreign inference traffic corrupts a wave.
The script refuses to start an arm unless the engine is idle, and records the
running-request gauge at every boundary so contamination is visible afterwards
rather than silently averaged in.
"""
import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

EP = os.environ.get("EP", "http://10.4.1.119:8000")
MODEL = os.environ.get("MODEL", "glm-5.3-flash")
WORDS = ["ledger", "reconcile", "batch", "shard", "entry", "audit", "commit", "replica",
         "cursor", "segment", "offset", "journal", "snapshot", "digest", "cohort", "epoch",
         "token", "lease", "quorum", "vector", "manifest", "checkpoint", "partition", "index"]


def scrape():
    """prefix-cache counters plus live gauges, or None if the scrape failed.

    None is propagated rather than coerced to zero: a failed scrape must never
    read as an idle engine or a zero-hit wave.
    """
    try:
        with urllib.request.urlopen(EP + "/metrics", timeout=15) as r:
            txt = r.read().decode()
    except Exception:
        return None
    want = {"vllm:prefix_cache_queries_total": "q", "vllm:prefix_cache_hits_total": "h",
            "vllm:kv_cache_usage_perc": "kv", "vllm:num_requests_running": "running",
            "vllm:num_requests_waiting": "waiting", "vllm:num_preemptions_total": "preempt"}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for metric, key in want.items():
            if line.startswith(metric):
                try:
                    out[key] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return out if len(out) == len(want) else None


def wait_idle(timeout=600):
    """Global counters make a busy engine uninterpretable. Refuse rather than
    publish a contaminated number."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = scrape()
        if s and s["running"] == 0 and s["waiting"] == 0:
            return s
        time.sleep(5)
    return None


def post(content, timeout):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 4, "temperature": 0}
    req = urllib.request.Request(EP + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["usage"]["prompt_tokens"]


def run_wave(prefixes, suffix, timeout):
    def one(p):
        try:
            return {"prompt": post(p + suffix, timeout)}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
    with ThreadPoolExecutor(max_workers=len(prefixes)) as ex:
        return list(ex.map(one, prefixes))


def hit(before, after):
    dq = after["q"] - before["q"]
    dh = after["h"] - before["h"]
    if dq <= 0:
        return None  # counters reset (engine restart) or no queries recorded
    return round(dh / dq, 4)


def arm(par, words, timeout):
    idle = wait_idle()
    if idle is None:
        return {"sessions": par, "skipped": "engine never went idle; global counters "
                                            "would be contaminated by foreign traffic"}
    rnd = random.Random(time.time_ns())
    prefixes = [" ".join(rnd.choice(WORDS) for _ in range(words)) for _ in range(par)]

    # SETTLE: the prefix-cache counters flush a scheduler step behind the HTTP
    # response. Scraping immediately attributes a wave's hits to the NEXT
    # window, which is how an early run reported cold_hit 0.83 / warm_hit 0.00
    # -- exactly inverted. Validated against a known-good repeat: at 120K a warm
    # wave reads 96.0% (50 of 52 blocks), matching the cloud-era 96.8%.
    settle = 5
    s0 = scrape()
    t0 = time.time()
    cold = run_wave(prefixes, "\n\nReply with the single word OK.", timeout)
    time.sleep(settle)
    s1 = scrape()
    t1 = time.time()
    warm = run_wave(prefixes, "\n\nNow reply with the single word DONE.", timeout)
    time.sleep(settle)
    s2 = scrape()
    t2 = time.time()

    if not all((s0, s1, s2)):
        return {"sessions": par, "skipped": "a metrics scrape failed; wave unmeasurable"}

    ok_cold = [r for r in cold if "error" not in r]
    ok_warm = [r for r in warm if "error" not in r]
    mean_prompt = int(sum(r["prompt"] for r in ok_warm) / len(ok_warm)) if ok_warm else None
    res = {
        "sessions": par, "words": words,
        "cold_wall_s": round(t1 - t0, 1), "warm_wall_s": round(t2 - t1, 1),
        "cold_ok": len(ok_cold), "warm_ok": len(ok_warm),
        "cold_hit": hit(s0, s1), "warm_hit": hit(s1, s2),
        "mean_prompt": mean_prompt,
        "kv_demand": mean_prompt * par if mean_prompt else None,
        "kv_after_cold": round(s1["kv"], 4), "kv_after_warm": round(s2["kv"], 4),
        "preemptions": int(s2["preempt"] - s0["preempt"]),
        "running_at_start": s0["running"],
    }
    errs = [r["error"] for r in cold + warm if "error" in r]
    if errs:
        res["errors"] = errs[:3]
        res["error_count"] = len(errs)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="3,6,9")
    ap.add_argument("--words", type=int, default=350000,
                    help="approx tokens per session; default is his measured mean context")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--out", default="/tmp/evict.jsonl")
    a = ap.parse_args()

    print(f"endpoint {EP} model {MODEL} words/session {a.words:,}")
    print("warm_hit is the discriminator; cold_hit is expected near 0\n", flush=True)
    with open(a.out, "a") as fh:
        for par in [int(x) for x in a.arms.split(",")]:
            r = arm(par, a.words, a.timeout)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(json.dumps(r), flush=True)
