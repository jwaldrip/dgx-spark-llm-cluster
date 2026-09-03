#!/usr/bin/env python3
"""Headline KPI: prefix-cache hit rate as a function of concurrent sessions.

Fallback rate was the wrong headline. Local capacity is bounded by KV, not by
throughput: per-call prompts average ~351K tokens, so a 2.65M-token pool holds
about 7.6 concurrent sessions. Above that the pool evicts, the prefix-cache hit
rate falls, and recompute scales as (1-H) -- so 96.8% -> 90.3% triples prefill
work. This script tests that hypothesis against collected data.

Reads ~/.omp/kpi/local-serving.jsonl (append-only, one row per scrape).

Method: prefix_cache_{queries,hits}_total are monotonic counters, so the hit
rate for an INTERVAL is delta(hits)/delta(queries) between consecutive rows --
never the lifetime ratio, which is dominated by history and moves too slowly to
show an effect. Each interval is bucketed by the concurrency observed at its
endpoints.

Known weakness, stated rather than hidden: num_requests_running is an
instantaneous gauge. An interval is labelled with max(running) at its two
endpoints, so brief concurrency spikes between scrapes are invisible and every
bucket label is a LOWER BOUND on true concurrency during that interval. Denser
sampling narrows this; it does not eliminate it.
"""
import json
import datetime
import os
import sys
from collections import defaultdict

PATH = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.omp/kpi/local-serving.jsonl")


def rows(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line while cron is mid-write


def contaminated_windows():
    """Windows that must never count toward the KPI, from freeze.json.

    Two kinds land here: bring-up and probe traffic (deliberately cache-hostile
    cold nonces), and synthetic experiment arms (saturating prefill). My own
    eviction experiments injected ~8M cold-prefix tokens after the 'clean'
    marker; without this exclusion the weekly verdict would read my probes as
    his workload and understate the hit rate.
    """
    for p in ("~/.omp/kpi/freeze.json", os.path.join(os.path.dirname(os.path.abspath(__file__)), "freeze.json")):
        fp = os.path.expanduser(p)
        if os.path.exists(fp):
            try:
                return json.load(open(fp)).get("contaminated_windows", []), fp
            except json.JSONDecodeError:
                pass
    return [], None


def row_ts(row):
    """Row timestamp as epoch seconds. The collector writes ISO-8601 'ts'
    (e.g. 2026-09-03T06:51:41Z); a bare epoch number is also tolerated."""
    ts = row.get("ts")
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return ts / 1000 if ts > 1e11 else float(ts)
    try:
        return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def is_contaminated(row, windows):
    ts = row_ts(row)
    if ts is None:
        return False  # unlabelled rows stay IN: exclusion needs evidence
    for w in windows:
        try:
            lo = datetime.datetime.fromisoformat(w["window_utc"][0].replace("Z", "+00:00")).timestamp()
            hi = datetime.datetime.fromisoformat(w["window_utc"][1].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError, IndexError):
            continue
        if lo <= ts <= hi:
            return True
    return False


def counter(row, name):
    return (row.get("counters") or {}).get(name)


def gauge(row, name):
    return (row.get("gauges") or {}).get(name)


def main():
    if not os.path.exists(PATH):
        sys.exit(f"no collector data at {PATH}")

    all_rows = list(rows(PATH))
    windows, wsrc = contaminated_windows()
    # A failed scrape must never read as a quiet interval, and a contaminated
    # interval must never read as workload. Both are excluded explicitly.
    # Keep the original index with each kept row. Pairing zip(good, good[1:])
    # would bridge an excluded window, attributing everything inside the gap
    # (e.g. a synthetic experiment's 2.2M cold queries) to one boundary
    # interval. Only originally-adjacent rows may form an interval.
    kept = [(i, r) for i, r in enumerate(all_rows)
            if r.get("scrape_ok")
            and counter(r, "prefix_cache_queries_total") is not None
            and not is_contaminated(r, windows)]
    skipped = len(all_rows) - len(kept)

    buckets = defaultdict(lambda: {"q": 0, "h": 0, "n": 0})
    kv_by_bucket = defaultdict(list)
    restarts = 0

    for (ia, a), (ib, b) in zip(kept, kept[1:]):
        if ib != ia + 1:
            continue  # gap: excluded rows between these two; interval unknowable
        dq = counter(b, "prefix_cache_queries_total") - counter(a, "prefix_cache_queries_total")
        dh = counter(b, "prefix_cache_hits_total") - counter(a, "prefix_cache_hits_total")
        if dq < 0 or dh < 0:
            restarts += 1  # engine restarted; counters reset, interval is meaningless
            continue
        if dq == 0:
            continue  # genuinely idle interval, not a data problem
        ra, rb = gauge(a, "requests_running"), gauge(b, "requests_running")
        conc = max(x for x in (ra, rb) if x is not None) if any(
            x is not None for x in (ra, rb)) else None
        if conc is None:
            continue
        key = int(conc)
        buckets[key]["q"] += dq
        buckets[key]["h"] += dh
        buckets[key]["n"] += 1
        kv = gauge(b, "kv_cache_usage_perc")
        if kv is not None:
            kv_by_bucket[key].append(kv)

    if not buckets:
        print(f"rows={len(all_rows)} usable={len(kept)} skipped={skipped}")
        print("No interval carried prefix-cache traffic yet. Not a failure: the")
        print("cluster has served no requests between scrapes. Re-run later.")
        return

    print(f"rows={len(all_rows)} usable={len(kept)} excluded={skipped} "
          f"engine_restarts={restarts}\n")
    print("concurrency is a LOWER BOUND (see module docstring)\n")
    print(f"{'sessions':>8} {'intervals':>9} {'queries':>14} {'hit rate':>9} "
          f"{'kv used':>8} {'recompute vs H=96.8%':>21}")
    print("-" * 76)
    base_miss = 1 - 0.968
    for k in sorted(buckets):
        b = buckets[k]
        h = b["h"] / b["q"]
        kv = kv_by_bucket.get(k)
        kv_s = f"{sum(kv) / len(kv):7.1%}" if kv else "      -"
        ratio = (1 - h) / base_miss if base_miss else float("inf")
        print(f"{k:>8} {b['n']:>9} {b['q']:>14,} {h:>8.1%} {kv_s} {ratio:>20.2f}x")

    tot_q = sum(b["q"] for b in buckets.values())
    tot_h = sum(b["h"] for b in buckets.values())
    print("-" * 76)
    print(f"{'all':>8} {sum(b['n'] for b in buckets.values()):>9} {tot_q:>14,} "
          f"{tot_h / tot_q:>8.1%}")

    print("\nVerdict on the eviction hypothesis:")
    hi = [k for k in buckets if k >= 7]
    lo = [k for k in buckets if k <= 3]
    if not hi:
        print("  UNDECIDED. No interval reached 7+ concurrent sessions, which is where")
        print("  the 2.65M KV pool is predicted to start evicting. The week has not yet")
        print("  exercised the condition, so three nodes are neither proven nor refuted.")
    elif not lo:
        print("  UNDECIDED. No low-concurrency baseline to compare against.")
    else:
        h_hi = sum(buckets[k]["h"] for k in hi) / sum(buckets[k]["q"] for k in hi)
        h_lo = sum(buckets[k]["h"] for k in lo) / sum(buckets[k]["q"] for k in lo)
        print(f"  H at <=3 sessions: {h_lo:.1%}")
        print(f"  H at >=7 sessions: {h_hi:.1%}   (delta {h_hi - h_lo:+.1%})")
        if h_hi < h_lo - 0.02:
            print(f"  DEGRADES under load. Recompute rises {(1 - h_hi) / (1 - h_lo):.2f}x.")
            print("  The eviction experiment measured this precisely: the cliff sits between")
            print("  41% and 82% of the nominal pool, i.e. ~3-4 concurrent sessions of his")
            print("  365K mean context on TP=3. TP=4 extends the cliff to ~5-6 sessions; it")
            print("  does NOT cover the 9-session peak (that needs ~6M tokens, ~TP=7).")
            print("  Action: cap concurrent sessions at 3-4, or accept cold re-prefill above")
            print("  the cliff. A single extra node buys margin, not peak coverage.")
        else:
            print("  HOLDS under load. KV headroom is sufficient at TP=3 for this workload;")
            print("  the fourth node would buy margin, not capability.")


if __name__ == "__main__":
    main()
