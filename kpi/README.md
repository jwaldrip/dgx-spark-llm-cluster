# One-week trial: KPIs for three ways of running the stack

The decision is not "is local good". It is which of three arms to run, measured on the
same axes:

    A  LOCAL         GLM-5.3-Flash NVFP4, three DGX Sparks, tensor-parallel 3
    B  CLOUD SUB     openai-codex gpt-5.6-sol + gpt-5.6-terra, subscription-metered
    C  CLOUD OPEN    the SAME model, GLM-5.3-Flash, via opencode-go or zai

Arm C is the one that makes this interesting, because it holds the model constant and
changes only where it runs. If C is close enough on latency and cheap enough, then A has
to justify itself on privacy, quota independence and fixed cost rather than on capability.

State it plainly up front: **cost probably does not decide this.** At roughly $0.10 per 1M
tokens and the measured main-lane volume, arm C lands near $30 a month with no cached-input
discount and near $3 with a 90% one. Neither number pays for three Sparks. So the trial
needs to test the axes that might actually differ, and treat cost as a tiebreak rather than
the headline.

## Tier 1: blocking. Any failure here disqualifies an arm regardless of everything else

| KPI | Why it is blocking | Source | Target |
|---|---|---|---|
| Fallback rate | Directly measures "the primary was not good enough or not there". A local endpoint that quietly serves half its turns from Anthropic is not a local endpoint. | `retry_fallback_applied` in session JSONL | under 1% of turns |
| Context failures | A turn that exceeds the ceiling hard-fails, it does not degrade. p95 is 782,561 and max 908,183. | provider errors, `turn_end` with error | zero |
| Output corruption | Arm A carries a locally patched upstream bug (vllm#54150). Its absence must be observed, not assumed. | U+FFFD count and repetition-lock scan over real outputs | zero |
| Node wedge needing power cycle | Arm A only. Unified memory has no relief valve; recovery needs physical access. Five occurred during bring-up. | `collect.sh` node reachability plus manual log | zero in steady state |
| Data retention / training use | Arm C only, and it is a contractual question not a performance one. Client work under confidentiality cannot go to a provider that trains on inputs. | provider terms | zero retention or equivalent |

## Tier 2: experience. What a turn actually feels like

Measure on REAL turns, not synthetic prompts. The gap is large: small prompts gave TTFT of
232 to 319 ms, while one real turn carrying the full tool surface at about 58K tokens gave
28.7 s. The second number is the one to report, because it is the one being lived.

| KPI | Source | Arm A measured so far |
|---|---|---|
| TTFT p50 / p95 on real turns | session JSONL | 28.7 s on a 58K-token turn |
| Turn wall-clock p50 / p95 | session JSONL | 40.8 s on the same turn |
| Cold versus warm turn split | prefix cache hit rate | 90.3% lifetime hits |
| Cold session start cost | manual, per context size | 79 s at 123K, 550-680 s at 790K |

The cold-versus-warm split matters more than either number alone. A continuing turn across
four sessions took 17.3 s against 272 s cold, so the experience is dominated by whether
sessions stay warm, not by raw throughput.

## Tier 3: capacity. Whether it holds up under his actual concurrency

Workload: 4 to 10 simultaneous sessions, observed peak 9 model-active.

| KPI | Source | Meaning of a bad reading |
|---|---|---|
| Peak concurrent model-active sessions | session JSONL overlap | above 12 means the admission cap is throttling |
| Preemptions | `vllm:num_preemptions_total` | any sustained rise means the cap is over-subscribed; 538 preemptions once collapsed throughput here |
| Requests waiting, reason=capacity | `vllm:num_requests_waiting_by_reason` | sustained nonzero means queueing, which is correct behaviour but a capacity signal |
| KV cache usage p95 | `vllm:kv_cache_usage_perc` | approaching 100% predicts the preemption cliff |

## Tier 4: cost per completed task, not cost per token

This is the framing that matters and the one Artificial Analysis uses. A model with a higher
unit price that solves a task in fewer turns is cheaper. His workload runs about 181 input
tokens per output token, so input pricing and caching dominate everything.

| KPI | Source |
|---|---|
| Assistant turns per human prompt | session JSONL, the best available proxy for corrective turns |
| Input and output tokens per task, separately | session JSONL usage records |
| Billed spend per task | provider billing for B and C |
| Amortized hardware plus power per task | arm A only |

Report cost per task for all three arms. Report cost per token as a supporting number only.

## Tier 5: risk and independence

| KPI | Applies to |
|---|---|
| Quota exhaustion events | arm B, the reason the advisor was disabled under a constrained subscription |
| Provider availability incidents | arms B and C |
| Fabric or collective errors | arm A, `NCCL WARN` count |
| Time spent operating the thing | arm A, and it is not zero: this stack carries two local vLLM patches and a padded config |

That last row deserves honesty. Arm A currently depends on a patched `modelopt.py` for an
unmerged upstream issue, an SM121 indexer patch, four vendor overlays, and a hand-padded
config. That is real maintenance surface, and a week of trouble-free running is evidence
about it rather than proof.

## How the data gets collected

`kpi/collect.sh` snapshots the serving state into `~/.omp/kpi/local-serving.jsonl`. Run it
every few minutes for the duration:

```bash
*/5 * * * * /Users/jwaldrip/dev/src/github.com/jwaldrip/dgx-spark-llm-cluster/kpi/collect.sh >/dev/null 2>&1
```

Two properties it was built to have, both learned the hard way:

- **Every vLLM counter is cumulative since engine start and resets on restart.** The
  collector stores raw totals plus the container start time so the report can compute
  deltas only within a continuous run. Diffing across a restart yields negative rates,
  which is how these dashboards end up lying.
- **A failed scrape is recorded as `scrape_ok: false`, never skipped.** An absent row and a
  zero row mean opposite things, and a gap that reads as a quiet period is how an outage
  gets reported as healthy. This was verified by running the collector against a loading
  endpoint and confirming it wrote a failure row rather than nothing.

Session-side KPIs come from `~/.omp/agent/sessions/**/*.jsonl` directly. Exclude synthetic
sessions matching `smoke|proof|writegate|routine|no-such-directory`, and note that subagent
files vastly outnumber main-lane files, so any token total that ignores them is wrong.

## What would change the recommendation

Stated in advance, so the trial can actually settle it rather than confirm a prior:

- **Fallback rate above a few percent** means arm A is not really serving the workload, and
  the honest move is arm C with the same model.
- **Any steady-state node wedge** makes arm A too expensive to operate regardless of its
  numbers, because recovery needs someone physically present.
- **Arm C shown to support cached input and 782K context** removes most of arm A's
  advantage, leaving privacy and fixed cost.
- **Arm C shown NOT to support 782K context** disqualifies it for the main lane the same
  way it disqualified Qwen, and arm A wins by default.
- **Provider terms permitting training on inputs** disqualifies arm C for client work
  whatever it costs.
- **Turns per task materially worse on GLM than on gpt-5.6-sol** would favour arm B despite
  its quota constraints, and this is the one axis where no trustworthy measurement exists
  yet: the only local head-to-head ran against a crippled GLM and was discarded.
