# Workload baseline: measured token, turn and per-call volume from the session history

Corpus: all OMP session transcripts under the sessions directory, plus the OMP usage databases. Window of record: files created 2026-08-08 through 2026-09-03 (the day of this analysis). Every figure is labelled [F] (computed from a named field in a named file class), [V] (verbatim from a page or file), or [I] (inference). No prompt text, no client names, no client-identifying paths. Synthetic sessions matching smoke, proof, writegate, routine, no-such-directory are excluded throughout.

## 1. Method

- Corpus: 3,678 .jsonl transcript files on disk at scan time; 3,645 scanned; 34 files sit in excluded synthetic directories. The corpus is live and counts are frozen at scan. [F]
- Classification rule, stated so it can be checked: a file directly inside a cwd directory matching ^YYYY-MM-DDTHH-MM-SS-mmmZ_<uuid>.jsonl is a MAIN session file; a .jsonl inside a session subdirectory is a CHILD file; the child file stem gives the lane (leading double-underscore = harness-builtin role such as the advisor; Tan- prefix = background dispatch; anything else = named task subagent). Entire cwd directories matching the synthetic regex are excluded with all contents. [F]
- Dedupe rule: parent transcripts replay parent-lane history into forked children, and parent files can hold replayed originals. Each assistant call is identified by its transcript event id and counted exactly once, attributed to the earliest-created carrying file (all mains sorted by filename date before all children). 29,021 duplicate occurrences carrying 10.89B tokens were suppressed. The unique-call total then agrees with the usage database within 0.6 percent. [F]
- Token fields used: per assistant message event, message.usage with input (uncached input), cacheRead, cacheWrite, output (includes reported reasoning tokens), reasoningTokens, and the harness-simulated cost block. Per-call latency: message.duration and message.ttft are present on 493,350 and 487,174 of 493,350 unique assistant calls, so TTFT IS recorded in transcripts and is mirrored into the stats database. [F]
- Turn boundaries: a human main prompt is a user-role message with attribution user in a main file (main files contain zero synthetic user-role messages by measurement). A main-lane turn runs from such a prompt to the last assistant call before the next one; mirrored child-lane calls inside a main file are excluded from main-lane turns via the dedupe winner. [F]
- Date attribution: event timestamps in UTC. Replayed events keep their original serving date, so daily totals represent actual service volume on that date. [F]

## 2. Inventory

- Main session files: 502 (created 2026-08-08..2026-09-03). [F]
- Child files, harness-builtin roles (advisor): 968. [F]
- Child files, reviewer-labelled stems: 104. [F]
- Child files, other named task subagents: 2,070, including 543 nested (child of child). [F]
- Files in excluded synthetic directories: 34. [F]
- Heaviest file-creation days: 08-13 (100 files), 08-25 (65), 08-20 (47). Traffic spans 27 days. [F]
- Five resumed sessions replay original July traffic (event dates 2026-07-10..08-07) into files created in August: 1.56B prompt tokens (main lane 0.97B, child 0.59B). Real service volume attributed to July, excluded from the window used below. [F]

## 3. Data-quality findings, all load-bearing

1. Mirror mechanism resolved: a position test across all 25 parent-child id-overlap pairs (17,221 shared ids) shows shared ids sit BEFORE a child file own call ids in every case, and the parent holds them before its own later calls: the shared ids are parent-lane originals replayed downward, children never originate parent content, and mirror-up is 2 ids. The lane split below is therefore exact, not approximate. [F]
2. The usage database at ~/.omp/stats.db (not ~/.omp/agent/stats.db, which does not exist) counts every (file, id) row including mirrors and synthetic directories, so its per-class split is inflated; its TOTALS agree with deduped transcript totals within 0.6 percent (93.06B vs 93.65B all-time prompt tokens). No vanished-file rows found. [F]
3. Local-arm cacheRead is recorded as ZERO in transcripts (spark-glm: raw-in 138.2M, cacheRead 0, cacheWrite 0) while the TTFT arithmetic proves the cache is hitting: median per-call prompt 92.3K tokens with median TTFT 5.50s implies about 17K tok/s effective ingestion against the 1,240 tok/s fresh-prefill ceiling, an implied effective hit rate of 92.8 percent, consistent with the cluster-reported 90.3 percent lifetime rate. The local endpoint serves cached prefixes but OMP does not map cached_tokens into cacheRead for it. Local cost and capacity models must use the cluster hit rate, not transcript cacheRead. [F]
4. Stats-database cost columns are harness-simulated list prices, not billing: codex totals $13,400.88 for the window ($9,581.81 sol, $3,564.46 terra, $161.53 luna, $80.72 gpt-5.5) and premium_requests is zero on every row; the sibling slice additionally reports zai cost columns overstated about 10x. Arm KPIs here use token columns only. [F]

## 4. Token volume per day, deduped, window 08-08..09-03

Prompt tokens split into the three components every provider prices differently: uncached input, cached input (cacheRead), cache write. Output includes reported reasoning tokens.

| day | main calls | main raw-in | main cacheRead | main cacheW | main output | child calls | child raw-in | child cacheRead | child cacheW | child output |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-08 | 432 | 27,203 | 78M | 4M | 290,107 | 279 | 510,710 | 14M | 875,565 | 106,545 |
| 2026-08-09 | 784 | 15,509 | 272M | 3M | 481,342 | 360 | 682,866 | 32M | 0 | 23,038 |
| 2026-08-10 | 3,481 | 746,107 | 1,370M | 19M | 2M | 6,107 | 13M | 709M | 5M | 3M |
| 2026-08-11 | 4,388 | 174,026 | 1,980M | 31M | 3M | 32,806 | 102M | 4,047M | 30M | 16M |
| 2026-08-12 | 6,040 | 2M | 2,864M | 46M | 4M | 62,064 | 109M | 9,070M | 29M | 12M |
| 2026-08-13 | 14,687 | 23M | 1,130M | 24M | 2M | 28,191 | 93M | 3,054M | 14M | 5M |
| 2026-08-14 | 1,963 | 15M | 518M | 5M | 928,337 | 23,703 | 58M | 2,767M | 7M | 3M |
| 2026-08-15 | 3,954 | 3M | 1,507M | 28M | 2M | 11,886 | 35M | 1,470M | 8M | 4M |
| 2026-08-16 | 1,151 | 9,625 | 517M | 14M | 765,802 | 5,186 | 5M | 397M | 2M | 1M |
| 2026-08-17 | 4,038 | 403,864 | 1,927M | 59M | 3M | 15,259 | 69M | 1,533M | 9M | 2M |
| 2026-08-18 | 2,817 | 3M | 1,166M | 25M | 2M | 17,044 | 28M | 1,769M | 5M | 3M |
| 2026-08-19 | 2,870 | 8M | 1,182M | 35M | 2M | 30,942 | 64M | 4,291M | 21M | 7M |
| 2026-08-20 | 4,550 | 41M | 1,911M | 72M | 3M | 32,362 | 140M | 4,420M | 24M | 7M |
| 2026-08-21 | 2,127 | 16M | 780M | 4M | 737,656 | 2,513 | 43M | 93M | 2M | 566,586 |
| 2026-08-22 | 2,997 | 728,602 | 1,281M | 41M | 2M | 31,235 | 32M | 4,178M | 17M | 4M |
| 2026-08-23 | 1,309 | 16,898 | 588M | 33M | 798,546 | 15,651 | 26M | 2,747M | 20M | 4M |
| 2026-08-24 | 1,692 | 6M | 680M | 32M | 1M | 18,001 | 13M | 3,866M | 85M | 9M |
| 2026-08-25 | 2,497 | 12M | 1,110M | 54M | 910,691 | 20,937 | 23M | 5,476M | 108M | 14M |
| 2026-08-26 | 4,172 | 13M | 1,166M | 0 | 380,083 | 22,817 | 122M | 5,727M | 538,910 | 7M |
| 2026-08-27 | 1,851 | 13M | 617M | 1M | 295,146 | 4,135 | 22M | 1,105M | 340,815 | 1M |
| 2026-08-28 | 921 | 13M | 199M | 0 | 285,650 | 3,879 | 11M | 649M | 6M | 2M |
| 2026-08-29 | 752 | 3M | 241M | 9M | 573,185 | 6,130 | 1M | 1,624M | 39M | 8M |
| 2026-08-30 | 842 | 1,683 | 408M | 30M | 578,868 | 3,186 | 6,364 | 665M | 17M | 4M |
| 2026-08-31 | 1,625 | 7M | 551M | 24M | 796,244 | 3,372 | 91M | 606M | 17M | 3M |
| 2026-09-01 | 1,255 | 6M | 442M | 20M | 825,210 | 890 | 90M | 35M | 1M | 644,450 |
| 2026-09-02 | 3,349 | 2M | 1,300M | 47M | 2M | 10,099 | 40M | 2,112M | 38M | 7M |
| 2026-09-03 | 1,511 | 17M | 722M | 24M | 990,614 | 2,803 | 98M | 404M | 9M | 2M |

- main-lane totals over 27 days: 78,055 calls, prompt 27.39B (raw-in 202M + cacheRead 26.51B + cacheWrite 684M), output 37.3M, error-stop calls 10,561. All-in prompt:output 735:1; uncached-input:output 5.4:1; cacheRead share of prompt 96.8 percent. [F]
- child-lane totals over 27 days: 411,837 calls, prompt 64.70B (raw-in 1327M + cacheRead 62.86B + cacheWrite 516M), output 131.2M, error-stop calls 4,405. All-in prompt:output 493:1; uncached-input:output 10.1:1; cacheRead share of prompt 97.2 percent. [F]
- Blended all-prompt:output across lanes: 547:1. The commonly quoted 181:1 workload ratio did not reproduce under any input definition: nearest are uncached-input-only (5.4:1 main, 10.1:1 child) and all-prompt (735:1 main, 493:1 child). Flagging that baseline as unreproducible is itself a finding. [F]

## 5. Turn volume

- Human main prompts: 2,691 over the window, mean 99.7/day. Prior-audit cross-check: the audit window 08-08..08-25 measures 2,073 by this definition vs the audit-reported 1,468; dev-repo sessions alone in that window give 1,394, so the audit number reconciles as a dev-repo-only count. [F]
- Child-lane prompts (turn starts fed to subagents): 268,533 over the window: 264,079 synthetic parent updates, 3,885 agent-attributed, 569 user-attributed, remainder old-schema unattributed. [F]
- Model-serving calls: 78,055 main-lane + 411,837 child-lane = 489,892 assistant API calls, i.e. 2,891 main-lane and 15,253 child-lane per day. [F]
- Daily human prompts by area: dev-repo sessions 1,841, personal and tmp directories 819, the cron daily-brief lane 31. Largest day 08-25 with 455 human prompts, a mass-parallel campaign day. [F]

## 6. Monthly projection, 30-day scaling of the observed window

Representative-day reasoning: the window has 27 traffic days and I scale the observed daily distribution three ways: mean-day, trimmed mean excluding the two largest days (08-13, the mass-fallback stress-test day, and the biggest orchestration day), and the p25..p75 band of daily volume scaled by 30. No weekday correction was applied because weekends also carry near-full load (08-15, 08-16). Spread drivers: large multi-agent orchestration sessions, advisor policy (section 10), and campaign days like 08-25.

| metric | mean-day x30 | trimmed x30 (ex top-2) | p25 x30 | p75 x30 |
|---|---|---|---|---|
| main-lane prompt tokens/mo | 30.4B | 26.9B | 16.0B | 40.1B |
| main-lane output tokens/mo | 41.4M | 36.3M | 19.7M | 60.6M |
| child-lane prompt tokens/mo | 71.9B | 59.6B | 20.2B | 122.2B |
| child-lane output tokens/mo | 145.8M | 120.9M | 64.0M | 202.5M |
| human main prompts/mo | 2,990 | 2,387 | 1,350 | 3,780 |

- Total prompt volume, BOTH lanes combined, mean-day basis: 102.3B/month; p25..p75 basis 35.9B..153.5B. The 30.4B figure in the table row above is main-lane only. [F]
- Calls: 87K main-lane and 458K child-lane assistant calls per month. [F]
- Advisor-off variant: removing the measured 44.5 percent advisor share of child volume projects child prompt at 39.9B/month, and the local era already measured child:main ratios of 0.27x, 1.62x and 0.67x per day against 2.36x for the cloud era. [F]

## 7. Per-turn comparison: cloud frontier vs local, identical metrics

Turn = one human prompt through the end of its main-lane activity, classified by the turn dominant serving model (modal by call count among the file own, non-mirrored assistant calls). Tokens are per-turn sums over main-lane calls only; subagent volume is in sections 4 and 10. Wall = prompt timestamp to last assistant-call completion. TTFT = first assistant call ttft after the prompt. The local sample is small and recent and that is stated plainly: local has served only 188 main-lane calls across 22 main-lane turns since the cutover, and 2 of its 22 turns overlap endpoint-loading windows.

| arm | turns | wall med | wall p95 | calls/turn med | calls/turn p95 | prompt tok/turn med | p95 | output/turn med | TTFT med | TTFT p95 | err turns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLOUD gpt-5.6-sol pre-cutover | 245 | 100.2s | 19,839s | 6.0 | 211.6 | 1.49M | 58.2M | 1.5k | 4.9s | 18.2s | 35 | arm 2 baseline |
| CLOUD gpt-5.6-sol post-cutover | 7 | 106.2s | 10,276s | 11.0 | 159.9 | 3.37M | 87.6M | 4.4k | 4.5s | 14.8s | 1 | fallback residual |
| CLOUD gpt-5.6-terra (task role) | 14 | 39.5s | 5,490s | 5.5 | 232.6 | 0.77M | 40.1M | 1.0k | 4.1s | 9.7s | 2 | mostly child-lane role |
| OPENWEIGHT glm-5.3-flash (opencode-go) | 16 | 69.2s | 12,899s | 5.5 | 382.8 | 0.32M | 127.7M | 1.1k | 7.1s | 14.3s | 5 | arm 3 context |
| LOCAL glm-5.3-flash, all turns | 22 | 122.8s | 2,539s | 2.5 | 18.9 | 0.17M | 2.6M | 1.1k | 19.2s | 52.9s | 1 | 3 days, incl loading windows |
| LOCAL clean turns | 20 | 100.8s | 2,531s | 2.5 | 17.1 | 0.10M | 1.7M | 1.1k | 19.2s | 47.7s | 0 | no error/aborted call, first TTFT < 60s |
| LOCAL degraded turns | 2 | 3,426.4s | 6,044s | 30.0 | 56.1 | 6.27M | 11.2M | 37.6k | 258.0s | 486.5s | 1 | overlap restart windows |

Readings, all [F]:
- On identical axes, local main-lane turns are not slower in wall-clock median (100.8s clean vs sol 100.2s pre-cutover) but run fewer assistant calls per turn (2.5 vs 6.0) and much smaller prompt volume per turn (about 0.10M vs 1.49M median). Local first-call TTFT median is 19.2s vs sol 4.9s: the local engine pays a visible cold-context prefill on the first call of a turn even with the prefix cache active.
- Degraded local turns, 2 of 22, show the loading-window signature: 6.27M prompt tokens per turn, 30 calls, TTFT 258s median, and they dominate naive local p95 figures. The 09-03 restart window produced 38 aborted calls, 09-02 produced 36 timeout-class errors, 09-01 produced 38 other-class errors. Local call-level error rate across the three days: 147 of 1,170 (12.6 percent); main-lane only 3 of 188 (1.6 percent). [F]
- Sol pre-cutover turns carrying at least one error: 35 of 245; sol call-level error rate 823 of 90,900 unique calls (0.91 percent), of which 721 are quota/rate-limit class. [F]
- Same-session pairs, the closest available apples-to-apples because the local sample has no direct sol overlap yet: in one client-monorepo session, sol turns (n=22) median wall 292s, 11.5 calls/turn, 3.70M prompt tokens/turn, TTFT 5.8s vs openweight-glm turns (n=8) median wall 219s, 19.5 calls/turn, 2.74M prompt tokens/turn, TTFT 10.9s; in a personal-project session sol (n=16) 55.4s, 6.0 calls, 0.98M, 2.8s vs openweight-glm (n=7) 34.9s, 3.0 calls, 0.27M, 6.0s. Mixed signal on corrective turns: openweight-glm made more calls per turn than sol in the first pair and fewer in the second. [F]
- Per-call latency from the same records: sol main-lane calls duration med 6.92s p95 18.1s, TTFT med 4.04s p95 6.3s; local calls duration med 29.54s p95 51.4s, TTFT med 5.50s p95 24.9s. [F]
- A prior GLM era (zai glm-5.3, a different checkpoint from arm 3) ran 08-17..08-25 with 22,267 calls; its main-lane turns (n=8) show wall med 2,447s and 51 calls/turn on very large prompts (17.9M/turn med). Included for completeness, not as an arm. [F]

## 8. Arm-2 availability: the subscription quota is the decisive failure

Verified as genuine quota exhaustion, not transport failure or poll-side error. Source: ~/.omp/agent/agent.db usage_history, 512 polls on the openai-codex primary limit between 08-08 and 09-03. [F]
- Status split: 343 ok, 56 warning, 113 exhausted. ALL 113 exhausted rows carry a server-issued resets_at timestamp (weekly window resets such as 08-18 00:01 and 08-20 04:01), which is quota metadata a transport failure cannot produce. [F]
- Timeline: 14 of 27 window days contain at least one exhausted or warning poll; the streaks 08-15..08-19 and 08-22..08-23 are exhausted for the entire day (18/18, 16/16, 24/24, 24/24, 15/15 polls non-ok on consecutive days). [F]
- Same-day transcript corroboration: gpt-5.6-sol error-stop assistant calls with the server-side usage-limit-reached code spike exactly on those days: 08-15: 111, 08-16: 59, 08-17: 205, 08-18: 120, 08-19: 172 quota-class error calls. The error text originates from the Codex API. [F]
- Reset behaviour visible: after exhausted streaks, ok polls resume with used_fraction below 50 percent (3 such resets observed). Poll cadence is roughly hourly. [F]
- Sibling-verified arm-3 context: opencode-go rolling-5h peaked at 99 percent used with 3 non-ok polls; weekly peaked at 66 percent, monthly at 55 percent; zai shows 97 polls since 08-29, all at 0 percent. [F]

## 9. Fallback events (model_change with resolvedModelIsFallback true)

Method note: durable session transcripts do NOT contain retry_fallback_applied, retry_fallback_succeeded or turn_end events; those exist only in streaming debug logs of nested runs. The durable signal is model_change carrying resolvedModelIsFallback true; the from-model is approximated as the previous model_change in the same file. [F]
- Total fallback model_change events: 11,681, of which 9,934 are one 08-13 stress-test session (a gemini self-pair with zero token volume; a policy experiment, not workload). Ex-outlier: 1,747 events. [F]
- Ex-outlier top from-to pairs: kimi-code/k3 to anthropic/claude-opus-5 447; kimi-code/k3 to zai/glm-5.3 178; spark-flash/qwen3.8-flash-next to anthropic/claude-sonnet-5 150; kimi-code/k3 to ollama/gpt-oss:120b 72; anthropic/claude-sonnet-5 to openai-codex/gpt-5.6-terra 67; anthropic/claude-opus-5 to openai-codex/gpt-5.6-sol 56; zai/glm-5.3 to anthropic/claude-opus-5 53; anthropic/claude-opus-5 to xai-oauth/grok-4.6 49. [F]
- LOCAL-ERA fallbacks (Sep 1-3): 230 events (11 on 09-01, 170 on 09-02, 49 on 09-03). Pairs originating from the local model: spark-glm/glm-5.3-flash to anthropic/claude-opus-5 40, to opencode-go/glm-5.3-flash 29, to spark-flash/qwen3.8-flash-next 4 (73 total; most occur inside child lanes, consistent with the 982 child-lane local calls carrying 144 errors, 14.7 percent). Main-file local-era fallback targets: sonnet-5 x9 on 09-02, opus-5 x2 and one gemini fallback on 09-03. On the degraded days roughly one in seven local child-lane calls fell back or errored; the clean days are far quieter. This is the single most important ongoing KPI. [F]

## 10. Subagent decomposition and the shrinkable term

- Child-lane prompt volume over the window by lane kind, from the child file stem: named-task subagents 35.4B (54.8 percent, 152,283 calls), advisor 28.8B (44.5 percent, 258,450 calls), background dispatch 0.5B (0.8 percent, 1,104 calls). [F]
- The advisor lane is a POLICY term: it is absent entirely in the local era (zero advisor calls; local-era child volume is named-task 2.77B plus dispatch 0.06B over three days) and it was the largest per-call prompt consumer in the cloud era because each advisor call receives the whole grown transcript. [F]
- Named-task subagent volume did NOT shrink in the local era: 0.92B/day locally vs 1.31B/day cloud era, same order. Fan-out is structural; the advisor is the optional part. [F]
- The 08-13 mass-fallback stress test produced 9,936 fallback events and 9,937 gemini calls but effectively zero token volume (usage rows empty on error). It inflates event counts, not volume. [F]

## 11. Usage-database cross-checks

- ~/.omp/stats.db mirrors transcript usage one-to-one and agrees with deduped transcript totals within 0.6 percent; its ttft/duration columns agree with transcript medians (sol dur 6.78s vs 6.92s transcripts; spark-glm dur 29.82s vs 29.54s). Its agent_type split (main 81,249 calls, advisor 259,845, subagent 152,477 all-time) double-counts parent-replayed rows as main and includes synthetic directories, so per-class numbers must be recomputed with the exclusions; per-day main-class deltas up to +390 percent trace entirely to those artifacts. [F]
- Its user_messages table (7,543 rows) excludes synthetic child prompts; transcript non-synthetic prompt count is 7,220 (2,691 main human + 569 child user + 3,885 child agent + 75 old-schema none). Agreement within 4 percent. [F]
- ~/.omp/agent/agent.db: usage_history (12,459 quota-poll rows, section 8), model_perf (spark-glm 208 samples, gen 69.6 ms/tok, mean TTFT 10.5s; sol 218 samples, gen 39.4 ms/tok, mean TTFT 4.9s), model_usage (51 rows), command_usage. model_perf corroborates transcript latencies. [F]
- Not found anywhere: per-call queue depth, per-session concurrency sampling, and any premium-request ledger for the codex subscription (premium_requests zero on all 493K rows). The subscription appears to meter only by quota-window status. [F]

## 12. Capacity audit of the parent servable-share arithmetic, adversarial

Confirmed parts: at the local cluster measured 90.3 percent prefix-hit rate, monthly recompute = 101.7B x 9.7 percent = 9.9B tokens; at the gated 1,240 tok/s fresh-prefill ceiling that is 8.0e6 seconds, i.e. 307 percent of a 30-day month, and the generation-only bound (187M output at 79.1 tok/s aggregate) is 91 percent, exactly as the parent quoted. The 27 percent point estimate sits in the neighbourhood of the prefill bound (the same inputs give me 33 percent). [F]
Four corrections change the answer materially:
1. Denominator: sound. Error-stop calls are 15.0K of 490K calls (3.1 percent) but consumed only about 0.04B prompt tokens (0.04 percent of volume) because they fail before prefill completes; ex-outlier fallback events (1,747) each trigger one re-send already inside the metered sum, bounded at roughly 0.3-0.4 percent of volume; compaction events (400) re-prefill a smaller context and would recur locally. Corrections are at most about 1 percent of volume, not a 30 percent term. [F]
2. cacheWrite: do NOT add it as local prefill work. The 1.2B window cacheWrite is an Anthropic billing concept; the local engine has no cacheWrite metric (measured zero on all 1,170 local calls). Local compute per call is uncached-suffix prefill (mean 2.6K, median 0 tokens per main-lane call, because most calls are served entirely from the prefix cache) plus full-context attention during decode. Counting cacheWrite again would overstate the local requirement by about 1.3 percent. [F]
3. Effective rate: the gated ceilings are small-prompt-regime numbers and the workload is not in that regime. Main-lane per-call prompt size: mean 351K, median 324K, p95 777K, max 997K (near the 1M cap); child-lane mean 157K, median 123K, p95 419K. Fresh-prefill rate is therefore irrelevant per call; what matters is recompute = prompt x (1 minus H) and KV capacity: the 2.55M-token KV pool holds about 7.3 sessions of the 351K mean context, observed peak concurrency is 9 model-active, so the top of the range evicts and re-prefills. The measured local hit rate (90.3 percent reported, 92.8 percent implied by TTFT) is consistent with exactly that eviction regime. [F]
4. The shrinkable term is the advisor: 44.5 percent of child prompt volume (28.8B window) and zero calls in the local era, because it is currently disabled. Named-task fan-out is structural and did not shrink. [F]

Servable-share scenarios, inputs [F], model [I]:

| scenario | monthly prompt requirement | recompute | prefill duty | implied servable share |
|---|---|---|---|---|
| advisor ON, H=90.3 pct (measured local hit regime) | 101.7B | 9.9B | 307 pct | ~33 pct |
| advisor OFF (local-era structure), H=90.3 pct | ~70.3B | 6.8B | 211 pct | ~47 pct |
| advisor ON, H=96.8 pct (cloud-era hit structure) | 101.7B | 3.3B | 101 pct | ~99 pct |
| advisor OFF, H=96.8 pct | ~70.3B | 2.3B | 70 pct | fully servable |

[I] The binding constraint is the prefix-hit rate under real concurrency, not raw prefill rate. The evaluation week decides between the 33-47 percent regime (90.3 percent hit, as measured) and the 99-100 percent regime (96.8 percent hit, reachable only if KV capacity keeps long sessions resident). The generation bound (91 percent) is not binding; the 1,033 tok/s aggregate figure is a 3,000-token-prompt-regime measurement and must not be extrapolated to 300K-context turns.

## 13. What the transcripts do not record, and the cheapest way to get it

- Per-call TTFT is recorded, but queue-vs-engine decomposition is not; cheapest source: vLLM metrics on the rank-0 node during the week, correlated by timestamp.
- Concurrency sampling is absent (no queue-depth or admission counters); cheapest source: the same vLLM metrics endpoint on a cron during the week.
- No premium-request ledger exists for the codex subscription; if the plan meters premium requests, the only source is the provider console during a live-subscription week.
- Local cacheRead is recorded as zero by OMP for spark-glm (reporting gap); the cluster prefix-hit counters are authoritative, and fixing the OMP cached_tokens mapping would make future KPIs self-consistent.
- retry_fallback_applied / retry_fallback_succeeded / turn_end exist only in nested-run debug logs; first-class fallback events in the KPI stream need either harness emission into the durable session file or a collector reading the debug logs.

## 14. Findings summary

1. Metered-basis volume is roughly 90x the per-turn-prompt basis: 27.4B main-lane prompt tokens over 27 days across 78K calls, not one prompt per turn. Any arm comparison priced per turn understates cost by about two orders of magnitude. [F]
2. The cacheRead share (96.8-97.2 percent cloud-era structure, zero-reported-but-92.8-percent-implied locally) is the hinge of every economic and capacity conclusion; it must be measured on the cluster during the evaluation week, not assumed. [F]
3. Arm 2 fails on availability, proven from server-side quota metadata plus same-day server error codes: 113 exhausted polls on 14 of 27 days including two multi-day full-exhaustion streaks, and 823 sol error-stop calls (721 quota-class). Dispositive independent of latency and cost. [F]
4. Local capacity at the measured hit regime serves roughly a third to a half of the metered workload with the advisor off, and essentially all of it if the hit rate reaches the cloud-era structure; the deciding experiment is hit-rate-under-real-concurrency, which is exactly what the one-week evaluation measures. [F]
5. The advisor is the only structural-policy lever in the volume (44.5 percent of child prompt tokens); named-task fan-out is structural.
6. Local-era per-turn wall-clock is already comparable to the cloud frontier baseline on identical axes (100.8s vs 100.2s medians) with fewer calls per turn (2.5 vs 6.0), a visible first-call TTFT penalty (19.2s vs 4.9s median), and a degraded-day error signature that must be excluded from KPI baselines.
