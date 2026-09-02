# Bisecting the TP=3 repetition lock on RedHatAI's checkpoint

docs/glm-tp3-reproduction.md picked RedHatAI's compressed-tensors checkpoint over the
published LibertAIDAI/marlin combination to sidestep vllm-project/vllm#54150. This document
records what that substitution actually did at TP=3: the topology loads and serves, the
generation is unusable, and most of the obvious explanations are now eliminated by direct
measurement rather than by assumption.

## What is proven: the topology, not the numerics

The RedHatAI checkpoint loads and serves at TP=3. 62.14 GiB per rank against FlyCockpit's
published 63.64 GiB, 2,621,440 KV tokens, `/health` 200. That is close enough to their figure
that padding, sharding, and placement are doing what they are supposed to do. What is not
proven is that the served output is correct, and it is not: every generation degenerates into
a pure repetition lock, `locklocklocklock...`, 1 distinct token out of 1, `content: null`,
`finish_reason: length`. The gate script in this repo reads generated text rather than
`/health`, per gotchas.md, and it fails this configuration correctly.

## The real loader defect, and why the obvious fix is wrong

At TP=3, `_load_per_tensor_weight_scale` in
`vllm/model_executor/layers/fused_moe/routed_experts.py` receives, for `shard_id='w2'`, a
per-expert-shaped buffer of shape `(288,)` where 288 is `n_routed_experts`. Only the current
expert's value is populated, at index `expert_id`; every other position is zero. `w13_*`
scales by contrast arrive as a genuine `(1,)` scalar on the same call path. Unpatched, the
`(288,)` buffer hits a `reshape(())` meant for a scalar and raises:

```text
RuntimeError: shape '[]' is invalid for input of size 288
```

The instrumented value confirms the shape is not noise: for `expert_id=0`,
`w2_input_global_scale` arrived as `[46.5, 0.0, 0.0, ...]`, and 46.5 is exactly expert 0's
`down_proj.input_global_scale` in the checkpoint. The loader is being handed the whole
per-expert scale tensor and only told which slot is real.

The fix that looks obvious, copying the buffer wholesale onto the parameter, is wrong. It
zeros the other 287 experts' `w2` input scales on every call, since each expert's load only
carries its own nonzero slot. That version loads and serves without error, which makes it easy
to mistake for correct, and it is exactly what produces `locklocklocklock`. The correct fix
selects the scalar out of the buffer: `loaded_weight.flatten()[expert_id]`. This is confirmed
to fix the crash. It does not fix the repetition lock, which means the lock has a separate
cause from this loader defect.

## Elimination table

Each row changed one variable, took a full load, and reset to the RedHatAI TP=3 baseline
before the next row.

    hypothesis                          change                              observed                       verdict
    MTP / speculative decoding          --speculative-config removed        loads, still locks             not the cause
    KV cache dtype                      fp8_e4m3 -> BF16 (auto)              loads, still locks             not the cause
    num_key_value_heads pad             64 -> 66, locally invented           loads, still locks             not the cause
    KDA linear_attn_config.num_heads    64 -> 66, locally invented           loads, still locks             not the cause
    KDA linear_num_heads                64 -> 66, locally invented           loads, still locks             not the cause
    MoE intermediate pad                baked 2304 instead of letting the   per-rank memory moved           overlay's own pad is
                                         overlay's own 2112 override fire    67.19 -> 62.14 GiB, still       correct; 2304 bypassed
                                                                             locks                           it and cost memory too
    our TP=2-era indexer patch          sparse_attn_indexer_kpool.py         loads, still locks              not the cause
                                         patch left in from TP=2
    fabric                              3-rank NCCL all_reduce, 2^20        exactly 6.0 (1+2+3) in fp32      fabric is correct
                                         elements, real mesh-plugin env      and bf16, zero mismatched
                                         (NCCL_NET=Mesh,                    elements on all three ranks
                                         NCCL_NET_PLUGIN=mesh,
                                         NCCL_IB_DISABLE=1,
                                         LD_LIBRARY_PATH=/opt/nccl-mesh,
                                         NCCL_SOCKET_IFNAME=enP7s7)
    w2 per-tensor scale handling        loaded_weight.flatten()[expert_id]  fixes the crash, load and       real defect, but not
                                         instead of wholesale copy           serve proceed, still locks      the lock's cause

The MoE pad row is worth reading twice: 67.19 GiB down to 62.14 GiB on the same checkpoint is
how we know the overlay's internal 2112 override actually fired once 2304 was removed and
2048 was let through, not merely that a different number was written to config.

## Two conclusions that cost real cycles

The overlay `model.py` handles KDA head sharding at TP=3 internally. Padding
`linear_attn_config.num_heads` or `linear_num_heads` in config is unnecessary and was locally
invented; it has no effect either way, so treat it as a false lead rather than a partial fix.

The overlay `model.py`'s `moe_intermediate_size` override to 2112 fires only when it sees
stock 2048 in config. Baking any other value, including a value that also satisfies the TP=3
divisibility and 16-element quant-group alignment checks such as 2304, silently bypasses the
overlay's own pad handling instead of cooperating with it. Leave the stock value in config and
let the overlay do the padding.

## Unproven leading hypothesis

TP=3's zero-padding, `num_attention_heads` 64 to 66 and MoE intermediate 2048 to 2112, may be
incompatible with a W4A4 checkpoint specifically. The mechanism would be that zero-padded
regions perturb dynamic activation quantization, which exists only in W4A4 and not in the
recipe's original W4A16 checkpoint. This is not proven; nothing here isolates padding as the
cause independent of the checkpoint's quantization scheme.

The symptom itself is documented elsewhere. `locklocklocklock` is listed on the LibertAIDAI
model card as a known input/activation-scale defect, fixed at the checkpoint level by shipping
`model-input-scales.safetensors` (2026-08-30). That fix lives in the checkpoint FlyCockpit's
recipe actually uses and that #54150 blocks here, not in RedHatAI's.

## Method notes for the next person

Two claims made during this bisect were premature and both were wrong. Record the tell, not
just the correction.

**A run staying alive is not the same as a run passing.** An overlay was declared isolated
because the process was still running after 100 seconds. It had not crashed, but it also had
not reached the point in the load or serve path where the failure actually occurs. Alive is
not passed; wait for the specific event the hypothesis predicts, not for the absence of a
crash.

**A syntax check is not a behavior check.** `bash -n` plus one manual expansion test was
treated as proof that `SPEC=` worked correctly in both the set and unset cases, when only the
unset case had actually been exercised. The empty-string case appended a literal `}` to the
output, because `${VAR-default}` only supplies the default when `VAR` is unset, not when it is
set to empty, and the default text itself contained JSON with its own braces, so the shell's
own expansion boundary landed inside the fallback value. Test every branch a conditional
expansion actually has, not the one that is easiest to trigger.

## Resolution: it works on the recipe's own checkpoint

The repetition lock was the checkpoint, not the topology. Switching to
`LibertAIDAI/GLM-5.3-Flash-NVFP4` (rev `caca4e6a`), the checkpoint the published recipe
actually uses and the one its four overlays were written against, plus the vllm#54150
overlay, produces a working endpoint.

    metric                     value
    per-rank weights           62.14 GiB   (published recipe: 63.64)
    KV cache, 1m variant       2,473,737 tokens
    max_model_len              1,048,576
    needle retrieval           found at 790,455 prompt tokens
    cold prefill, 790K         678 s
    warm repeat, same prompt   9 s
    prefix cache hits          790,272 of 1,580,910 queries

The gate passes: 0 U+FFFD across 6 runs, a well-formed structured tool call, and a clean
multi-turn tool-result continuation, with no repetition lock anywhere.

The 75x gap between the cold and warm 790K prefill is the number that matters for this
workload, which runs 181 prompt tokens per generated token. It is also why the padding
question was worth chasing rather than settling for a smaller context.

### One more defect found on the way to 1m

The `fast` variant (262,144) passed the gate, then the `1m` variant returned HTTP 500 after
678 s of prefill:

    launch_persistent_topk: persistent_topk would oversubscribe and the FilteredTopK
    fallback requires >=128KB smem per block (have 101376).
    total_ctas=124 > num_sms*occupancy=48

That is the sparse attention indexer's top-k kernel exceeding GB10's shared memory as
`total_ctas` grows with sequence length. It is exactly what `sparse_attn_indexer_kpool.py`
guards, and it was absent because the bisect had swapped that patch out for the stock file
and it was never swapped back. Restoring it fixed the 1m variant.

The lesson is narrow and worth stating: when a bisect neutralizes a component, put it back
before testing a DIFFERENT axis. The indexer patch was irrelevant to the repetition lock and
load-bearing for long context, and testing context length while it was still disabled
produced a failure that had nothing to do with context length.

### Checkpoint verdict

    checkpoint                        TP=3 loads   TP=3 correct   needs
    LibertAIDAI (ModelOpt, W4A16)     yes          yes            modelopt-vllm54150.diff
    RedHatAI (compressed-tensors,     yes          NO             routed_experts-w2-scale.diff
      W4A4)                                                       to load at all

RedHatAI remains the better checkpoint in the abstract, being structurally immune to #54150,
and its loader defect is genuinely fixed here. But it emits a repetition lock at TP=3 that
survived eliminating every other variable, and the leading explanation is that padding
perturbs the dynamic activation quantization that only a W4A4 scheme has. That is a
hypothesis, not a proven cause, and it is the open thread if RedHatAI is ever wanted at TP=3.

## Closing the gap to the published recipe

The first working TP=3 endpoint was a deliberately conservative variant: BF16 KV, no
speculative decoding, eager mode, gpu-memory-utilization 0.80. That was the right call
during a bisect and the wrong place to leave it. Each axis was then re-enabled one at a
time, with the fidelity gate and a 790,455-token needle after every change.

    axis                        gen tok/s   TTFT ms   prefill tok/s   KV tokens   gate
    eager, BF16 KV, no spec         16.75      2753            24.3   1,919,006   pass
    compiled                        20.45       247           275.1   1,891,946   pass
    compiled + fp8 KV               20.43       236           276.0   3,566,043   pass
    compiled + fp8 KV + MTP         29.62       318           207.7   2,941,956   pass

Shipped defaults are the last row. Per-rank weights land at 63.64 GiB, which is the
published recipe's figure exactly, and 29.62 tok/s is within 5% of its reported 31.3 tok/s
warm on the same 1m variant.

Compiled is the change that matters most for this workload. Generation improved 22%, but
prefill improved 11x and TTFT 11x, on a lane that runs 181 prompt tokens per generated
token. Eager mode was costing far more than the generation number suggested.

fp8 KV nearly doubles the pool for no measured throughput cost. It is gated rather than
assumed because GLM ships no KV scaling factors, so vLLM substitutes unit scale silently;
the 790K needle still returns the exact string under it.

MTP is worth its KV cost: 1.45x generation for 17% of the pool.

gpu-memory-utilization stays at 0.80 rather than the recipe's 0.85. At 0.85 rank 0 settled
at 1.4 GiB available of 121 GiB, and rank 0 also runs the API server. On hardware whose
failure mode is a wedge needing a physical power cycle, that is not a trade worth 6 GiB of
KV. At 0.80 with MTP the ranks sit at 6, 12 and 12 GiB.

### The fabric, which was the biggest surprise

NVIDIA publishes a three-Spark NCCL page prescribing `NCCL_IB_SUBNET_AWARE_ROUTING=1` with
`NCCL_NET_PLUGIN=none`. That is stock NCCL's answer to the same problem the published
recipe's third-party mesh plugin was built for: on a pairwise triangle no single HCA sees
all three peers. Measured here with a 3-rank bf16 all-reduce, both correct to zero
mismatched elements:

    size     mesh plugin    NVIDIA stock IB
     8 MB     2.03 GB/s       8.42 GB/s
    64 MB    12.50 GB/s      22.53 GB/s
   256 MB    12.12 GB/s      22.35 GB/s

Stock IB wins 1.8x on large messages and 4.1x on small ones, and small messages are what
per-token decode all-reduces are. `FABRIC=nvidia` is now the default; `FABRIC=mesh` is kept
as a fallback. The supported path was better than the community workaround here, which is
not the usual direction.

### Two mistakes in this round

MTP appeared to give no benefit at all on its first measurement, 20.45 against 20.43. It
had never engaged: `SPEC` defaults to empty and needs `SPEC=mtp`, and omitting the variable
is not the same as setting it. `speculative_config=None` in the engine's own startup line
is what caught it. There is now a `SPEC=mtp` shorthand and a one-line verification in the
comment, because passing that JSON through ssh plus a shell is how a flag silently fails to
arrive.

Then MTP would not load at all: `MergedColumnParallelLinear` asserts each output size
divides `tp_size`, and the MTP layer builds its MLP from `moe_intermediate_size` 2048, which
3 does not divide. The MTP layer is constructed in `mtp.py`, reads `config.json` directly,
and never sees the overlay's in-code override to 2112. The fix is to bake 2112 into
`config.json`, which is the value the overlay would have chosen anyway. This is the same
class of trap recorded earlier in this document: an override that reaches most of the model
but not all of it.
