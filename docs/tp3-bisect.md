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
