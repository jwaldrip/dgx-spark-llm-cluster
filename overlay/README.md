# Local vLLM overlays

Two patches against the vendor image
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (image id `35c6f70ffcba`,
vLLM `0.1.dev20051+g487ecf187`). Neither is upstream. Both are applied the same
way the published three-node recipe applies its own four overlays: extract the
file from the image, patch it, and bind-mount the result read-only over its path
inside the container.

The image is the source of truth for the base files, so these are stored as
diffs rather than whole vendored copies. Materialize one like this:

```bash
cid=$(docker create ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 true)
docker cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py" /tmp/
docker rm -f "$cid"
patch /tmp/modelopt.py < overlay/modelopt-vllm54150.diff
python3 -m py_compile /tmp/modelopt.py   # never mount a file you have not compiled
```

Then pass it through the launcher's `EXTRA_MOUNTS`:

```bash
EXTRA_MOUNTS='-v /home/jwaldrip/glm-tp3/fix54150/modelopt.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro'
```

## `routed_experts-w2-scale.diff`

Fixes a real crash at tensor-parallel 3, and the subtler wrong fix for it.

`_load_per_tensor_weight_scale` assumes `loaded_weight` is a scalar and calls
`reshape(())`. With `RedHatAI/GLM-5.3-Flash-NVFP4` at TP=3 it instead receives,
for `shard_id="w2"`, a per-expert-shaped buffer of `(288,)` where 288 is
`n_routed_experts`, carrying only the current expert's value at index
`expert_id` and zeros everywhere else. Unpatched:

    RuntimeError: shape '[]' is invalid for input of size 288

Measured directly by instrumenting the call site: `w2_input_global_scale`
arrives as `[46.5, 0.0, 0.0, 0.0, 0.0, 0.0]` for `expert_id=0`, and `46.5` is
exactly expert 0's `down_proj.input_global_scale` in the checkpoint. The
`w13_*` scales by contrast arrive as a genuine `(1,)` scalar, so only `w2`
needs this.

The patch selects `loaded_weight.flatten()[expert_id]`.

**The obvious fix is wrong and fails silently.** Copying the buffer wholesale
over the parameter, which shape-matches and looks right, writes zeros into the
other 287 experts' scales on every call. That loads cleanly, reports `/health`
200, and then emits a pure repetition lock. It cost a full diagnostic cycle.
See docs/tp3-bisect.md.

## `modelopt-vllm54150.diff`

Fix for [vllm-project/vllm#54150](https://github.com/vllm-project/vllm/issues/54150),
which is open with no PR referencing it. Verbatim from the issue thread, comment
by @shinyquagsire23, attributed to "Fable". Every line it removes matches this
image's tree exactly, which is unsurprising: the issue's own environment block
names commit `487ecf187`, the same commit this image reports.

`ModelOptNvFp4FusedMoE.process_weights_after_loading` needs one global scale per
expert for the fused w13 GEMM, but ModelOpt checkpoints carry separate global
scales for w1 (gate) and w3 (up). The code picks `[:, 0]`, the gate scale, and
applies it to the up shard too, mis-scaling up-projection output by
`ws2_gate/ws2_up` per expert. Reported ratio mean 1.15, p90 1.29, max 10.0, on
69.1% of expert pairs.

The patch takes the per-expert max as the shared global scale and folds each
shard's ratio into its E4M3 block scales instead. Ratios are then at most 1, so
nothing overflows and no clamping is needed.

This matters for agent serving specifically: the corruption shows up as U+FFFD
replacement characters, and a corrupted token inside a tool-call block desyncs
the parser into a repetition lock. Reference measurement from the issue is 86
U+FFFD across 6 runs on the affected checkpoint against 0 across 6 on an
unaffected one.

A second candidate fix exists in the same thread from @mechramc, with stronger
validation (an exact numeric restoration plus a 46-pass zero-corruption battery
on 2x GB10 SM121, the same hardware family as this cluster). It was not chosen
first because it was posted as a code snippet rather than a diff, it clamps to
the E4M3 maximum of 448 rather than avoiding overflow structurally, and it
assumes `w13_weight_scale`'s dim-1 is already per-TP-rank. The author reports
their own earlier version broke on exactly that assumption, which is the one
this cluster would be exercising at TP=3. Try it second if the chosen fix does
not hold.

## Which checkpoint needs which

    checkpoint                        scheme   needs w2-scale   needs 54150
    RedHatAI/GLM-5.3-Flash-NVFP4      W4A4     yes              no
    LibertAIDAI/GLM-5.3-Flash-NVFP4   W4A16    unknown          yes

RedHatAI is structurally immune to #54150 because its gate and up scales are
shared by construction rather than merely equal in practice. LibertAIDAI is the
checkpoint the published recipe actually uses and the one #54150 was reported
against.
