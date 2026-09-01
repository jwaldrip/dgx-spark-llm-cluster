# Choosing a GLM-5.3-Flash checkpoint for two 128 GB nodes

GLM-5.3-Flash is 320B total, 18B active, 45 layers plus one MTP layer, 288 routed experts
at top-8, NoPE MLA, 34 KDA linear-attention layers plus 11 sparse-attention layers, native
1M positions, with a vision tower. Same architecture in every checkpoint below.

Two nodes give 242 GiB visible. Every relevant dimension divides by 2, 4 and 8, so tensor
parallel is arithmetically legal at any of those degrees. Fit is what eliminates candidates.

## The official release does not fit

The live `zai-org/GLM-5.3-Flash` revision is no longer BF16. It is FP8 block-128:

```text
62 safetensors shards   328,337,455,672 B   =   305.78 GiB
per rank at TP2                                 152.9  GiB   >  121 GiB visible
```

It does not fit two Sparks, before counting KV, activations or runtime. It needs three.

## Both NVFP4 quantizations fit, and only one is usable

| checkpoint | quant | resident | per rank | verdict |
|---|---|---:|---:|---|
| `RedHatAI/GLM-5.3-Flash-NVFP4` | compressed-tensors, W4A4 | 184.29 GiB | 92.1 GiB | **use this** |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | ModelOpt, W4A16 | 181.29 GiB | 90.6 GiB | corrupts output on vLLM |
| `local-inference-lab/GLM-5.3-Flash-NVFP4` | ModelOpt, MIXED_PRECISION | 180.91 GiB | 90.5 GiB | needs the author's vLLM fork |

Sizes computed from `https://huggingface.co/api/models/<id>/tree/main?recursive=true`, not
from the model card. Do not use the `usedStorage` field as a download size; it counts
non-main refs and pre-GC revision blobs and overstates every repo here.

## Why the W4A16 checkpoint is unusable, and why it is not the checkpoint's fault

[vllm-project/vllm#54150](https://github.com/vllm-project/vllm/issues/54150). vLLM's
`modelopt.py` takes `layer.w13_weight_scale_2[:, 0]` as a single global scale for the fused
w13 GEMM. ModelOpt checkpoints carry **separate** global scales for w1 (gate) and w3 (up).
Picking one mis-scales the other shard's dequantization, measured at up to 10x on this
model.

Corruption counts from the issue, identical Korean prompt, six runs each, temperature 0 and
0.7, counting U+FFFD replacement characters:

```text
LibertAIDAI ModelOpt NVFP4        86
LibertAIDAI rev cf5434c00b        62
dealignai uncensored              94
RedHatAI compressed-tensors        0
```

Independently reproduced on two DGX Sparks: ModelOpt 4/9/8 against compressed-tensors
0/0/0. The reporter ruled out context length, sampling, MTP, PDL, abliteration and
ignore-list completeness. A fix was posted 2026-08-30 and is not merged.

This is a **loader** bug, not a weights bug, so it will presumably disappear upstream. Until
that patch is in your image, W4A16 GLM NVFP4 is off the table on vLLM. That matters
disproportionately for agent work: a corrupted token landing inside a tool-call block
desyncs the parser, and the observed failure mode is a repetition lock rather than an error.

The tradeoff you accept in exchange is real. RedHatAI quantizes activations to 4-bit, where
the ModelOpt builds are weight-only at 16-bit activations, so expect somewhat weaker hard
reasoning. Correct output beats a few points of reasoning when the consumer is a tool loop.

## The same bug class does not necessarily fire on other ModelOpt checkpoints

Worth checking rather than assuming, because the precondition is specific. The bug needs the
checkpoint to carry *distinct* gate and up global scales. If a quantizer emitted one shared
per-expert global scale for both shards, index 0 is correct for both and the mis-scale is
mathematically zero.

Checking a checkpoint directly, without loading it:

```python
# read weight_scale_2 tensors from the safetensors header and compare gate/up pairs
# per expert; distinct values mean exposed, identical values mean safe
```

On the Qwen3.8-Flash-Next NVFP4 checkpoint in this cluster, all 24,576 gate/up pairs were
identical, worst ratio 1.0000, and a Hangul plus emoji probe returned zero U+FFFD across
six runs. Same quantizer family, not exposed.

## Verify the checkpoint landed intact

```bash
snap=~/.cache/huggingface/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4/snapshots/<rev>
du -sbL "$snap"                                                  # expect 197,881,157,135
find ~/.cache/huggingface/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4 -name '*.incomplete' | wc -l
```

Both nodes should report byte-identical totals and zero incomplete files.

## The missing vision template is a non-issue

Published recipes warn that `chat_template_mm.jinja` must be present or image requests
return 500. No GLM-5.3-Flash repo ships that file: not `zai-org`, not `RedHatAI`, not
`LibertAIDAI`. Each ships only `chat_template.jinja` plus `processor_config.json`.

That template handles multimodal content parts by itself. Verified against the live TP2
endpoint with a generated solid-blue PNG, base64 inline:

```text
prompt   "What single color fills this entire image? Answer with one word."
          + data:image/png;base64,... (32x32, RGB 0,64,255)
response 'Blue'    finish_reason stop    prompt_tokens 43
```

Build the fixture yourself rather than trusting a base64 blob you cannot decode. An earlier
run of this check used a hand-written base64 string labelled "solid red" that was not red,
and the model's answer of "White" looked like a vision failure when the fixture was the
thing that was wrong.
