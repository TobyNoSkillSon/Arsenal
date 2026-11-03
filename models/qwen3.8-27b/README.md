# Qwen3.8 27B

The only Arsenal chat LLM. Vendor bf16 checkpoint, Apache-2.0, pinned revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` from `Qwen/Qwen3.8-27B`.

Native MLX Swift text inference uses the existing family dependency pins. The checkpoint has 64 hybrid
linear/full-attention layers, a vision tower and one MTP layer. Text v1 drops vision/MTP parameters via the
upstream sanitizer; images/video and tool/schema decoding are honestly unsupported.

The vendor context is 262,144 tokens. The runnable default is 8192 total prompt+output tokens to bound KV
memory; load config can raise this to 32768. Requests that do not fit are rejected, not truncated.
Fast/Exact is unavailable. Affine group-64 int8/int4 are derived locally from the vendor bf16, never from
community quantized weights. A functional smoke is not a complete quality gate or controlled benchmark.

Local development weights are clone-copied from the existing vendor checkpoint, never downloaded or modified
in its owner folder. See `log/` for source compatibility, clone/hash verification and real smoke evidence.

## Ran for real
Vendor bf16 and locally derived int4 passed the real-supervisor stream/cancel/unload smoke on this Mac.
Download size: **55.59 GB** (31 files). Int4 output: **15.16 GB**, derived in 25.1 seconds.
Both returned `17 plus 25 equals 42.` No controlled benchmark or full quality gate is claimed.
Int8 is a prepared derive recipe, not a tested precision. See `log/003-real-supervisor-smoke.md` for the
cold/warm caveat, load/first-token/token-rate/peak-RAM figures and exact evidence.
- context: Arsenal runs up to 32768 tokens (default 8192 per request config); the vendor model supports 262144. Longer context is future work (hybrid linear attention keeps KV small).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Vision-language chat model; the Arsenal chat op takes text only. [model card, 2026-08, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Maker's settings. Thinking: temperature 1.0, top_p 0.95, top_k 20, presence penalty 0.0. Non-thinking: temperature 0.7, top_p 0.80, top_k 20, presence penalty 1.5. [model card, 2026-08, first-hand (maker)]
- Arsenal call: `arsenal_run(model:"qwen3.8-27b", operation:"chat", input:{"messages":[{"role":"user","content":"Hello"}],"thinking":true})` [ops/chat.input.json, first-hand]

**Good and bad**
- Good at: The maker's figures lead on SWE-bench Pro (61.7) and LiveCodeBench v6 (90.3). [model card, first-hand (maker)]
- Bad at: One user saw about 1 in 10 tool calls fail under vLLM. No replies captured. [discussion 219, undated, first-hand (user)]
- Against the obvious alternatives: The card trails Opus 4.6 Max on GPQA Diamond (89.2 vs 91.3) and HLE (30.8 vs 40.0). One 4-bit vLLM run had IFBench at 69.77% against 59.30% for Qwen3.6-27B. [model card; discussion 225, undated, first-hand (self-run)]

**Complaints, bugs and fixes**
- Complaint or known bug: On SGLang, one user got 20,000 to 55,000+ token thinking runs on large coding prompts; a 4-bit Mac user reported a similar loop. [discussion 178, opened Aug 27 (year not shown), first-hand (users)]
- Fix or workaround: Reported by one user, not confirmed: an explicit thinking_budget (2,048 to 16,384) with temperature 0.6 and top_p 0.95 stopped the loops. No maintainer or second user confirmed it. [discussion 178, first-hand (user)]

**Evidence**
- Strength: thin. Maker figures and single-user reports; no Qwen-team reply in the discussions read.
- Sources:
  - Model card, https://huggingface.co/Qwen/Qwen3.8-27B, 2026-08, first-hand (maker).
  - Discussion 178, https://huggingface.co/Qwen/Qwen3.8-27B/discussions/178, Aug 27 (year not shown), first-hand.
  - Discussion 219, https://huggingface.co/Qwen/Qwen3.8-27B/discussions/219, undated, first-hand.
  - Discussion 225, https://huggingface.co/Qwen/Qwen3.8-27B/discussions/225, undated, first-hand.
