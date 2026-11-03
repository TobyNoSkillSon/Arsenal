+++
hypothesis = "Live faults (int4 runaway, a dropped first line, padded tables) come from the port, not the weights."
change = "Carry the prefill MROPE deltas into decode, keep RoPE in the activation dtype, cap the MLX buffer cache, decode as the maker client does."
measurement = "Quick check, 20 Oct 2026, M5 Max: two pages (three text lines; a 4×3 table) per tier through the worker with real weights, max_tokens 512, MLX limit 12 GiB."
verdict = "kept"
numbers = { fp16_peak_gb = 5.66, int8_peak_gb = 4.26, int4_peak_gb = 3.54, kv_kb_per_token = 115 }
links = ["../README.md", "001-integration.md"]
+++

# Decode positions, activation dtype and generation memory, 20 October 2026

Live run through Arsenal 3.5.2 (all tiers): the text page lost its first line, the table came back with 8 columns and every
price one cell right, and int4 generated for 92 s while the worker grew to tens of gigabytes before failing with no layout
lines.

- Decode positions. The pinned mlx-swift-lm `TokenIterator` keeps no `LMOutput.State` from a `.logits` prepare, so each
  decode step lost the prefill's MROPE deltas and used 1D RoPE at the cache offset: after a 1,369-token layout image the
  generated tokens sat about 1,300 positions too far. The layout's first line came out as `<nl><fcel>038 148 636 252>…`
  and table rows gained `<ecel>` cells. `TeleOCRVL` now keeps the prefill deltas for the decode steps that follow.
  With only the dtype and decoding changes below, FP16 still produced the faulty output; with this change all three tiers
  read both pages.
- Activation dtype. Text MROPE cos/sin stayed float32, which promoted queries, keys, the KV cache and every later
  activation to float32. They now take the activations' dtype, as `modeling_naviocr.py` does (`cos.to(dtype=x.dtype)`).
  KV memory falls from about 229 to 115 KB per token.
- Generation memory. MLX's buffer cache defaults to most of the Mac's memory; decoding frees buffers that grow every step
  and are never reused, so the cache grew without bound (full-size random graph: 3.0 GB cache after 1,024 tokens, over
  8 GB footprint by 2,048). The worker caps the cache at 256 MB: random FP16 graph peak 2.11 → 2.46 GB from 1,024 to
  4,096 tokens, cache flat at about 270 MB.
- Decoding. Penalties now match NaviDC-OCR `TeleOCR_client.py`: no repetition penalty, presence/frequency on
  recognition kinds, `no_repeat_ngram_size` 100 on every call (MinerU's processor, checked against a transcription).

Outputs per tier: three blocks (title, title, text) with the exact line text; the table as 4 rows × 3 columns with the
exact cells; layout calls stop after 72 and 25 tokens. Peak worker footprints above include Load. Raw generations and
memory traces are in `lab/teleocr-fix/` (not tracked).
