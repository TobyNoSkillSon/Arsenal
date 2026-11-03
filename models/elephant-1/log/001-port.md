+++
hypothesis = "A Swift MLX port of the Qwen3.5 text decoder with the maker's block-bidirectional full attention, slot_lm readout and qev request path reproduces elephant-1's decisions at FP16."
change = "New native family elephant (runner in runners/swift, linked as Workers/Elephant): Qwen3.5 decoder adapted from mlx-swift-lm, qev prompt/answers/schema ported, d1-3B's ordered wire, FP16 cast at Get through IntegrationKit (cast:elephant-1-v1)."
measurement = "Weight-free prompt and answer goldens against qev; token, answer and per-layer parity against the maker's DecisionEngine on MPS, captured once in BF16 and once in FP16, bounds frozen first."
verdict = "kept"
numbers = { max_abs_dp_vs_maker_fp16 = 0.000974, max_abs_dp_vs_maker_bf16 = 0.011567, maker_bf16_vs_fp16_floor = 0.011092, argmax_flips = 0, branches = 15, worst_layer_cosine = 0.999992, max_abs_last_token = 595.5, peak_mlx_gb = 17.9 }
links = ["README.md", "gate/golden/golden.json"]
+++

# 001 Port (2 Nov 2026)

Runtime choice. No bundled family runs Qwen3.5 with this checkpoint's attention: mlx-swift-lm has the decoder, but its
layers are internal and its full-attention mask is causal for any multi-token input. The kit runner was ruled out
because its wire parses requests with JSONSerialization, which loses object key order, and a choice's option order sets
its answer letters. So elephant is its own native family on d1-3B's pattern: the runner lives here, Workers/Elephant
links it into the one native graph, and the request path keeps key order (d1-3B's PyJSON and byte-level wire).

The maker's path is OneJev `qev` @81ce62f with the checkpoint's `serve.patch`: render each question (qev-labels-v2),
cut the longest common token prefix that ends at the state block, prefill it once, then score each suffix from that
cache. Full-attention layers are block-bidirectional (prefix block, then question block); in the sequential path a
forward is either the prefix alone or one suffix after the whole cached prefix, so the mask is none in both. DeltaNet
layers stay causal and recurrent. The readout is the softmax over the LM-head logits of the slot labels (A..Z, then
two-letter labels, 256 slots) in float32; only those 256 head rows are kept. Answers use qev's assembly and TypeSafe's
confidence formulas, rounded to six places as Python rounds.

Precision. The checkpoint is BF16; Get casts it to FP16 file by file. The maker's own BF16 and FP16 runs differ by at
most 0.0111 in probability with no answer flips; the residual stream peaks at 1,248. FP16 is kept.

Parity (bounds frozen before the first candidate run). Nine fixed requests, 15 branches: the four Arsenal decision
bench cases, the card's ticket, a nested JSON state with structured criteria, a 30-option choice (two-letter labels),
Polish, and questions with no instructions. Token ids identical for every prefix and suffix; max |dp| 0.00097 against
the maker's FP16 run and 0.0116 against its BF16 run; no argmax changes. Components at the last token of both passes
(embedding, 32 layers, final norm, 68 points): worst cosine 0.999992 (prefix layer 21).

Not ported: the image path (`qev/mm_engine.py`: the vision encoder, `<image:N>` placement and M-RoPE positions);
image requests are refused with `unsupported_media`. Debias over option shifts (an HTTP query flag in qev) is not
offered; Arsenal's decide contract has no such field.
