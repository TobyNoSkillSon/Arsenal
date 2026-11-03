# Real check, 2 Nov 2026

Requests through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: the pinned `Qwen/Qwen3-Embedding-8B` files imported from the local Hugging Face cache and cast from BF16 to FP16
  (`cast:embed-qwen3-v2`).
- `embed` on three documents and one query; compared with Qwen's own model-card code run once on the GPU in FP16
  (`lab/evidence/qwen3-embedding-8b/maker-2026-11-02`).
- First run: cosine 0.46 to 0.70 against Qwen's vectors. Two causes: Arsenal did not append the `<|endoftext|>` token
  that Qwen's tokenizer adds and that last-token pooling reads, and it wrote "Query: " with a space where the card
  writes "Query:". Fixed in `Workers/Embed` (Qwen3-Embedding only; Harrier unchanged) with a prompt test.
- After the fix: input tokens 25 and 28 as in the reference; cosine 0.999996 to 0.999998 per text, largest element
  difference 0.00034, similarity matrix within 0.0006. The query ranks the two zebra sentences first (0.62, 0.50) and
  the revenue sentence last (0.09). Golden: `gate/golden/golden.json`.
- Not checked: long inputs near the 32k context, other instructions, reduced `dimensions`, int8 and int4.
