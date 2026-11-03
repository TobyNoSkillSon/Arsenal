+++
hypothesis = "Expose Vireo's qualified speech engines through Arsenal v2 without changing their dtype recipes."
change = "V2 settings, schemas, deterministic runner selection, saved voice persistence and optimization inventory."
measurement = "Source comparison against Vireo 9366da5e14dcafe846fc41565c94f4f145eaf7cc; real-cycle evidence follows separately."
verdict = "kept"
numbers = { dtype_changes = 0, performance_claims = 0 }
links = ["optimizations.json", "Workers/Voices/PORTED-FROM.md", "docs/evidence/mb-v2-voices/"]
+++

# V2 folder and operation contract

Exact components: qwen-fused-rope, qwen-predictor-tables, qwen-fused-sampler. Fast components: qwen-fused-rope, qwen-predictor-tables, qwen-fused-sampler. Every kept component remains on by default in its mode and retains its self-test/fallback. Lab-only pipeline/CFG/SmallM candidates remain off (not accepted levers). Pocket has no accepted optimization components; Fast and Exact use the same path.

Dtypes come from Vireo's presence gate: bf16, int4, int8. Kokoro and Pocket int4 are absent because they did not pass; Qwen int8/int4 are affine group 64 from the 16-bit talker, not community quants. Breeze int8 is affine group 64; int4 is the pinned vendor MXFP4 group 32, never an affine derive.

Settings apply to requests, not load, because voice, language, speed, instructions and reference are request inputs. Speed is only exposed for Kokoro; other engines reject non-unit speed. Reference settings map to nested reference.path/text. Saved voices are scoped to model IDs under support/voices; design saves a description plus a generated preview, cloning copies the bounded reference clip and transcript. Later speaks use the saved conditioning via the same engine, and voices lists presets plus saved IDs.
