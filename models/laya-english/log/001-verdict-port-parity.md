+++
hypothesis = "Verdict's Swift engine, copied unchanged and driven over ArsenalWire, reproduces Verdict's Laya fixture gate."
change = "Ported VerdictEngine + helper request rules into arsenal-worker-decisions; pinned mlx-swift 9019419, swift-transformers 1.3.4."
measurement = "Verdict fixture regression, 3 sets x 202 items, Exact and Fast, through the wire runner (M5 Max, gpulock)."
verdict = "kept"
numbers = { items_per_set = 202, sets = 3, exact_mismatches = 0, fast_fp16_changed_answers = 0, fast_fp16_max_dp = 0.0066, fp16_median_item_ms = 7.28, int8_median_item_ms = 8.94, int4_median_item_ms = 8.47, load_s_fp16 = 0.14, mlx_peak_mb_fp16 = 1792.0, mlx_peak_mb_int8 = 1447.0, mlx_peak_mb_int4 = 1322.0 }
links = ["gate/gate.toml", "gate/parity-fp16-exact.json", "gate/parity-fp16.json", "gate/parity-int8-exact.json", "gate/parity-int4-exact.json", "../../Workers/Decisions/PORTED-FROM.md"]
+++

## What was done
Copied Verdict's engine (Laya network, prompt, fast byte-BPE tokenizer, windowed attention, small-M GEMM route) verbatim into
`Workers/Decisions` (see `Workers/Decisions/PORTED-FROM.md`) and put Verdict's helper request rules (System One validation,
`/v1/judge` grouping, stock-path fallback, finite check, answer formatting) behind the ArsenalWire runner. The weights are the
pinned snapshot `20aed815fc6acde75733882e7ec0e3f28aeb9717`, clone-copied from the local Hugging Face cache (no download).

## Measurement
`scripts/mlock gpulock decisions -- python3 Workers/Decisions/scripts/parity.py laya laya-english all [--exact]` on an M5 Max,
macOS 26, 2 Oct 2026. Verdict's own gate (`lab/tests/test_laya_regression.py`): the 202 oracle items of each fixture set
(`laya-english-0/8/4bit.json.gz`), each judged alone over the wire, compared with the oracle's wire answers.

- **Exact** (Optimized · Exact): 0 mismatches at fp16, int8 and int4 — byte-identical answers, rounding included.
- **Fast** at fp16 adds the small-M GEMM (gate passed on this chip: logits rel 1.6e-03, 14/14 answers unchanged, 0 near-ties, persisted).
  It is not bit-identical: 197/202 items differ in the 4th decimal, max |Δp| 0.0066,
  0 changed answers — inside Verdict's noise ruling (≤ 0.5 % changed, |Δp| ≤ 0.01). At int8/int4
  the kernel is ineligible, so Fast = Exact (0 mismatches).
- System One `decide` for every item returned exactly Verdict's System One view of the same answers (0 mismatches); 12
  oracle-exact cases plus over-context 422s are saved as golden files for the API worker.
- Single-item latency (median, Exact, under the GPU lock, not a benchmark): fp16 7.28 ms,
  int8 8.94 ms, int4 8.47 ms. Load 0.14 s
  (weights map lazily; the first request pays the rest).
- Memory: MLX peak 1792.0 MB at fp16, 1447.0 MB at int8,
  1322.0 MB at int4 (quantizing at load reads the fp16 tensors first, so int8 peaks
  above int4 by less than the weight ratio suggests). Footprint after the 202 items: 1413.0 MB.

## Next
The bench harness's quick-suite result (`results/<chip>.json`) is produced by `arsenal-bench` once the app supervises this worker.
