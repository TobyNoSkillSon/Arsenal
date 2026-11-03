+++
hypothesis = "int8/int4 can be stored as quantized tensors at Get, and the worker needs no MLX cache bound."
change = "Kept Verdict's quantize-at-load for int8/int4 (derived view = 16-bit files + arsenal-derived.json); bounded the MLX buffer cache at 1024 MB like Verdict's helper."
measurement = "First fp16 Fast run without the cache bound: footprint 10039 MB after 202 items (mlx_cache 9015 MB); with it 1345 MB."
verdict = "kept"
numbers = { footprint_mb_unbounded = 10039, mlx_cache_mb_unbounded = 9015, footprint_mb_bounded = 1345, convert_seconds_per_view = 0.3 }
links = ["gate/parity-fp16.json", "../../Workers/Decisions/Sources/arsenal-worker-decisions/Convert.swift"]
+++

## Pre-quantized storage: rejected for v1
Verdict, and the Python oracle its 8/4-bit fixtures come from, quantize the fp16 Linears with `MLX.quantized(group 64)` while
loading. Writing those tensors out at Get and loading them back would save a little load-time memory but changes the code path
the fixture gate qualified (QuantizedLinear construction from stored scales/biases), so it would need its own requalification.
The derived int8/int4 view is therefore the 16-bit snapshot plus `arsenal-derived.json` (recipe
`mlx-affine-g64:verdict-at-load`); in Arsenal's content-addressed store the blobs are shared, so the view costs no disk. The 9×202
fixture gate passed byte-for-byte on this path (log 001). Cost: an int8/int4 load reads the 16-bit weights first (MLX peak
1447 MB at int8 vs 1792 MB at fp16 for Laya English).

## MLX cache bound: kept
The first fp16 run left a 10 GB footprint: MLX's buffer cache keeps every activation shape. Verdict's helper sets
`Memory.cacheLimit` to 1024 MB and clears the cache after a request that leaves more; the worker now does the same
(`ARSENAL_DECISIONS_CACHE_LIMIT_MB`). Footprint after the same 202 items: 1345 MB, latency unchanged (median 6.85 vs 9.93 ms
first run, which also paid the small-M gate's one-time kernel check).
