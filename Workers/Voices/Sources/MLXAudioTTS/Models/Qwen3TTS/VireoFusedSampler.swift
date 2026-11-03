import Foundation
@preconcurrency import MLX

/// Vireo fast path: the code predictor's per-codebook sampling (top-k, temperature, `categorical`) as one Metal
/// dispatch per sub-step instead of ~20 MLX kernels, plus one dispatch per frame that derives the keys and uniform
/// draws of all sub-steps from the global key exactly as the sub-steps' `categorical` calls would.
///
/// It replays stock `sampleToken(logits, temperature:, topP: 1, topK:, minP: 0)` bit for bit (finite logits):
/// - keys: `MLXRandom.globalState.next()` = `split(state)` = `bits([2, 2], key: state)` → (new state, subkey); the
///   key kernel walks that chain (threefry2x32 as MLX's `rbitsc`) and hands the state after the last split back to
///   the global state, so every later draw (the talker's next token) sees the stock key;
/// - `categorical` of one row is MLX's inverse-CDF path (`categorical_inverse_cdf`): x = float(logits), m = max x,
///   w = exp(x − m), u = uniform([1]) · sum(w), token = searchsorted(exclusive cumsum(w), u, right) − 1. The float
///   sums depend on the reduction order, so the kernel runs MLX's own layouts: V/4 threads with 4 consecutive
///   elements each, `all_reduce` (sequential 4 → simd_sum → simd_sum of the simdgroup sums) for sum(w) and
///   `contiguous_scan` (per-thread inclusive scan → simd exclusive prefix → simdgroup prefix) for the cumsum, then
///   the same binary search;
/// - top-k: stock masks positions ≥ k of a *stable* ascending sort of −logits (MLX's arg-partition is its block merge
///   sort: NaN last, ties in index order); the kernel bisects bf16's 16-bit order for the k-th key and keeps ties at
///   the boundary by index; then `filtered / T(temperature)` in the logits dtype.
/// The model's self-test compares a whole seeded generation with stock before the path serves.
enum VireoFusedSampler {
    static let threefry = """
        inline uint2 vireo_threefry(uint2 key, uint2 count) {
            uint ks[3] = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDAu};
            const uint rot[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
            uint2 v = uint2(count.x + ks[0], count.y + ks[1]);
            for (int i = 0; i < 5; ++i) {
                for (int r = 0; r < 4; ++r) {
                    uint rr = rot[i % 2][r];
                    v.x += v.y;
                    v.y = (v.y << rr) | (v.y >> (32 - rr));
                    v.y ^= v.x;
                }
                v.x += ks[(i + 1) % 3];
                v.y += ks[(i + 2) % 3] + uint(i) + 1u;
            }
            return v;
        }
    """

    /// For S consecutive `categorical` draws from `key`: the subkeys [S, 2], the uniform draw of each ([S], float32,
    /// `uniform([1], key: subkey)`), and the key state after the S splits.
    static let keySource = """
        uint s = thread_position_in_grid.x;
        int S = params[0];
        if (int(s) >= S) return;
        uint2 state = uint2(key[0], key[1]);
        for (uint t = 0; t < s; ++t) {
            uint2 a = vireo_threefry(state, uint2(0u, 2u));
            uint2 b = vireo_threefry(state, uint2(1u, 3u));
            state = uint2(a.x, b.x);
        }
        uint2 a = vireo_threefry(state, uint2(0u, 2u));
        uint2 b = vireo_threefry(state, uint2(1u, 3u));
        uint2 sub = uint2(a.y, b.y);
        subkeys[2 * s] = sub.x; subkeys[2 * s + 1] = sub.y;
        if (int(s) == S - 1) { new_key[0] = a.x; new_key[1] = b.x; }
        uint word = vireo_threefry(sub, uint2(0u, 0u)).x;
        float u = float(word) / 4294967296.0f;
        uniforms[s] = metal::min(u, as_type<float>(0x3F7FFFFFu));
    """

    /// One sample from the last row of `logits` [1, L, V] (bf16) with uniform draw `step`: token [1, 1] uint32.
    /// One threadgroup of V/4 threads (MLX's reduction and scan layouts for V ≤ 4096, V % 128 == 0).
    static let sampleSource = """
        uint t = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        int V = params[0], L = params[1], K = params[2], s = params[3];
        uint NSG = uint(V) / 128u;
        const device T* row = logits + (L - 1) * V;
        threadgroup uint counts[32];
        threadgroup float fshared[32];
        threadgroup ushort dkeys[4096];
        threadgroup float cdf[4096];

        // Descending 16-bit key of each logit: smaller = earlier in the stable sort of -logits (NaN last).
        T lv[4];
        ushort dk[4];
        for (int i = 0; i < 4; ++i) {
            int j = int(t) * 4 + i;
            lv[i] = row[j];
            float f = float(lv[i]);
            ushort d = 0xFFFF;
            if (!metal::isnan(f)) {
                uint u = as_type<uint>(f);
                if ((u & 0x7FFFFFFFu) == 0u) u = 0u;
                uint asc = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                d = ushort((~asc) >> 16);
            }
            dk[i] = d;
            dkeys[j] = d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Top-k: keep the first K of that order (bisection for the K-th key tau, boundary ties by index).
        bool useK = K > 0 && K < V;
        uint tau = 0xFFFFu, needTies = 0u;
        if (useK) {
            uint lo = 0u, hi = 0xFFFFu;
            while (lo < hi) {
                uint mid = (lo + hi) >> 1;
                uint c = 0u;
                for (int i = 0; i < 4; ++i) c += uint(dk[i]) <= mid ? 1u : 0u;
                c = simd_sum(c);
                if (lane == 0) counts[sg] = c;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint total = 0u;
                for (uint w = 0; w < NSG; ++w) total += counts[w];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (total >= uint(K)) hi = mid; else lo = mid + 1u;
            }
            tau = lo;
            uint c = 0u;
            for (int i = 0; i < 4; ++i) c += uint(dk[i]) < tau ? 1u : 0u;
            c = simd_sum(c);
            if (lane == 0) counts[sg] = c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint below = 0u;
            for (uint w = 0; w < NSG; ++w) below += counts[w];
            needTies = uint(K) - below;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // filtered / T(temperature), promoted to float32 (categorical's astype).
        T tv = temp[0];
        float x[4];
        for (int i = 0; i < 4; ++i) {
            int j = int(t) * 4 + i;
            bool keep = true;
            if (useK) {
                if (uint(dk[i]) > tau) keep = false;
                else if (uint(dk[i]) == tau) {
                    uint before = 0u;
                    for (int q = 0; q < j; ++q) before += (uint(dkeys[q]) == tau) ? 1u : 0u;
                    keep = before < needTies;
                }
            }
            T y = (keep ? lv[i] : T(-INFINITY)) / tv;
            x[i] = float(y);
        }

        // m = max(x) (exact in any order).
        float mx = metal::max(metal::max(x[0], x[1]), metal::max(x[2], x[3]));
        mx = simd_max(mx);
        if (lane == 0) fshared[sg] = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m = fshared[0];
        for (uint w = 1; w < NSG; ++w) m = metal::max(m, fshared[w]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float wv[4];
        bool minf = metal::isinf(m);
        for (int i = 0; i < 4; ++i) wv[i] = minf ? (x[i] == m ? 1.0f : 0.0f) : metal::precise::exp(x[i] - m);

        // sum(w): MLX all_reduce layout.
        float total = 0.0f;
        for (int i = 0; i < 4; ++i) total = wv[i] + total;
        total = simd_sum(total);
        if (lane == 0) fshared[sg] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = lane < NSG ? fshared[lane] : 0.0f;
        total = simd_sum(total);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) fshared[31] = total;   // simdgroup 0's value (MLX writes lid 0's)

        // Exclusive cumsum(w): MLX contiguous_scan layout.
        float v[4];
        v[0] = wv[0];
        for (int i = 1; i < 4; ++i) v[i] = wv[i] + v[i - 1];
        float prevThread = simd_prefix_exclusive_sum(v[3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumAll = fshared[31];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 31) fshared[sg] = prevThread + v[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float p = simd_prefix_exclusive_sum(lane < NSG ? fshared[lane] : 0.0f);
            fshared[lane] = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sgPrefix = fshared[sg];
        for (int i = 0; i < 4; ++i) {
            float o = v[i] + 0.0f;
            o = o + sgPrefix;
            o = o + prevThread;
            int j = int(t) * 4 + i;
            if (j + 1 < V) cdf[j + 1] = o;
        }
        if (t == 0) cdf[0] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // searchsorted(cdf, u * sum, right) - 1 (MLX's binary search with its NaN-aware LessThan).
        if (t == 0) {
            float val = uniforms[s] * sumAll;
            uint lo = 0u, hi = uint(V);
            while (lo < hi) {
                uint mid = lo + (hi - lo) / 2u;
                float mv = cdf[mid];
                bool an = metal::isnan(val), bn = metal::isnan(mv);
                bool lt = (an | bn) ? ((!an) & bn) : (val < mv);
                if (!lt) lo = mid + 1u; else hi = mid;
            }
            token[0] = lo - 1u;
        }
    """

    static let keyKernel = MLXFast.metalKernel(
        name: "vireo_qwen3tts_categorical_keys",
        inputNames: ["key", "params"],
        outputNames: ["subkeys", "uniforms", "new_key"],
        source: keySource,
        header: threefry
    )

    static let sampleKernel = MLXFast.metalKernel(
        name: "vireo_qwen3tts_topk_categorical",
        inputNames: ["logits", "uniforms", "params", "temp"],
        outputNames: ["token"],
        source: sampleSource
    )

    /// The draws of one frame: subkeys [S, 2] (stock fallback) and uniforms [S] (fused kernel).
    struct Draws { let subkeys: MLXArray; let uniforms: MLXArray }

    /// Whether the fused sampler replays `sampleToken` for these settings (else the caller uses stock).
    static func eligible(temperature: Float, topP: Float, minP: Float, vocab: Int, dtype: DType) -> Bool {
        temperature > 0 && (topP <= 0 || topP >= 1) && minP <= 0 && vocab % 128 == 0 && vocab <= 4096 && dtype == .bfloat16
    }

    /// Keys and uniforms for `steps` draws; advances `MLXRandom.globalState` exactly as `steps` `categorical` calls.
    static func draw(steps: Int) -> Draws {
        let state = MLXRandom.globalState.innerState()[0]
        let out = keyKernel(
            [state, MLXArray([Int32(steps)])],
            grid: (steps, 1, 1),
            threadGroup: (steps, 1, 1),
            outputShapes: [[steps, 2], [steps], [2]],
            outputDTypes: [.uint32, .float32, .uint32]
        )
        state._updateInternal(out[2])
        return Draws(subkeys: out[0], uniforms: out[1])
    }

    /// Stock-equivalent token [1, 1] (uint32) from `logits` [1, L, V] with draw `step`.
    static func sample(logits: MLXArray, draws: Draws, step: Int, topK: Int, temperature: Float) -> MLXArray {
        let (L, V) = (logits.dim(-2), logits.dim(-1))
        guard logits.dtype == .bfloat16, V % 128 == 0, V <= 4096 else {
            // The kernel's top-k keys are bf16's 16-bit order: other dtypes run the stock ops with the same subkey.
            return stockSample(logits, topK: topK, temperature: temperature, key: draws.subkeys[step])
        }
        let params = MLXArray([Int32(V), Int32(L), Int32(topK), Int32(step)])
        let temp = temperature.asMLXArray(dtype: logits.dtype).reshaped(1)
        let out = sampleKernel(
            [logits, draws.uniforms, params, temp],
            template: [("T", logits.dtype)],
            grid: (V / 4, 1, 1),
            threadGroup: (V / 4, 1, 1),
            outputShapes: [[1, 1]],
            outputDTypes: [.uint32]
        )
        return out[0]
    }

    /// Stock sampling path of `sampleToken` for the code predictor (top-k, temperature, `categorical`).
    static func stockSample(_ logits: MLXArray, topK: Int, temperature: Float, key: MLXArray? = nil) -> MLXArray {
        let logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1)
        var filtered = logitsSlice
        let vocab = logitsSlice.dim(-1)
        if topK > 0, topK < vocab {
            let maskIdx = argPartition(-logitsSlice, kth: topK - 1, axis: -1)[0..., topK...]
            let negInf = MLXArray.full(maskIdx.shape, values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
            filtered = putAlong(filtered, maskIdx, values: negInf, axis: -1)
        }
        return categorical(filtered / temperature, key: key).reshaped(1, 1)
    }

    static func unitCheck() -> Int { unitCheckReport().mismatches }

    /// Kernel check against stock: (1) subkeys, uniforms and the key state after 15 draws; (2) the sampler with each
    /// subkey vs stock `categorical(…, key:)` on bf16 logits on a coarse grid (many exact ties at the top-k boundary)
    /// with some ±0, L = 1 and 2, top-k 50 / 7 / 0 / ≥ V; (3) end to end: 15 draws from one seed each way (tokens
    /// and the key state). Reseeds the global key (callers reseed before any seeded work).
    static func unitCheckReport() -> (mismatches: Int, detail: String) {
        var parts: [String] = []
        let V = 2048, steps = 15
        MLXRandom.seed(424_242)
        let stockSubkeys = (0 ..< steps).map { _ in MLXRandom.globalState.next() }
        let stockUniforms = concatenated(stockSubkeys.map { MLXRandom.uniform(0 ..< 1, [1], key: $0) }, axis: 0)
        let stockKey = MLXRandom.globalState.innerState()[0]
        let stockKeys = stacked(stockSubkeys, axis: 0)
        eval(stockKeys, stockUniforms, stockKey)
        MLXRandom.seed(424_242)
        let draws = draw(steps: steps)
        let fusedKey = MLXRandom.globalState.innerState()[0]
        eval(draws.subkeys, draws.uniforms, fusedKey)
        let keyDiff = (stockKey .!= fusedKey).asType(.int32).sum().item(Int.self)
            + (stockKeys .!= draws.subkeys).asType(.int32).sum().item(Int.self)
        let uniformDiff = (stockUniforms .!= draws.uniforms).asType(.int32).sum().item(Int.self)
        parts.append("keys \(keyDiff) · uniforms \(uniformDiff)/\(steps)")

        var samplerDiff = 0, endToEnd = 0
        for (n, (L, topK)) in [(1, 50), (2, 50), (1, 0), (1, 4096), (2, 7)].enumerated() {
            let base = MLXRandom.normal([steps, L, V], key: MLXRandom.key(UInt64(10 + n))) * 3
            var logits = ((base * 4).round() / 4).asType(.bfloat16)
            let zeroMask = MLXRandom.uniform(0 ..< 1, [steps, L, V], key: MLXRandom.key(UInt64(20 + n))) .< 0.01
            logits = which(zeroMask, MLXArray(Float(-0.0)).asType(.bfloat16), logits)
            eval(logits)
            let rows = (0 ..< steps).map { logits[$0 ..< ($0 + 1)] }

            let ref = concatenated(rows.enumerated().map { stockSample($1, topK: topK, temperature: 0.9, key: draws.subkeys[$0]) }, axis: 1)
            let got = concatenated(rows.enumerated().map { sample(logits: $1, draws: draws, step: $0, topK: topK, temperature: 0.9) }, axis: 1)
            eval(ref, got)
            let d = ref.dtype != got.dtype ? steps : (ref .!= got).asType(.int32).sum().item(Int.self)
            if d > 0 { parts.append("case \(n) (L \(L), k \(topK)): sampler \(d)/\(steps)") }
            samplerDiff += d

            MLXRandom.seed(424_242)
            let stock = concatenated(rows.map { stockSample($0, topK: topK, temperature: 0.9) }, axis: 1)
            let sKey = MLXRandom.globalState.innerState()[0]
            eval(stock, sKey)
            MLXRandom.seed(424_242)
            let frame = draw(steps: steps)
            let fused = concatenated(rows.enumerated().map { sample(logits: $1, draws: frame, step: $0, topK: topK, temperature: 0.9) }, axis: 1)
            let fKey = MLXRandom.globalState.innerState()[0]
            eval(fused, fKey)
            endToEnd += stock.dtype != fused.dtype ? steps : (stock .!= fused).asType(.int32).sum().item(Int.self)
            endToEnd += (sKey .!= fKey).asType(.int32).sum().item(Int.self)
        }
        parts.append("sampler \(samplerDiff) · end-to-end \(endToEnd)")
        MLXRandom.seed(0)
        return (keyDiff + uniformDiff + samplerDiff + endToEnd, parts.joined(separator: "; "))
    }
}
