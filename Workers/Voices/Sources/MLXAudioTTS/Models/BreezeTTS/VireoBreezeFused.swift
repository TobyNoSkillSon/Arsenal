import Foundation
@preconcurrency import MLX
import MLXNN

/// Vireo fast path (`VIREO_BREEZE_FUSED`): the depth decoder's one-token step with fewer dependent dispatches.
/// Per layer, stock MLX runs ~20 kernels (RMSNorm, q/k/v, RoPE q, RoPE k, two KV-cache slice updates that each copy
/// the buffer inside a lazy frame, SDPA, the head-transpose copy, o, add, RMSNorm, gate/up, SiLU, multiply, down,
/// add); at M = 1–2 each costs a few µs of latency on the dependent chain, 12 layers × 14 steps per codec frame.
/// Here the layer runs 14 (measured: the attention kernel and the compiled SiLU-gate pay; a residual-add + RMSNorm
/// kernel did not, lab/perf/NOTES.md):
/// - `attention`: RoPE of q and of the new k, attention over the cached positions plus the new one, float softmax,
///   and the new frame-sized KV buffers (old positions + the new one) in one dispatch (one simdgroup per row and query
///   head; the first head of each KV group writes the buffers). RoPE replays MLX's `rope_single` (inverse frequency
///   `exp2(-d/(D/2) · log2 base)`, `fast::cos/sin`, float products rounded once), so rotated keys match stock.
/// - `siluGate`: `silu(gate) · up` as one compiled elementwise kernel.
/// Inexact only in the attention sums' order (float softmax, one simdgroup reduction); the model's self-test bounds
/// it with the teacher-forced depth distributions like the KV cache itself. Frame-sized caches only (≤ 32 positions).
enum VireoBreezeFused {
    static let capacity = 16

    static let attentionSource = """
        uint row = threadgroup_position_in_grid.y;
        uint head = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint kvh = head / GROUP;
        int off = offset;
        const device T* qr = q + row * NQ * HD + head * HD;
        const device T* kr = k + row * NKV * HD + kvh * HD;
        const device T* vr = v + row * NKV * HD + kvh * HD;
        float qv[4], kn[4], vn[4];
        float pos = rope_scale * float(off);
        for (int j = 0; j < 2; j++) {
            uint d = lane + 32 * j;
            float invFreq = metal::exp2(-(float(d) / float(HD / 2)) * log2_base);
            float theta = pos * invFreq;
            float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
            float a = float(qr[d]), b = float(qr[d + HD / 2]);
            qv[j] = float(T(a * c - b * s)); qv[j + 2] = float(T(a * s + b * c));
            a = float(kr[d]); b = float(kr[d + HD / 2]);
            kn[j] = float(T(a * c - b * s)); kn[j + 2] = float(T(a * s + b * c));
            vn[j] = float(vr[d]); vn[j + 2] = float(vr[d + HD / 2]);
        }
        const device T* kc = k_cache + ((row * NKV + kvh) * CAP) * HD;
        const device T* vc = v_cache + ((row * NKV + kvh) * CAP) * HD;
        float scores[CAP];
        float m = -INFINITY;
        for (int p = 0; p <= off; p++) {
            float part = 0;
            for (int j = 0; j < 4; j++) {
                uint d = lane + 32 * (j % 2) + (HD / 2) * (j / 2);
                part += qv[j] * ((p == off) ? kn[j] : float(kc[p * HD + d]));
            }
            float sc = simd_sum(part) * scale;
            scores[p] = sc;
            m = max(m, sc);
        }
        float den = 0;
        for (int p = 0; p <= off; p++) { scores[p] = metal::exp(scores[p] - m); den += scores[p]; }
        float acc[4] = {0, 0, 0, 0};
        for (int p = 0; p <= off; p++) {
            for (int j = 0; j < 4; j++) {
                uint d = lane + 32 * (j % 2) + (HD / 2) * (j / 2);
                acc[j] += scores[p] * ((p == off) ? vn[j] : float(vc[p * HD + d]));
            }
        }
        device T* o = out + row * NQ * HD + head * HD;
        for (int j = 0; j < 4; j++) { uint d = lane + 32 * (j % 2) + (HD / 2) * (j / 2); o[d] = T(acc[j] / den); }
        if (head % GROUP == 0) {
            device T* ko = k_out + ((row * NKV + kvh) * CAP) * HD;
            device T* vo = v_out + ((row * NKV + kvh) * CAP) * HD;
            for (int p = 0; p < CAP; p++) {
                for (int j = 0; j < 4; j++) {
                    uint d = lane + 32 * (j % 2) + (HD / 2) * (j / 2);
                    ko[p * HD + d] = (p < off) ? kc[p * HD + d] : (p == off) ? T(kn[j]) : T(0);
                    vo[p * HD + d] = (p < off) ? vc[p * HD + d] : (p == off) ? T(vn[j]) : T(0);
                }
            }
        }
    """

    static let attentionKernel = MLXFast.metalKernel(
        name: "vireo_breeze_depth_attention",
        inputNames: ["q", "k", "v", "k_cache", "v_cache", "offset", "scale", "log2_base", "rope_scale"],
        outputNames: ["out", "k_out", "v_out"],
        source: attentionSource
    )

    static let siluGate: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { gate, up in
        silu(gate) * up
    }

    /// Versions of the kernels above; part of the gate key through the fast-path label.
    static let version = "vireo_breeze_depth_attention 1"

    private static let offsets = (0 ..< 64).map { MLXArray(Int32($0)) }
    static func offsetArray(_ offset: Int) -> MLXArray { offset < offsets.count ? offsets[offset] : MLXArray(Int32(offset)) }

    /// q `[R, NQ·HD]`, k/v `[R, NKV·HD]` (pre-RoPE projections), caches `[R, NKV, CAP, HD]` holding `offset` positions.
    /// Returns the attention output `[R, NQ·HD]` (heads concatenated, as `o_proj` takes it) and the new caches.
    static func attention(q: MLXArray, k: MLXArray, v: MLXArray, keys: MLXArray, values: MLXArray, offset: Int,
                          queryHeads: Int, kvHeads: Int, headDim: Int, ropeBase: Float, ropeScale: Float = 1)
        -> (MLXArray, MLXArray, MLXArray)
    {
        precondition(offset < keys.dim(2), "fused depth attention: cache full")
        return attention(q: q, k: k, v: v, keys: keys, values: values, offsetArray: offsetArray(offset),
                         queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim, ropeBase: ropeBase, ropeScale: ropeScale)
    }

    /// Same, with the offset as a 0-d int32 array.
    static func attention(q: MLXArray, k: MLXArray, v: MLXArray, keys: MLXArray, values: MLXArray, offsetArray: MLXArray,
                          queryHeads: Int, kvHeads: Int, headDim: Int, ropeBase: Float, ropeScale: Float = 1)
        -> (MLXArray, MLXArray, MLXArray)
    {
        let rows = q.dim(0), cap = keys.dim(2)
        precondition(headDim == 128 && q.dim(1) == queryHeads * headDim && queryHeads % kvHeads == 0,
                     "fused depth attention: unsupported shape")
        let outputs = attentionKernel(
            [q, k, v, keys, values, offsetArray,
             MLXArray(Foundation.pow(Float(headDim), -0.5)), MLXArray(Foundation.log2(ropeBase)), MLXArray(ropeScale)],
            template: [("T", q.dtype), ("NQ", queryHeads), ("NKV", kvHeads), ("HD", headDim), ("CAP", cap),
                       ("GROUP", queryHeads / kvHeads)],
            grid: (queryHeads * 32, rows, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[rows, queryHeads * headDim], keys.shape, values.shape],
            outputDTypes: [q.dtype, keys.dtype, values.dtype]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    /// Child-process check (no model): the kernels against the stock ops on fixed random inputs, bf16, rows 1 and 2,
    /// every offset of a frame. Returns (max relative RMS of the attention output, max |Δ| of the rotated cache).
    static func unitCheck() -> (attention: Float, cache: Float) {
        var worst: (Float, Float) = (0, 0)
        let (nq, nkv, hd, base): (Int, Int, Int, Float) = (8, 2, 128, 500_000)
        let rope = RoPE(dimensions: hd, traditional: false, base: base, scale: 1)
        for rows in [1, 2] {
            var keys = MLXArray.zeros([rows, nkv, capacity, hd], dtype: .bfloat16)
            var values = MLXArray.zeros([rows, nkv, capacity, hd], dtype: .bfloat16)
            var stockK = [MLXArray](), stockV = [MLXArray]()
            for offset in 0 ..< capacity {
                let seed = UInt64(rows * 100 + offset)
                let q = MLXRandom.normal([rows, nq * hd], key: MLXRandom.key(seed)).asType(.bfloat16)
                let k = MLXRandom.normal([rows, nkv * hd], key: MLXRandom.key(seed + 1000)).asType(.bfloat16)
                let v = MLXRandom.normal([rows, nkv * hd], key: MLXRandom.key(seed + 2000)).asType(.bfloat16)
                let (out, nk, nv) = attention(q: q, k: k, v: v, keys: keys, values: values, offset: offset,
                                              queryHeads: nq, kvHeads: nkv, headDim: hd, ropeBase: base)
                keys = nk; values = nv
                let qh = rope(q.reshaped([rows, 1, nq, hd]).transposed(0, 2, 1, 3), offset: offset)
                stockK.append(rope(k.reshaped([rows, 1, nkv, hd]).transposed(0, 2, 1, 3), offset: offset))
                stockV.append(v.reshaped([rows, 1, nkv, hd]).transposed(0, 2, 1, 3))
                let allK = concatenated(stockK, axis: 2), allV = concatenated(stockV, axis: 2)
                let reference = MLXFast.scaledDotProductAttention(queries: qh, keys: allK, values: allV,
                                                                  scale: Foundation.pow(Float(hd), -0.5), mask: .none)
                    .transposed(0, 2, 1, 3).reshaped([rows, -1])
                let rel = relativeRMS(out, reference)
                let cacheError = Swift.max(
                    abs(keys[0..., 0..., ..<(offset + 1), 0...].asType(.float32) - allK.asType(.float32)).max().item(Float.self),
                    abs(values[0..., 0..., ..<(offset + 1), 0...].asType(.float32) - allV.asType(.float32)).max().item(Float.self))
                worst.0 = Swift.max(worst.0, rel.isFinite ? rel : .infinity)
                worst.1 = Swift.max(worst.1, cacheError.isFinite ? cacheError : .infinity)
            }
        }
        return worst
    }

    static func relativeRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a32 = a.asType(.float32), b32 = b.asType(.float32)
        let num = sqrt(mean(square(a32 - b32))).item(Float.self)
        let den = sqrt(mean(square(b32))).item(Float.self)
        return den > 0 ? num / den : num
    }
}
