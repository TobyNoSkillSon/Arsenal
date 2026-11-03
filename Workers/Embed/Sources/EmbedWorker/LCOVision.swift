// SPDX-License-Identifier: MIT
// Qwen2_5OmniVisionEncoder (Qwen2.5-VL's window-attention ViT with separate q/k/v projections), from Transformers'
// modeling_qwen2_5_omni.py and vision_utils.py; module layout follows mlx-swift-lm's Qwen25VL.swift (MIT).
// Packed variable-length attention is computed per segment, batching segments of equal length, so no dense
// sequence×sequence mask is ever built.
import Foundation
import ArsenalWire
import IntegrationKit
import MLX
import MLXFast
import MLXNN

struct LCOGrid: Equatable { let t, h, w: Int }

enum LCOVisionLayout {
    /// get_vision_position_ids: (h, w) per patch, laid out block-major over merge×merge blocks, repeated per frame.
    static func positions(_ grids: [LCOGrid], merge: Int) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        for g in grids {
            var frame: [(Int, Int)] = []
            for bh in 0..<(g.h / merge) {
                for bw in 0..<(g.w / merge) {
                    for mh in 0..<merge {
                        for mw in 0..<merge {
                            frame.append((bh * merge + mh, bw * merge + mw))
                        }
                    }
                }
            }
            for _ in 0..<g.t { out += frame }
        }
        return out
    }
    /// get_vision_window_index: merged-unit order for window attention and the window boundaries (in patches).
    static func windows(_ grids: [LCOGrid], merge: Int, windowSize: Int, patchSize: Int) -> (index: [Int], cu: [Int]) {
        let size = windowSize / merge / patchSize, unit = merge * merge
        var index: [Int] = [], cu = [0], base = 0
        for g in grids {
            let gh = g.h / merge, gw = g.w / merge
            let padH = size - gh % size, padW = size - gw % size
            let nh = (gh + padH) / size, nw = (gw + padW) / size
            for t in 0..<g.t {
                for wh in 0..<nh {
                    for ww in 0..<nw {
                        var count = 0
                        for r in 0..<size {
                            for c in 0..<size {
                                let y = wh * size + r, x = ww * size + c
                                if y < gh && x < gw { index.append(base + t * gh * gw + y * gw + x); count += 1 }
                            }
                        }
                        cu.append(cu.last! + count * unit)
                    }
                }
            }
            base += g.t * gh * gw
        }
        var unique: [Int] = []
        for v in cu where unique.last != v { unique.append(v) }
        return (index, unique)
    }
    /// get_vision_cu_seqlens: one full-attention segment per frame.
    static func frames(_ grids: [LCOGrid]) -> [Int] {
        var cu = [0]
        for g in grids { for _ in 0..<g.t { cu.append(cu.last! + g.h * g.w) } }
        return cu
    }
}

/// Segments of a packed sequence grouped by length: gather indices per group and the inverse order.
struct LCOSegments {
    let groups: [(length: Int, indices: MLXArray, count: Int)]
    let inverse: MLXArray
    init(_ cu: [Int]) {
        var byLength: [Int: [Int]] = [:]
        for i in 1..<cu.count where cu[i] > cu[i - 1] { byLength[cu[i] - cu[i - 1], default: []].append(cu[i - 1]) }
        var order: [Int] = [], groups: [(Int, MLXArray, Int)] = []
        for length in byLength.keys.sorted() {
            let starts = byLength[length]!
            let flat = starts.flatMap { s in (s..<(s + length)).map { $0 } }
            order += flat; groups.append((length, MLXArray(flat.map(Int32.init)), starts.count))
        }
        var inv = [Int32](repeating: 0, count: order.count)
        for (position, source) in order.enumerated() { inv[source] = Int32(position) }
        self.groups = groups; inverse = MLXArray(inv)
    }
}

final class LCOVisionAttention: Module {
    @ModuleInfo var q: Linear
    @ModuleInfo var k: Linear
    @ModuleInfo var v: Linear
    @ModuleInfo var proj: Linear
    let heads: Int
    init(_ width: Int, heads: Int) {
        self.heads = heads
        _q.wrappedValue = Linear(width, width); _k.wrappedValue = Linear(width, width); _v.wrappedValue = Linear(width, width)
        _proj.wrappedValue = Linear(width, width)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, segments: LCOSegments) -> MLXArray {
        let (s, w) = (x.dim(0), x.dim(1)), d = w / heads
        func rotate(_ t: MLXArray) -> MLXArray {
            // apply_rotary_pos_emb_vision: rotation in FP32, cast back to the activation dtype.
            let f = t.asType(.float32)
            return (f * cos + LCORope.rotateHalf(f) * sin).asType(t.dtype)
        }
        let qs = rotate(q(x).reshaped(s, heads, d)), ks = rotate(k(x).reshaped(s, heads, d)), vs = v(x).reshaped(s, heads, d)
        var outputs: [MLXArray] = []
        for group in segments.groups {
            func take(_ t: MLXArray) -> MLXArray { t[group.indices].reshaped(group.count, group.length, heads, d).transposed(0, 2, 1, 3) }
            let o = MLXFast.scaledDotProductAttention(queries: take(qs), keys: take(ks), values: take(vs), scale: pow(Float(d), -0.5), mask: .none)
            outputs.append(o.transposed(0, 2, 1, 3).reshaped(group.count * group.length, w))
        }
        return proj(concatenated(outputs, axis: 0)[segments.inverse])
    }
}

final class LCOVisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ width: Int, _ hidden: Int) {
        _gate.wrappedValue = Linear(width, hidden); _up.wrappedValue = Linear(width, hidden); _down.wrappedValue = Linear(hidden, width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class LCOVisionBlock: Module {
    @ModuleInfo var norm1: RMSNorm
    @ModuleInfo var norm2: RMSNorm
    @ModuleInfo var attn: LCOVisionAttention
    @ModuleInfo var mlp: LCOVisionMLP
    init(_ c: LCOConfiguration.Vision) {
        _norm1.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6); _norm2.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _attn.wrappedValue = LCOVisionAttention(c.hiddenSize, heads: c.heads); _mlp.wrappedValue = LCOVisionMLP(c.hiddenSize, c.intermediateSize)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, segments: LCOSegments) -> MLXArray {
        let h = x + attn(norm1(x), cos: cos, sin: sin, segments: segments)
        return h + mlp(norm2(h))
    }
}

/// Conv3d with kernel = stride = patch: one matrix over the processor's (channel, time, row, column) patch layout.
final class LCOPatchEmbed: Module {
    @ModuleInfo var proj: Linear
    init(_ c: LCOConfiguration.Vision) {
        _proj.wrappedValue = Linear(c.inChannels * c.temporalPatchSize * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

final class LCOPatchMerger: Module {
    @ModuleInfo(key: "ln_q") var lnQ: RMSNorm
    @ModuleInfo var mlp: (Linear, GELU, Linear)
    let width: Int
    init(_ c: LCOConfiguration.Vision) {
        width = c.hiddenSize * c.spatialMergeSize * c.spatialMergeSize
        _lnQ.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _mlp.wrappedValue = (Linear(width, width), GELU(), Linear(width, c.outHiddenSize))
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mlp.2(mlp.1(mlp.0(lnQ(x).reshaped(-1, width)))) }
}

final class LCOVisionEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: LCOPatchEmbed
    @ModuleInfo var blocks: [LCOVisionBlock]
    @ModuleInfo var merger: LCOPatchMerger
    let config: LCOConfiguration.Vision
    init(_ c: LCOConfiguration.Vision) {
        config = c
        _patchEmbed.wrappedValue = LCOPatchEmbed(c)
        _blocks.wrappedValue = (0..<c.depth).map { _ in LCOVisionBlock(c) }
        _merger.wrappedValue = LCOPatchMerger(c)
    }

    /// pixels [patches, 3·2·14·14] → merged features [patches / 4, out_hidden_size], in the processor's order.
    func callAsFunction(_ pixels: MLXArray, grids: [LCOGrid], cancellation: NativeCancellation) throws -> MLXArray {
        let merge = config.spatialMergeSize, unit = merge * merge
        let (window, cuWindow) = LCOVisionLayout.windows(grids, merge: merge, windowSize: config.windowSize, patchSize: config.patchSize)
        let windowArray = MLXArray(window.map(Int32.init))
        var x = patchEmbed(pixels.asType(patchEmbed.proj.weight.dtype))
        let s = x.dim(0)
        guard s == window.count * unit else { throw WorkerFailure(code: "invalid_request", message: "Vision patches do not match their grids.") }
        func permute(_ t: MLXArray) -> MLXArray { t.reshaped(s / unit, unit, -1)[windowArray].reshaped(s, -1) }
        x = permute(x)
        // Axial rotary angles: head_dim/4 frequencies for rows, the same for columns, then repeated for the rotated half.
        let headDim = config.hiddenSize / config.heads, spatial = headDim / 2
        let inv = (0..<(spatial / 2)).map { Float(1 / pow(10000.0, Double(2 * $0) / Double(spatial))) }
        let positions = LCOVisionLayout.positions(grids, merge: merge)
        var angles = [Float](repeating: 0, count: s * headDim)
        for (i, (r, c)) in positions.enumerated() {
            for j in 0..<(spatial / 2) {
                let a = Float(r) * inv[j], b = Float(c) * inv[j]
                angles[i * headDim + j] = a; angles[i * headDim + spatial / 2 + j] = b
                angles[i * headDim + spatial + j] = a; angles[i * headDim + spatial + spatial / 2 + j] = b
            }
        }
        let rotary = permute(MLXArray(angles, [s, headDim])).reshaped(s, 1, headDim)
        let cos = MLX.cos(rotary), sin = MLX.sin(rotary)
        let full = LCOSegments(LCOVisionLayout.frames(grids)), windowed = LCOSegments(cuWindow)
        for (i, block) in blocks.enumerated() {
            try cancellation.check()
            x = block(x, cos: cos, sin: sin, segments: config.fullAttentionBlocks.contains(i) ? full : windowed)
            eval(x)
        }
        let merged = merger(x)
        return merged[argSort(windowArray, axis: 0)]
    }
}
