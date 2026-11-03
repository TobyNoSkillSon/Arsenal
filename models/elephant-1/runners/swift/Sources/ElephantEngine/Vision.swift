// swiftlint:disable identifier_name shorthand_operator
// SPDX-License-Identifier: MIT
// elephant-1's vision encoder (Qwen3.5's, frozen in the fine-tune) and the multimodal rotary positions of a prefix that
// holds images. Ported from Arsenal's Ovis port (Workers/Embed/Sources/EmbedWorker/OvisVision.swift, itself from
// mlx-swift-lm's Qwen3-VL vision tower, MIT) without deepstack mergers (this checkpoint has none) and with its own
// configuration type. References: transformers 5.19.0 Qwen3_5VisionModel and Qwen3_5Model.get_rope_index.
import Foundation
import MLX
import MLXFast
import MLXNN

public struct THW: Equatable, Sendable {
    public let t: Int, h: Int, w: Int
    public init(_ t: Int, _ h: Int, _ w: Int) { self.t = t; self.h = h; self.w = w }
}

public struct VisionConfig: Sendable {
    public let depth: Int, hidden: Int, heads: Int, intermediate: Int, patch: Int, temporal: Int, merge: Int, out: Int, positions: Int
    public let imageTokenId: Int, videoTokenId: Int, visionStartTokenId: Int, visionEndTokenId: Int

    public init(json: [String: Any]) throws {
        guard let v = json["vision_config"] as? [String: Any] else { throw PromptError("config.json lacks vision_config") }
        func int(_ d: [String: Any], _ k: String) throws -> Int {
            guard let x = d[k] as? Int else { throw PromptError("config.json \(k) missing") }
            return x
        }
        depth = try int(v, "depth"); hidden = try int(v, "hidden_size"); heads = try int(v, "num_heads")
        intermediate = try int(v, "intermediate_size"); patch = try int(v, "patch_size"); temporal = try int(v, "temporal_patch_size")
        merge = try int(v, "spatial_merge_size"); out = try int(v, "out_hidden_size"); positions = try int(v, "num_position_embeddings")
        imageTokenId = try int(json, "image_token_id"); videoTokenId = try int(json, "video_token_id")
        visionStartTokenId = try int(json, "vision_start_token_id"); visionEndTokenId = try int(json, "vision_end_token_id")
        guard (v["deepstack_visual_indexes"] as? [Int] ?? []).isEmpty, v["hidden_act"] as? String ?? "gelu_pytorch_tanh" == "gelu_pytorch_tanh" else {
            throw PromptError("Only the Qwen3.5 vision encoder without deepstack is implemented.")
        }
    }
}

enum VisionOps {
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, 0..<half]], axis: -1)
    }

    /// Vision rotary: freqs [L, D/2] float32; HF applies it in float32 and casts back.
    static func applyRotary(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let c = expandedDimensions(tiled(expandedDimensions(cos(freqs), axis: 1), repetitions: [1, 1, 2]), axis: 0)
        let s = expandedDimensions(tiled(expandedDimensions(sin(freqs), axis: 1), repetitions: [1, 1, 2]), axis: 0)
        let x32 = x.asType(.float32)
        return ((x32 * c) + (rotateHalf(x32) * s)).asType(x.dtype)
    }
}

final class VisionPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d
    let patch: Int, temporal: Int, hidden: Int
    init(_ c: VisionConfig) {
        patch = c.patch; temporal = c.temporal; hidden = c.hidden
        let kernel = IntOrTriple([c.temporal, c.patch, c.patch])
        _proj.wrappedValue = Conv3d(inputChannels: 3, outputChannels: c.hidden, kernelSize: kernel, stride: kernel, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let states = x.reshaped(-1, 3, temporal, patch, patch).movedAxis(source: 1, destination: 4)
        return proj(states).reshaped(-1, hidden)
    }
}

final class VisionMerger: Module {
    let width: Int
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear
    init(_ c: VisionConfig) {
        width = c.hidden * c.merge * c.merge
        _norm.wrappedValue = LayerNorm(dimensions: c.hidden, eps: 1e-6)
        _fc1.wrappedValue = Linear(width, width)
        _fc2.wrappedValue = Linear(width, c.out)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(norm(x).reshaped(-1, width)))) }
}

final class VisionAttention: Module {
    let heads: Int, headDim: Int, scale: Float
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ c: VisionConfig) {
        heads = c.heads; headDim = c.hidden / c.heads; scale = Foundation.pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(c.hidden, 3 * c.hidden, bias: true)
        _proj.wrappedValue = Linear(c.hidden, c.hidden)
    }
    /// Attention within each image (frame) segment only; no mask over the concatenation.
    func callAsFunction(_ x: MLXArray, boundaries: [Int], rotary: MLXArray) -> MLXArray {
        let L = x.dim(0)
        let parts = split(qkv(x).reshaped(L, 3, heads, headDim).transposed(1, 0, 2, 3), parts: 3, axis: 0)
        let q = VisionOps.applyRotary(parts[0][0], freqs: rotary).reshaped(1, L, heads, headDim).transposed(0, 2, 1, 3)
        let k = VisionOps.applyRotary(parts[1][0], freqs: rotary).reshaped(1, L, heads, headDim).transposed(0, 2, 1, 3)
        let v = parts[2][0].reshaped(1, L, heads, headDim).transposed(0, 2, 1, 3)
        var segments: [MLXArray] = []
        for i in 1..<boundaries.count {
            let r = boundaries[i - 1]..<boundaries[i]
            let o = MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., r, 0...], keys: k[0..., 0..., r, 0...], values: v[0..., 0..., r, 0...], scale: scale, mask: .none)
            eval(o)
            segments.append(o)
        }
        return proj(concatenated(segments, axis: 2).transposed(0, 2, 1, 3).reshaped(L, -1))
    }
}

final class VisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear
    init(_ c: VisionConfig) {
        _fc1.wrappedValue = Linear(c.hidden, c.intermediate, bias: true)
        _fc2.wrappedValue = Linear(c.intermediate, c.hidden, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

final class VisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP
    init(_ c: VisionConfig) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hidden, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hidden, eps: 1e-6)
        _attention.wrappedValue = VisionAttention(c)
        _mlp.wrappedValue = VisionMLP(c)
    }
    func callAsFunction(_ x: MLXArray, boundaries: [Int], rotary: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x), boundaries: boundaries, rotary: rotary)
        return h + mlp(norm2(h))
    }
}

public final class VisionModel: Module {
    public let config: VisionConfig
    let gridPerSide: Int
    @ModuleInfo(key: "patch_embed") var patchEmbed: VisionPatchEmbed
    @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
    @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
    @ModuleInfo(key: "merger") var merger: VisionMerger

    public init(_ c: VisionConfig) {
        config = c
        gridPerSide = Int(Double(c.positions).squareRoot())
        _patchEmbed.wrappedValue = VisionPatchEmbed(c)
        _posEmbed.wrappedValue = Embedding(embeddingCount: c.positions, dimensions: c.hidden)
        _blocks.wrappedValue = (0..<c.depth).map { _ in VisionBlock(c) }
        _merger.wrappedValue = VisionMerger(c)
    }

    /// 2D rotary over (row, column) in merge-block order: freqs [patches, headDim/2], float32 (theta 10,000).
    func rotary(_ grids: [THW]) -> MLXArray {
        let headDim = config.hidden / config.heads, dim = headDim / 2
        let maxHW = grids.map { Swift.max($0.h, $0.w) }.max() ?? 1
        let invFreq = 1 / MLX.pow(MLXArray(Float(10_000)), MLXArray(stride(from: 0, to: dim, by: 2).map(Float.init)) / Float(dim))
        let table = outer(MLXArray((0..<maxHW).map(Float.init)), invFreq)
        var rows: [Int32] = [], cols: [Int32] = []
        let m = config.merge
        for g in grids {
            for _ in 0..<g.t {
                for bh in 0..<(g.h / m) {
                    for bw in 0..<(g.w / m) {
                        for ih in 0..<m {
                            for iw in 0..<m { rows.append(Int32(bh * m + ih)); cols.append(Int32(bw * m + iw)) }
                        }
                    }
                }
            }
        }
        return concatenated([table[MLXArray(rows)], table[MLXArray(cols)]], axis: -1)
    }

    /// The learned 48x48 position table bilinearly resampled to each grid (align corners), in merge-block order
    /// (transformers vision_utils get_vision_interpolation_indices_and_weights).
    func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
        let n = gridPerSide, maxIndex = Float(n - 1)
        var parts: [MLXArray] = []
        for g in grids {
            let hs = (0..<g.h).map { g.h == 1 ? 0 : Float($0) * maxIndex / Float(g.h - 1) }
            let ws = (0..<g.w).map { g.w == 1 ? 0 : Float($0) * maxIndex / Float(g.w - 1) }
            var idx: [[Int32]] = Array(repeating: [], count: 4), wts: [[Float]] = Array(repeating: [], count: 4)
            for y in hs {
                let y0 = Int(y), y1 = Swift.min(y0 + 1, n - 1), dy = y - Float(y0)
                for x in ws {
                    let x0 = Int(x), x1 = Swift.min(x0 + 1, n - 1), dx = x - Float(x0)
                    idx[0].append(Int32(y0 * n + x0)); wts[0].append((1 - dy) * (1 - dx))
                    idx[1].append(Int32(y0 * n + x1)); wts[1].append((1 - dy) * dx)
                    idx[2].append(Int32(y1 * n + x0)); wts[2].append(dy * (1 - dx))
                    idx[3].append(Int32(y1 * n + x1)); wts[3].append(dy * dx)
                }
            }
            // transformers 5.19: half-precision table rows times float32 weights, summed in float32, then cast.
            var e = MLXArray.zeros([g.h * g.w, config.hidden], dtype: .float32)
            for c in 0..<4 {
                e = e + posEmbed(MLXArray(idx[c])).asType(.float32) * expandedDimensions(MLXArray(wts[c]), axis: -1)
            }
            e = e.asType(posEmbed.weight.dtype)
            let m = config.merge
            e = tiled(e, repetitions: [g.t, 1]).reshaped(g.t, g.h / m, m, g.w / m, m, config.hidden)
            parts.append(e.transposed(0, 1, 3, 2, 4, 5).reshaped(-1, config.hidden))
        }
        return concatenated(parts, axis: 0)
    }

    /// pixels [patches, 3*T*P*P] -> merged features [patches / merge^2, out].
    public func callAsFunction(_ pixels: MLXArray, grids: [THW], probe: Probe? = nil, cancelled: () -> Bool = { false }) throws -> MLXArray {
        var h = patchEmbed(pixels) + positionalEmbeddings(grids)
        let rot = rotary(grids)
        var boundaries = [0]
        for g in grids { for _ in 0..<g.t { boundaries.append(boundaries.last! + g.h * g.w) } }
        for block in blocks {
            if cancelled() { throw Cancelled() }
            h = block(h, boundaries: boundaries, rotary: rot)
            eval(h)
        }
        let out = merger(h)
        probe?("vision", out)
        return out
    }

    /// Checkpoint keys of the vision tower to module keys; the Conv3d weight to MLX's channels-last layout.
    public static func map(_ key: String, _ value: MLXArray) -> (String, MLXArray)? {
        guard key.hasPrefix("model.visual.") else { return nil }
        let name = "visual." + key.dropFirst("model.visual.".count)
        if name == "visual.patch_embed.proj.weight", value.ndim == 5, value.dim(-1) != 3 { return (name, value.transposed(0, 2, 3, 4, 1)) }
        return (name, value)
    }
}

public enum MultimodalPositions {
    public static func rope(_ ids: [Int], grids: [THW], imageTokenId: Int, merge: Int) throws -> (positions: [[Int32]], delta: Int) {
        try rope(ids, blocks: grids.map { (imageTokenId, $0) }, merge: merge)
    }

    /// Qwen3.5's get_rope_index for one sequence: text tokens advance all three axes together; a vision block (an image,
    /// or one frame pair of a video: get_rope_index splits a video into t grids of [1, h, w]) of t x (h/m) x (w/m)
    /// tokens takes (offset + t, offset + row, offset + column), and the next token starts at offset + max(h/m, w/m).
    /// `blocks` are the vision blocks in prompt order with their token id. Returns the [3, L] positions and the delta
    /// (next position - L) later tokens add.
    public static func rope(_ ids: [Int], blocks: [(token: Int, grid: THW)], merge: Int) throws -> (positions: [[Int32]], delta: Int) {
        var axes: [[Int32]] = [[], [], []]
        var cursor = 0, offset = 0, block = 0
        let tokens = Set(blocks.map(\.token))
        while cursor < ids.count {
            if tokens.contains(ids[cursor]) {
                guard block < blocks.count, blocks[block].token == ids[cursor] else { throw PromptError("Vision blocks are out of order.") }
                let g = blocks[block].grid, token = blocks[block].token
                block += 1
                let h = g.h / merge, w = g.w / merge, count = g.t * h * w
                guard cursor + count <= ids.count, ids[cursor..<(cursor + count)].allSatisfy({ $0 == token }) else {
                    throw PromptError("A vision grid does not match its token block.")
                }
                for t in 0..<g.t {
                    for r in 0..<h {
                        for c in 0..<w {
                            axes[0].append(Int32(offset + t)); axes[1].append(Int32(offset + r)); axes[2].append(Int32(offset + c))
                        }
                    }
                }
                offset += Swift.max(h, w)
                cursor += count
            } else {
                for a in 0..<3 { axes[a].append(Int32(offset)) }
                offset += 1
                cursor += 1
            }
        }
        guard block == blocks.count else { throw PromptError("Some vision blocks have no tokens.") }
        return (axes, offset - ids.count)
    }
}

// swiftlint:enable identifier_name shorthand_operator
