// SPDX-License-Identifier: MIT
// The d1-3B language model on MLX: LiquidAI's hybrid System One tree (the checkpoint's `hybrid.py`), packed passes,
// pruned tail. Ported from the folder's earlier Python MLX runner (runners/mlx/engine.py, git history), which the
// same gate qualified; every rounding below is the vendor's bf16 arithmetic.
import Foundation
import MLX
import MLXFast

// `a += b` on an MLXArray rewrites the shared array object in place; residual updates stay `a = a + b`
// (marked per line for SwiftLint's shorthand rule).

/// Branch queries per attention product: bounds a tree's mask at `branchBlock` x (trunk + branches) booleans.
let branchBlock = 1024

/// Host-side description of one packed forward: one or more trees on one axis. A tree is a trunk (the state) and
/// branches (the questions) that each continue it: a branch token sees the whole trunk and its own earlier tokens,
/// its convolution continues from the trunk's last inputs, and its positions restart where the trunk ends. A plain
/// chain is a tree without branches.
public struct Layout {
    public var ids: [Int32] = []
    var pos: [Int32] = []
    var tree: [Int32] = []
    var branch: [Int32] = []
    var prev1: [Int32] = [] // index + 1 into the inputs, 0 = zero padding
    var prev2: [Int32] = []
    public var leaves: [Int] = []
    var trees = 0
    var spans: [(start: Int, end: Int, trunk: Int)] = []
    var branches = 0
    var hasBranches = false

    public init() {}
    public var count: Int { ids.count }

    /// Adds one tree; returns the packed index of each branch's last token (or the trunk's, without branches).
    @discardableResult
    public mutating func add(trunk: [Int], branches list: [[Int]]) -> [Int] {
        let start = ids.count
        let t = Int32(trees)
        trees += 1
        let p = trunk.count
        for (i, token) in trunk.enumerated() {
            let g = start + i
            ids.append(Int32(token))
            pos.append(Int32(i))
            tree.append(t)
            branch.append(-1)
            prev1.append(Int32(i >= 1 ? g : 0)) // (g - 1) + 1
            prev2.append(Int32(i >= 2 ? g - 1 : 0)) // (g - 2) + 1
        }
        let last = start + p - 1
        if list.isEmpty {
            leaves.append(last)
            spans.append((start, ids.count, p))
            return [last]
        }
        hasBranches = true
        var out: [Int] = []
        for b in list {
            let bid = Int32(branches)
            branches += 1
            let base = ids.count
            for (o, token) in b.enumerated() {
                let g = base + o
                ids.append(Int32(token))
                pos.append(Int32(p + o))
                tree.append(t)
                branch.append(bid)
                let p1 = o >= 1 ? g - 1 : (p >= 1 ? last : -1)
                let p2 = o >= 2 ? g - 2 : (o == 1 ? (p >= 1 ? last : -1) : (p >= 2 ? last - 1 : -1))
                prev1.append(Int32(p1 + 1))
                prev2.append(Int32(p2 + 1))
            }
            out.append(base + b.count - 1)
            leaves.append(base + b.count - 1)
        }
        spans.append((start, ids.count, p))
        return out
    }
}

/// The short convolution's elementwise block in one kernel: B*x, the 3-tap causal conv along each token's predecessors
/// (prev indices +1, 0 = zero padding), then C*. Same roundings as the stock-op path (`convOps`): the bf16 products,
/// the taps summed in fp32 and rounded once, the bf16 product with C.
private let convSource = """
        uint d = thread_position_in_grid.x;
        uint t = thread_position_in_grid.y;
        const uint W = 3 * D;
        int p1 = prev1[t], p2 = prev2[t];
        float here = float(bfloat16_t(float(y[t * W + d]) * float(y[t * W + 2 * D + d])));
        float a = p2 ? float(bfloat16_t(float(y[(p2 - 1) * W + d]) * float(y[(p2 - 1) * W + 2 * D + d]))) : 0.0f;
        float b = p1 ? float(bfloat16_t(float(y[(p1 - 1) * W + d]) * float(y[(p1 - 1) * W + 2 * D + d]))) : 0.0f;
        float z = (a * taps[d] + b * taps[D + d]) + here * taps[2 * D + d];
        out[t * D + d] = bfloat16_t(float(y[t * W + D + d]) * float(bfloat16_t(z)));
    """

private let convKernel = MLXFast.metalKernel(name: "d1_short_conv", inputNames: ["y", "prev1", "prev2", "taps"], outputNames: ["out"], source: convSource)

func convFused(_ y: MLXArray, _ prev1: MLXArray, _ prev2: MLXArray, _ taps: MLXArray) -> MLXArray {
    let t = y.dim(0)
    let d = y.dim(1) / 3
    return convKernel(
        [y, prev1, prev2, taps], template: [("D", d)], grid: (d, t, 1), threadGroup: (256, 1, 1),
        outputShapes: [[t, d]], outputDTypes: [y.dtype])[0]
}

func convOps(_ y: MLXArray, _ prev1: MLXArray, _ prev2: MLXArray, _ taps: MLXArray, rows: MLXArray? = nil) -> MLXArray {
    let d = y.dim(1) / 3
    let b = y[0..., 0..<d]
    let c = y[0..., d..<(2 * d)]
    let u = y[0..., (2 * d)...]
    let bx = b * u
    let padded = concatenated([MLX.zeros([1, d], dtype: bx.dtype), bx], axis: 0)
    let here = rows.map { bx[$0] } ?? bx
    // The vendor's bf16 depthwise conv1d: products summed in fp32, rounded once.
    let z = (padded[prev2].asType(.float32) * taps[0] + padded[prev1].asType(.float32) * taps[1] + here.asType(.float32) * taps[2]).asType(bx.dtype)
    return (rows.map { c[$0] } ?? c) * z
}

/// The fused kernel against the stock-op path on a small tree-shaped input: bit-identical or unused.
func convKernelCheck() -> Bool {
    let t = 37
    let d = 512
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    func next() -> Float { // deterministic values in [-3, 3)
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float(Int64(bitPattern: state >> 11) % 6000) / 1000
    }
    let y = MLXArray((0..<(t * 3 * d)).map { _ in next() }, [t, 3 * d]).asType(.bfloat16)
    var p1 = (0..<t).map { Int32($0) }
    var p2 = (0..<t).map { Int32(max($0 - 1, 0)) }
    p1[20] = 7
    p2[20] = 6
    p2[21] = 7 // a branch continuing from position 6
    let taps = MLXArray((0..<(3 * d)).map { _ in next() }, [3, d]).asType(.bfloat16)
    let a = convFused(y, MLXArray(p1), MLXArray(p2), taps)
    let b = convOps(y, MLXArray(p1), MLXArray(p2), taps)
    return (a .== b).all().item(Bool.self)
}

private func swigluOps(_ g: MLXArray, _ u: MLXArray) -> MLXArray {
    // torch's bf16 silu: computed in fp32, rounded once; then the product, rounded once.
    let g32 = g.asType(.float32)
    return (g32 * sigmoid(g32)).asType(g.dtype) * u
}

/// Shapeless: one trace for every pass length.
private let swiglu = compile(shapeless: true, swigluOps)

/// Query (rows) may see key (columns): same tree, not later, and either the trunk or the query's own branch.
func visible(_ qp: MLXArray, _ qt: MLXArray, _ qb: MLXArray, _ kp: MLXArray, _ kt: MLXArray, _ kb: MLXArray) -> MLXArray {
    let sameTree = qt.expandedDimensions(axis: 1) .== kt.expandedDimensions(axis: 0)
    let notLater = kp.expandedDimensions(axis: 0) .<= qp.expandedDimensions(axis: 1)
    let trunkOrOwn = (kb.expandedDimensions(axis: 0) .== -1) .|| (kb.expandedDimensions(axis: 0) .== qb.expandedDimensions(axis: 1))
    return sameTree .&& notLater .&& trunkOrOwn
}

struct Layer {
    let conv: Bool
    let operatorNorm: MLXArray
    let ffnNorm: MLXArray
    let w13: D1Matrix
    let w2: D1Matrix
    // conv
    var inProj: D1Matrix?
    var outProj: D1Matrix?
    var taps: MLXArray?
    // attention
    var qkv: D1Matrix?
    var kv: D1Matrix?
    var o: D1Matrix?
    var qNorm: MLXArray?
    var kNorm: MLXArray?
}

/// The d1 language model in MLX at the checkpoint's bf16.
public final class D1Model: @unchecked Sendable {
    public let imageTokenID: Int
    let dim: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let eps: Float
    let scale: Float
    let inv: MLXArray
    public let embed: D1Matrix
    let finalNorm: MLXArray
    let layers: [Layer]
    let hiddenDim: Int
    let lastAttention: Int
    public let convKernelOK: Bool

    public init(weights: URL, store: SafetensorsFile) throws {
        let cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: weights.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
        guard let tc = cfg["text_config"] as? [String: Any], let imageToken = cfg["image_token_id"] as? Int,
            let dim = tc["hidden_size"] as? Int, let heads = tc["num_attention_heads"] as? Int, let kvHeads = tc["num_key_value_heads"] as? Int,
            let eps = tc["norm_eps"] as? Double, let kinds = tc["layer_types"] as? [String],
            let rope = tc["rope_parameters"] as? [String: Any], let theta = rope["rope_theta"] as? Double
        else { throw JSONError("config.json is not the d1-3B LFM2-VL configuration") }
        imageTokenID = imageToken
        self.dim = dim
        self.heads = heads
        self.kvHeads = kvHeads
        let headDim = dim / heads
        self.headDim = headDim
        self.eps = Float(eps)
        scale = Float(1 / Double(headDim).squareRoot())
        // As the vendor: 1 / theta ** (arange(0, D, 2) / D) in fp32; every pair is rotated.
        let half = (0..<(headDim / 2)).map { i -> Float in 1 / powf(Float(theta), Float(2 * i) / Float(headDim)) }
        inv = MLXArray(half + half)
        let prefix = "model.language_model."
        func w(_ name: String) throws -> MLXArray { try store.take(prefix + name) }
        func m(_ name: String) throws -> D1Matrix { try store.matrix(prefix + name) }
        embed = try m("embed_tokens.weight")
        finalNorm = try w("embedding_norm.weight")
        var layers: [Layer] = []
        for (i, kind) in kinds.enumerated() {
            let p = "layers.\(i)."
            // w1 and w3 as one GEMM: one wide product runs nearer the GEMM peak than two at small M.
            var layer = Layer(
                conv: kind == "conv", operatorNorm: try w(p + "operator_norm.weight"), ffnNorm: try w(p + "ffn_norm.weight"),
                w13: try D1Matrix.concatenate([m(p + "feed_forward.w1.weight"), m(p + "feed_forward.w3.weight")]),
                w2: try m(p + "feed_forward.w2.weight"))
            if layer.conv {
                layer.inProj = try m(p + "conv.in_proj.weight")
                layer.outProj = try m(p + "conv.out_proj.weight")
                layer.taps = try w(p + "conv.conv.weight").reshaped([dim, 3]).transposed()
            } else {
                let qkv = try D1Matrix.concatenate(
                    [m(p + "self_attn.q_proj.weight"), m(p + "self_attn.k_proj.weight"), m(p + "self_attn.v_proj.weight")])
                layer.qkv = qkv
                layer.kv = qkv.rows(dim..<qkv.shape[0])
                layer.o = try m(p + "self_attn.out_proj.weight")
                layer.qNorm = try w(p + "self_attn.q_layernorm.weight")
                layer.kNorm = try w(p + "self_attn.k_layernorm.weight")
            }
            let arrays =
                [layer.operatorNorm, layer.ffnNorm] + [layer.taps, layer.qNorm, layer.kNorm].compactMap { $0 }
                + ([layer.w13, layer.w2] + [layer.inProj, layer.outProj, layer.qkv, layer.o].compactMap { $0 }).flatMap(\.arrays)
            eval(arrays)
            layers.append(layer)
        }
        self.layers = layers
        hiddenDim = layers[0].w2.inputs
        lastAttention = kinds.lastIndex { $0 != "conv" } ?? 0
        eval(embed.arrays + [finalNorm, inv])
        convKernelOK = convKernelCheck()
    }

    // MARK: pieces

    /// Half-split RoPE in fp32 over (T, H, D), rounded once to the activation dtype.
    private func rope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let half = headDim / 2
        let rot = concatenated([-x32[.ellipsis, half...], x32[.ellipsis, ..<half]], axis: -1)
        return (x32 * cos + rot * sin).asType(x.dtype)
    }

    private func conv(_ layer: Layer, _ x: MLXArray, _ prev1: MLXArray, _ prev2: MLXArray, rows: MLXArray? = nil) -> MLXArray {
        let y = layer.inProj!(x)
        let z = rows == nil && convKernelOK ? convFused(y, prev1, prev2, layer.taps!) : convOps(y, prev1, prev2, layer.taps!, rows: rows)
        return layer.outProj!(z)
    }

    enum Mask {
        case causal
        case array(MLXArray)
        /// Per tree: its keys start at `start`; query blocks [lo, hi) with their mask (nil = causal).
        case trees([(start: Int, blocks: [(lo: Int, hi: Int, mask: MLXArray?)])])
    }

    private func attention(_ layer: Layer, _ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray, _ mask: Mask, rows: MLXArray? = nil) -> MLXArray {
        let t = x.dim(0)
        let kvd = kvHeads * headDim
        var q: MLXArray
        var k: MLXArray
        var v: MLXArray
        var qc = cos
        var qs = sin
        if let rows {
            let kv = layer.kv!(x)
            k = kv[0..., ..<kvd]
            v = kv[0..., kvd...]
            q = layer.qkv!.rows(0..<dim)(x[rows])
            qc = cos[rows]
            qs = sin[rows]
        } else {
            let qkv = layer.qkv!(x)
            q = qkv[0..., ..<dim]
            k = qkv[0..., dim..<(dim + kvd)]
            v = qkv[0..., (dim + kvd)...]
        }
        let n = q.dim(0)
        q = MLXFast.rmsNorm(q.reshaped([n, heads, headDim]), weight: layer.qNorm!, eps: eps)
        k = MLXFast.rmsNorm(k.reshaped([t, kvHeads, headDim]), weight: layer.kNorm!, eps: eps)
        v = v.reshaped([t, kvHeads, headDim])
        q = rope(q, qc, qs)
        k = rope(k, cos, sin)
        q = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        k = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        v = v.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let y: MLXArray
        switch mask {
        case .causal:
            y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal)
        case .array(let m):
            y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .array(m))
        case .trees(let trees):
            // Trees attend only within themselves: products per tree, never T x T masks.
            var parts: [MLXArray] = []
            for tree in trees {
                for block in tree.blocks {
                    let qq = q[0..., 0..., block.lo..<block.hi]
                    let kk = k[0..., 0..., tree.start..<block.hi]
                    let vv = v[0..., 0..., tree.start..<block.hi]
                    parts.append(
                        MLXFast.scaledDotProductAttention(
                            queries: qq, keys: kk, values: vv, scale: scale, mask: block.mask.map { .array($0) } ?? .causal))
                }
            }
            y = concatenated(parts, axis: 2)
        }
        return layer.o!(y[0].transposed(1, 0, 2).reshaped([n, dim]))
    }

    private func mlp(_ layer: Layer, _ x: MLXArray) -> MLXArray {
        let y = layer.w13(x)
        return layer.w2(swiglu(y[0..., ..<hiddenDim], y[0..., hiddenDim...]))
    }

    // MARK: forward

    /// Rows each layer after the last attention layer must produce: only the answer slots and, through each
    /// convolution, their two predecessors. Returns the query rows of the last attention layer and, per later layer,
    /// (outputs, prev1, prev2) indexed into that layer's input rows, and the leaves' rows in the final output.
    private func tailPlan(_ layout: Layout) -> (queries: [Int32], plans: [([Int32], [Int32], [Int32])], leaves: [Int32]) {
        let p1All = layout.prev1.map { Int($0) - 1 }
        let p2All = layout.prev2.map { Int($0) - 1 }
        var need = Array(Set(layout.leaves)).sorted()
        var plans: [([Int32], [Int32], [Int32])] = []
        for _ in stride(from: layers.count - 1, to: lastAttention, by: -1) {
            let p1 = need.map { p1All[$0] }
            let p2 = need.map { p2All[$0] }
            let rowsIn = Array(Set(need + p1.filter { $0 >= 0 } + p2.filter { $0 >= 0 })).sorted()
            var index: [Int: Int] = [:]
            for (i, r) in rowsIn.enumerated() { index[r] = i }
            let local = { (r: [Int]) in r.map { Int32($0 >= 0 ? index[$0]! + 1 : 0) } }
            plans.append((need.map { Int32(index[$0]!) }, local(p1), local(p2)))
            need = rowsIn
        }
        plans.reverse()
        let unique = Array(Set(layout.leaves)).sorted()
        var at: [Int: Int] = [:]
        for (i, r) in unique.enumerated() { at[r] = i }
        return (need.map { Int32($0) }, plans, layout.leaves.map { Int32(at[$0]!) })
    }

    /// Final-norm hidden states at `layout.leaves`, (R, dim). `embeds` replaces the token embeddings (pictures).
    var intake: D1ComponentSession?
    var intakePrefix = ""

    public func hidden(_ layout: Layout, embeds: MLXArray? = nil) -> MLXArray {
        let ids = MLXArray(layout.ids)
        var h = embeds ?? embed[ids]
        intake?.output(intakePrefix + "embedding", h)
        let prev1 = MLXArray(layout.prev1)
        let prev2 = MLXArray(layout.prev2)
        let pos = MLXArray(layout.pos)
        let tree = MLXArray(layout.tree)
        let br = MLXArray(layout.branch)
        let mask: Mask
        if layout.spans.count == 1 && !layout.hasBranches {
            mask = .causal
        } else {
            var trees: [(start: Int, blocks: [(lo: Int, hi: Int, mask: MLXArray?)])] = []
            for span in layout.spans {
                var blocks: [(lo: Int, hi: Int, mask: MLXArray?)] = [(span.start, span.start + span.trunk, nil)]
                for lo in stride(from: span.start + span.trunk, to: span.end, by: branchBlock) {
                    let hi = min(lo + branchBlock, span.end)
                    blocks.append(
                        (lo, hi, visible(pos[lo..<hi], tree[lo..<hi], br[lo..<hi], pos[span.start..<hi], tree[span.start..<hi], br[span.start..<hi])))
                }
                trees.append((span.start, blocks))
            }
            mask = .trees(trees)
        }
        let (queryRows, plans, leaves) = tailPlan(layout)
        let rows = MLXArray(queryRows)
        let rowsMask = visible(pos[rows], tree[rows], br[rows], pos, tree, br)
        let angles = pos.asType(.float32).expandedDimensions(axis: 1) * inv.expandedDimensions(axis: 0)
        let cos = MLX.cos(angles).expandedDimensions(axis: 1)
        let sin = MLX.sin(angles).expandedDimensions(axis: 1)
        for (i, layer) in layers[...lastAttention].enumerated() {
            h = intake?.input(intakePrefix + "layer-\(i)", h) ?? h
            let x = MLXFast.rmsNorm(h, weight: layer.operatorNorm, eps: eps)
            if layer.conv {
                h = h + conv(layer, x, prev1, prev2) // swiftlint:disable:this shorthand_operator
            } else if i < lastAttention {
                h = h + attention(layer, x, cos, sin, mask) // swiftlint:disable:this shorthand_operator
            } else { // the last attention layer: every key, only the queries the tail reads
                h = h[rows] + attention(layer, x, cos, sin, .array(rowsMask), rows: rows)
            }
            h = h + mlp(layer, MLXFast.rmsNorm(h, weight: layer.ffnNorm, eps: eps)) // swiftlint:disable:this shorthand_operator
            intake?.output(intakePrefix + "layer-\(i)", h)
        }
        for (offset, pair) in zip(layers[(lastAttention + 1)...], plans).enumerated() {
            let (layer, plan) = pair
            let i = lastAttention + 1 + offset
            h = intake?.input(intakePrefix + "layer-\(i)", h) ?? h
            let x = MLXFast.rmsNorm(h, weight: layer.operatorNorm, eps: eps)
            let r = MLXArray(plan.0)
            h = h[r] + conv(layer, x, MLXArray(plan.1), MLXArray(plan.2), rows: r)
            h = h + mlp(layer, MLXFast.rmsNorm(h, weight: layer.ffnNorm, eps: eps)) // swiftlint:disable:this shorthand_operator
            intake?.output(intakePrefix + "layer-\(i)", h)
        }
        return MLXFast.rmsNorm(h[MLXArray(leaves)], weight: finalNorm, eps: eps)
    }

    /// Only the selected option rows are promoted: logits must not round near ties to BF16.
    static func readout(_ hidden: MLXArray, rows: MLXArray) -> MLXArray {
        matmul(hidden.asType(.float32), rows.asType(.float32).T)
    }
    public func logits(_ hidden: MLXArray, _ tokens: [Int]) -> MLXArray {
        let rows = embed[MLXArray(tokens.map { Int32($0) })]
        return Self.readout(hidden, rows: rows)
    }
}
