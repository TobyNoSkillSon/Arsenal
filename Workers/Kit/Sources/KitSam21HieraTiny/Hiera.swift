// SPDX-License-Identifier: MIT
// SAM 2.1 image encoder: Hiera backbone and FPN neck, ported from transformers models/sam2/modeling_sam2.py
// (Sam2HieraDetModel, Sam2MultiScaleBlock, Sam2MultiScaleAttention, Sam2VisionNeck). Channels-last throughout.
// Property keys follow the checkpoint (vision_encoder.backbone.*, vision_encoder.neck.*).
import Foundation
import MLX
import MLXFast
import MLXNN

/// The Hiera numbers from config.json's vision_config.backbone_config (tiny and large differ only here).
struct HieraConfig {
    var hidden: Int
    var dims: [Int]
    var blocksPerStage: [Int]
    var heads: [Int]
    var windows: [Int]
    var globalBlocks: Set<Int>
    var background: [Int]
    var channelList: [Int]
    var eps: Float

    init(_ config: [String: Any]) throws {
        guard let vision = config["vision_config"] as? [String: Any], let b = vision["backbone_config"] as? [String: Any] else {
            throw RunnerError.failed("config.json has no vision_config.backbone_config.")
        }
        func ints(_ object: [String: Any], _ key: String) throws -> [Int] {
            guard let values = object[key] as? [NSNumber] else { throw RunnerError.failed("config.json lacks \(key).") }
            return values.map(\.intValue)
        }
        hidden = (b["hidden_size"] as? NSNumber)?.intValue ?? 96
        dims = try ints(b, "embed_dim_per_stage")
        blocksPerStage = try ints(b, "blocks_per_stage")
        heads = try ints(b, "num_attention_heads_per_stage")
        windows = try ints(b, "window_size_per_stage")
        globalBlocks = Set(try ints(b, "global_attention_blocks"))
        background = try ints(b, "window_positional_embedding_background_size")
        channelList = try ints(vision, "backbone_channel_list")
        eps = (b["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        guard (b["num_query_pool_stages"] as? NSNumber)?.intValue ?? 3 == 3, (b["mlp_ratio"] as? NSNumber)?.doubleValue ?? 4 == 4 else {
            throw RunnerError.failed("This runner expects 3 query-pool stages and MLP ratio 4.")
        }
    }
}

/// 2x2 max pooling, stride 2, on [B, H, W, C] (H and W even in every place Hiera pools).
func maxPool2(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    return x.reshaped([b, h / 2, 2, w / 2, 2, c]).max(axes: [2, 4])
}

final class HieraAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let heads: Int
    let pool: Bool

    init(dim: Int, dimOut: Int, heads: Int, pool: Bool) {
        _qkv.wrappedValue = Linear(dim, dimOut * 3)
        _proj.wrappedValue = Linear(dimOut, dimOut)
        self.heads = heads
        self.pool = pool
    }

    /// x: [B, H, W, C]. With pooling the queries are max-pooled 2x2 and the output is [B, H/2, W/2, C'].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        var h = x.dim(1), w = x.dim(2)
        let parts = qkv(x).reshaped([b, h * w, 3, heads, -1])
        let headDim = parts.dim(4)
        var q = parts[0..., 0..., 0]
        let k = parts[0..., 0..., 1], v = parts[0..., 0..., 2]
        if pool {
            q = maxPool2(q.reshaped([b, h, w, heads * headDim]))
            h /= 2
            w /= 2
            q = q.reshaped([b, h * w, heads, headDim])
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
            scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return proj(out.transposed(0, 2, 1, 3).reshaped([b, h, w, heads * headDim]))
    }
}

/// Linear layers with an activation between them (transformers Sam2FeedForward: proj_in, layers, proj_out).
final class FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "layers") var layers: [Linear]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    let useGelu: Bool
    let sigmoidOutput: Bool

    init(_ inDim: Int, _ hidden: Int, _ outDim: Int, layers count: Int, gelu useGelu: Bool = false, sigmoidOutput: Bool = false) {
        _projIn.wrappedValue = Linear(inDim, hidden)
        _layers.wrappedValue = (0..<max(0, count - 2)).map { _ in Linear(hidden, hidden) }
        _projOut.wrappedValue = Linear(hidden, outDim)
        self.useGelu = useGelu
        self.sigmoidOutput = sigmoidOutput
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        func act(_ y: MLXArray) -> MLXArray { useGelu ? gelu(y) : relu(y) }
        var y = act(projIn(x))
        for layer in layers { y = act(layer(y)) }
        y = projOut(y)
        return sigmoidOutput ? sigmoid(y) : y
    }
}

func windowPartition(_ x: MLXArray, _ size: Int) -> (MLXArray, Int, Int) {
    let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    let ph = (size - h % size) % size, pw = (size - w % size) % size
    let padded = ph > 0 || pw > 0 ? MLX.padded(x, widths: [0, .init((0, ph)), .init((0, pw)), 0]) : x
    let hp = h + ph, wp = w + pw
    let windows = padded.reshaped([b, hp / size, size, wp / size, size, c]).transposed(0, 1, 3, 2, 4, 5)
        .reshaped([-1, size, size, c])
    return (windows, hp, wp)
}

func windowUnpartition(_ windows: MLXArray, _ size: Int, padded: (Int, Int), original: (Int, Int)) -> MLXArray {
    let (hp, wp) = padded
    let nh = hp / size, nw = wp / size
    let b = windows.dim(0) / (nh * nw)
    let x = windows.reshaped([b, nh, nw, size, size, -1]).transposed(0, 1, 3, 2, 4, 5).reshaped([b, hp, wp, -1])
    return hp == original.0 && wp == original.1 ? x : x[0..., 0 ..< original.0, 0 ..< original.1, 0...]
}

final class HieraBlock: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: HieraAttention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: FeedForward
    @ModuleInfo(key: "proj") var proj: Linear?
    let window: Int
    let pool: Bool

    init(dim: Int, dimOut: Int, heads: Int, window: Int, pool: Bool, eps: Float) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        _attn.wrappedValue = HieraAttention(dim: dim, dimOut: dimOut, heads: heads, pool: pool)
        _norm2.wrappedValue = LayerNorm(dimensions: dimOut, eps: eps)
        _mlp.wrappedValue = FeedForward(dimOut, dimOut * 4, dimOut, layers: 2, gelu: true)
        _proj.wrappedValue = dim != dimOut ? Linear(dim, dimOut) : nil
        self.window = window
        self.pool = pool
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        let normed = norm1(x)
        if let proj {
            residual = proj(normed)
            if pool { residual = maxPool2(residual) }
        }
        var h = x.dim(1), w = x.dim(2)
        var y = normed
        var paddedHW = (h, w)
        if window > 0 {
            let (windows, hp, wp) = windowPartition(normed, window)
            y = windows
            paddedHW = (hp, wp)
        }
        y = attn(y)
        var size = window
        if pool {
            size = window / 2
            h = residual.dim(1)
            w = residual.dim(2)
            paddedHW = (h + (size - h % size) % size, w + (size - w % size) % size)
        }
        if window > 0 { y = windowUnpartition(y, size, padded: paddedHW, original: (h, w)) }
        let out = residual + y
        return out + mlp(norm2(out))
    }
}

final class PatchEmbed: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ hidden: Int) {
        _projection.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: hidden, kernelSize: IntOrPair(7), stride: IntOrPair(4), padding: IntOrPair(3))
    }
}

final class HieraBackbone: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ParameterInfo(key: "pos_embed_window") var posEmbedWindow: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [HieraBlock]
    let stageEnds: Set<Int>
    /// The positional embedding at the 256x256 patch grid, computed once after the weights load.
    var gridPE: MLXArray?

    init(_ c: HieraConfig) {
        _patchEmbed.wrappedValue = PatchEmbed(c.hidden)
        // Kept in PyTorch's [1, C, h, w] layout, as in the checkpoint; only read in preparePositions().
        _posEmbed.wrappedValue = MLXArray.zeros([1, c.hidden, c.background[0], c.background[1]])
        _posEmbedWindow.wrappedValue = MLXArray.zeros([1, c.hidden, c.windows[0], c.windows[0]])
        var blocks: [HieraBlock] = []
        var ends: Set<Int> = []
        var total = 0
        for (stage, count) in c.blocksPerStage.enumerated() {
            for index in 0..<count {
                let first = stage > 0 && index == 0
                let dim = first ? c.dims[stage - 1] : c.dims[stage]
                var window = first ? c.windows[stage - 1] : c.windows[stage]
                if c.globalBlocks.contains(total) { window = 0 }
                blocks.append(HieraBlock(
                    dim: dim, dimOut: c.dims[stage], heads: c.heads[stage], window: window, pool: first, eps: c.eps))
                total += 1
            }
            ends.insert(total - 1)
        }
        _blocks.wrappedValue = blocks
        stageEnds = ends
    }

    /// pos_embed resized bicubically to the grid (F.interpolate, align_corners=False) plus the tiled window embedding.
    func preparePositions(grid: Int) {
        let pe = posEmbed.asType(.float32)[0]  // [C, h, w]
        let c = pe.dim(0), h = pe.dim(1), w = pe.dim(2)
        let ry = bicubicMatrix(out: grid, input: h), rx = bicubicMatrix(out: grid, input: w)
        let resized = ry.matmul(pe).matmul(rx.transposed())  // [C, grid, grid]
        let win = posEmbedWindow.asType(.float32)[0]
        let ws = win.dim(1)
        let tiled = tiled(win, repetitions: [1, grid / ws, grid / ws])
        gridPE = (resized + tiled).transposed(1, 2, 0).reshaped([1, grid, grid, c]).asType(.float16)
        eval(gridPE!)
    }

    /// The four stage outputs (stride 4, 8, 16, 32), each [1, H, W, C].
    func callAsFunction(_ pixels: MLXArray) -> [MLXArray] {
        var x = patchEmbed.projection(pixels)
        x = x + (gridPE ?? MLXArray.zeros(like: x))
        var outs: [MLXArray] = []
        for (i, block) in blocks.enumerated() {
            x = block(x)
            if stageEnds.contains(i) { outs.append(x) }
        }
        return outs
    }
}

/// torch's bicubic upsampling as a matrix (A = -0.75, align_corners=False, edge indices clamped).
func bicubicMatrix(out: Int, input: Int) -> MLXArray {
    let a = -0.75
    func near(_ x: Double) -> Double { ((a + 2) * x - (a + 3)) * x * x + 1 }
    func far(_ x: Double) -> Double { ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
    var m = [Float](repeating: 0, count: out * input)
    let scale = Double(input) / Double(out)
    for o in 0..<out {
        let real = (Double(o) + 0.5) * scale - 0.5
        let i0 = Int(real.rounded(.down))
        let t = real - Double(i0)
        let weights = [far(t + 1), near(t), near(1 - t), far(2 - t)]
        for (k, weight) in weights.enumerated() {
            let index = min(max(i0 - 1 + k, 0), input - 1)
            m[o * input + index] += Float(weight)
        }
    }
    return MLXArray(m, [out, input])
}

final class NeckConvs: Module {
    @ModuleInfo(key: "convs") var convs: [Conv2d]

    init(_ channels: [Int]) {
        _convs.wrappedValue = channels.map { Conv2d(inputChannels: $0, outputChannels: 256, kernelSize: IntOrPair(1)) }
    }

    /// Stage outputs (stride 4 ... 32) to FPN levels at stride 4, 8 and 16; top-down fusion at levels 2 and 3 only.
    func callAsFunction(_ xs: [MLXArray]) -> [MLXArray] {
        let l3 = convs[0](xs[3])
        let l2 = convs[1](xs[2]) + upsampleNearest2x(l3)
        return [convs[3](xs[0]), convs[2](xs[1]), l2]
    }
}

func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
    let n = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    return broadcast(x.reshaped([n, h, 1, w, 1, c]), to: [n, h, 2, w, 2, c]).reshaped([n, h * 2, w * 2, c])
}

final class VisionEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: HieraBackbone
    @ModuleInfo(key: "neck") var neck: NeckConvs

    init(_ c: HieraConfig) {
        _backbone.wrappedValue = HieraBackbone(c)
        _neck.wrappedValue = NeckConvs(c.channelList)
    }
}
