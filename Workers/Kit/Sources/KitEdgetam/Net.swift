// SPDX-License-Identifier: MIT
// EdgeTAM image path, ported from facebookresearch/EdgeTAM (sam2/): timm repvit_m1 trunk (RepViT, legacy
// RepVggDw), FpnNeck, SAM PromptEncoder and two-way MaskDecoder. Property and @ModuleInfo keys follow the
// checkpoint's names exactly. Weights are already channel-last (MLX layout) in the mlx-community checkpoint.
import Foundation
import MLX
import MLXFast
import MLXNN

/// PyTorch BatchNorm2d in eval mode (running statistics, eps 1e-5).
final class EvalBatchNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    init(_ channels: Int) {
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
        _runningMean.wrappedValue = MLXArray.zeros([channels])
        _runningVar.wrappedValue = MLXArray.ones([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) * rsqrt(runningVar + 1e-5) * weight + bias
    }
}

/// timm ConvNorm: a bias-free convolution followed by BatchNorm.
final class ConvNorm: Module, UnaryLayer {
    @ModuleInfo(key: "c") var c: Conv2d
    @ModuleInfo(key: "bn") var bn: EvalBatchNorm

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int = 1, stride: Int = 1, padding: Int = 0, groups: Int = 1) {
        _c.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: IntOrPair(kernel),
            stride: IntOrPair(stride), padding: IntOrPair(padding), groups: groups, bias: false)
        _bn.wrappedValue = EvalBatchNorm(outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { bn(c(x)) }
}

/// Channel-last LayerNorm2d from sam2_utils (eps 1e-6 over channels).
final class LayerNorm2d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(_ channels: Int) {
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let u = mean(x, axis: -1, keepDims: true)
        let s = mean(square(x - u), axis: -1, keepDims: true)
        return weight * ((x - u) * rsqrt(s + 1e-6)) + bias
    }
}

/// A plain learned table (nn.Embedding or a bare parameter in the checkpoint), key "weight".
final class ParamTable: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ rows: Int, _ cols: Int) {
        _weight.wrappedValue = MLXArray.zeros([rows, cols])
    }
}

/// timm SqueezeExcite with a ReLU bottleneck and sigmoid gate; channel-last.
final class SqueezeExcite: Module {
    @ModuleInfo(key: "fc1") var fc1: Conv2d
    @ModuleInfo(key: "fc2") var fc2: Conv2d

    init(_ channels: Int, reduced: Int) {
        _fc1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: reduced, kernelSize: IntOrPair(1))
        _fc2.wrappedValue = Conv2d(inputChannels: reduced, outputChannels: channels, kernelSize: IntOrPair(1))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The spatial mean is summed in float32: a float16 sum over 4096 to 65536 positions overflows.
        let pooled = mean(x.asType(.float32), axes: [1, 2], keepDims: true).asType(x.dtype)
        return x * sigmoid(fc2(relu(fc1(pooled))))
    }
}

/// RepVggDw in legacy mode: depthwise kxk ConvNorm + depthwise 1x1 ConvNorm + identity.
final class RepVggDw: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: ConvNorm
    @ModuleInfo(key: "conv1") var conv1: ConvNorm

    init(_ dim: Int, kernel: Int) {
        _conv.wrappedValue = ConvNorm(dim, dim, kernel: kernel, padding: (kernel - 1) / 2, groups: dim)
        _conv1.wrappedValue = ConvNorm(dim, dim, kernel: 1, groups: dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) + conv1(x) + x }
}

/// RepViT channel mixer: 1x1 expand, GELU, 1x1 project.
final class RepVitMlp: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: ConvNorm
    @ModuleInfo(key: "conv2") var conv2: ConvNorm

    init(_ inDim: Int, _ hidden: Int) {
        _conv1.wrappedValue = ConvNorm(inDim, hidden)
        _conv2.wrappedValue = ConvNorm(hidden, inDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(gelu(conv1(x))) }
}

final class RepViTBlock: Module, UnaryLayer {
    @ModuleInfo(key: "token_mixer") var tokenMixer: RepVggDw
    @ModuleInfo(key: "se") var se: SqueezeExcite?
    @ModuleInfo(key: "channel_mixer") var channelMixer: RepVitMlp

    init(_ dim: Int, useSE: Bool) {
        _tokenMixer.wrappedValue = RepVggDw(dim, kernel: 3)
        _se.wrappedValue = useSE ? SqueezeExcite(dim, reduced: makeDivisible(dim / 4)) : nil
        _channelMixer.wrappedValue = RepVitMlp(dim, dim * 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = tokenMixer(x)
        if let se { y = se(y) }
        return y + channelMixer(y)
    }
}

/// timm make_divisible(v, 8) with round_limit 0, as used for the SE reduction width.
func makeDivisible(_ v: Int) -> Int { max(8, (v + 4) / 8 * 8) }

final class RepViTDownsample: Module, UnaryLayer {
    @ModuleInfo(key: "pre_block") var preBlock: RepViTBlock
    @ModuleInfo(key: "spatial_downsample") var spatialDownsample: ConvNorm
    @ModuleInfo(key: "channel_downsample") var channelDownsample: ConvNorm
    @ModuleInfo(key: "ffn") var ffn: RepVitMlp

    init(inDim: Int, outDim: Int) {
        _preBlock.wrappedValue = RepViTBlock(inDim, useSE: false)
        _spatialDownsample.wrappedValue = ConvNorm(inDim, inDim, kernel: 3, stride: 2, padding: 1, groups: inDim)
        _channelDownsample.wrappedValue = ConvNorm(inDim, outDim)
        _ffn.wrappedValue = RepVitMlp(outDim, outDim * 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = channelDownsample(spatialDownsample(preBlock(x)))
        return ffn(y) + y
    }
}

final class RepViTStage: Module, UnaryLayer {
    @ModuleInfo(key: "downsample") var downsample: RepViTDownsample?
    @ModuleInfo(key: "blocks") var blocks: [RepViTBlock]

    init(inDim: Int, outDim: Int, depth: Int, hasDownsample: Bool) {
        _downsample.wrappedValue = hasDownsample ? RepViTDownsample(inDim: inDim, outDim: outDim) : nil
        // timm alternates SE on every block, starting with the first one of each stage.
        _blocks.wrappedValue = (0..<depth).map { RepViTBlock(outDim, useSE: $0 % 2 == 0) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        if let downsample { y = downsample(y) }
        for block in blocks { y = block(y) }
        return y
    }
}

final class RepViTStem: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: ConvNorm
    @ModuleInfo(key: "conv2") var conv2: ConvNorm

    init(_ inChannels: Int, _ outChannels: Int) {
        _conv1.wrappedValue = ConvNorm(inChannels, outChannels / 2, kernel: 3, stride: 2, padding: 1)
        _conv2.wrappedValue = ConvNorm(outChannels / 2, outChannels, kernel: 3, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(gelu(conv1(x))) }
}

/// The RepViT-M1 body. `features` returns the four stage outputs (stride 4, 8, 16, 32).
final class RepViTBody: Module {
    @ModuleInfo(key: "stem") var stem: RepViTStem
    @ModuleInfo(key: "stages_0") var stages0: RepViTStage
    @ModuleInfo(key: "stages_1") var stages1: RepViTStage
    @ModuleInfo(key: "stages_2") var stages2: RepViTStage
    @ModuleInfo(key: "stages_3") var stages3: RepViTStage

    override init() {
        _stem.wrappedValue = RepViTStem(3, 48)
        _stages0.wrappedValue = RepViTStage(inDim: 48, outDim: 48, depth: 2, hasDownsample: false)
        _stages1.wrappedValue = RepViTStage(inDim: 48, outDim: 96, depth: 2, hasDownsample: true)
        _stages2.wrappedValue = RepViTStage(inDim: 96, outDim: 192, depth: 14, hasDownsample: true)
        _stages3.wrappedValue = RepViTStage(inDim: 192, outDim: 384, depth: 2, hasDownsample: true)
    }

    func features(_ x: MLXArray) -> [MLXArray] {
        let x0 = stages0(stem(x))
        let x1 = stages1(x0)
        let x2 = stages2(x1)
        let x3 = stages3(x2)
        return [x0, x1, x2, x3]
    }
}

final class Trunk: Module {
    @ModuleInfo(key: "body") var body: RepViTBody

    override init() {
        _body.wrappedValue = RepViTBody()
    }
}

final class NeckConv: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ inChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: 256, kernelSize: IntOrPair(1))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// FpnNeck with fpn_top_down_levels [2, 3] and nearest upsampling. Position encodings are not needed here.
final class FpnNeck: Module {
    @ModuleInfo(key: "convs") var convs: [NeckConv]

    override init() {
        _convs.wrappedValue = [384, 192, 96, 48].map { NeckConv($0) }
    }

    /// `xs` are the stage outputs in order (stride 4 ... 32). Returns levels 0...2 (stride 4, 8, 16); the stride-32 level is dropped (scalp 1).
    func callAsFunction(_ xs: [MLXArray]) -> [MLXArray] {
        let l3 = convs[0](xs[3])
        let l2 = convs[1](xs[2]) + upsampleNearest2x(l3)
        let l1 = convs[2](xs[1])
        let l0 = convs[3](xs[0])
        return [l0, l1, l2]
    }
}

func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
    let n = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    let expanded = broadcast(x.reshaped([n, h, 1, w, 1, c]), to: [n, h, 2, w, 2, c])
    return expanded.reshaped([n, h * 2, w * 2, c])
}

final class ImageEncoder: Module {
    @ModuleInfo(key: "trunk") var trunk: Trunk
    @ModuleInfo(key: "neck") var neck: FpnNeck

    override init() {
        _trunk.wrappedValue = Trunk()
        _neck.wrappedValue = FpnNeck()
    }
}

/// Random-Fourier positional encoding (sam2 PositionEmbeddingRandom), with the checkpoint's gaussian matrix.
final class GaussianPE: Module {
    @ParameterInfo(key: "positional_encoding_gaussian_matrix") var matrix: MLXArray

    override init() {
        _matrix.wrappedValue = MLXArray.zeros([2, 128])
    }

    /// Coordinates in [0, 1], shape [N, 2], to [N, 256].
    func encode(_ coords: MLXArray) -> MLXArray {
        let t = ((coords.asType(.float32) * 2 - 1).matmul(matrix.asType(.float32))) * (2 * Float.pi)
        return concatenated([sin(t), cos(t)], axis: -1)
    }

    /// The dense positional encoding over a h x w grid, [1, h, w, 256].
    func denseGrid(h: Int, w: Int) -> MLXArray {
        var coords = [Float](repeating: 0, count: h * w * 2)
        for i in 0..<h {
            for j in 0..<w {
                let k = (i * w + j) * 2
                coords[k] = (Float(j) + 0.5) / Float(w)
                coords[k + 1] = (Float(i) + 0.5) / Float(h)
            }
        }
        return encode(MLXArray(coords, [h * w, 2])).reshaped([1, h, w, 256])
    }
}

/// A parameter-free activation, used as a placeholder in the sequential blocks below.
final class Activation: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(x) }
}

/// Sequential blocks are arrays so that the checkpoint's numbered keys (0, 1, 3, ...) line up with the indices.
/// The mask-input branch of the prompt encoder is loaded but not run by this op.
func makeMaskDownscaling() -> [Module] {
    [
        Conv2d(inputChannels: 1, outputChannels: 4, kernelSize: IntOrPair(2), stride: IntOrPair(2)),
        LayerNorm2d(4), Activation(),
        Conv2d(inputChannels: 4, outputChannels: 16, kernelSize: IntOrPair(2), stride: IntOrPair(2)),
        LayerNorm2d(16), Activation(),
        Conv2d(inputChannels: 16, outputChannels: 256, kernelSize: IntOrPair(1)),
    ]
}

final class PromptEncoder: Module {
    static let imageSize: Float = 1024
    static let embeddingSize = 64

    @ModuleInfo(key: "pe_layer") var peLayer: GaussianPE
    @ModuleInfo(key: "point_embeddings") var pointEmbeddings: [ParamTable]
    @ModuleInfo(key: "not_a_point_embed") var notAPointEmbed: ParamTable
    @ModuleInfo(key: "mask_downscaling") var maskDownscaling: [Module]
    @ModuleInfo(key: "no_mask_embed") var noMaskEmbed: ParamTable

    override init() {
        _peLayer.wrappedValue = GaussianPE()
        _pointEmbeddings.wrappedValue = (0..<4).map { _ in ParamTable(1, 256) }
        _notAPointEmbed.wrappedValue = ParamTable(1, 256)
        _maskDownscaling.wrappedValue = makeMaskDownscaling()
        _noMaskEmbed.wrappedValue = ParamTable(1, 256)
    }

    /// Sparse prompt tokens, [1, N, 256]. Each prompt is (x, y) normalized to [0, 1] with a label:
    /// 0 background point, 1 foreground point, 2 and 3 box corners. The maker's padding row (label -1) is appended.
    func sparse(_ prompts: [(x: Float, y: Float, label: Int32)]) -> MLXArray {
        let rows = prompts + [(x: Float(0), y: Float(0), label: Int32(-1))]
        var coords = [Float](repeating: 0, count: rows.count * 2)
        var valid = [Float](repeating: 0, count: rows.count)
        var table = [Int32](repeating: 4, count: rows.count)
        for (i, row) in rows.enumerated() where row.label >= 0 {
            // The maker shifts each point by half a pixel of the 1024 frame, then divides by 1024.
            coords[2 * i] = row.x + 0.5 / PromptEncoder.imageSize
            coords[2 * i + 1] = row.y + 0.5 / PromptEncoder.imageSize
            valid[i] = 1
            table[i] = row.label
        }
        let encoded = peLayer.encode(MLXArray(coords, [rows.count, 2])) * MLXArray(valid, [rows.count, 1])
        let labelTable = concatenated(pointEmbeddings.map { $0.weight } + [notAPointEmbed.weight], axis: 0)
        let labels = take(labelTable, MLXArray(table), axis: 0)
        return (encoded + labels).expandedDimensions(axis: 0).asType(.float16)
    }

    /// The dense prompt (no mask input): no_mask_embed broadcast over the 64 x 64 embedding, [1, 64, 64, 256].
    func denseNoMask() -> MLXArray {
        let s = PromptEncoder.embeddingSize
        return broadcast(noMaskEmbed.weight.reshaped([1, 1, 1, 256]), to: [1, s, s, 256])
    }
}

final class SAMAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int
    let internalDim: Int

    init(dim: Int, heads: Int, downsample: Int = 1) {
        self.heads = heads
        internalDim = dim / downsample
        _qProj.wrappedValue = Linear(dim, internalDim)
        _kProj.wrappedValue = Linear(dim, internalDim)
        _vProj.wrappedValue = Linear(dim, internalDim)
        _outProj.wrappedValue = Linear(internalDim, dim)
    }

    func callAsFunction(q: MLXArray, k: MLXArray, v: MLXArray) -> MLXArray {
        let b = q.dim(0), nq = q.dim(1), nk = k.dim(1)
        let headDim = internalDim / heads
        let qh = qProj(q).reshaped([b, nq, heads, headDim]).transposed(0, 2, 1, 3)
        let kh = kProj(k).reshaped([b, nk, heads, headDim]).transposed(0, 2, 1, 3)
        let vh = vProj(v).reshaped([b, nk, heads, headDim]).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return outProj(out.transposed(0, 2, 1, 3).reshaped([b, nq, internalDim]))
    }
}

/// Linear layers with ReLU between them; the last layer optionally followed by a sigmoid.
final class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [Linear]
    let sigmoidOutput: Bool

    init(_ inDim: Int, _ hidden: Int, _ outDim: Int, layers count: Int, sigmoidOutput: Bool = false) {
        let inputs = [inDim] + Array(repeating: hidden, count: count - 1)
        let outputs = Array(repeating: hidden, count: count - 1) + [outDim]
        _layers.wrappedValue = zip(inputs, outputs).map { Linear($0.0, $0.1) }
        self.sigmoidOutput = sigmoidOutput
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for (i, layer) in layers.enumerated() {
            y = i < layers.count - 1 ? relu(layer(y)) : layer(y)
        }
        return sigmoidOutput ? sigmoid(y) : y
    }
}

final class TwoWayBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SAMAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "cross_attn_token_to_image") var tokenToImage: SAMAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "norm4") var norm4: LayerNorm
    @ModuleInfo(key: "cross_attn_image_to_token") var imageToToken: SAMAttention
    let skipFirstPE: Bool

    init(skipFirstPE: Bool) {
        let d = 256
        _selfAttn.wrappedValue = SAMAttention(dim: d, heads: 8)
        _norm1.wrappedValue = LayerNorm(dimensions: d)
        _tokenToImage.wrappedValue = SAMAttention(dim: d, heads: 8, downsample: 2)
        _norm2.wrappedValue = LayerNorm(dimensions: d)
        _mlp.wrappedValue = MLP(d, 2048, d, layers: 2)
        _norm3.wrappedValue = LayerNorm(dimensions: d)
        _norm4.wrappedValue = LayerNorm(dimensions: d)
        _imageToToken.wrappedValue = SAMAttention(dim: d, heads: 8, downsample: 2)
        self.skipFirstPE = skipFirstPE
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, queryPE: MLXArray, keyPE: MLXArray) -> (MLXArray, MLXArray) {
        var q = queries
        if skipFirstPE {
            q = selfAttn(q: q, k: q, v: q)
        } else {
            let qp = q + queryPE
            q = q + selfAttn(q: qp, k: qp, v: q)
        }
        q = norm1(q)
        q = q + tokenToImage(q: q + queryPE, k: keys + keyPE, v: keys)
        q = norm2(q)
        q = q + mlp(q)
        q = norm3(q)
        let keysOut = norm4(keys + imageToToken(q: keys + keyPE, k: q + queryPE, v: q))
        return (q, keysOut)
    }
}

final class TwoWayTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [TwoWayBlock]
    @ModuleInfo(key: "final_attn_token_to_image") var finalAttn: SAMAttention
    @ModuleInfo(key: "norm_final_attn") var normFinal: LayerNorm

    override init() {
        _layers.wrappedValue = (0..<2).map { TwoWayBlock(skipFirstPE: $0 == 0) }
        _finalAttn.wrappedValue = SAMAttention(dim: 256, heads: 8, downsample: 2)
        _normFinal.wrappedValue = LayerNorm(dimensions: 256)
    }

    /// image, pe: [1, HW, 256]; tokens: [1, T, 256]. Returns the processed tokens and image keys.
    func callAsFunction(image: MLXArray, pe: MLXArray, tokens: MLXArray) -> (MLXArray, MLXArray) {
        var queries = tokens
        var keys = image
        for layer in layers {
            (queries, keys) = layer(queries: queries, keys: keys, queryPE: tokens, keyPE: pe)
        }
        let attn = finalAttn(q: queries + tokens, k: keys + pe, v: keys)
        return (normFinal(queries + attn), keys)
    }
}

/// output_upscaling: ConvTranspose, LayerNorm2d, GELU, ConvTranspose, GELU (indices 0 to 4).
final class MaskDecoder: Module {
    @ModuleInfo(key: "transformer") var transformer: TwoWayTransformer
    @ModuleInfo(key: "iou_token") var iouToken: ParamTable
    @ModuleInfo(key: "mask_tokens") var maskTokens: ParamTable
    @ModuleInfo(key: "obj_score_token") var objScoreToken: ParamTable
    @ModuleInfo(key: "output_upscaling") var upscaling: [Module]
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d
    @ModuleInfo(key: "output_hypernetworks_mlps") var hypernetworks: [MLP]
    @ModuleInfo(key: "iou_prediction_head") var iouHead: MLP
    @ModuleInfo(key: "pred_obj_score_head") var objScoreHead: MLP

    override init() {
        _transformer.wrappedValue = TwoWayTransformer()
        _iouToken.wrappedValue = ParamTable(1, 256)
        _maskTokens.wrappedValue = ParamTable(4, 256)
        _objScoreToken.wrappedValue = ParamTable(1, 256)
        _upscaling.wrappedValue = [
            ConvTransposed2d(inputChannels: 256, outputChannels: 64, kernelSize: IntOrPair(2), stride: IntOrPair(2)),
            LayerNorm2d(64), Activation(),
            ConvTransposed2d(inputChannels: 64, outputChannels: 32, kernelSize: IntOrPair(2), stride: IntOrPair(2)),
            Activation(),
        ]
        _convS0.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 32, kernelSize: IntOrPair(1))
        _convS1.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 64, kernelSize: IntOrPair(1))
        _hypernetworks.wrappedValue = (0..<4).map { _ in MLP(256, 256, 32, layers: 3) }
        _iouHead.wrappedValue = MLP(256, 256, 4, layers: 3, sigmoidOutput: true)
        _objScoreHead.wrappedValue = MLP(256, 256, 1, layers: 3)
    }

    /// Low-resolution mask logits [1, 4, 256, 256] and IoU predictions [1, 4]. Mask 0 is the single-mask output, 1...3 the multimask outputs.
    /// The high-resolution skips are already projected by conv_s0 and conv_s1 (as the maker does in forward_image).
    func predict(
        embedding: MLXArray, highRes0: MLXArray, highRes1: MLXArray,
        dense: MLXArray, sparse: MLXArray, pe: MLXArray
    ) -> (masks: MLXArray, iou: MLXArray) {
        // Output tokens: object score, IoU, then four mask tokens, as in the maker's predict_masks.
        let outputTokens = concatenated([objScoreToken.weight, iouToken.weight, maskTokens.weight], axis: 0)
            .expandedDimensions(axis: 0)
        let tokens = concatenated([outputTokens, sparse], axis: 1)
        let src = embedding + dense
        let h = src.dim(1), w = src.dim(2), c = src.dim(3)
        let (hs, keys) = transformer(
            image: src.reshaped([1, h * w, c]), pe: pe.reshaped([1, h * w, c]), tokens: tokens)
        let iouTokenOut = hs[0..., 1, 0...]
        let maskTokensOut = hs[0..., 2 ..< 6, 0...]
        let feature = keys.reshaped([1, h, w, c])

        let up0 = upscaling[0] as! ConvTransposed2d, norm1 = upscaling[1] as! LayerNorm2d
        let up3 = upscaling[3] as! ConvTransposed2d
        var up = gelu(norm1(up0(feature) + highRes1))
        up = gelu(up3(up) + highRes0)
        let flat = up.reshaped([1, up.dim(1) * up.dim(2), up.dim(3)])
        let tokensPerMask = (0..<4).map { hypernetworks[$0](maskTokensOut[0..., $0, 0...]) }
        let hyper = stacked(tokensPerMask, axis: 1)
        let masks = hyper.matmul(flat.transposed(0, 2, 1)).reshaped([1, 4, up.dim(1), up.dim(2)])
        let iou = iouHead(iouTokenOut)
        return (masks, iou)
    }
}

/// The EdgeTAM image model: image encoder, prompt encoder and mask decoder, with the checkpoint's key names.
final class EdgeTAMImage: Module {
    @ModuleInfo(key: "image_encoder") var imageEncoder: ImageEncoder
    @ModuleInfo(key: "sam_prompt_encoder") var promptEncoder: PromptEncoder
    @ModuleInfo(key: "sam_mask_decoder") var maskDecoder: MaskDecoder
    @ParameterInfo(key: "no_mem_embed") var noMemEmbed: MLXArray

    override init() {
        _imageEncoder.wrappedValue = ImageEncoder()
        _promptEncoder.wrappedValue = PromptEncoder()
        _maskDecoder.wrappedValue = MaskDecoder()
        _noMemEmbed.wrappedValue = MLXArray.zeros([1, 1, 256])
    }

    /// Only the image path loads. Video memory, perceiver and object-pointer tensors are skipped.
    static func keepsForImagePath(_ key: String) -> String? {
        if key.hasPrefix("image_encoder.") || key.hasPrefix("sam_mask_decoder.") || key.hasPrefix("sam_prompt_encoder.") || key == "no_mem_embed" {
            return key
        }
        return nil
    }

    /// Image embedding [1, 64, 64, 256] (with no_mem_embed added, as the maker's predictor does) and the two high-resolution skips.
    func encode(_ pixels: MLXArray) -> (embedding: MLXArray, highRes0: MLXArray, highRes1: MLXArray) {
        let levels = imageEncoder.neck(imageEncoder.trunk.body.features(pixels))
        let embedding = levels[2] + noMemEmbed.reshaped([1, 1, 1, 256])
        return (embedding, maskDecoder.convS0(levels[0]), maskDecoder.convS1(levels[1]))
    }
}

/// Bilinear resize as a matrix (align_corners=False, no antialias), so that Ry * mask * Rx^T matches F.interpolate.
func bilinearMatrix(out: Int, input: Int) -> MLXArray {
    var m = [Float](repeating: 0, count: out * input)
    let scale = Double(input) / Double(out)
    for o in 0..<out {
        let src = max(0, (Double(o) + 0.5) * scale - 0.5)
        let i0 = min(Int(src.rounded(.down)), input - 1)
        let i1 = min(i0 + 1, input - 1)
        let lambda = Float(src - Double(i0))
        m[o * input + i0] += 1 - lambda
        m[o * input + i1] += lambda
    }
    return MLXArray(m, [out, input])
}
