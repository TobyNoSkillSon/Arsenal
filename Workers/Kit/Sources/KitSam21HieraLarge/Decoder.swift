// SPDX-License-Identifier: MIT
// SAM 2.1 prompt encoder and two-way mask decoder, ported from transformers models/sam2/modeling_sam2.py
// (Sam2PromptEncoder, Sam2TwoWayTransformer, Sam2MaskDecoder) with the checkpoint's key names. The same network as
// the EdgeTAM runner's (which follows the original sam2 key names). Channels-last.
import Foundation
import MLX
import MLXFast
import MLXNN

/// A learned table (nn.Embedding), key "weight".
final class Table: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    init(_ rows: Int, _ cols: Int) { _weight.wrappedValue = MLXArray.zeros([rows, cols]) }
}

/// Channel-last LayerNorm over channels, eps 1e-6 (Sam2LayerNorm channels_first in the maker's layout).
final class ChannelNorm: Module, UnaryLayer {
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

/// Random-Fourier positional encoding (Sam2PositionalEmbedding), key "positional_embedding" [2, 128].
final class FourierPE: Module {
    @ParameterInfo(key: "positional_embedding") var matrix: MLXArray
    override init() { _matrix.wrappedValue = MLXArray.zeros([2, 128]) }

    /// Coordinates in [0, 1], [N, 2], to [N, 256] (float32).
    func encode(_ coords: MLXArray) -> MLXArray {
        let t = (coords.asType(.float32) * 2 - 1).matmul(matrix.asType(.float32)) * (2 * Float.pi)
        return concatenated([sin(t), cos(t)], axis: -1)
    }

    /// The image-wide encoding over an n x n grid of cell centres, [1, n, n, 256].
    func grid(_ n: Int) -> MLXArray {
        var coords = [Float](repeating: 0, count: n * n * 2)
        for i in 0..<n {
            for j in 0..<n {
                coords[(i * n + j) * 2] = (Float(j) + 0.5) / Float(n)
                coords[(i * n + j) * 2 + 1] = (Float(i) + 0.5) / Float(n)
            }
        }
        return encode(MLXArray(coords, [n * n, 2])).reshaped([1, n, n, 256])
    }
}

/// One prompt: (x, y) as fractions of the image, label 0 background, 1 foreground, 2 and 3 box corners.
typealias Prompt = (x: Float, y: Float, label: Int32)

final class PromptEncoder: Module {
    @ModuleInfo(key: "shared_embedding") var shared: FourierPE
    @ModuleInfo(key: "point_embed") var pointEmbed: Table
    @ModuleInfo(key: "not_a_point_embed") var notAPoint: Table
    @ModuleInfo(key: "no_mask_embed") var noMask: Table

    override init() {
        _shared.wrappedValue = FourierPE()
        _pointEmbed.wrappedValue = Table(4, 256)
        _notAPoint.wrappedValue = Table(1, 256)
        _noMask.wrappedValue = Table(1, 256)
    }

    /// Sparse tokens [1, N + 1, 256]: each prompt shifted half a pixel of the 1024 frame, plus the padding token
    /// (not_a_point_embed) the maker appends.
    func sparse(_ prompts: [Prompt]) -> MLXArray {
        let n = prompts.count + 1
        var coords = [Float](repeating: 0, count: n * 2)
        var valid = [Float](repeating: 0, count: n)
        var labels = [Int32](repeating: 4, count: n)
        for (i, p) in prompts.enumerated() {
            coords[2 * i] = p.x + 0.5 / 1024
            coords[2 * i + 1] = p.y + 0.5 / 1024
            valid[i] = 1
            labels[i] = p.label
        }
        let encoded = shared.encode(MLXArray(coords, [n, 2])) * MLXArray(valid, [n, 1])
        let table = concatenated([pointEmbed.weight, notAPoint.weight], axis: 0).asType(.float32)
        return (encoded + take(table, MLXArray(labels), axis: 0)).expandedDimensions(axis: 0).asType(.float16)
    }
}

final class DecoderAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let heads = 8
    let inner: Int

    init(downsample: Int) {
        inner = 256 / downsample
        _qProj.wrappedValue = Linear(256, inner)
        _kProj.wrappedValue = Linear(256, inner)
        _vProj.wrappedValue = Linear(256, inner)
        _oProj.wrappedValue = Linear(inner, 256)
    }

    func callAsFunction(q: MLXArray, k: MLXArray, v: MLXArray) -> MLXArray {
        let b = q.dim(0), nq = q.dim(1), nk = k.dim(1), hd = inner / heads
        let qh = qProj(q).reshaped([b, nq, heads, hd]).transposed(0, 2, 1, 3)
        let kh = kProj(k).reshaped([b, nk, heads, hd]).transposed(0, 2, 1, 3)
        let vh = vProj(v).reshaped([b, nk, heads, hd]).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: 1 / Float(hd).squareRoot(), mask: .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped([b, nq, inner]))
    }
}

final class TwoWayBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DecoderAttention
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "cross_attn_token_to_image") var tokenToImage: DecoderAttention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: FeedForward
    @ModuleInfo(key: "layer_norm3") var norm3: LayerNorm
    @ModuleInfo(key: "layer_norm4") var norm4: LayerNorm
    @ModuleInfo(key: "cross_attn_image_to_token") var imageToToken: DecoderAttention
    let skipFirstPE: Bool

    init(skipFirstPE: Bool) {
        _selfAttn.wrappedValue = DecoderAttention(downsample: 1)
        _norm1.wrappedValue = LayerNorm(dimensions: 256)
        _tokenToImage.wrappedValue = DecoderAttention(downsample: 2)
        _norm2.wrappedValue = LayerNorm(dimensions: 256)
        _mlp.wrappedValue = FeedForward(256, 2048, 256, layers: 2)
        _norm3.wrappedValue = LayerNorm(dimensions: 256)
        _norm4.wrappedValue = LayerNorm(dimensions: 256)
        _imageToToken.wrappedValue = DecoderAttention(downsample: 2)
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
        q = norm2(q + tokenToImage(q: q + queryPE, k: keys + keyPE, v: keys))
        q = norm3(q + mlp(q))
        let k = norm4(keys + imageToToken(q: keys + keyPE, k: q + queryPE, v: q))
        return (q, k)
    }
}

final class TwoWayTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [TwoWayBlock]
    @ModuleInfo(key: "final_attn_token_to_image") var finalAttn: DecoderAttention
    @ModuleInfo(key: "layer_norm_final_attn") var normFinal: LayerNorm

    override init() {
        _layers.wrappedValue = (0..<2).map { TwoWayBlock(skipFirstPE: $0 == 0) }
        _finalAttn.wrappedValue = DecoderAttention(downsample: 2)
        _normFinal.wrappedValue = LayerNorm(dimensions: 256)
    }

    func callAsFunction(image: MLXArray, pe: MLXArray, tokens: MLXArray) -> (MLXArray, MLXArray) {
        var q = tokens, k = image
        for layer in layers { (q, k) = layer(queries: q, keys: k, queryPE: tokens, keyPE: pe) }
        return (normFinal(q + finalAttn(q: q + tokens, k: k + pe, v: k)), k)
    }
}

final class MaskDecoder: Module {
    @ModuleInfo(key: "iou_token") var iouToken: Table
    @ModuleInfo(key: "mask_tokens") var maskTokens: Table
    @ModuleInfo(key: "obj_score_token") var objScoreToken: Table
    @ModuleInfo(key: "transformer") var transformer: TwoWayTransformer
    @ModuleInfo(key: "upscale_conv1") var up1: ConvTransposed2d
    @ModuleInfo(key: "upscale_conv2") var up2: ConvTransposed2d
    @ModuleInfo(key: "upscale_layer_norm") var upNorm: ChannelNorm
    @ModuleInfo(key: "output_hypernetworks_mlps") var hypernetworks: [FeedForward]
    @ModuleInfo(key: "iou_prediction_head") var iouHead: FeedForward
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d

    override init() {
        _iouToken.wrappedValue = Table(1, 256)
        _maskTokens.wrappedValue = Table(4, 256)
        _objScoreToken.wrappedValue = Table(1, 256)
        _transformer.wrappedValue = TwoWayTransformer()
        _up1.wrappedValue = ConvTransposed2d(inputChannels: 256, outputChannels: 64, kernelSize: IntOrPair(2), stride: IntOrPair(2))
        _up2.wrappedValue = ConvTransposed2d(inputChannels: 64, outputChannels: 32, kernelSize: IntOrPair(2), stride: IntOrPair(2))
        _upNorm.wrappedValue = ChannelNorm(64)
        _hypernetworks.wrappedValue = (0..<4).map { _ in FeedForward(256, 256, 32, layers: 3) }
        _iouHead.wrappedValue = FeedForward(256, 256, 4, layers: 3, sigmoidOutput: true)
        _convS0.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 32, kernelSize: IntOrPair(1))
        _convS1.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 64, kernelSize: IntOrPair(1))
    }

    /// All four mask logits [B, 4, 256, 256] and IoU predictions [B, 4] (mask 0 single-mask, 1...3 multimask).
    /// `sparse` is [B, T, 256]: B prompt sets against the same image.
    func callAsFunction(image: Encoded, dense: MLXArray, sparse: MLXArray, pe: MLXArray) -> (masks: MLXArray, iou: MLXArray) {
        let b = sparse.dim(0)
        let output = concatenated([objScoreToken.weight, iouToken.weight, maskTokens.weight], axis: 0)
        let tokens = concatenated([broadcast(output.expandedDimensions(axis: 0), to: [b, 6, 256]), sparse], axis: 1)
        let src = image.embedding + dense
        let h = src.dim(1), w = src.dim(2)
        let flatImage = broadcast(src.reshaped([1, h * w, 256]), to: [b, h * w, 256])
        let flatPE = broadcast(pe.reshaped([1, h * w, 256]), to: [b, h * w, 256])
        let (hs, keys) = transformer(image: flatImage, pe: flatPE, tokens: tokens)
        let feature = keys.reshaped([b, h, w, 256])
        var up = gelu(upNorm(up1(feature) + image.highRes1))
        up = gelu(up2(up) + image.highRes0)
        let flat = up.reshaped([b, up.dim(1) * up.dim(2), up.dim(3)])
        let hyper = stacked((0..<4).map { hypernetworks[$0](hs[0..., 2 + $0, 0...]) }, axis: 1)  // [B, 4, 32]
        let masks = hyper.matmul(flat.transposed(0, 2, 1)).reshaped([b, 4, up.dim(1), up.dim(2)])
        return (masks, iouHead(hs[0..., 1, 0...]))
    }
}

/// The encoded image: the 64x64 embedding (with no_memory_embedding added) and the two projected high-resolution skips.
struct Encoded {
    let embedding: MLXArray
    let highRes0: MLXArray
    let highRes1: MLXArray
}

/// SAM 2.1's image predictor path with the checkpoint's names. Video memory, object pointers and the mask-prompt
/// branch are not loaded.
final class Sam2Image: Module {
    @ModuleInfo(key: "vision_encoder") var visionEncoder: VisionEncoder
    @ModuleInfo(key: "prompt_encoder") var promptEncoder: PromptEncoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: MaskDecoder
    @ModuleInfo(key: "shared_image_embedding") var sharedImageEmbedding: FourierPE
    @ParameterInfo(key: "no_memory_embedding") var noMemory: MLXArray
    var imagePE: MLXArray?

    init(_ c: HieraConfig) {
        _visionEncoder.wrappedValue = VisionEncoder(c)
        _promptEncoder.wrappedValue = PromptEncoder()
        _maskDecoder.wrappedValue = MaskDecoder()
        _sharedImageEmbedding.wrappedValue = FourierPE()
        _noMemory.wrappedValue = MLXArray.zeros([1, 1, 256])
    }

    /// Keys this path uses, with PyTorch conv weights turned channels-last. Everything else is dropped.
    static func prepare(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        let keep = ["vision_encoder.", "mask_decoder.", "prompt_encoder.shared_embedding.", "prompt_encoder.point_embed.",
                    "prompt_encoder.not_a_point_embed.", "prompt_encoder.no_mask_embed.", "shared_image_embedding.", "no_memory_embedding"]
        for (key, value) in weights {
            guard keep.contains(where: { key.hasPrefix($0) }), !key.hasPrefix("mask_decoder.pred_obj_score_head.") else { continue }
            if key.hasSuffix(".weight") && value.ndim == 4 {
                let transposed = key.hasPrefix("mask_decoder.upscale_conv")
                out[key] = transposed ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }

    func prepare() {
        visionEncoder.backbone.preparePositions(grid: 256)
        imagePE = sharedImageEmbedding.grid(64).asType(.float16)
        eval(imagePE!)
    }

    func encode(_ pixels: MLXArray) -> Encoded {
        let levels = visionEncoder.neck(visionEncoder.backbone(pixels))
        return Encoded(
            embedding: levels[2] + noMemory.reshaped([1, 1, 1, 256]),
            highRes0: maskDecoder.convS0(levels[0]), highRes1: maskDecoder.convS1(levels[1]))
    }

    func dense() -> MLXArray {
        broadcast(promptEncoder.noMask.weight.reshaped([1, 1, 1, 256]), to: [1, 64, 64, 256])
    }
}
