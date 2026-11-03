// SPDX-License-Identifier: MIT
// ViTPose (usyd-community/vitpose-base-simple). Ported from transformers' modeling_vitpose.py and
// modeling_vitpose_backbone.py (network) and image_processing_vitpose.py (crop, heatmap decoding).
// Input: the whole photo is treated as one person box, cropped to 256x192 as the processor does.
import Foundation
import MLX
import MLXFast
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.pose"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["estimate_pose"]

// Geometry from config.json and preprocessor_config.json.
private let inputHeight = 256
private let inputWidth = 192
private let patchSize = 16
private let gridHeight = inputHeight / patchSize  // 16 (with the checkpoint's padding=2 patch conv)
private let gridWidth = 12
private let tokenCount = gridHeight * gridWidth  // 192
private let hidden = 768
private let heads = 12
private let mlpHidden = 3072
private let layerCount = 12
private let upscale = 4  // config.json scale_factor
private let heatHeight = gridHeight * upscale  // 64
private let heatWidth = gridWidth * upscale  // 48
private let imageMean: [Float] = [0.485, 0.456, 0.406]
private let imageStd: [Float] = [0.229, 0.224, 0.225]
// config.json id2label, in channel order.
private let keypointNames = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]

final class Model {
    let net: VitPose

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, *.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = VitPose(keypoints: keypointNames.count)
        var weights = try Weights.load(weightsDir)
        // PyTorch Conv2d [out, in, kh, kw] becomes MLX [out, kh, kw, in].
        for key in ["backbone.embeddings.patch_embeddings.projection.weight", "head.conv.weight"] {
            guard let tensor = weights[key] else { throw RunnerError.failed("Checkpoint has no \(key).") }
            weights[key] = tensor.transposed(0, 2, 3, 1)
        }
        try Weights.apply(weights, to: net)
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let loaded = try Images.load(try params.file("image"))
        let width = loaded.width, height = loaded.height
        let pixels = loaded.pixels.asType(.float32).reshaped([-1]).asArray(Float.self)

        // Whole photo as the person box (no detector): box_to_center_and_scale with the model's 192:256 aspect.
        let box = wholeBoxCenterScale(width: Double(width), height: Double(height))
        let crop = cropToInput(pixels, width: width, height: height, center: box.center, target: box.target)
        let input = MLXArray(crop, [1, inputHeight, inputWidth, 3]).asType(.float16)

        let heat = net.heatmaps(input)  // [1, 64, 48, 17]
        let values = heat.asType(.float32).transposed(0, 3, 1, 2).reshaped([-1]).asArray(Float.self)
        let plane = heatHeight * heatWidth

        var keypoints: [[String: Any]] = []
        for (k, name) in keypointNames.enumerated() {
            let map = Array(values[(k * plane)..<((k + 1) * plane)])
            let logMap = logHeatmap(map)
            // Argmax, then DARK sub-pixel refinement on the blurred log-heatmap.
            var best = 0
            for i in 1..<map.count where map[i] > map[best] { best = i }
            let refined = darkRefine(logMap, x: best % heatWidth, y: best / heatWidth)
            let (x, y) = toImage(refined, target: box.target, center: box.center)
            keypoints.append(["name": name, "x": x, "y": y, "score": Double(map[best])])
        }
        return ["keypoints": keypoints, "image_width": width, "image_height": height]
    }
}

// MARK: - Geometry (image_processing_vitpose.py)

/// box_to_center_and_scale for the whole photo. Returns the center and the crop size in pixels (scale * 200).
private func wholeBoxCenterScale(width: Double, height: Double) -> (center: (Double, Double), target: (Double, Double)) {
    let aspect = Double(inputWidth) / Double(inputHeight)
    var w = width, h = height
    if w > aspect * h { h = w / aspect } else if w < aspect * h { w = h * aspect }
    // The processor pads the box by 1.25 (scale * padding_factor, then * 200).
    return ((width * 0.5, height * 0.5), (w * 1.25, h * 1.25))
}

/// get_warp_matrix with rotation 0, inverted and sampled bilinearly (scipy order=1, zeros outside the photo).
/// Output is 256x192x3 values 0...1, normalized with the ImageNet mean and std.
private func cropToInput(
    _ pixels: [Float], width: Int, height: Int, center: (Double, Double), target: (Double, Double)
) -> [Float] {
    let scaleX = Double(inputWidth - 1) / target.0
    let scaleY = Double(inputHeight - 1) / target.1
    var out = [Float](repeating: 0, count: inputHeight * inputWidth * 3)
    for yo in 0..<inputHeight {
        let ys = Double(yo) / scaleY + center.1 - 0.5 * target.1
        for xo in 0..<inputWidth {
            let xs = Double(xo) / scaleX + center.0 - 0.5 * target.0
            guard xs >= 0, ys >= 0, xs <= Double(width - 1), ys <= Double(height - 1) else { continue }
            let x0 = Int(xs.rounded(.down)), y0 = Int(ys.rounded(.down))
            let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
            let fx = Float(xs - Double(x0)), fy = Float(ys - Double(y0))
            for c in 0..<3 {
                let top = pixels[(y0 * width + x0) * 3 + c] * (1 - fx) + pixels[(y0 * width + x1) * 3 + c] * fx
                let bottom = pixels[(y1 * width + x0) * 3 + c] * (1 - fx) + pixels[(y1 * width + x1) * 3 + c] * fx
                let value = top * (1 - fy) + bottom * fy
                out[(yo * inputWidth + xo) * 3 + c] = (value - imageMean[c]) / imageStd[c]
            }
        }
    }
    return out
}

/// transform_preds: heatmap coordinates back to original photo pixels.
private func toImage(_ point: (Double, Double), target: (Double, Double), center: (Double, Double)) -> (Double, Double) {
    let scaleX = target.0 / Double(heatWidth - 1)
    let scaleY = target.1 / Double(heatHeight - 1)
    return (point.0 * scaleX + center.0 - target.0 * 0.5, point.1 * scaleY + center.1 - target.1 * 0.5)
}

// MARK: - Heatmap decoding (post_dark_unbiased_data_processing, gaussian sigma 0.8, radius 5)

private let blurWeights: [Double] = {
    let radius = 5, sigma = 0.8
    let raw = (-radius...radius).map { exp(-0.5 * Double($0 * $0) / (sigma * sigma)) }
    let total = raw.reduce(0, +)
    return raw.map { $0 / total }
}()

/// scipy's 'reflect' boundary: (d c b a | a b c d | d c b a).
private func reflectIndex(_ i: Int, _ n: Int) -> Int {
    let period = 2 * n
    var j = ((i % period) + period) % period
    if j >= n { j = period - 1 - j }
    return j
}

/// Separable Gaussian blur over rows then columns, then clip to 0.001...50 and take the log.
private func logHeatmap(_ map: [Float]) -> [Float] {
    let radius = (blurWeights.count - 1) / 2
    var rows = [Double](repeating: 0, count: map.count)
    for y in 0..<heatHeight {
        for x in 0..<heatWidth {
            var sum = 0.0
            for t in -radius...radius {
                sum += blurWeights[t + radius] * Double(map[reflectIndex(y + t, heatHeight) * heatWidth + x])
            }
            rows[y * heatWidth + x] = sum
        }
    }
    var out = [Float](repeating: 0, count: map.count)
    for y in 0..<heatHeight {
        for x in 0..<heatWidth {
            var sum = 0.0
            for t in -radius...radius {
                sum += blurWeights[t + radius] * rows[y * heatWidth + reflectIndex(x + t, heatWidth)]
            }
            out[y * heatWidth + x] = Float(log(min(max(sum, 0.001), 50)))
        }
    }
    return out
}

/// The DARK refinement at integer peak (x, y): one Taylor step with the Hessian inverse, edge-clamped.
private func darkRefine(_ logMap: [Float], x: Int, y: Int) -> (Double, Double) {
    func at(_ px: Int, _ py: Int) -> Double {
        let cx = min(max(px, 0), heatWidth - 1), cy = min(max(py, 0), heatHeight - 1)
        return Double(logMap[cy * heatWidth + cx])
    }
    let center = at(x, y)
    let right = at(x + 1, y), left = at(x - 1, y)
    let down = at(x, y + 1), up = at(x, y - 1)
    let downRight = at(x + 1, y + 1), upLeft = at(x - 1, y - 1)
    let dx = 0.5 * (right - left), dy = 0.5 * (down - up)
    let eps = Double(Float.ulpOfOne)
    let dxx = right - 2 * center + left + eps
    let dyy = down - 2 * center + up + eps
    let dxy = 0.5 * (downRight - right - down + center + center - left - up + upLeft)
    let det = dxx * dyy - dxy * dxy
    let ox = (dyy * dx - dxy * dy) / det
    let oy = (-dxy * dx + dxx * dy) / det
    return (Double(x) - ox, Double(y) - oy)
}

// MARK: - Network (modeling_vitpose.py + modeling_vitpose_backbone.py), channels-last

/// Patch embedding: 16x16 conv, stride 16, padding 2 (the checkpoint's setting).
final class PatchEmbeddings: Module {
    var projection: Conv2d
    init(_ channels: Int) {
        projection = Conv2d(
            inputChannels: 3, outputChannels: channels, kernelSize: .init(patchSize), stride: .init(patchSize),
            padding: .init(2), bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(x) }
}

final class Embeddings: Module {
    var patch_embeddings: PatchEmbeddings
    var position_embeddings: MLXArray
    init(_ channels: Int) {
        patch_embeddings = PatchEmbeddings(channels)
        position_embeddings = MLXArray.zeros([1, tokenCount + 1, channels])
        super.init()
    }
    /// Patch tokens plus the position grid plus the class-slot position, which the checkpoint adds to every token.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let batch = pixels.dim(0)
        let patches = patch_embeddings(pixels).reshaped([batch, tokenCount, hidden])
        let grid = position_embeddings[0..., 1..., 0...]
        let slot = position_embeddings[0..., ..<1, 0...]
        return patches + grid + slot
    }
}

final class SelfAttention: Module {
    var query: Linear
    var key: Linear
    var value: Linear
    init(_ channels: Int) {
        query = Linear(channels, channels, bias: true)
        key = Linear(channels, channels, bias: true)
        value = Linear(channels, channels, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0), count = x.dim(1), headDim = hidden / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, count, heads, headDim]).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(
            queries: split(query(x)), keys: split(key(x)), values: split(value(x)),
            scale: pow(Float(headDim), -0.5), mask: .none)
        return out.transposed(0, 2, 1, 3).reshaped([batch, count, hidden])
    }
}

final class SelfOutput: Module {
    var dense: Linear
    init(_ channels: Int) {
        dense = Linear(channels, channels, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

final class AttentionBlock: Module {
    var attention: SelfAttention
    var output: SelfOutput
    init(_ channels: Int) {
        attention = SelfAttention(channels)
        output = SelfOutput(channels)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(attention(x)) }
}

final class MLP: Module {
    var fc1: Linear
    var fc2: Linear
    init(_ channels: Int) {
        fc1 = Linear(channels, mlpHidden, bias: true)
        fc2 = Linear(mlpHidden, channels, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class EncoderLayer: Module {
    var attention: AttentionBlock
    var mlp: MLP
    var layernorm_before: LayerNorm
    var layernorm_after: LayerNorm
    init(_ channels: Int) {
        attention = AttentionBlock(channels)
        mlp = MLP(channels)
        layernorm_before = LayerNorm(dimensions: channels, eps: 1e-12)
        layernorm_after = LayerNorm(dimensions: channels, eps: 1e-12)
        super.init()
    }
    /// Two residual connections, as in VitPoseBackboneLayer.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mid = attention(layernorm_before(x)) + x
        return mlp(layernorm_after(mid)) + mid
    }
}

final class Encoder: Module {
    var layer: [EncoderLayer]
    init(_ channels: Int) {
        layer = (0..<layerCount).map { _ in EncoderLayer(channels) }
        super.init()
    }
}

final class Backbone: Module {
    var embeddings: Embeddings
    var encoder: Encoder
    var layernorm: LayerNorm
    init(_ channels: Int) {
        embeddings = Embeddings(channels)
        encoder = Encoder(channels)
        layernorm = LayerNorm(dimensions: channels, eps: 1e-12)
        super.init()
    }
    /// Tokens [N, 192, 768] after the final LayerNorm (the stage12 feature map).
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var x = embeddings(pixels)
        for block in encoder.layer { x = block(x) }
        return layernorm(x)
    }
}

/// VitPoseSimpleDecoder: ReLU, bilinear 4x upsample (align_corners false), 3x3 conv to keypoint heatmaps.
final class SimpleDecoder: Module {
    var conv: Conv2d
    init(keypoints: Int) {
        conv = Conv2d(
            inputChannels: hidden, outputChannels: keypoints, kernelSize: .init(3), stride: .init(1),
            padding: .init(1), bias: true)
        super.init()
    }
    func callAsFunction(_ feature: MLXArray) -> MLXArray {
        let up = Upsample(scaleFactor: .array([Float(upscale), Float(upscale)]), mode: .linear(alignCorners: false))
        return conv(up(relu(feature)))
    }
}

final class VitPose: Module {
    var backbone: Backbone
    var head: SimpleDecoder
    init(keypoints: Int) {
        backbone = Backbone(hidden)
        head = SimpleDecoder(keypoints: keypoints)
        super.init()
    }
    /// Heatmaps [1, 64, 48, keypoints] for one 256x192 channels-last input.
    func heatmaps(_ pixels: MLXArray) -> MLXArray {
        let tokens = backbone(pixels)
        return head(tokens.reshaped([pixels.dim(0), gridHeight, gridWidth, hidden]))
    }
}
