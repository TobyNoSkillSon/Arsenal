// SPDX-License-Identifier: MIT
// Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf). Ported from transformers'
// modeling_depth_anything.py (neck, head), modeling_dinov2.py (Dinov2Backbone) and
// image_processing_dpt.py (resize with keep_aspect_ratio to a multiple of 14, ImageNet normalization,
// post_process_depth_estimation's bicubic upsample to the photo's size).
// Output: relative inverse depth (disparity: larger is nearer) at the photo's size, as FP16 safetensors
// and a 16-bit grayscale PNG preview normalized to the map's own range.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXFast
import MLXNN
import UniformTypeIdentifiers

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "perception"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["estimate_depth"]

/// Geometry and widths from config.json and preprocessor_config.json.
struct DepthConfig {
    var hidden = 384, heads = 6, layers = 12, mlpRatio = 4, patch = 14, layerNormEps: Float = 1e-6
    var outIndices = [3, 6, 9, 12]
    var neckSizes = [48, 96, 192, 384]
    var factors: [Double] = [4, 2, 1, 0.5]
    var fusion = 64, headHidden = 32
    var metric = false, maxDepth: Float = 1
    var target = 518, multiple = 14
    var mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]

    init(config: [String: Any], preprocessor: [String: Any]) {
        let backbone = config["backbone_config"] as? [String: Any] ?? [:]
        hidden = backbone["hidden_size"] as? Int ?? hidden
        heads = backbone["num_attention_heads"] as? Int ?? heads
        layers = backbone["num_hidden_layers"] as? Int ?? layers
        mlpRatio = backbone["mlp_ratio"] as? Int ?? mlpRatio
        layerNormEps = (backbone["layer_norm_eps"] as? NSNumber)?.floatValue ?? layerNormEps
        outIndices = backbone["out_indices"] as? [Int] ?? outIndices
        patch = config["patch_size"] as? Int ?? patch
        neckSizes = config["neck_hidden_sizes"] as? [Int] ?? neckSizes
        factors = (config["reassemble_factors"] as? [NSNumber])?.map(\.doubleValue) ?? factors
        fusion = config["fusion_hidden_size"] as? Int ?? fusion
        headHidden = config["head_hidden_size"] as? Int ?? headHidden
        metric = (config["depth_estimation_type"] as? String) == "metric"
        maxDepth = (config["max_depth"] as? NSNumber)?.floatValue ?? maxDepth
        target = (preprocessor["size"] as? [String: Any])?["height"] as? Int ?? target
        multiple = preprocessor["ensure_multiple_of"] as? Int ?? multiple
        mean = (preprocessor["image_mean"] as? [NSNumber])?.map(\.floatValue) ?? mean
        std = (preprocessor["image_std"] as? [NSNumber])?.map(\.floatValue) ?? std
    }
}

final class Model {
    let config: DepthConfig
    let net: DepthAnything

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, *.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = DepthConfig(config: try Files.json(weightsDir), preprocessor: try Files.json(weightsDir, "preprocessor_config.json"))
        net = DepthAnything(config)
        var weights = try Weights.load(weightsDir)
        // PyTorch Conv2d [out, in, kh, kw] -> MLX [out, kh, kw, in]; ConvTranspose2d [in, out, kh, kw] -> [out, kh, kw, in].
        let transposedConvs = Set(config.factors.enumerated().filter { $0.element > 1 }.map { "neck.reassemble_stage.layers.\($0.offset).resize.weight" })
        for (key, value) in weights where value.ndim == 4 {
            weights[key] = transposedConvs.contains(key) ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
        }
        // mask_token is used only for masked-image pretraining.
        try Weights.apply(weights, to: net, rename: { $0 == "backbone.embeddings.mask_token" ? nil : $0 })
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let url: URL
        if params.raw["image"] != nil {
            url = try params.file("image")
        } else if let images = params.raw["images"] as? [Any], images.count == 1 {
            url = try Params(["image": images[0]]).file("image")
        } else {
            throw RunnerError.invalid("This model estimates depth for one image at a time; send image or a one-item images list.")
        }
        if let mode = params.raw["mode"] as? String, mode != "relative" {
            throw RunnerError.invalid("This model gives relative depth only; it has no metric mode.")
        }
        if params.bool("points", default: false) || params.bool("intrinsics", default: false) {
            throw RunnerError.invalid("This model gives a depth map only; it does not estimate points or camera intrinsics.")
        }
        let photo = try Images.load(url)
        let (height, width) = resizeTarget(height: photo.height, width: photo.width)
        guard height >= config.patch, width >= config.patch else {
            throw RunnerError.invalid("The image is too thin: its short side would be under \(config.patch) pixels after resizing.")
        }
        // Bicubic with antialiasing (torchvision resize on uint8, as DPTImageProcessor), then rescale and normalize.
        var pixels = Resample.apply(
            (photo.pixels * 255).expandedDimensions(axis: 0),
            rows: Resample.antialiasedBicubic(from: photo.height, to: height),
            cols: Resample.antialiasedBicubic(from: photo.width, to: width))
        pixels = clip(pixels.round(), min: 0, max: 255) / 255
        let input = Images.normalize(pixels, mean: config.mean, std: config.std).asType(.float16)
        try params.checkCancelled()

        let predicted = try net(input, checkpoint: { try params.checkCancelled() })  // [1, h, w, 1]
        // post_process_depth_estimation: bicubic (align_corners false) to the photo's own size.
        let depth = Resample.apply(
            predicted.asType(.float32),
            rows: Resample.bicubic(from: predicted.dim(1), to: photo.height),
            cols: Resample.bicubic(from: predicted.dim(2), to: photo.width))[0, 0..., 0..., 0]
        eval(depth)
        try params.checkCancelled()

        let root = try outputFolder(params)
        let tensor = root.appendingPathComponent("view-0.safetensors")
        try save(arrays: ["depth": depth.asType(.float16)], url: tensor)
        let preview = root.appendingPathComponent("view-0-depth.png")
        try writePreview(depth, to: preview)
        return [
            "files": [
                try fileReceipt(tensor, id: "view-0", mime: "application/octet-stream", label: "Depth values (FP16 safetensors)", order: 1),
                try fileReceipt(preview, id: "view-0-depth", mime: "image/png", label: "Depth map", order: 0, size: (photo.width, photo.height)),
            ],
            "mode": "relative",
            "depth_unit": "relative",
            "views": [["index": 0, "file_ids": ["view-0", "view-0-depth"]]],
        ]
    }

    /// DPT get_resize_output_image_size: scale as little as possible, keep the aspect ratio, round to a multiple of 14.
    func resizeTarget(height: Int, width: Int) -> (Int, Int) {
        var scaleH = Double(config.target) / Double(height), scaleW = Double(config.target) / Double(width)
        if abs(1 - scaleW) < abs(1 - scaleH) { scaleH = scaleW } else { scaleW = scaleH }
        func constrain(_ value: Double) -> Int {
            // Python's round() rounds halves to even.
            Int((value / Double(config.multiple)).rounded(.toNearestOrEven)) * config.multiple
        }
        return (constrain(scaleH * Double(height)), constrain(scaleW * Double(width)))
    }
}

// MARK: - Files

/// Arsenal passes out_dir (a job folder); `kit try` does not, so fall back to a job folder under ARSENAL_SUPPORT_DIR.
private func outputFolder(_ params: Params) throws -> URL {
    let root: URL
    if let dir = params.raw["out_dir"] as? String, !dir.isEmpty {
        root = URL(fileURLWithPath: dir)
    } else {
        let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory
        root = support.appendingPathComponent("jobs").appendingPathComponent(UUID().uuidString)
    }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func fileReceipt(_ url: URL, id: String, mime: String, label: String, order: Int, size: (Int, Int)? = nil) throws -> [String: Any] {
    var receipt: [String: Any] = [
        "id": id, "path": url.path, "name": url.lastPathComponent, "mime": mime, "label": label, "order": order,
        "bytes": try Files.size(url), "sha256": try Files.sha256(url),
    ]
    if let size { receipt["width"] = size.0; receipt["height"] = size.1 }
    return receipt
}

/// A 16-bit grayscale PNG, the map's finite values stretched to 0...65535 (brighter is nearer for disparity).
private func writePreview(_ depth: MLXArray, to url: URL) throws {
    let values = depth.asType(.float32).asArray(Float.self)
    let finite = values.filter(\.isFinite)
    let lower = finite.min() ?? 0, upper = finite.max() ?? 1
    let span = max(upper - lower, Float.ulpOfOne)
    var bytes = [UInt8]()
    bytes.reserveCapacity(values.count * 2)
    for value in values {
        let normalized = value.isFinite ? min(1, max(0, (value - lower) / span)) : 0
        let sample = UInt16(normalized * 65535)
        bytes.append(UInt8(sample >> 8)); bytes.append(UInt8(sample & 255))
    }
    let h = depth.dim(0), w = depth.dim(1)
    guard let provider = CGDataProvider(data: Data(bytes) as CFData),
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: w * 2, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageByteOrderInfo.order16Big.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent),
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw RunnerError.failed("Could not encode the depth preview.") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw RunnerError.failed("Could not write the depth preview.") }
}

// MARK: - Resampling as separable [out, in] weight matrices (PyTorch semantics)

enum Resample {
    static func cubic(_ distance: Double, a: Double) -> Double {
        let x = abs(distance)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// interpolate(mode="bicubic", align_corners=False, antialias=False): A = -0.75, edge taps clamped.
    static func bicubic(from n: Int, to m: Int) -> MLXArray {
        var w = [Float](repeating: 0, count: m * n)
        let scale = Double(n) / Double(m)
        for i in 0..<m {
            let source = (Double(i) + 0.5) * scale - 0.5
            let base = Int(source.rounded(.down)), t = source - Double(base)
            for k in -1...2 { w[i * n + min(max(base + k, 0), n - 1)] += Float(cubic(t - Double(k), a: -0.75)) }
        }
        return MLXArray(w, [m, n])
    }

    /// Antialiased bicubic (torchvision/PIL): A = -0.5, the kernel widened by the downscale factor, weights normalized.
    static func antialiasedBicubic(from n: Int, to m: Int) -> MLXArray {
        var w = [Float](repeating: 0, count: m * n)
        let scale = Double(n) / Double(m), stretch = max(scale, 1), support = 2 * stretch
        for i in 0..<m {
            let center = (Double(i) + 0.5) * scale
            let low = max(Int(center - support + 0.5), 0), high = min(Int(center + support + 0.5), n)
            var taps = [Double]()
            for j in low..<high { taps.append(cubic((Double(j) - center + 0.5) / stretch, a: -0.5)) }
            let total = taps.reduce(0, +)
            for (offset, tap) in taps.enumerated() { w[i * n + low + offset] = Float(total != 0 ? tap / total : tap) }
        }
        return MLXArray(w, [m, n])
    }

    /// interpolate(mode="bilinear") with or without align_corners.
    static func bilinear(from n: Int, to m: Int, alignCorners: Bool) -> MLXArray {
        var w = [Float](repeating: 0, count: m * n)
        for i in 0..<m {
            let source: Double
            if alignCorners {
                source = m > 1 ? Double(i) * Double(n - 1) / Double(m - 1) : 0
            } else {
                source = max((Double(i) + 0.5) * Double(n) / Double(m) - 0.5, 0)
            }
            let x0 = min(Int(source.rounded(.down)), n - 1), x1 = min(x0 + 1, n - 1)
            let lambda = Float(source - Double(x0))
            w[i * n + x0] += 1 - lambda
            w[i * n + x1] += lambda
        }
        return MLXArray(w, [m, n])
    }

    /// x [B, H, W, C] -> [B, H', W', C] with rows [H', H] and cols [W', W].
    static func apply(_ x: MLXArray, rows: MLXArray, cols: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let r = rows.asType(x.dtype), k = cols.asType(x.dtype)
        let tall = matmul(r, x.reshaped([b, h, w * c])).reshaped([b, r.dim(0), w, c])
        let wide = matmul(tall.transposed(0, 1, 3, 2), k.transposed(1, 0))
        return wide.transposed(0, 1, 3, 2)
    }

    static func bilinear(_ x: MLXArray, height: Int, width: Int, alignCorners: Bool) -> MLXArray {
        if x.dim(1) == height && x.dim(2) == width && alignCorners { return x }
        return apply(
            x, rows: bilinear(from: x.dim(1), to: height, alignCorners: alignCorners),
            cols: bilinear(from: x.dim(2), to: width, alignCorners: alignCorners))
    }
}

// MARK: - DINOv2 backbone (modeling_dinov2.py), channels-last

final class PatchEmbeddings: Module {
    var projection: Conv2d
    init(_ c: DepthConfig) {
        projection = Conv2d(
            inputChannels: 3, outputChannels: c.hidden, kernelSize: .init(c.patch), stride: .init(c.patch), padding: .init(0), bias: true)
        super.init()
    }
}

final class Embeddings: Module {
    var cls_token: MLXArray
    var patch_embeddings: PatchEmbeddings
    var position_embeddings: MLXArray
    init(_ c: DepthConfig) {
        cls_token = MLXArray.zeros([1, 1, c.hidden])
        patch_embeddings = PatchEmbeddings(c)
        position_embeddings = MLXArray.zeros([1, 1370, c.hidden])
        super.init()
    }

    /// Class token plus patch tokens, with the 37x37 position grid resized bicubically (in float32) to the patch grid.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let patches = patch_embeddings.projection(pixels)  // [B, ph, pw, C]
        let b = patches.dim(0), ph = patches.dim(1), pw = patches.dim(2), dim = patches.dim(3)
        let tokens = concatenated([broadcast(cls_token, to: [b, 1, dim]), patches.reshaped([b, ph * pw, dim])], axis: 1)
        let positions = position_embeddings.dim(1) - 1
        let side = Int(Double(positions).squareRoot())
        var position = position_embeddings
        if !(ph * pw == positions && ph == pw) {
            let grid = position_embeddings[0..., 1..., 0...].reshaped([1, side, side, dim]).asType(.float32)
            let resized = Resample.apply(grid, rows: Resample.bicubic(from: side, to: ph), cols: Resample.bicubic(from: side, to: pw))
            position = concatenated(
                [position_embeddings[0..., ..<1, 0...], resized.reshaped([1, ph * pw, dim]).asType(position_embeddings.dtype)], axis: 1)
        }
        return tokens + position
    }
}

final class SelfAttention: Module {
    var query: Linear
    var key: Linear
    var value: Linear
    let heads: Int
    init(_ c: DepthConfig) {
        query = Linear(c.hidden, c.hidden, bias: true)
        key = Linear(c.hidden, c.hidden, bias: true)
        value = Linear(c.hidden, c.hidden, bias: true)
        heads = c.heads
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), dim = x.dim(2), headDim = dim / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(
            queries: split(query(x)), keys: split(key(x)), values: split(value(x)), scale: pow(Float(headDim), -0.5), mask: .none)
        return out.transposed(0, 2, 1, 3).reshaped([b, n, dim])
    }
}

final class SelfOutput: Module {
    var dense: Linear
    init(_ c: DepthConfig) {
        dense = Linear(c.hidden, c.hidden, bias: true)
        super.init()
    }
}

final class AttentionBlock: Module {
    var attention: SelfAttention
    var output: SelfOutput
    init(_ c: DepthConfig) {
        attention = SelfAttention(c)
        output = SelfOutput(c)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output.dense(attention(x)) }
}

final class LayerScale: Module {
    var lambda1: MLXArray
    init(_ c: DepthConfig) {
        lambda1 = MLXArray.ones([c.hidden])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * lambda1 }
}

final class MLP: Module {
    var fc1: Linear
    var fc2: Linear
    init(_ c: DepthConfig) {
        fc1 = Linear(c.hidden, c.hidden * c.mlpRatio, bias: true)
        fc2 = Linear(c.hidden * c.mlpRatio, c.hidden, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class Dinov2Layer: Module {
    var norm1: LayerNorm
    var attention: AttentionBlock
    var layer_scale1: LayerScale
    var norm2: LayerNorm
    var mlp: MLP
    var layer_scale2: LayerScale
    init(_ c: DepthConfig) {
        norm1 = LayerNorm(dimensions: c.hidden, eps: c.layerNormEps)
        attention = AttentionBlock(c)
        layer_scale1 = LayerScale(c)
        norm2 = LayerNorm(dimensions: c.hidden, eps: c.layerNormEps)
        mlp = MLP(c)
        layer_scale2 = LayerScale(c)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mid = x + layer_scale1(attention(norm1(x)))
        return mid + layer_scale2(mlp(norm2(mid)))
    }
}

final class Encoder: Module {
    var layer: [Dinov2Layer]
    init(_ c: DepthConfig) {
        layer = (0..<c.layers).map { _ in Dinov2Layer(c) }
        super.init()
    }
}

final class Backbone: Module {
    var embeddings: Embeddings
    var encoder: Encoder
    var layernorm: LayerNorm
    let outIndices: [Int]
    init(_ c: DepthConfig) {
        embeddings = Embeddings(c)
        encoder = Encoder(c)
        layernorm = LayerNorm(dimensions: c.hidden, eps: c.layerNormEps)
        outIndices = c.outIndices
        super.init()
    }
    /// Dinov2Backbone feature maps: hidden state after layer i (stage i, 1-based) for each out index, layer-normed, class token kept.
    func callAsFunction(_ pixels: MLXArray, checkpoint: () throws -> Void) rethrows -> [MLXArray] {
        var x = embeddings(pixels)
        var features: [MLXArray] = []
        for (index, block) in encoder.layer.enumerated() {
            x = block(x)
            if outIndices.contains(index + 1) { features.append(layernorm(x)) }
            if index % 4 == 3 { eval(x); try checkpoint() }
        }
        return features
    }
}

// MARK: - Neck and head (modeling_depth_anything.py), channels-last

final class ReassembleLayer: Module {
    var projection: Conv2d
    @ModuleInfo(key: "resize") var resize: Module?
    init(_ c: DepthConfig, channels: Int, factor: Double) {
        projection = Conv2d(inputChannels: c.hidden, outputChannels: channels, kernelSize: .init(1), bias: true)
        if factor > 1 {
            let k = Int(factor)
            _resize.wrappedValue = ConvTransposed2d(
                inputChannels: channels, outputChannels: channels, kernelSize: .init(k), stride: .init(k), padding: .init(0), bias: true)
        } else if factor < 1 {
            _resize.wrappedValue = Conv2d(
                inputChannels: channels, outputChannels: channels, kernelSize: .init(3), stride: .init(Int(1 / factor)), padding: .init(1),
                bias: true)
        } else {
            _resize.wrappedValue = nil
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = projection(x)
        switch resize {
        case let up as ConvTransposed2d: return up(y)
        case let down as Conv2d: return down(y)
        default: return y
        }
    }
}

final class ReassembleStage: Module {
    var layers: [ReassembleLayer]
    init(_ c: DepthConfig) {
        layers = zip(c.neckSizes, c.factors).map { ReassembleLayer(c, channels: $0.0, factor: $0.1) }
        super.init()
    }
}

final class PreActResidualLayer: Module {
    var convolution1: Conv2d
    var convolution2: Conv2d
    init(_ c: DepthConfig) {
        convolution1 = Conv2d(inputChannels: c.fusion, outputChannels: c.fusion, kernelSize: .init(3), padding: .init(1), bias: true)
        convolution2 = Conv2d(inputChannels: c.fusion, outputChannels: c.fusion, kernelSize: .init(3), padding: .init(1), bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { convolution2(relu(convolution1(relu(x)))) + x }
}

final class FusionLayer: Module {
    var projection: Conv2d
    var residual_layer1: PreActResidualLayer
    var residual_layer2: PreActResidualLayer
    init(_ c: DepthConfig) {
        projection = Conv2d(inputChannels: c.fusion, outputChannels: c.fusion, kernelSize: .init(1), bias: true)
        residual_layer1 = PreActResidualLayer(c)
        residual_layer2 = PreActResidualLayer(c)
        super.init()
    }
    func callAsFunction(_ hidden: MLXArray, residual: MLXArray?, size: (Int, Int)?) -> MLXArray {
        var x = hidden
        if var residual {
            if residual.shape != x.shape {
                residual = Resample.bilinear(residual, height: x.dim(1), width: x.dim(2), alignCorners: false)
            }
            x = x + residual_layer1(residual)
        }
        x = residual_layer2(x)
        let target = size ?? (x.dim(1) * 2, x.dim(2) * 2)
        x = Resample.bilinear(x, height: target.0, width: target.1, alignCorners: true)
        return projection(x)
    }
}

final class FusionStage: Module {
    var layers: [FusionLayer]
    init(_ c: DepthConfig) {
        layers = c.neckSizes.map { _ in FusionLayer(c) }
        super.init()
    }
}

final class Neck: Module {
    var reassemble_stage: ReassembleStage
    var convs: [Conv2d]
    var fusion_stage: FusionStage
    init(_ c: DepthConfig) {
        reassemble_stage = ReassembleStage(c)
        convs = c.neckSizes.map {
            Conv2d(inputChannels: $0, outputChannels: c.fusion, kernelSize: .init(3), padding: .init(1), bias: false)
        }
        fusion_stage = FusionStage(c)
        super.init()
    }
    /// The last fused map (the head reads index -1).
    func callAsFunction(_ hidden: [MLXArray], ph: Int, pw: Int) -> MLXArray {
        var features: [MLXArray] = []
        for (index, state) in hidden.enumerated() {
            let b = state.dim(0), dim = state.dim(2)
            let grid = state[0..., 1..., 0...].reshaped([b, ph, pw, dim])
            features.append(convs[index](reassemble_stage.layers[index](grid)))
        }
        let reversed = Array(features.reversed())
        var fused: MLXArray? = nil
        for (index, layer) in fusion_stage.layers.enumerated() {
            let size = index + 1 < reversed.count ? (reversed[index + 1].dim(1), reversed[index + 1].dim(2)) : nil
            if let previous = fused {
                fused = layer(previous, residual: reversed[index], size: size)
            } else {
                fused = layer(reversed[index], residual: nil, size: size)
            }
        }
        return fused!
    }
}

final class DepthHead: Module {
    var conv1: Conv2d
    var conv2: Conv2d
    var conv3: Conv2d
    let patch: Int, metric: Bool, maxDepth: Float
    init(_ c: DepthConfig) {
        conv1 = Conv2d(inputChannels: c.fusion, outputChannels: c.fusion / 2, kernelSize: .init(3), padding: .init(1), bias: true)
        conv2 = Conv2d(inputChannels: c.fusion / 2, outputChannels: c.headHidden, kernelSize: .init(3), padding: .init(1), bias: true)
        conv3 = Conv2d(inputChannels: c.headHidden, outputChannels: 1, kernelSize: .init(1), bias: true)
        patch = c.patch; metric = c.metric; maxDepth = c.maxDepth
        super.init()
    }
    func callAsFunction(_ x: MLXArray, ph: Int, pw: Int) -> MLXArray {
        var y = conv1(x)
        y = Resample.bilinear(y, height: ph * patch, width: pw * patch, alignCorners: true)
        y = conv3(relu(conv2(y))).asType(.float32)
        return (metric ? sigmoid(y) : relu(y)) * maxDepth
    }
}

final class DepthAnything: Module {
    var backbone: Backbone
    var neck: Neck
    var head: DepthHead
    let patch: Int
    init(_ c: DepthConfig) {
        backbone = Backbone(c)
        neck = Neck(c)
        head = DepthHead(c)
        patch = c.patch
        super.init()
    }
    /// pixels [1, H, W, 3] normalized, H and W multiples of the patch size -> predicted depth [1, H, W, 1] float32.
    func callAsFunction(_ pixels: MLXArray, checkpoint: () throws -> Void) rethrows -> MLXArray {
        let ph = pixels.dim(1) / patch, pw = pixels.dim(2) / patch
        let hidden = try backbone(pixels, checkpoint: checkpoint)
        return head(neck(hidden, ph: ph, pw: pw), ph: ph, pw: pw)
    }
}
