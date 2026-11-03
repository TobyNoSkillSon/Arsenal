// SPDX-License-Identifier: MIT
// UVDoc (PaddlePaddle) document unwarping, ported from transformers' modeling_uvdoc.py and image_processing_uvdoc.py.
// The network predicts a coarse 2-D sampling grid; the grid is upsampled to the photo's size and the photo is resampled
// through it (grid_sample, bilinear, align_corners=True), giving the flattened page at the photo's own size.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import UniformTypeIdentifiers

let modelKind = "local.dewarp"
let modelOps = ["dewarp"]

final class Model {
    let net: UVDocModel
    let height: Int
    let width: Int

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        let processor = (try? Files.json(weightsDir, "preprocessor_config.json")) ?? [:]
        let size = processor["size"] as? [String: Any] ?? [:]
        height = size["height"] as? Int ?? 712
        width = size["width"] as? Int ?? 488
        net = try UVDocModel(config: config)
        var weights = try Weights.load(weightsDir)
        for (key, value) in weights where value.ndim == 4 { weights[key] = value.transposed(0, 2, 3, 1) }
        try Weights.apply(weights, to: net)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "dewarp" else { throw RunnerError.invalid("Unknown op \(op).") }
        let url = try params.file("image")
        // The flattened page comes out at the photo's size, so each side stays at most 8192 (checked before decoding).
        _ = try checkImageHeader(url)
        let photo = try Images.load(url)  // [H, W, 3] RGB 0...1, the photo's own size
        guard photo.width >= 8, photo.height >= 8 else { throw RunnerError.invalid("The image is too small (at least 8×8 pixels).") }
        guard photo.width * photo.height <= 50_000_000 else { throw RunnerError.invalid("The image is too large (at most 50 megapixels).") }
        // The processor feeds BGR, resized with bilinear align_corners=True (no antialias) to 712×488.
        let bgr = photo.pixels[0..., 0..., MLXArray([2, 1, 0] as [Int32])]
        let resized = Resize.bilinearAligned(bgr, height: height, width: width)
        try params.checkCancelled()
        let mesh = net(resized.expandedDimensions(axis: 0).asType(.float16))[0].asType(.float32)  // [h, w, 2] (x, y) in -1...1
        eval(mesh)
        try params.checkCancelled()
        // post_process_document_rectification: upsample the grid to the photo's size, then grid_sample the photo.
        let grid = Resize.bilinearAligned(mesh, height: photo.height, width: photo.width)
        let flat = Sample.grid(photo.pixels, grid: grid)
        // The maker converts with a truncating cast to uint8.
        let output = floor(clip(flat, min: 0, max: 1) * 255) / 255
        eval(output)
        try params.checkCancelled()

        let root = try outputFolder(params)
        let image = root.appendingPathComponent("dewarped.png")
        try Images.savePNG(output, to: image)
        let gridFile = root.appendingPathComponent("grid.safetensors")
        try save(arrays: ["grid": mesh.asType(.float16)], url: gridFile)
        return [
            "files": [
                try fileReceipt(image, id: "dewarped", mime: "image/png", label: "Flattened page", order: 0, size: (photo.width, photo.height)),
                try fileReceipt(gridFile, id: "grid", mime: "application/octet-stream", label: "Unwarp grid (FP16 safetensors)", order: 1),
            ]
        ]
    }
}

// MARK: - Network

/// Conv2d + BatchNorm2d + optional activation (UVDocConvLayer).
final class ConvLayer: Module, UnaryLayer {
    enum Activation { case none, relu, prelu }
    @ModuleInfo var convolution: Conv2d
    @ModuleInfo var normalization: BatchNorm
    @ModuleInfo var activation: PReLU?
    let act: Activation
    let reflectPad: Int

    init(_ inC: Int, _ outC: Int, kernel: Int = 3, stride: Int = 1, padding: Int = 0, dilation: Int = 1, bias: Bool = false,
         activation: Activation = .relu, reflect: Bool = false) {
        reflectPad = reflect ? padding : 0
        _convolution.wrappedValue = Conv2d(
            inputChannels: inC, outputChannels: outC, kernelSize: .init(kernel), stride: .init(stride),
            padding: .init(reflect ? 0 : padding), dilation: .init(dilation), bias: bias)
        _normalization.wrappedValue = BatchNorm(featureCount: outC, eps: 1e-5)
        _activation.wrappedValue = activation == .prelu ? PReLU() : nil
        act = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convolution(reflectPad > 0 ? reflectPad2D(x, reflectPad) : x)
        h = normalization(h)
        switch act {
        case .none: return h
        case .relu: return relu(h)
        case .prelu: return activation!(h)
        }
    }
}

/// nn.PReLU() with a single shared slope.
final class PReLU: Module, UnaryLayer {
    var weight: MLXArray
    override init() { weight = MLXArray([Float(0.25)]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { maximum(x, 0) + weight.asType(x.dtype) * minimum(x, 0) }
}

/// padding_mode="reflect" on H and W of an [N, H, W, C] tensor.
func reflectPad2D(_ x: MLXArray, _ p: Int) -> MLXArray {
    func indices(_ n: Int) -> MLXArray {
        let before = (1...p).reversed().map { Int32($0) }
        let after = (1...p).map { Int32(n - 1 - $0) }
        return MLXArray(before + (0..<n).map(Int32.init) + after)
    }
    let rows = take(x, indices(x.dim(1)), axis: 1)
    return take(rows, indices(x.dim(2)), axis: 2)
}

final class ResidualBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv_down") var convDown: ConvLayer?
    @ModuleInfo(key: "conv_start") var convStart: ConvLayer
    @ModuleInfo(key: "conv_final") var convFinal: ConvLayer

    init(_ inC: Int, _ outC: Int, kernel: Int, dilation: Int, downsample: Bool) {
        let stride = downsample ? 2 : 1, padding = dilation * 2
        _convDown.wrappedValue = downsample
            ? ConvLayer(inC, outC, kernel: kernel, stride: stride, padding: kernel / 2, bias: true, activation: .none) : nil
        _convStart.wrappedValue = ConvLayer(inC, outC, kernel: kernel, stride: stride, padding: padding, dilation: dilation, bias: true)
        _convFinal.wrappedValue = ConvLayer(outC, outC, kernel: kernel, padding: padding, dilation: dilation, bias: true, activation: .none)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = convDown.map { $0(x) } ?? x
        return relu(convFinal(convStart(x)) + residual)
    }
}

final class ResNetStage: Module, UnaryLayer {
    var layers: [ResidualBlock]
    init(_ layers: [ResidualBlock]) { self.layers = layers }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers.reduce(x) { $1($0) } }
}

final class ResNet: Module, UnaryLayer {
    @ModuleInfo(key: "resnet_head") var head: [ConvLayer]
    @ModuleInfo(key: "resnet_down") var down: [ResNetStage]
    init(head: [ConvLayer], down: [ResNetStage]) { _head.wrappedValue = head; _down.wrappedValue = down }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down.reduce(head.reduce(x) { $1($0) }) { $1($0) } }
}

final class BridgeBlock: Module, UnaryLayer {
    var blocks: [ConvLayer]
    init(_ blocks: [ConvLayer]) { self.blocks = blocks }
    func callAsFunction(_ x: MLXArray) -> MLXArray { blocks.reduce(x) { $1($0) } }
}

final class Bridge: Module {
    var bridge: [BridgeBlock]
    init(_ bridge: [BridgeBlock]) { self.bridge = bridge }
}

final class Backbone: Module {
    var resnet: ResNet
    var bridge: Bridge
    init(resnet: ResNet, bridge: Bridge) { self.resnet = resnet; self.bridge = bridge }
    /// Every bridge block reads the same ResNet features; their outputs are the feature maps.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let features = resnet(x)
        return bridge.bridge.map { $0(features) }
    }
}

final class PointPositions: Module, UnaryLayer {
    @ModuleInfo(key: "conv_down") var convDown: ConvLayer
    @ModuleInfo(key: "conv_up") var convUp: Conv2d
    let pad: Int
    init(_ c0: [Int], _ c1: [Int], kernel: Int) {
        pad = kernel / 2
        _convDown.wrappedValue = ConvLayer(c0[0], c0[1], kernel: kernel, padding: kernel / 2, activation: .prelu, reflect: true)
        _convUp.wrappedValue = Conv2d(inputChannels: c1[0], outputChannels: c1[1], kernelSize: .init(kernel), padding: .init(0), bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { convUp(reflectPad2D(convDown(x), pad)) }
}

final class Head: Module {
    @ModuleInfo(key: "bridge_connector") var bridgeConnector: ConvLayer
    @ModuleInfo(key: "out_point_positions2D") var outPoints: PointPositions
    init(connector: ConvLayer, points: PointPositions) { _bridgeConnector.wrappedValue = connector; _outPoints.wrappedValue = points }
    func callAsFunction(_ x: MLXArray) -> MLXArray { outPoints(bridgeConnector(x)) }
}

final class UVDocModel: Module, UnaryLayer {
    var backbone: Backbone
    var head: Head

    init(config: [String: Any]) throws {
        guard let backboneConfig = config["backbone_config"] as? [String: Any],
            let headSpec = backboneConfig["resnet_head"] as? [[Int]],
            let resnetSpec = backboneConfig["resnet_configs"] as? [[[Any]]],
            let stageSpec = backboneConfig["stage_configs"] as? [[[Int]]],
            let connector = config["bridge_connector"] as? [Int],
            let points = config["out_point_positions2D"] as? [[Int]]
        else { throw RunnerError.failed("config.json is not a UVDoc config.") }
        let kernel = config["kernel_size"] as? Int ?? 5
        let heads = headSpec.map { ConvLayer($0[0], $0[1], kernel: kernel, stride: 2, padding: kernel / 2) }
        let stages = resnetSpec.map { stage in
            ResNetStage(stage.map { spec in
                ResidualBlock(
                    (spec[0] as? Int) ?? 0, (spec[1] as? Int) ?? 0, kernel: kernel, dilation: (spec[2] as? Int) ?? 1,
                    downsample: (spec[3] as? Bool) ?? false)
            })
        }
        let bridges = stageSpec.map { bridge in BridgeBlock(bridge.map { ConvLayer($0[0], $0[0], padding: $0[1], dilation: $0[1]) }) }
        backbone = Backbone(resnet: ResNet(head: heads, down: stages), bridge: Bridge(bridges))
        head = Head(
            connector: ConvLayer(connector[0] * stageSpec.count, connector[1], kernel: 1),
            points: PointPositions(points[0], points[1], kernel: kernel))
    }

    /// [N, H, W, 3] BGR 0...1 → [N, h, w, 2] sampling grid.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        head(concatenated(backbone(x), axis: -1))
    }
}

// MARK: - Resampling

enum Resize {
    /// Source taps and weights of F.interpolate(mode="bilinear", align_corners=True) along one axis.
    static func taps(from n: Int, to m: Int) -> (lower: MLXArray, upper: MLXArray, t: MLXArray) {
        var lower = [Int32](), upper = [Int32](), t = [Float]()
        for i in 0..<m {
            let source = m > 1 ? Double(i) * Double(n - 1) / Double(m - 1) : 0
            let lo = min(Int(source.rounded(.down)), n - 1)
            lower.append(Int32(lo)); upper.append(Int32(min(lo + 1, n - 1))); t.append(Float(source - Double(lo)))
        }
        return (MLXArray(lower), MLXArray(upper), MLXArray(t))
    }

    /// [H, W, C] float32 → [height, width, C]. Elementwise lerps (a float32 matmul here loses precision on the GPU).
    static func bilinearAligned(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let x = x.asType(.float32)
        let r = taps(from: x.dim(0), to: height)
        let rt = r.t.reshaped([height, 1, 1])
        let rows = take(x, r.lower, axis: 0) * (1 - rt) + take(x, r.upper, axis: 0) * rt
        let c = taps(from: x.dim(1), to: width)
        let ct = c.t.reshaped([1, width, 1])
        return take(rows, c.lower, axis: 1) * (1 - ct) + take(rows, c.upper, axis: 1) * ct
    }
}

enum Sample {
    /// F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True).
    /// image [H, W, C], grid [Ho, Wo, 2] with (x, y) in -1...1; returns [Ho, Wo, C].
    static func grid(_ image: MLXArray, grid: MLXArray) -> MLXArray {
        let h = image.dim(0), w = image.dim(1), c = image.dim(2)
        let flat = image.asType(.float32).reshaped([h * w, c])
        let ix = (grid[0..., 0..., 0] + 1) / 2 * Float(w - 1)
        let iy = (grid[0..., 0..., 1] + 1) / 2 * Float(h - 1)
        let x0 = floor(ix), y0 = floor(iy)
        let x1 = x0 + 1, y1 = y0 + 1
        let wx1 = ix - x0, wx0 = 1 - wx1, wy1 = iy - y0, wy0 = 1 - wy1
        func corner(_ x: MLXArray, _ y: MLXArray, _ weight: MLXArray) -> MLXArray {
            let valid = (x .>= 0) .&& (x .<= Float(w - 1)) .&& (y .>= 0) .&& (y .<= Float(h - 1))
            let xi = clip(x, min: 0, max: Float(w - 1)).asType(.int32), yi = clip(y, min: 0, max: Float(h - 1)).asType(.int32)
            let values = take(flat, (yi * Int32(w) + xi).flattened(), axis: 0).reshaped([x.dim(0), x.dim(1), c])
            return values * (weight * valid.asType(.float32)).expandedDimensions(axis: -1)
        }
        return corner(x0, y0, wx0 * wy0) + corner(x1, y0, wx1 * wy0) + corner(x0, y1, wx0 * wy1) + corner(x1, y1, wx1 * wy1)
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

/// Reads the image's pixel size from its header, before decoding, and refuses a side over `maxSide` pixels
/// (so a huge or extreme-aspect file is refused without allocating its pixels).
private func checkImageHeader(_ url: URL, maxSide: Int = 8192) throws -> (width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
          let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue
    else { throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.") }
    guard width >= 1, height >= 1, width <= maxSide, height <= maxSide else {
        throw RunnerError.invalid("The image is \(width)x\(height); each side must be 1 to \(maxSide) pixels.")
    }
    return (width, height)
}
