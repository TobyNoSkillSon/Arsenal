// SPDX-License-Identifier: MIT
// Real-ESRGAN x4plus (Comfy-Org/Real-ESRGAN, RealESRGAN_x4plus.safetensors). The network is BasicSR's RRDBNet
// (basicsr/archs/rrdbnet_arch.py: num_feat 64, num_block 23, num_grow_ch 32, scale 4) and the inference path is
// Real-ESRGAN's RealESRGANer (realesrgan/utils.py): RGB in 0...1, tile_process with tile_pad 10, clamp, round to 8 bits,
// and a Lanczos-class resize of the 4x result when a smaller scale or a target height is asked for (outscale).
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import UniformTypeIdentifiers

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "restoration"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["upscale_image"]

private let netScale = 4
/// Input pixels per tile side; each tile also reads `tilePad` pixels of its neighbours (RealESRGANer tile_pad = 10).
private let tileSize = 256
private let tilePad = 10
/// Bounds: the 4x result is held as 8-bit RGBA, so 64 megapixels is 256 MB.
private let maxNetPixels = 64 * 1024 * 1024
private let maxSide = 16384

final class Model {
    let net: RRDBNet

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = RRDBNet()
        var weights = try Weights.load(weightsDir)
        // PyTorch Conv2d [out, in, kh, kw] becomes MLX [out, kh, kw, in].
        for (key, value) in weights where key.hasSuffix(".weight") && value.ndim == 4 {
            weights[key] = value.transposed(0, 2, 3, 1)
        }
        try Weights.apply(weights, to: net)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let url = try params.file("image")
        let loaded = try Images.load(url)
        let width = loaded.width, height = loaded.height
        let netWidth = width * netScale, netHeight = height * netScale
        guard netWidth * netHeight <= maxNetPixels, netWidth <= maxSide, netHeight <= maxSide else {
            throw RunnerError.invalid(
                "The image is \(width)x\(height); at most 4 megapixels (e.g. 2048x2048) and 4096 pixels per side can be upscaled.")
        }
        // Output size: target_height wins, else scale (2 or 4, default 4), as RealESRGANer's outscale.
        var outHeight = netHeight, outWidth = netWidth
        if let target = (params.raw["target_height"] as? NSNumber)?.intValue {
            guard target >= 1, target <= maxSide else { throw RunnerError.invalid("target_height must be 1 to \(maxSide).") }
            outHeight = target
            outWidth = max(1, Int((Double(width) * Double(target) / Double(height)).rounded()))
            guard outWidth <= maxSide, outWidth * outHeight <= maxNetPixels else {
                throw RunnerError.invalid("target_height \(target) gives a \(outWidth)x\(target) image; at most 64 megapixels and \(maxSide) pixels per side.")
            }
        } else if let scale = (params.raw["scale"] as? NSNumber)?.intValue {
            guard scale == 2 || scale == 4 else { throw RunnerError.invalid("scale must be 2 or 4.") }
            outWidth = width * scale; outHeight = height * scale
        }

        let pixels = loaded.pixels.asType(.float16)  // [H, W, 3], RGB 0...1
        var rgba = [UInt8](repeating: 255, count: netWidth * netHeight * 4)
        // tile_process: each tile reads tilePad extra pixels on every side (clamped to the image) and keeps its own part.
        let tilesX = (width + tileSize - 1) / tileSize, tilesY = (height + tileSize - 1) / tileSize
        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                try params.checkCancelled()
                let x0 = tx * tileSize, y0 = ty * tileSize
                let x1 = min(x0 + tileSize, width), y1 = min(y0 + tileSize, height)
                let px0 = max(x0 - tilePad, 0), py0 = max(y0 - tilePad, 0)
                let px1 = min(x1 + tilePad, width), py1 = min(y1 + tilePad, height)
                let tile = pixels[py0..<py1, px0..<px1, 0...].expandedDimensions(axis: 0)
                let out = clip(net(tile)[0].asType(.float32), min: 0, max: 1)
                let keep = out[((y0 - py0) * netScale)..<((y1 - py0) * netScale), ((x0 - px0) * netScale)..<((x1 - px0) * netScale), 0...]
                let bytes = (keep * 255).round().asType(.uint8).asArray(UInt8.self)
                let tw = (x1 - x0) * netScale, th = (y1 - y0) * netScale
                for row in 0..<th {
                    let dst = ((y0 * netScale + row) * netWidth + x0 * netScale) * 4
                    let src = row * tw * 3
                    for col in 0..<tw {
                        rgba[dst + col * 4] = bytes[src + col * 3]
                        rgba[dst + col * 4 + 1] = bytes[src + col * 3 + 1]
                        rgba[dst + col * 4 + 2] = bytes[src + col * 3 + 2]
                    }
                }
                Memory.clearCache()
            }
        }

        var image = try makeImage(&rgba, width: netWidth, height: netHeight)
        if outWidth != netWidth || outHeight != netHeight {
            image = try resize(image, width: outWidth, height: outHeight)
        }
        let dir = try outputDirectory(params)
        let file = dir.appendingPathComponent("upscaled.png")
        try writePNG(image, to: file)
        return ["files": [[
            "id": "image", "path": file.path, "name": "upscaled.png", "mime": "image/png", "label": "Upscaled",
            "bytes": try Files.size(file), "sha256": try Files.sha256(file), "width": outWidth, "height": outHeight,
        ]]]
    }
}

// MARK: - Files

/// Arsenal passes `out_dir` (a job folder under <support>/jobs/); outside Arsenal, a new folder under $ARSENAL_SUPPORT_DIR/jobs.
private func outputDirectory(_ params: Params) throws -> URL {
    let dir: URL
    if let given = params.raw["out_dir"] as? String, !given.isEmpty {
        dir = URL(fileURLWithPath: given)
    } else {
        let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] ?? NSTemporaryDirectory()
        dir = URL(fileURLWithPath: support).appendingPathComponent("jobs").appendingPathComponent(UUID().uuidString)
    }
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func makeImage(_ rgba: inout [UInt8], width: Int, height: Int) throws -> CGImage {
    guard let context = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
        let image = context.makeImage()
    else { throw RunnerError.failed("Could not assemble the upscaled image.") }
    return image
}

/// High-quality (Lanczos-class) resize of the 4x result, as RealESRGANer's cv2.resize(INTER_LANCZOS4) for outscale.
private func resize(_ image: CGImage, width: Int, height: Int) throws -> CGImage {
    guard let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw RunnerError.failed("Could not resize the image.") }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let out = context.makeImage() else { throw RunnerError.failed("Could not resize the image.") }
    return out
}

private func writePNG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw RunnerError.failed("Could not write \(url.lastPathComponent).")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw RunnerError.failed("Could not write \(url.lastPathComponent).") }
}

// MARK: - Network (basicsr/archs/rrdbnet_arch.py), channels-last

private func conv(_ inC: Int, _ outC: Int) -> Conv2d {
    Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: true)
}

private func lrelu(_ x: MLXArray) -> MLXArray { leakyRelu(x, negativeSlope: 0.2) }

/// Nearest-neighbour 2x upsample (F.interpolate(scale_factor=2, mode='nearest')) on [N, H, W, C].
private func upNearest2(_ x: MLXArray) -> MLXArray {
    let n = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    let wide = broadcast(x.reshaped([n, h, 1, w, 1, c]), to: [n, h, 2, w, 2, c])
    return wide.reshaped([n, h * 2, w * 2, c])
}

final class ResidualDenseBlock: Module {
    var conv1: Conv2d, conv2: Conv2d, conv3: Conv2d, conv4: Conv2d, conv5: Conv2d
    init(feat: Int = 64, grow: Int = 32) {
        conv1 = conv(feat, grow)
        conv2 = conv(feat + grow, grow)
        conv3 = conv(feat + 2 * grow, grow)
        conv4 = conv(feat + 3 * grow, grow)
        conv5 = conv(feat + 4 * grow, feat)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = lrelu(conv1(x))
        let x2 = lrelu(conv2(concatenated([x, x1], axis: -1)))
        let x3 = lrelu(conv3(concatenated([x, x1, x2], axis: -1)))
        let x4 = lrelu(conv4(concatenated([x, x1, x2, x3], axis: -1)))
        let x5 = conv5(concatenated([x, x1, x2, x3, x4], axis: -1))
        return x5 * 0.2 + x
    }
}

final class RRDB: Module {
    var rdb1 = ResidualDenseBlock(), rdb2 = ResidualDenseBlock(), rdb3 = ResidualDenseBlock()
    func callAsFunction(_ x: MLXArray) -> MLXArray { rdb3(rdb2(rdb1(x))) * 0.2 + x }
}

final class RRDBNet: Module {
    var conv_first = conv(3, 64)
    var body: [RRDB] = (0..<23).map { _ in RRDB() }
    var conv_body = conv(64, 64)
    var conv_up1 = conv(64, 64)
    var conv_up2 = conv(64, 64)
    var conv_hr = conv(64, 64)
    var conv_last = conv(64, 3)

    /// [N, H, W, 3] RGB 0...1 to [N, 4H, 4W, 3].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let feat = conv_first(x)
        var body = feat
        for block in self.body { body = block(body) }
        var up = feat + conv_body(body)
        up = lrelu(conv_up1(upNearest2(up)))
        up = lrelu(conv_up2(upNearest2(up)))
        return conv_last(lrelu(conv_hr(up)))
    }
}
