// High-level pipeline API: CGImage in, mask out.
//
// Mirrors the shape of `MoGePipeline` in mlx-swift-MoGe and
// `DepthAnything3Pipeline` in mlx-swift-da3. Users should reach for this
// API by default; the bare `BiRefNet` Module is the lower-level escape hatch.

import CoreGraphics
import Foundation
import ImageIO
import MLX

#if canImport(UIKit)
    import UIKit
    public typealias PlatformImage = UIImage
#elseif canImport(AppKit)
    import AppKit
    public typealias PlatformImage = NSImage
#endif

// MARK: - Normalisation

/// ImageNet normalisation used by every BiRefNet training recipe.
public struct ImageNorm: Sendable {
    public let mean: (Float, Float, Float)
    public let std: (Float, Float, Float)
    public init(mean: (Float, Float, Float), std: (Float, Float, Float)) {
        self.mean = mean
        self.std = std
    }
    public static let imageNet = ImageNorm(
        mean: (0.485, 0.456, 0.406),
        std:  (0.229, 0.224, 0.225)
    )
}

public enum BiRefNetPipelineError: Error {
    case invalidImage
    case maskRenderFailed
}

// MARK: - Prediction

public struct BiRefNetPrediction {
    public let outputs: [String: MLXArray]
    public let sourceHeight: Int
    public let sourceWidth: Int

    public init(outputs: [String: MLXArray], sourceHeight: Int, sourceWidth: Int) {
        self.outputs = outputs
        self.sourceHeight = sourceHeight
        self.sourceWidth = sourceWidth
    }

    /// Sigmoid mask at the model's working resolution (e.g. 1024×1024),
    /// shape `(1, H, W, 1)`.
    public var mask: MLXArray? { outputs["mask"] }

    /// Sigmoid mask resampled back to the source image resolution,
    /// shape `(1, sourceHeight, sourceWidth, 1)`.
    public var maskAtSource: MLXArray? { outputs["mask_source"] }

    /// Render the source-resolution mask to an 8-bit grayscale `CGImage`.
    /// White = foreground, black = background.
    public func maskCGImage() throws -> CGImage {
        guard let m = outputs["mask_source_bytes"] ?? maskAtSource else { throw BiRefNetPipelineError.maskRenderFailed }
        return try BiRefNetPipeline.renderGrayscale(mask: m)
    }
}

// MARK: - Pipeline

public struct BiRefNetPipeline {
    public let model: BiRefNet
    public let dtype: DType
    public let inputSize: (width: Int, height: Int)
    public let norm: ImageNorm

    public init(
        model: BiRefNet,
        dtype: DType = .float16,
        inputSize: (width: Int, height: Int) = (1024, 1024),
        norm: ImageNorm = .imageNet
    ) {
        self.model = model
        self.dtype = dtype
        self.inputSize = inputSize
        self.norm = norm
    }

    /// CGImage → row-major NHWC float tensor, normalised with `norm`, cast
    /// to `self.dtype` so the matmul chain stays at the configured precision
    /// (otherwise MLX promotes everything to f32 when activations and weights
    /// disagree).
    public func preprocess(_ image: CGImage) -> MLXArray {
        let raw = BiRefNetPipeline.cgImageToNHWC(
            image, targetW: inputSize.width, targetH: inputSize.height, norm: norm)
        return dtype == .float32 ? raw : raw.asType(dtype)
    }

    /// Run the model on a preprocessed `(1, H, W, 3)` tensor and return a
    /// dictionary `{"mask": sigmoid output @ model resolution}`. Forces
    /// `eval` so the user can time the call meaningfully.
    public func predict(_ input: MLXArray) -> [String: MLXArray] {
        var logits = model(input)
        logits = sigmoid(logits)
        eval(logits)
        return ["mask": logits]
    }

    /// One-call convenience: CGImage → `BiRefNetPrediction`. The prediction
    /// always carries both the model-resolution mask (`mask`) and the
    /// source-resolution mask (`mask_source`).
    public func callAsFunction(_ image: CGImage) -> BiRefNetPrediction {
        let input = preprocess(image)
        var outputs = predict(input)
        if let m = outputs["mask"] {
            // Vendor ToPILImage quantizes before Pillow's default bicubic resize.
            let bytes = Self.sourceMaskBytes(m, width: image.width, height: image.height)
            outputs["mask_source_bytes"] = MLXArray(bytes, [1, image.height, image.width, 1])
            outputs["mask_source"] = MLXArray(bytes, [1, image.height, image.width, 1]).asType(.float32) / 255
        }
        return BiRefNetPrediction(
            outputs: outputs,
            sourceHeight: image.height,
            sourceWidth: image.width
        )
    }
}

// MARK: - Loading

extension BiRefNetPipeline {

    /// Load the pinned original PyTorch-layout `.safetensors` checkpoint.
    /// Layout views are made in memory; the stored FP16 weights stay intact.
    ///
    /// - Parameters:
    ///   - path: path to either the weights file directly, or a directory
    ///           containing `model.safetensors`.
    ///   - dtype: working precision; defaults to `.float16` (the right
    ///            default for an attention/conv-heavy model on Apple
    ///            Silicon — see fast-mlx skill notes).
    ///   - config: model configuration; defaults to
    ///             ``BiRefNetConfig/swinLargeDefault`` which matches the
    ///             flagship `ZhengPeng7/BiRefNet` checkpoint. Override
    ///             when loading a variant (e.g. `_lite` with `.swinV1T`,
    ///             or a 2K HR variant with `inputSize = (2048, 2048)`).
    public static func fromPretrained(
        _ path: String,
        dtype: DType = .float16,
        config: BiRefNetConfig = .swinLargeDefault,
        bits: Int? = nil
    ) throws -> BiRefNetPipeline {
        let raw = URL(fileURLWithPath: path)
        let weightsURL: URL
        var isDir: ObjCBool = false
        FileManager.default.fileExists(atPath: raw.path, isDirectory: &isDir)
        if isDir.boolValue {
            weightsURL = raw.appendingPathComponent("model.safetensors")
        } else {
            weightsURL = raw
        }

        let model = BiRefNet(config: config)
        try model.loadWeights(url: weightsURL, dtype: dtype, strict: true, bits: bits)
        return BiRefNetPipeline(model: model, dtype: dtype,
                                 inputSize: config.inputSize)
    }

    public static func fromPretrained(
        url: URL,
        dtype: DType = .float16,
        config: BiRefNetConfig = .swinLargeDefault
    ) throws -> BiRefNetPipeline {
        try fromPretrained(url.path, dtype: dtype, config: config)
    }
}

// MARK: - Top-level convenience

public enum MLXBiRefNet {
    public static func fromPretrained(
        _ path: String,
        dtype: DType = .float16
    ) throws -> BiRefNetPipeline {
        try BiRefNetPipeline.fromPretrained(path, dtype: dtype)
    }
}

// MARK: - Convenience: bare-model `segment(image:)`

extension BiRefNet {

    /// Convenience wrapper that builds a one-shot pipeline and segments a
    /// single `CGImage`. Use ``BiRefNetPipeline`` directly if you're going
    /// to make many calls — preprocessing parameters and dtype are configured
    /// once on the pipeline, not on every call.
    public func segment(
        _ image: CGImage,
        inputSize: (width: Int, height: Int) = (1024, 1024),
        norm: ImageNorm = .imageNet,
        dtype: DType = .float16
    ) throws -> CGImage {
        let pipeline = BiRefNetPipeline(
            model: self, dtype: dtype,
            inputSize: inputSize, norm: norm)
        let pred = pipeline(image)
        return try pred.maskCGImage()
    }

    // Back-compat: a few tests + the CLI call these directly. They stay as
    // pass-throughs to the pipeline's implementation.
    public static func preprocess(
        image: CGImage, targetW: Int, targetH: Int, norm: ImageNorm
    ) throws -> MLXArray {
        BiRefNetPipeline.cgImageToNHWC(image, targetW: targetW, targetH: targetH, norm: norm)
    }

    public static func renderGrayscale(mask: MLXArray) throws -> CGImage {
        try BiRefNetPipeline.renderGrayscale(mask: mask)
    }
}

// MARK: - Bitmap helpers

extension BiRefNetPipeline {

    /// CGImage → row-major NHWC float32 tensor, ImageNet-normalised, at
    /// `(targetW, targetH)`. The tensor stays in float32; the pipeline casts
    /// to `dtype` separately so test code can ask for an unquantised version.
    static func cgImageToNHWC(
        _ image: CGImage, targetW: Int, targetH: Int, norm: ImageNorm
    ) -> MLXArray {
        let pixels = resizeToRGBPixels(image: image, width: targetW, height: targetH) ?? []
        precondition(pixels.count == targetW * targetH * 3,
                     "preprocess: failed to extract \(targetW)x\(targetH) RGB pixels")

        var floats = [Float](repeating: 0, count: pixels.count)
        let m = (norm.mean.0, norm.mean.1, norm.mean.2)
        let invS = (1.0 / norm.std.0, 1.0 / norm.std.1, 1.0 / norm.std.2)
        let inv255: Float = 1.0 / 255.0
        for i in 0..<(targetW * targetH) {
            let r = Float(pixels[i * 3 + 0]) * inv255
            let g = Float(pixels[i * 3 + 1]) * inv255
            let b = Float(pixels[i * 3 + 2]) * inv255
            floats[i * 3 + 0] = (r - m.0) * invS.0
            floats[i * 3 + 1] = (g - m.1) * invS.1
            floats[i * 3 + 2] = (b - m.2) * invS.2
        }
        return MLXArray(floats, [1, targetH, targetW, 3])
    }

    /// `(1, H, W, 1)` float in `[0, 1]` → 8-bit grayscale `CGImage`.
    static func renderGrayscale(mask: MLXArray) throws -> CGImage {
        precondition(mask.ndim == 4 && mask.dim(0) == 1 && mask.dim(3) == 1)
        let H = mask.dim(1)
        let W = mask.dim(2)
        let bytes = maskBytes(mask)
        let cs = CGColorSpaceCreateDeviceGray()
        let bitmapInfo: UInt32 = CGImageAlphaInfo.none.rawValue
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let img = CGImage(
                width: W, height: H,
                bitsPerComponent: 8, bitsPerPixel: 8,
                bytesPerRow: W, space: cs,
                bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent)
        else { throw BiRefNetPipelineError.maskRenderFailed }
        return img
    }

    static func sourceMaskBytes(_ mask: MLXArray, width: Int, height: Int) -> [UInt8] {
        ImageBoundary.resize(maskBytes(mask), width: mask.dim(2), height: mask.dim(1), channels: 1,
            targetWidth: width, targetHeight: height, filter: .bicubic)
    }

    static func maskBytes(_ mask: MLXArray) -> [UInt8] {
        if mask.dtype == .uint8 { return mask.asArray(UInt8.self) }
        // torchvision ToPILImage multiplies in the NumPy tensor dtype, then truncates to uint8.
        if mask.dtype == .float16 {
            return mask.asArray(Float16.self).map { UInt8(clamping: Int(min(Float16(1), max(0, $0)) * Float16(255))) }
        }
        return mask.asType(.float32).asArray(Float.self).map { UInt8(clamping: Int(min(1, max(0, $0)) * 255)) }
    }

    static func resizeToRGBPixels(image: CGImage, width: Int, height: Int) -> [UInt8]? {
        guard let pixels = ImageBoundary.straightRGB(image) else { return nil }
        return ImageBoundary.resize(pixels, width: image.width, height: image.height, channels: 3,
            targetWidth: width, targetHeight: height, filter: .bilinear)
    }

}

// MARK: - I/O helpers

public func loadCGImage(url: URL) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { return nil }
    return img
}

public func saveCGImage(_ image: CGImage, url: URL) throws {
    let type: CFString
    switch url.pathExtension.lowercased() {
    case "png": type = "public.png" as CFString
    case "jpg", "jpeg": type = "public.jpeg" as CFString
    default: type = "public.png" as CFString
    }
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type, 1, nil) else {
        throw BiRefNetPipelineError.maskRenderFailed
    }
    try finalizeImage(image, destination: dest)
}

// Shared by file output and the failing-consumer regression fixture.
func finalizeImage(_ image: CGImage, destination: CGImageDestination) throws {
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw BiRefNetPipelineError.maskRenderFailed }
}
