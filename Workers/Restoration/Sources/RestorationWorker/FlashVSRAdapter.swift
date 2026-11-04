// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import FlashVSRNative
import MLX
import MLXNN
import ArsenalWire
import AVFoundation

/// FlashVSR v1.1: one-step streaming diffusion video super-resolution. The whole clip is one stream with one seed: the
/// DiT's K/V cache, the LQ projector's and the decoder's carried state run from the first frame to the last.
final class FlashVSRAdapter: NativeModel {
    let architecture = "flashvsr"
    let operations: Set<String> = ["upscale_video"]
    private var pipeline: FlashVSRPipeline?
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] {
        guard let pipeline else { return [:] }
        var result: [String: MLXArray] = [:]
        for (prefix, module) in [("dit.", pipeline.dit as Module), ("lq_proj.", pipeline.lqProj), ("tcdecoder.", pipeline.decoder)] {
            for (name, value) in module.parameters().flattened() { result[prefix + name] = value }
        }
        return result
    }
    /// Get casts the BF16 release to FP16 file by file; nothing is quantized (convolutions and projections stay 16-bit).
    static let conversion = NativeConversion(recipe: "cast:restoration-flashvsr", eligible: { _, _ in false })
    /// Padded output pixels per frame. 1920×1152 is the largest size the port's author measured (33.7 GB peak, BF16);
    /// larger frames follow the same per-pixel line up to this bound (about 57 GB, derived, not measured).
    static let maxOutputPixels = 2048 * 2048
    /// Freed-buffer cache bound during a run: without it MLX keeps the working set's high water (14 to 18 GB) cached.
    static let cacheLimit = 2 << 30

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let half: DType = context.precision == "bf16" ? .bfloat16 : .float16
        guard ["fp16", "bf16"].contains(context.precision) else { throw MediaBridge.error("FlashVSR runs at FP16 or BF16 only.") }
        if context.randomWeights {
            // Plumbing fixture: one 128-wide DiT block, a narrow projector and decoder; same graph and schedule.
            let dit = FlashVSRDiT(.tiny()), lq = LQProjector(outDim: 128, layers: 1, hidden: (64, 64)), decoder = TCDecoder(channels: [32, 32, 32, 32])
            for module in [dit as Module, lq, decoder] { try NativeParameters.initializeRandom(module, half: half) }
            dit.prepare(context: MLXArray.zeros([1, 512, FlashVSRDiTConfig.tiny().textDim], dtype: half))
            pipeline = FlashVSRPipeline(dit: dit, lqProj: lq, decoder: decoder)
        } else {
            pipeline = try FlashVSRPipeline.load(directory: context.weights, dtypes: FlashVSRDTypes(dit: half, lqProj: half, decoder: half))
        }
        try cancellation.check()
    }

    /// Output size for a source frame: even sides from `target_height` or `scale`, then the padded model size (multiples of 128).
    static func geometry(height: Int, width: Int, scale: Double, targetHeight: Int?) throws -> (height: Int, width: Int, paddedHeight: Int, paddedWidth: Int) {
        guard scale.isFinite, (1...4).contains(scale), height > 0, width > 0 else { throw MediaBridge.error("scale must be between 1 and 4.") }
        let h = (targetHeight ?? Int((Double(height) * scale).rounded())) / 2 * 2
        let w = Int((Double(width) * Double(h) / Double(height)).rounded()) / 2 * 2
        guard h >= height / 2 * 2, h <= height * 4, w >= 2 else { throw MediaBridge.error("FlashVSR enlarges from one to four times; target_height must be in that range.") }
        let ph = (h + 127) / 128 * 128, pw = (w + 127) / 128 * 128
        guard ph <= 8192, pw <= 8192, ph * pw <= maxOutputPixels else {
            throw MediaBridge.error("Output \(w)×\(h) exceeds FlashVSR's bound of about 4.2 megapixels per frame; lower scale or target_height.")
        }
        return (h, w, ph, pw)
    }

    /// 32BGRA frame → (H, W, 3) float32 RGB in 0…255, as the port's driver reads it.
    static func rgb255(_ sample: CMSampleBuffer) throws -> MLXArray {
        guard let buffer = CMSampleBufferGetImageBuffer(sample), CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else {
            throw MediaBridge.error("Missing video frame pixels.")
        }
        CVPixelBufferLockBaseAddress(buffer, .readOnly); defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer), row = CVPixelBufferGetBytesPerRow(buffer)
        guard width > 0, height > 0, width <= 16_000_000 / height, let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw MediaBridge.error("Video frame exceeds 16 megapixels.")
        }
        var packed = [UInt8](repeating: 0, count: width * height * 4)
        packed.withUnsafeMutableBytes { dst in for y in 0..<height { memcpy(dst.baseAddress! + y * width * 4, base + y * row, width * 4) } }
        let bgra = MLXArray(packed, [height, width, 4])
        return concatenated([bgra[.ellipsis, 2..<3], bgra[.ellipsis, 1..<2], bgra[.ellipsis, 0..<1]], axis: -1).asType(.float32)
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "upscale_video", let pipeline else { throw MediaBridge.error("FlashVSR is not loaded or the operation is unsupported.") }
        let root = try MediaBridge.output(input), scale = input["scale"]?.number ?? 2
        let rawHeight = input["target_height"]?.number
        guard rawHeight == nil || (rawHeight!.isFinite && rawHeight!.rounded() == rawHeight && (2...8192).contains(rawHeight!)) else {
            throw MediaBridge.error("target_height must be an integer from 2 to 8192.")
        }
        let rawSeed = input["seed"]?.number ?? 0
        guard rawSeed.isFinite, rawSeed.rounded() == rawSeed, (0...Double(UInt32.max)).contains(rawSeed) else { throw MediaBridge.error("seed must be a uint32.") }
        let options = FlashVSROptions(colorFix: input["color_correction"]?.bool ?? true)
        let source = try MediaBridge.path(input["video"]), reader = try MediaBridge.wait { try await NativeVideoReader(url: source) }
        defer { reader.cancel() }
        let previousCache = Memory.cacheLimit
        Memory.cacheLimit = Self.cacheLimit
        defer { Memory.cacheLimit = previousCache; Memory.clearCache() }
        let url = root.appendingPathComponent("upscaled.mp4"), expected = MediaBridge.frameEstimate(source)
        var writer: NativeVideoWriter?, done = false, written = 0, origin: Double?, times: [CMTime] = []
        var stream: FlashVSRStream?, size: (height: Int, width: Int, paddedHeight: Int, paddedWidth: Int)?
        var matrices: (y: MLXArray, x: MLXArray)?, sourceSize = (0, 0)
        defer { if !done { writer?.cancel(); try? FileManager.default.removeItem(at: url) } }
        try MediaBridge.begin(total: expected, label: "Upscaling", emit: emit)
        func write(_ frames: [MLXArray]) throws {
            guard let g = size else { return }
            for frame in frames {
                try cancellation.check()
                let image = ((frame[0..<g.height, 0..<g.width] + 1) / 2).expandedDimensions(axis: 0)
                if writer == nil { writer = try NativeVideoWriter(url: url, width: g.width, height: g.height, transform: reader.transform) }
                try writer!.append(MediaBridge.buffer(image), time: times[written], cancellation: cancellation); written += 1
                try MediaBridge.progress(written, total: expected, label: "Upscaling", emit: emit)
            }
        }
        while let sample = try reader.next() {
            try cancellation.check()
            let timestamp = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            if origin == nil { origin = timestamp }
            let frame = try Self.rgb255(sample)
            if size == nil {
                // The first frame fixes the output size and starts the clip's one stream.
                let g = try Self.geometry(height: frame.dim(0), width: frame.dim(1), scale: scale, targetHeight: rawHeight.map(Int.init))
                size = g; sourceSize = (frame.dim(0), frame.dim(1))
                matrices = (FlashVSRPreprocess.bicubicMatrix(inN: frame.dim(0), outN: g.height), FlashVSRPreprocess.bicubicMatrix(inN: frame.dim(1), outN: g.width))
                stream = FlashVSRStream(pipeline: pipeline, height: g.paddedHeight, width: g.paddedWidth, options: options, seed: UInt64(rawSeed))
            }
            guard let g = size, let stream, sourceSize == (frame.dim(0), frame.dim(1)) else {
                throw MediaBridge.error("Video frames change size.")
            }
            var up = FlashVSRPreprocess.bicubic8(frame, outH: g.height, outW: g.width, matrices: matrices)
            if g.paddedHeight > g.height || g.paddedWidth > g.width {
                up = padded(up, widths: [.init((0, g.paddedHeight - g.height)), .init((0, g.paddedWidth - g.width)), 0], mode: .edge)
            }
            times.append(CMTime(seconds: timestamp - origin!, preferredTimescale: 60000))
            try write(stream.push(FlashVSRPreprocess.toModel(up)))
        }
        guard let stream else { throw MediaBridge.error("Video has no frames.") }
        try write(stream.finish())
        guard let writer, written == times.count else { throw MediaBridge.error("FlashVSR returned \(written) frames for \(times.count).") }
        try MediaBridge.wait { try await writer.finish() }; done = true
        do {
            try MediaBridge.keepAudio(from: source, video: url, origin: origin ?? 0, cancellation: cancellation)
        } catch {
            try? FileManager.default.removeItem(at: url); throw error
        }
        return ["files": [.object(try NativeMedia.file(url, root: root, mime: "video/mp4", id: "video"))]]
    }
    func unload() { pipeline = nil; Memory.clearCache() }
}
