// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import SeedVR2Native
import RIFENative
import MLX
import MLXNN
import MLXRandom
import ArsenalWire
import AVFoundation

final class SeedVR2Adapter: NativeModel {
    let architecture = "seedvr2"
    let operations: Set<String> = ["upscale_image", "upscale_video"]
    private var transformer: SeedVR2Transformer?
    private var vae: SeedVR2VAE?
    private var text: MLXArray?
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (prefix, module) in [("transformer.", transformer as Module?), ("vae.", vae as Module?)] {
            for (name, value) in module?.parameters().flattened() ?? [] { result[prefix + name] = value }
        }
        if let text { result["pos_emb.embedding"] = text.squeezed(axis: 0) }
        return result
    }
    static func eligible(_ name: String, _ shape: [Int]) -> Bool { shape.count == 2 && name.hasSuffix(".weight") && shape[1] % 64 == 0 }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        var config = SeedVR2Config.r3B
        let small = context.randomWeights && context.config["small"]?.bool == true
        if small { config.vidDim = 128; config.heads = 1; config.txtInDim = 64; config.numLayers = 2; config.mmLayers = 1; config.expandRatio = 2 }
        let transformer = SeedVR2Transformer(config), vae = SeedVR2VAE(blockOut: small ? [64, 64, 64, 64] : [128, 256, 512, 512])
        if context.randomWeights {
            try NativeParameters.initializeRandom(transformer); try NativeParameters.initializeRandom(vae)
            text = MLXArray.zeros([1, 58, config.txtInDim], dtype: .float16)
            if let bits = try NativePrecisionTier(context.precision).bits {
                for module in [transformer as Module, vae as Module] {
                    quantize(model: module, groupSize: 64, bits: bits) { name, layer in
                        guard let linear = layer as? Linear else { return false }; return Self.eligible(name + ".weight", linear.weight.shape)
                    }
                }
            }
        } else {
            let tier = try NativePrecisionTier(context.precision)
            _ = try tier.load(transformer, archives: [context.weights.appendingPathComponent("transformer.safetensors")], map: { ($0, $1) }, eligible: Self.eligible)
            try cancellation.check()
            _ = try tier.load(vae, archives: [context.weights.appendingPathComponent("vae.safetensors")], map: { ($0, $1) }, eligible: Self.eligible)
            guard let embedding = try loadArrays(url: context.weights.appendingPathComponent("pos_emb.safetensors"))["embedding"], embedding.shape == [58, 5120] else {
                throw MediaBridge.error("Invalid SeedVR2 positive embedding.")
            }
            text = embedding.asType(.float16).expandedDimensions(axis: 0)
        }
        transformer.train(false); vae.train(false); self.transformer = transformer; self.vae = vae
    }
    /// Output size for a source frame: even sides, at most 8192 per side and 16 megapixels.
    static func outputSize(height sourceHeight: Int, width sourceWidth: Int, scale: Double, targetHeight: Int?) throws -> (height: Int, width: Int) {
        guard scale.isFinite, (1...8).contains(scale), sourceHeight > 0, sourceWidth > 0 else { throw MediaBridge.error("Invalid SeedVR2 batch or scale.") }
        let h = (targetHeight ?? Int((Double(sourceHeight) * scale).rounded())) / 2 * 2
        let w = Int((Double(sourceWidth) * Double(h) / Double(sourceHeight)).rounded()) / 2 * 2
        guard (1...8192).contains(h), (1...8192).contains(w), w <= 16_000_000 / h else { throw MediaBridge.error("Upscale output exceeds the pixel bound.") }
        return (h, w)
    }
    /// Pixel-frames of the VAE's full-resolution activations held per spatial tile. At roughly 4 KB of FP16/FP32
    /// activations per output pixel-frame in the last decoder stage this keeps one tile near 8 GiB (derived, not measured).
    static let vaePixelFrames = 2_097_152
    /// Spatial VAE tile side in output pixels for a chunk of `frames`: 1,408 for an image, 640 for five frames, 256 for 21.
    static func vaeTile(frames: Int) -> Int { Swift.max(256, Int((Double(vaePixelFrames) / Double(Swift.max(1, frames))).squareRoot()) / 64 * 64) }
    /// Video frames per diffusion pass (4n+1). Upstream runs a whole clip in one pass with one noise draw; Arsenal bounds the
    /// pass to 21 frames at 720p output and shortens it in steps of four for larger frames, never below five.
    static func chunkFrames(outputPixels: Int) -> Int {
        var frames = 21
        while frames > 5 && frames * outputPixels > 21 * 1280 * 720 { frames -= 4 }
        return frames
    }
    /// Latent frames the causal VAE makes from `frames` input frames once padded to 4n+1.
    static func latentFrames(_ frames: Int) -> Int { (frames - 1 + 3) / 4 + 1 }
    /// One noise draw per request: shape [1,16,latent frames,H/8,W/8] for the longest chunk; shorter chunks take a prefix.
    static func noise(seed: UInt64, height: Int, width: Int, frames: Int) -> MLXArray {
        let ph = (height + 15) / 16 * 16, pw = (width + 15) / 16 * 16
        return MLXRandom.normal([1, 16, latentFrames(frames), ph / 8, pw / 8], key: MLXRandom.key(seed)).asType(.float16)
    }
    private func upscale(
        _ images: [MLXArray], height h: Int, width w: Int, noise: MLXArray, softness: Double, colorCorrection: Bool, cancellation: NativeCancellation,
        stage: (Int, String) throws -> Void = { _, _ in }
    ) throws -> [MLXArray] {
        try cancellation.check()
        guard let transformer, let vae, let text, let first = images.first, images.allSatisfy({ $0.shape == first.shape }) else {
            throw MediaBridge.error("Invalid SeedVR2 batch or scale.")
        }
        let ph = (h + 15) / 16 * 16, pw = (w + 15) / 16 * 16
        let prepared = try images.map { image -> MLXArray in
            var result = try MediaBridge.lanczos(image, width: w, height: h)
            if softness > 0 {
                let factor = 1 + softness * 7
                result = try MediaBridge.lanczos(result, width: Swift.max(1, Int(Double(w) / factor)), height: Swift.max(1, Int(Double(h) / factor)))
                result = try MediaBridge.lanczos(result, width: w, height: h)
            }
            return result
        }
        let resized = prepared.map { image in
            padded(image, widths: [.init((0, 0)), .init((0, ph - h)), .init((0, pw - w)), .init((0, 0))]) * 2 - 1
        }
        var video = stacked(resized, axis: 1).transposed(0, 4, 1, 2, 3)
        // Causal VAE expands time by four. Repeat the final frame to complete a 4n+1 chunk, as upstream cut_videos does.
        let wanted = images.count
        let paddedCount = ((wanted - 1 + 3) / 4) * 4 + 1
        if paddedCount > wanted { video = concatenated([video, repeated(video[0..., 0..., (wanted - 1)..<wanted], count: paddedCount - wanted, axis: 2)], axis: 2) }
        let tile = Self.vaeTile(frames: paddedCount)
        try stage(1, "Encoding")
        let initial = vae.encode(video, tile: tile); eval(initial); try cancellation.check()
        guard initial.dim(2) == Self.latentFrames(paddedCount), noise.dim(2) >= initial.dim(2), noise.dim(3) == initial.dim(3), noise.dim(4) == initial.dim(4) else {
            throw MediaBridge.error("SeedVR2 latent shape does not match its noise.")
        }
        let condition = concatenated([initial, MLXArray.ones([1, 1, initial.dim(2), initial.dim(3), initial.dim(4)], dtype: .float16)], axis: 1)
        let chunkNoise = noise[0..., 0..., 0..<initial.dim(2)]
        try stage(2, "Restoring")
        let prediction = transformer(concatenated([chunkNoise, condition], axis: 1), text, timestep: MLXArray(Float(1000)).asType(.float16)); eval(prediction)
        try cancellation.check()
        try stage(3, "Decoding")
        let decoded = vae.decode(chunkNoise - prediction, tile: tile); eval(decoded); try cancellation.check()
        guard decoded.dim(2) == paddedCount, decoded.dim(2) == SeedVR2VAE.decodedFrames(latentFrames: initial.dim(2)) else {
            throw MediaBridge.error("SeedVR2 decoder returned \(decoded.dim(2)) frames for \(paddedCount).")
        }
        return (0..<wanted).map { index in
            var result = decoded[0..., 0..., index, ..<h, ..<w]
            if colorCorrection { result = SeedVR2ColorCorrect.labTransfer(content: result, style: prepared[index].transposed(0, 3, 1, 2) * 2 - 1) }
            return ((result.transposed(0, 2, 3, 1) + 1) / 2).asType(.float16)
        }
    }
    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        let root = try MediaBridge.output(input), scale = input["scale"]?.number ?? 2
        let rawHeight = input["target_height"]?.number
        guard rawHeight == nil || (rawHeight!.isFinite && rawHeight!.rounded() == rawHeight && (2...8192).contains(rawHeight!)) else {
            throw MediaBridge.error("target_height must be an integer from 2 to 8192.")
        }
        let target = rawHeight.map(Int.init), rawSeed = input["seed"]?.number ?? Double(UInt32.random(in: 0...UInt32.max))
        guard rawSeed.isFinite, rawSeed.rounded() == rawSeed, (0...Double(UInt32.max)).contains(rawSeed) else { throw MediaBridge.error("seed must be a uint32.") }
        let seed = UInt64(rawSeed), softness = input["softness"]?.number ?? 0
        guard softness.isFinite, (0...1).contains(softness) else { throw MediaBridge.error("softness must be between zero and one.") }
        let colorCorrection = input["color_correction"]?.bool ?? true
        if operation == "upscale_image" {
            let image = try MediaBridge.tensor(NativeMedia.image(MediaBridge.path(input["image"])))
            let size = try Self.outputSize(height: image.dim(1), width: image.dim(2), scale: scale, targetHeight: target)
            let noise = Self.noise(seed: seed, height: size.height, width: size.width, frames: 1)
            let result = try upscale(
                [image], height: size.height, width: size.width, noise: noise, softness: softness, colorCorrection: colorCorrection, cancellation: cancellation,
                stage: { step, label in try MediaBridge.stage(step, of: 3, label, emit: emit) })[0]
            let url = root.appendingPathComponent("upscaled.png"); try MediaBridge.png(result, at: url)
            return ["files": [.object(try NativeMedia.file(url, root: root, mime: "image/png", id: "image"))]]
        }
        guard operation == "upscale_video" else { throw MediaBridge.error("Unsupported SeedVR2 operation.") }
        let source = try MediaBridge.path(input["video"]), reader = try MediaBridge.wait { try await NativeVideoReader(url: source) }
        defer { reader.cancel() }
        let url = root.appendingPathComponent("upscaled.mp4"), expected = MediaBridge.frameEstimate(source)
        var writer: NativeVideoWriter?, done = false, index = 0, origin: Double?
        var size: (height: Int, width: Int)?, chunk = 5, noise: MLXArray?
        defer { if !done { writer?.cancel(); try? FileManager.default.removeItem(at: url) } }
        try MediaBridge.begin(total: expected, label: "Upscaling", emit: emit)
        while true {
            var images: [MLXArray] = [], times: [CMTime] = []
            while images.count < chunk {
                try cancellation.check(); guard let sample = try reader.next() else { break }
                let timestamp = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                if origin == nil { origin = timestamp }
                let image = try MediaBridge.tensor(sample)
                if size == nil {
                    // The first frame fixes the output size, the chunk length and the clip's single noise draw.
                    let first = try Self.outputSize(height: image.dim(1), width: image.dim(2), scale: scale, targetHeight: target)
                    size = first; chunk = Self.chunkFrames(outputPixels: first.height * first.width)
                    noise = Self.noise(seed: seed, height: first.height, width: first.width, frames: chunk)
                }
                images.append(image); times.append(CMTime(seconds: timestamp - origin!, preferredTimescale: 60000))
            }
            if images.isEmpty { break }
            guard let outSize = size, let clipNoise = noise else { throw MediaBridge.error("Video has no frames.") }
            let frames = try upscale(
                images, height: outSize.height, width: outSize.width, noise: clipNoise, softness: softness, colorCorrection: colorCorrection, cancellation: cancellation)
            if writer == nil { writer = try NativeVideoWriter(url: url, width: frames[0].dim(2), height: frames[0].dim(1), transform: reader.transform) }
            for (frame, time) in zip(frames, times) {
                try writer!.append(MediaBridge.buffer(frame), time: time, cancellation: cancellation); index += 1
                try MediaBridge.progress(index, total: expected, label: "Upscaling", emit: emit)
            }
        }
        guard let writer else { throw MediaBridge.error("Video has no frames.") }
        try MediaBridge.wait { try await writer.finish() }; done = true
        do {
            try MediaBridge.keepAudio(from: source, video: url, origin: origin ?? 0, cancellation: cancellation)
        } catch {
            try? FileManager.default.removeItem(at: url); throw error
        }
        return ["files": [.object(try NativeMedia.file(url, root: root, mime: "video/mp4", id: "video"))]]
    }
    func unload() { transformer = nil; vae = nil; text = nil }
}
