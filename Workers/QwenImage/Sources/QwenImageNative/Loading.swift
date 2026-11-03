// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import Tokenizers

/// Checkpoint names → module names, and the quantization rule shared by Get (int8/int4 derivation) and Load.
public enum QwenImageWeights {
    /// Text encoder: `model.visual.*` → `visual.*`, `model.language_model.*` → `language_model.*`. The LM head and the
    /// final norm are never read (the pipeline takes the last layer's output before that norm).
    public static func encoderName(_ key: String) -> String? {
        if key == "lm_head.weight" || key == "model.language_model.norm.weight" { return nil }
        if key.hasPrefix("model.visual.") { return "visual." + key.dropFirst("model.visual.".count) }
        if key.hasPrefix("model.language_model.") { return "language_model." + key.dropFirst("model.language_model.".count) }
        return key
    }
    /// Transformer: `modulation.1` (nn.Sequential index) → `modulation.linear`.
    public static func transformerName(_ key: String) -> String {
        key.hasPrefix("modulation.1.") ? "modulation.linear." + key.dropFirst("modulation.1.".count) : key
    }
    /// VAE: temporal convolutions never run on one frame; `resample.1` → `conv`.
    public static func vaeName(_ key: String) -> String? {
        if key.contains(".time_conv.") { return nil }
        return key.replacingOccurrences(of: ".resample.1.", with: ".conv.")
    }
    /// VAE tensors in MLX layout: OIHW convolutions → OHWI, γ [C,1,1(,1)] → [C].
    public static func vaeTensor(_ name: String, _ value: MLXArray) -> MLXArray {
        if name.hasSuffix(".gamma") { return value.reshaped(-1) }
        if name.hasSuffix(".weight"), value.ndim == 4 { return value.transposed(0, 2, 3, 1) }
        return value
    }
    /// Every Linear or Embedding matrix whose input width is a multiple of 64 packs to int8/int4; the vision position
    /// table (read by bilinear weights), the patch kernel (5-D), convolutions, norms and vectors stay half.
    public static func eligible(_ name: String, _ shape: [Int]) -> Bool {
        shape.count == 2 && name.hasSuffix(".weight") && shape[1] % 64 == 0 && !name.contains("pos_embed")
    }
}

/// The byte tokenizer used only by random-weight lab smokes: the real graph runs on reduced dimensions, the checkpoint's
/// tokenizer is not involved. Special tokens map to their configured IDs.
public struct ByteTokens {
    public let imagePad: Int, visionStart: Int, visionEnd: Int, imStart: Int, imEnd: Int
    public init(config: Qwen3VLConfig) {
        imagePad = config.imageToken; visionStart = config.visionStart; visionEnd = config.visionEnd
        imStart = config.visionStart; imEnd = config.visionEnd
    }
    public func encode(_ text: String) -> [Int] {
        let specials = ["<|image_pad|>": imagePad, "<|vision_start|>": visionStart, "<|vision_end|>": visionEnd, "<|im_start|>": imStart, "<|im_end|>": imEnd]
        var rest = text[...], out: [Int] = []
        while !rest.isEmpty {
            if let s = specials.first(where: { rest.hasPrefix($0.key) }) { out.append(s.value); rest = rest.dropFirst(s.key.count); continue }
            out += String(rest.removeFirst()).utf8.map { Int($0) + 1 }
        }
        return out
    }
}

/// Loads the checkpoint's tokenizer from `processor/` without any network access.
public enum LocalTokenizer {
    public static func load(_ directory: URL) throws -> any Tokenizer {
        let box = Box(), done = DispatchSemaphore(value: 0)
        Task {
            do { box.result = .success(try await AutoTokenizer.from(modelFolder: directory)) } catch { box.result = .failure(error) }
            done.signal()
        }
        done.wait()
        return try box.result!.get()
    }
    private final class Box: @unchecked Sendable { var result: Result<any Tokenizer, Error>? }
}

/// Small configurations for the lab-only random-weight smoke (same graph, reduced widths).
public enum SmallConfigs {
    public static var encoder: Qwen3VLConfig {
        var c = Qwen3VLConfig()
        c.text.hidden = 64; c.text.layers = 2; c.text.heads = 2; c.text.kvHeads = 1; c.text.headDim = 32; c.text.intermediate = 128
        c.text.vocab = 512; c.text.mropeSection = [6, 5, 5]
        c.vision.depth = 2; c.vision.hidden = 64; c.vision.heads = 2; c.vision.intermediate = 128; c.vision.outHidden = 64
        c.vision.positions = 16; c.vision.deepstack = [0, 1]
        c.imageToken = 300; c.visionStart = 301; c.visionEnd = 302
        return c
    }
    public static var transformer: TransformerConfig {
        var c = TransformerConfig()
        c.layers = 2; c.heads = 2; c.headDim = 32; c.axes = [8, 12, 12]; c.contextDim = 64
        return c
    }
    public static var vae: VAEConfig {
        var c = VAEConfig()
        c.baseDim = 8; c.decoderBaseDim = 8
        c.latentsMean = (0..<64).map { Float($0 % 7) / 10 - 0.3 }; c.latentsStd = (0..<64).map { 1 + Float($0 % 5) / 10 }
        return c
    }
}
