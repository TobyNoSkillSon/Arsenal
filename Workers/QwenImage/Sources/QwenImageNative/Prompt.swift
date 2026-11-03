// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// QwenImage21Pipeline's raw prompt templates (not the chat template: the checkpoint expects this tokenization).
/// The system turn is dropped from the encoder output; each condition image is `<imageN><|vision_start|>` + one
/// `<|image_pad|>` per merged 2×2 patch group + `<|vision_end|>`, images separated by a space.
public struct PromptLayout: Equatable {
    public static let system = "Comprehend and analyze the provided prompt."
    public static let systemTurn = "<|im_start|>system\n\(system)<|im_end|>\n"

    /// Text pieces between image-pad runs; `pads[i]` pads follow `pieces[i]`.
    public let pieces: [String]
    public let pads: [Int]

    public init(prompt: String, imageSlots: [Int]) {
        // Qwen has no BOS token: an empty prompt would leave the encoder nothing to read.
        let text = prompt.isEmpty ? " " : prompt
        var pieces: [String] = [], current = Self.systemTurn + "<|im_start|>user\n"
        for i in imageSlots.indices {
            current += (i == 0 ? "" : " ") + "<image\(i + 1)><|vision_start|>"
            pieces.append(current); current = "<|vision_end|>"
        }
        pieces.append(current + text + "<|im_end|>\n<|im_start|>assistant\n")
        self.pieces = pieces; pads = imageSlots
    }

    /// The full template text with every pad written out (for tests and inspection).
    public var text: String {
        var s = ""
        for (i, piece) in pieces.enumerated() {
            s += piece
            if i < pads.count { s += String(repeating: "<|image_pad|>", count: pads[i]) }
        }
        return s
    }

    /// Token IDs: each piece tokenized on its own (added tokens split text first, as in Transformers) with the pad
    /// runs inserted between them.
    public func tokens(_ encode: (String) -> [Int], imagePad: Int) -> [Int32] {
        var ids: [Int32] = []
        for (i, piece) in pieces.enumerated() {
            ids += encode(piece).map(Int32.init)
            if i < pads.count { ids += Array(repeating: Int32(imagePad), count: pads[i]) }
        }
        return ids
    }

    public static func dropIndex(_ encode: (String) -> [Int]) -> Int { encode(systemTurn).count }
}

/// Qwen2-VL image processor layout for an image already sized to a multiple of 32: [−1, 1] values, the frame repeated
/// over the temporal patch, patches ordered by 2×2 merge group, each flattened as (channel, time, row, column).
public enum VisionPatches {
    public static func make(rgb: [UInt8], width: Int, height: Int, config: Qwen3VLConfig.Vision) throws -> (MLXArray, VisionGrid) {
        let p = config.patch, m = config.merge, factor = p * m
        guard width % factor == 0, height % factor == 0, rgb.count == width * height * 3 else {
            throw QwenImageError("Vision input must be a multiple of \(factor) pixels")
        }
        let gh = height / p, gw = width / p
        let x = MLXArray(rgb, [height, width, 3]).asType(.float32) / 127.5 - 1
        // [gh/m, m, P, gw/m, m, P, C] → [gh/m, gw/m, m, m, C, P, P]
        var y = x.reshaped(gh / m, m, p, gw / m, m, p, 3).transposed(0, 3, 1, 4, 6, 2, 5)
        y = broadcast(expandedDimensions(y, axis: 5), to: [gh / m, gw / m, m, m, 3, config.temporalPatch, p, p])
        return (y.reshaped(gh * gw, 3 * config.temporalPatch * p * p), VisionGrid(t: 1, h: gh, w: gw))
    }
}

public enum ImageSizing {
    /// calculate_dimensions: the size with `area` pixels at `ratio` (w/h), each side rounded (half to even) to 32.
    public static func dimensions(area: Int, ratio: Double) -> (width: Int, height: Int) {
        let width = (Double(area) * ratio).squareRoot(), height = width / ratio
        return (Int((width / 32).rounded(.toNearestOrEven)) * 32, Int((height / 32).rounded(.toNearestOrEven)) * 32)
    }
}
