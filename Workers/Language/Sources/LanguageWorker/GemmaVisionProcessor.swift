// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM

/// Gemma 4 image processor.
///
/// PORTED-FROM: Transformers `models/gemma4/image_processing_gemma4.py` (get_aspect_ratio_preserving_size,
/// convert_image_to_patches, pad_along_first_dim) and `processing_gemma4.py` (replace_image_token), Apache-2.0;
/// reimplemented, no code vendored. It replaces the pinned mlx-swift-lm `Gemma4Processor`, which forced 800×800 and
/// always 280 soft tokens. Each image keeps its aspect ratio at multiples of pooling_kernel_size·patch_size (48 px)
/// within max_soft_tokens·k² patches (2,520), is resized with Pillow-style antialiased bicubic on sRGB bytes, rescaled
/// to [0, 1] (Gemma normalizes in the patch embedder) and expands to `<|image>` + n × `<|image|>` + `<image|>`.
struct GemmaVisionConfiguration: Decodable, Sendable {
    /// image_processing_gemma4.py _SUPPORTED_SOFT_TOKENS.
    static let supportedSoftTokens = [70, 140, 280, 560, 1120]
    let patchSize: Int
    let maxSoftTokens: Int
    let poolingKernelSize: Int

    enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size", maxSoftTokens = "max_soft_tokens", poolingKernelSize = "pooling_kernel_size"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        patchSize = Int(try c.decodeIfPresent(Double.self, forKey: .patchSize) ?? 16)
        maxSoftTokens = Int(try c.decodeIfPresent(Double.self, forKey: .maxSoftTokens) ?? 280)
        poolingKernelSize = Int(try c.decodeIfPresent(Double.self, forKey: .poolingKernelSize) ?? 3)
        guard patchSize > 0, poolingKernelSize > 0, Self.supportedSoftTokens.contains(maxSoftTokens) else {
            throw VLMError.processing("Gemma 4 max_soft_tokens must be one of \(Self.supportedSoftTokens).")
        }
    }
}

struct GemmaVisionProcessor: UserInputProcessor {
    /// One image's encoder input: `[rows·columns, 3·P·P]` patches in [0, 1] and `[rows·columns, 2]` (x, y) positions.
    struct EncoderInput {
        let patches: MLXArray
        let positions: MLXArray
        let rows: Int
        let columns: Int
        var patchCount: Int { rows * columns }
    }

    let config: GemmaVisionConfiguration
    let tokenizer: any MLXLMCommon.Tokenizer
    let imageTokenID: Int
    let boiTokenID: Int
    let eoiTokenID: Int?

    init(_ config: GemmaVisionConfiguration, tokenizer: any MLXLMCommon.Tokenizer, imageTokenID: Int, boiTokenID: Int, eoiTokenID: Int?) {
        self.config = config
        self.tokenizer = tokenizer
        self.imageTokenID = imageTokenID
        self.boiTokenID = boiTokenID
        self.eoiTokenID = eoiTokenID
    }

    /// get_aspect_ratio_preserving_size → (height, width).
    static func targetSize(height: Int, width: Int, patchSize: Int, maxPatches: Int, poolingKernelSize: Int) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else { throw VLMError.imageProcessingFailure("Image has no pixels.") }
        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / Double(height * width)).squareRoot()
        let side = poolingKernelSize * patchSize
        var targetHeight = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var targetWidth = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        guard targetHeight > 0 || targetWidth > 0 else { throw VLMError.imageProcessingFailure("Gemma 4 cannot resize \(width)×\(height).") }
        let maxSide = (maxPatches / (poolingKernelSize * poolingKernelSize)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
        }
        guard Double(targetHeight * targetWidth) <= targetPixels else {
            throw VLMError.imageProcessingFailure("Gemma 4 resize of \(width)×\(height) exceeds \(maxPatches) patches.")
        }
        return (targetHeight, targetWidth)
    }

    func encoderInput(_ image: CIImage) throws -> EncoderInput {
        var rgb = ImagePixels.rgb(image)
        let p = config.patchSize, k = config.poolingKernelSize
        let target = try Self.targetSize(
            height: rgb.height, width: rgb.width, patchSize: p, maxPatches: config.maxSoftTokens * k * k, poolingKernelSize: k)
        if target.height != rgb.height || target.width != rgb.width {
            rgb = PILResample.resize(rgb, width: target.width, height: target.height, filter: .bicubic)
        }
        let rows = rgb.height / p, columns = rgb.width / p
        // do_rescale only (do_normalize is false): [3, H, W] in [0, 1] → [rows·columns, P·P·3] ordered (py, px, c).
        let values = ImagePixels.normalized(rgb, mean: [0, 0, 0], std: [1, 1, 1])[0]
        let patches = values.reshaped(3, rows, p, columns, p).transposed(1, 3, 2, 4, 0).reshaped(rows * columns, p * p * 3)
        var positions = [Int32]()
        positions.reserveCapacity(rows * columns * 2)
        for y in 0..<rows { for x in 0..<columns { positions += [Int32(x), Int32(y)] } }
        return EncoderInput(patches: patches, positions: MLXArray(positions, [rows * columns, 2]), rows: rows, columns: columns)
    }

    /// processing_gemma4.py replace_image_token on the templated ids: each `<|image|>` placeholder becomes
    /// `<|image>` + n × `<|image|>` + `<image|>` for its image's soft-token count.
    func expand(_ tokens: [Int], softTokens: [Int]) throws -> [Int] {
        let placeholders = tokens.filter { $0 == imageTokenID }.count
        guard placeholders == softTokens.count else {
            throw VLMError.processing("The prompt has \(placeholders) image placeholders for \(softTokens.count) images.")
        }
        var result: [Int] = [], next = softTokens.makeIterator()
        for token in tokens {
            guard token == imageTokenID, let count = next.next() else { result.append(token); continue }
            result.append(boiTokenID)
            result += Array(repeating: imageTokenID, count: count)
            if let eoiTokenID { result.append(eoiTokenID) }
        }
        return result
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else { throw VLMError.processing("Video input is not supported by this model.") }
        let messages = Gemma4MessageGenerator().generate(from: input)
        var tokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        guard !input.images.isEmpty else {
            let prompt = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)))
        }
        let encoded = try input.images.map { try encoderInput($0.asCIImage()) }
        let k = config.poolingKernelSize
        tokens = try expand(tokens, softTokens: encoded.map { $0.patchCount / (k * k) })
        // pad_along_first_dim: zero patches and (-1, -1) positions up to the longest image in this request.
        let longest = encoded.map(\.patchCount).max() ?? 0
        let patches = encoded.map { item -> MLXArray in
            let missing = longest - item.patchCount
            let rows = missing > 0 ? concatenated([item.patches, MLXArray.zeros([missing, item.patches.dim(1)], dtype: item.patches.dtype)], axis: 0) : item.patches
            return rows.expandedDimensions(axis: 0)
        }
        let positions = encoded.map { item -> MLXArray in
            let missing = longest - item.patchCount
            let rows = missing > 0 ? concatenated([item.positions, MLX.full([missing, 2], values: MLXArray(Int32(-1)))], axis: 0) : item.positions
            return rows.expandedDimensions(axis: 0)
        }
        let prompt = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)),
            image: LMInput.ProcessedImage(
                pixels: concatenated(patches, axis: 0), positionIds: concatenated(positions, axis: 0),
                frames: encoded.map { THW(1, $0.rows, $0.columns) }))
    }
}
