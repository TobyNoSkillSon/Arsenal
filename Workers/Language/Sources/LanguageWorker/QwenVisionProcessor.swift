// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM

/// Image processor for the Qwen3-VL / Qwen3.5 language models.
///
/// PORTED-FROM: mlx-swift-lm bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57 `Libraries/MLXVLM/Models/Qwen3VL.swift`
/// `Qwen3VLProcessor.prepare` (MIT, see Upstream/TeleOCR/LICENSE-upstream). Changes against the pinned processor, each
/// following Transformers `Qwen2VLImageProcessor(Fast)` as configured by the checkpoints:
/// - pixels are read as colour-managed sRGB bytes (the pinned image path skipped `toSRGB()` and fed linear light);
/// - resizing is Pillow's antialiased bicubic on those bytes (`PILResample`), not Core Image's unfiltered bicubic;
/// - `smart_resize` uses Python rounding (half to even) and the configured min/max pixels, which `processorConfiguration`
///   narrows by the folder's `min_pixels`/`max_pixels` (the pinned configuration only read `size`);
/// - an optional maker pre-resize (Chandra's `scale_to_fit`, Lanczos) runs first;
/// - `UserInput.processing.maxPixels` can only lower the budget for one call (describe/tag caps); videos are refused.
struct QwenVisionConfiguration: Decodable, Sendable {
    struct ScaleToFit: Decodable, Sendable {
        let maxPixels: Int
        let minPixels: Int
        let grid: Int
        enum CodingKeys: String, CodingKey { case maxPixels = "max_pixels", minPixels = "min_pixels", grid }
        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            maxPixels = Int(try c.decode(Double.self, forKey: .maxPixels))
            minPixels = Int(try c.decode(Double.self, forKey: .minPixels))
            grid = Int(try c.decode(Double.self, forKey: .grid))
        }
        init(maxPixels: Int, minPixels: Int, grid: Int) { self.maxPixels = maxPixels; self.minPixels = minPixels; self.grid = grid }
    }

    let imageMean: [Float]
    let imageStd: [Float]
    let minPixels: Int
    let maxPixels: Int
    let mergeSize: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let scaleToFit: ScaleToFit?

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean", imageStd = "image_std", minPixels = "min_pixels", maxPixels = "max_pixels"
        case mergeSize = "merge_size", patchSize = "patch_size", temporalPatchSize = "temporal_patch_size"
        case scaleToFit = "arsenal_scale_to_fit"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageMean = try c.decode([Float].self, forKey: .imageMean)
        imageStd = try c.decode([Float].self, forKey: .imageStd)
        // Transformers' Qwen2VLImageProcessor defaults when a checkpoint names neither form.
        minPixels = Int(try c.decodeIfPresent(Double.self, forKey: .minPixels) ?? 56 * 56)
        maxPixels = Int(try c.decodeIfPresent(Double.self, forKey: .maxPixels) ?? 28 * 28 * 1280)
        mergeSize = Int(try c.decodeIfPresent(Double.self, forKey: .mergeSize) ?? 2)
        patchSize = Int(try c.decodeIfPresent(Double.self, forKey: .patchSize) ?? 14)
        temporalPatchSize = Int(try c.decodeIfPresent(Double.self, forKey: .temporalPatchSize) ?? 2)
        scaleToFit = try c.decodeIfPresent(ScaleToFit.self, forKey: .scaleToFit)
        guard imageMean.count == 3, imageStd.count == 3, minPixels > 0, maxPixels >= minPixels else {
            throw VLMError.processing("Invalid Qwen vision processor configuration.")
        }
    }
}

struct QwenVisionProcessor: UserInputProcessor {
    let config: QwenVisionConfiguration
    let tokenizer: any MLXLMCommon.Tokenizer

    init(_ config: QwenVisionConfiguration, tokenizer: any MLXLMCommon.Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Transformers `smart_resize` (image_processing_qwen2_vl.py) with Python's round-half-to-even.
    static func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else { throw VLMError.imageProcessingFailure("Image has no pixels.") }
        guard Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw VLMError.imageProcessingFailure("Absolute aspect ratio must be smaller than 200: \(width) × \(height)")
        }
        let f = Double(factor)
        var h = max(factor, Int((Double(height) / f).rounded(.toNearestOrEven)) * factor)
        var w = max(factor, Int((Double(width) / f).rounded(.toNearestOrEven)) * factor)
        if h * w > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / f).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / f).rounded(.down)) * factor)
        } else if h * w < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / f).rounded(.up)) * factor
            w = Int((Double(width) * beta / f).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// Chandra `chandra/model/util.py` scale_to_fit: fit the pixel count into [min, max] on a `grid` lattice, trading
    /// one grid step at a time on the side that best keeps the aspect ratio. Returns nil when the size is unchanged.
    static func scaleToFit(width: Int, height: Int, fit: QwenVisionConfiguration.ScaleToFit) -> (width: Int, height: Int)? {
        guard width > 0, height > 0 else { return nil }
        let aspect = Double(width) / Double(height), pixels = width * height
        var scale = 1.0
        if pixels > fit.maxPixels {
            scale = (Double(fit.maxPixels) / Double(pixels)).squareRoot()
        } else if pixels < fit.minPixels {
            scale = (Double(fit.minPixels) / Double(pixels)).squareRoot()
        }
        let grid = Double(fit.grid)
        var columns = max(1, Int((Double(width) * scale / grid).rounded(.toNearestOrEven)))
        var rows = max(1, Int((Double(height) * scale / grid).rounded(.toNearestOrEven)))
        while columns * rows * fit.grid * fit.grid > fit.maxPixels {
            if columns == 1 && rows == 1 { break }
            if columns == 1 { rows -= 1; continue }
            if rows == 1 { columns -= 1; continue }
            let widthLoss = abs(Double(columns - 1) / Double(rows) - aspect)
            let heightLoss = abs(Double(columns) / Double(rows - 1) - aspect)
            if widthLoss < heightLoss { columns -= 1 } else { rows -= 1 }
        }
        let result = (width: columns * fit.grid, height: rows * fit.grid)
        return result.width == width && result.height == height ? nil : result
    }

    /// One image as `[patches, C·T·P·P]` plus its grid, after the maker pre-resize and smart_resize.
    func pixels(_ image: CIImage, minPixels: Int? = nil, maxPixels: Int? = nil) throws -> (MLXArray, THW) {
        var rgb = ImagePixels.rgb(image)
        if let fit = config.scaleToFit, let size = Self.scaleToFit(width: rgb.width, height: rgb.height, fit: fit) {
            rgb = PILResample.resize(rgb, width: size.width, height: size.height, filter: .lanczos)
        }
        let maximum = min(config.maxPixels, maxPixels ?? Int.max)
        let minimum = min(max(config.minPixels, minPixels ?? 0), maximum)
        let target = try Self.smartResize(
            height: rgb.height, width: rgb.width, factor: config.patchSize * config.mergeSize, minPixels: minimum, maxPixels: maximum)
        if target.height != rgb.height || target.width != rgb.width {
            rgb = PILResample.resize(rgb, width: target.width, height: target.height, filter: .bicubic)
        }
        let array = ImagePixels.normalized(rgb, mean: config.imageMean, std: config.imageStd)
        return try TeleOCRUtilities.patchify(
            images: [array], mergeSize: config.mergeSize, patchSize: config.patchSize, temporalPatchSize: config.temporalPatchSize)
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else { throw VLMError.processing("Video input is not supported by this model.") }
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        if input.images.isEmpty {
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)))
        }
        let processed = try input.images.map { try pixels($0.asCIImage(), minPixels: input.processing.minPixels, maxPixels: input.processing.maxPixels) }
        let frames = processed.map(\.1)
        promptTokens = try TeleOCRUtilities.replacePaddingTokens(
            in: promptTokens, frames: frames, paddingToken: "<|image_pad|>", mergeSize: config.mergeSize, tokenizer: tokenizer)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: LMInput.ProcessedImage(pixels: concatenated(processed.map(\.0)), frames: frames))
    }
}
