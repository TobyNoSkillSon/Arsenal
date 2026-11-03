// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM

/// LFM2-VL / LFM2.5-VL image processor.
///
/// PORTED-FROM: Transformers `models/lfm2_vl/image_processing_lfm2_vl.py` (Lfm2VlImageProcessor: smart_resize,
/// _is_image_too_large, crop_image_to_patches, convert_image_to_patches) and `processing_lfm2_vl.py`
/// (Lfm2VlProcessor._build_image_tokens), Apache-2.0; reimplemented, no code vendored. It replaces the pinned
/// mlx-swift-lm `LFM2VLProcessor`, which stretched every image to 512-px multiples as one sequence, emitted no
/// tiling, thumbnail or `<|image_start|>`/row-column markers, hard-coded image token 396 and left count mismatches to a
/// fatalError in the model.
///
/// Small images (≤ 2× the 256-token budget after rounding) are resized once to 64–256 tokens. Larger images are split
/// into 512-px tiles on the grid (2–10 tiles) whose aspect ratio is closest to the image, followed by a thumbnail at the
/// single-tile size. Resizing is Pillow-style antialiased bilinear on colour-managed sRGB bytes (Transformers resizes
/// with BILINEAR, antialias=True).
struct LFMVisionConfiguration: Decodable, Sendable {
    let imageMean: [Float]
    let imageStd: [Float]
    let downsampleFactor: Int
    let doImageSplitting: Bool
    let minTiles: Int
    let maxTiles: Int
    let useThumbnail: Bool
    let minImageTokens: Int
    let maxImageTokens: Int
    let encoderPatchSize: Int
    let tileSize: Int
    let maxPixelsTolerance: Double
    let useImageSpecialTokens: Bool

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean", imageStd = "image_std", downsampleFactor = "downsample_factor", doImageSplitting = "do_image_splitting"
        case minTiles = "min_tiles", maxTiles = "max_tiles", useThumbnail = "use_thumbnail", minImageTokens = "min_image_tokens"
        case maxImageTokens = "max_image_tokens", encoderPatchSize = "encoder_patch_size", tileSize = "tile_size"
        case maxPixelsTolerance = "max_pixels_tolerance", useImageSpecialTokens = "use_image_special_tokens"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func integer(_ key: CodingKeys, _ fallback: Int) throws -> Int { Int(try c.decodeIfPresent(Double.self, forKey: key) ?? Double(fallback)) }
        // Transformers' Lfm2VlImageProcessor class defaults.
        imageMean = try c.decodeIfPresent([Float].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([Float].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        downsampleFactor = try integer(.downsampleFactor, 2)
        doImageSplitting = try c.decodeIfPresent(Bool.self, forKey: .doImageSplitting) ?? true
        minTiles = try integer(.minTiles, 2)
        maxTiles = try integer(.maxTiles, 10)
        useThumbnail = try c.decodeIfPresent(Bool.self, forKey: .useThumbnail) ?? true
        minImageTokens = try integer(.minImageTokens, 64)
        maxImageTokens = try integer(.maxImageTokens, 256)
        encoderPatchSize = try integer(.encoderPatchSize, 16)
        tileSize = try integer(.tileSize, 512)
        maxPixelsTolerance = try c.decodeIfPresent(Double.self, forKey: .maxPixelsTolerance) ?? 2
        useImageSpecialTokens = try c.decodeIfPresent(Bool.self, forKey: .useImageSpecialTokens) ?? true
        guard imageMean.count == 3, imageStd.count == 3, downsampleFactor > 0, encoderPatchSize > 0, tileSize % encoderPatchSize == 0,
            minImageTokens > 0, maxImageTokens >= minImageTokens, !doImageSplitting || (minTiles >= 1 && minTiles <= maxTiles)
        else { throw VLMError.processing("Invalid LFM2-VL processor configuration.") }
    }
}

struct LFMVisionProcessor: UserInputProcessor {
    /// The resized size and tile grid for one image, and its image-token counts.
    struct Layout: Equatable {
        let rows: Int
        let columns: Int
        /// Single-tile (or thumbnail) size from smart_resize.
        let width: Int
        let height: Int
        let tileTokens: Int
        let imageTokens: Int
        var tiled: Bool { rows > 1 || columns > 1 }
        var totalTokens: Int { tiled ? rows * columns * tileTokens + imageTokens : imageTokens }
    }

    let config: LFMVisionConfiguration
    let tokenizer: any MLXLMCommon.Tokenizer
    let imageTokenID: Int

    init(_ config: LFMVisionConfiguration, tokenizer: any MLXLMCommon.Tokenizer, imageTokenID: Int) {
        self.config = config
        self.tokenizer = tokenizer
        self.imageTokenID = imageTokenID
    }

    private static func roundByFactor(_ value: Int, _ factor: Int) -> Int {
        Int((Double(value) / Double(factor)).rounded(.toNearestOrEven)) * factor
    }

    /// Lfm2VlImageProcessor.smart_resize → (width, height).
    func smartResize(height: Int, width: Int) -> (width: Int, height: Int) {
        let factor = config.encoderPatchSize * config.downsampleFactor
        let perToken = config.encoderPatchSize * config.encoderPatchSize * config.downsampleFactor * config.downsampleFactor
        let minimum = config.minImageTokens * perToken, maximum = config.maxImageTokens * perToken
        var h = max(factor, Self.roundByFactor(height, factor)), w = max(factor, Self.roundByFactor(width, factor))
        if h * w > maximum {
            let beta = (Double(height * width) / Double(maximum)).squareRoot()
            h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < minimum {
            let beta = (Double(minimum) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (w, h)
    }

    /// _is_image_too_large: more than `max_pixels_tolerance` × the single-tile budget after rounding.
    func isTooLarge(height: Int, width: Int) -> Bool {
        let factor = config.encoderPatchSize * config.downsampleFactor
        let h = max(config.encoderPatchSize, Self.roundByFactor(height, factor)), w = max(config.encoderPatchSize, Self.roundByFactor(width, factor))
        let budget = config.maxImageTokens * config.encoderPatchSize * config.encoderPatchSize * config.downsampleFactor * config.downsampleFactor
        return Double(h * w) > Double(budget) * config.maxPixelsTolerance
    }

    /// _target_ratios + find_closest_aspect_ratio → (columns, rows).
    func grid(width: Int, height: Int) -> (columns: Int, rows: Int) {
        var ratios: [(Int, Int)] = []
        for n in config.minTiles...config.maxTiles {
            for w in 1...n {
                for h in 1...n where config.minTiles <= w * h && w * h <= config.maxTiles && !ratios.contains(where: { $0 == (w, h) }) {
                    ratios.append((w, h))
                }
            }
        }
        // Python's sorted() is stable; ties in w·h never tie in aspect difference except for equal ratios, ordered by area.
        ratios = ratios.enumerated().sorted { ($0.element.0 * $0.element.1, $0.offset) < ($1.element.0 * $1.element.1, $1.offset) }.map(\.element)
        let aspect = Double(width) / Double(height), area = Double(width * height)
        var bestDifference = Double.infinity, best = (1, 1)
        for ratio in ratios {
            let difference = abs(aspect - Double(ratio.0) / Double(ratio.1))
            if difference < bestDifference {
                bestDifference = difference; best = ratio
            } else if difference == bestDifference, area > 0.5 * Double(config.tileSize * config.tileSize * ratio.0 * ratio.1) {
                best = ratio
            }
        }
        return (best.0, best.1)
    }

    func layout(width: Int, height: Int) -> Layout {
        let splitting = config.doImageSplitting && !(config.minTiles == 1 && config.maxTiles == 1)
        let resized = smartResize(height: height, width: width)
        let (columns, rows) = splitting && isTooLarge(height: height, width: width) ? grid(width: width, height: height) : (columns: 1, rows: 1)
        let tilePatches = config.tileSize / config.encoderPatchSize
        let tileSide = Int((Double(tilePatches) / Double(config.downsampleFactor)).rounded(.up))
        let imageTokens =
            Int((Double(resized.height / config.encoderPatchSize) / Double(config.downsampleFactor)).rounded(.up))
            * Int((Double(resized.width / config.encoderPatchSize) / Double(config.downsampleFactor)).rounded(.up))
        // A grid of one tile is not tiled: crop_image_to_patches adds no thumbnail and the processor emits no markers.
        return Layout(rows: rows, columns: columns, width: resized.width, height: resized.height, tileTokens: tileSide * tileSide, imageTokens: imageTokens)
    }

    /// _build_image_tokens for one image.
    func replacement(_ layout: Layout) -> String {
        var parts: [String] = []
        if config.useImageSpecialTokens { parts.append("<|image_start|>") }
        let image = "<image>"
        if layout.tiled {
            for row in 0..<layout.rows {
                for column in 0..<layout.columns {
                    if config.useImageSpecialTokens { parts.append("<|img_row_\(row + 1)_col_\(column + 1)|>") }
                    parts.append(String(repeating: image, count: layout.tileTokens))
                }
            }
            if config.useThumbnail {
                if config.useImageSpecialTokens { parts.append("<|img_thumbnail|>") }
                parts.append(String(repeating: image, count: layout.imageTokens))
            }
        } else {
            parts.append(String(repeating: image, count: layout.imageTokens))
        }
        if config.useImageSpecialTokens { parts.append("<|image_end|>") }
        return parts.joined()
    }

    /// Replaces each single image placeholder in the templated prompt with its expansion, and checks the result
    /// carries exactly the features the vision tower will produce.
    func expand(_ tokens: [Int], layouts: [Layout]) throws -> [Int] {
        let placeholders = tokens.indices.filter { tokens[$0] == imageTokenID }
        guard placeholders.count == layouts.count else {
            throw VLMError.processing("The prompt has \(placeholders.count) image placeholders for \(layouts.count) images.")
        }
        var result: [Int] = [], cursor = 0
        for (index, layout) in zip(placeholders, layouts) {
            result += tokens[cursor..<index]
            result += tokenizer.encode(text: replacement(layout), addSpecialTokens: false)
            cursor = index + 1
        }
        result += tokens[cursor...]
        let expected = layouts.reduce(0) { $0 + featureCount($1) }
        let actual = result.filter { $0 == imageTokenID }.count
        guard actual == expected else {
            throw VLMError.processing("LFM2-VL image tokens (\(actual)) do not match vision features (\(expected)).")
        }
        return result
    }

    /// Features the model produces for one image: each encoder input contributes ceil(h/2)·ceil(w/2) after pixel unshuffle.
    func featureCount(_ layout: Layout) -> Int {
        let side = { (patches: Int) in Int((Double(patches) / Double(self.config.downsampleFactor)).rounded(.up)) }
        let tile = side(config.tileSize / config.encoderPatchSize)
        let single = side(layout.height / config.encoderPatchSize) * side(layout.width / config.encoderPatchSize)
        guard layout.tiled else { return single }
        return layout.rows * layout.columns * tile * tile + (config.useThumbnail ? single : 0)
    }

    /// The encoder inputs for one image: row-major tiles, then the thumbnail, each as `[patches, P·P·C]` with its grid.
    func encoderInputs(_ image: CIImage) -> (layout: Layout, patches: [(MLXArray, THW)]) {
        let rgb = ImagePixels.rgb(image)
        let layout = self.layout(width: rgb.width, height: rgb.height)
        var crops: [RGBImage] = []
        if layout.tiled {
            let tile = config.tileSize
            let resized = PILResample.resize(rgb, width: tile * layout.columns, height: tile * layout.rows, filter: .bilinear)
            for row in 0..<layout.rows {
                for column in 0..<layout.columns {
                    var bytes = [UInt8](repeating: 0, count: tile * tile * 3)
                    for y in 0..<tile {
                        let source = ((row * tile + y) * resized.width + column * tile) * 3
                        bytes.replaceSubrange((y * tile * 3)..<((y + 1) * tile * 3), with: resized.bytes[source..<(source + tile * 3)])
                    }
                    crops.append(RGBImage(width: tile, height: tile, bytes: bytes))
                }
            }
            if config.useThumbnail { crops.append(PILResample.resize(rgb, width: layout.width, height: layout.height, filter: .bilinear)) }
        } else {
            crops.append(PILResample.resize(rgb, width: layout.width, height: layout.height, filter: .bilinear))
        }
        let p = config.encoderPatchSize
        let patches = crops.map { crop -> (MLXArray, THW) in
            let rows = crop.height / p, columns = crop.width / p
            // convert_image_to_patches: [C, H, W] → [rows·columns, P·P·C] with (py, px, c) inside a patch.
            let normalized = ImagePixels.normalized(crop, mean: config.imageMean, std: config.imageStd)[0]
            let grid = normalized.reshaped(3, rows, p, columns, p).transposed(1, 3, 2, 4, 0).reshaped(rows * columns, p * p * 3)
            return (grid, THW(1, rows, columns))
        }
        return (layout, patches)
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else { throw VLMError.processing("Video input is not supported by this model.") }
        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var tokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        if input.images.isEmpty {
            let prompt = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)))
        }
        var layouts: [Layout] = [], inputs: [(MLXArray, THW)] = []
        for image in input.images {
            let encoded = encoderInputs(try image.asCIImage())
            layouts.append(encoded.layout); inputs += encoded.patches
        }
        tokens = try expand(tokens, layouts: layouts)
        // Pad every encoder input to the longest one; the local LFM2-VL tower masks the padding from attention.
        let longest = inputs.map { $0.0.dim(0) }.max() ?? 0
        let padded = inputs.map { entry -> MLXArray in
            let patches = entry.0, missing = longest - patches.dim(0)
            let rows = missing > 0 ? concatenated([patches, MLXArray.zeros([missing, patches.dim(1)], dtype: patches.dtype)], axis: 0) : patches
            return rows.expandedDimensions(axis: 0)
        }
        let prompt = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: prompt, mask: ones(like: prompt).asType(.int8)),
            image: LMInput.ProcessedImage(pixels: concatenated(padded, axis: 0), frames: inputs.map(\.1)))
    }
}
