// swiftlint:disable force_cast shorthand_operator
// SPDX-License-Identifier: MIT
// LCO-Embedding Omni 7B: one embedding space for text, images (and PDF pages), audio and video. Served as its makers
// document it: the sentence-transformers chat template (additional_chat_templates/sentence_transformers.jinja), the
// Qwen2.5-Omni processor and qwen_omni_utils media rules, last-token pooling of the final hidden state, L2 normalized.
import AVFoundation
import Foundation
import Hub
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Tokenizers

final class LCOGraph: Module {
    @ModuleInfo(key: "model") var language: LCOTextModel
    @ModuleInfo(key: "visual") var vision: LCOVisionEncoder
    @ModuleInfo(key: "audio_tower") var audio: LCOAudioEncoder
    let config: LCOConfiguration
    init(_ config: LCOConfiguration) {
        self.config = config
        _language.wrappedValue = LCOTextModel(config.text)
        _vision.wrappedValue = LCOVisionEncoder(config.vision)
        _audio.wrappedValue = LCOAudioEncoder(config.audio)
    }
    /// Checkpoint key → graph key and layout; nil skips tensors the embedding path never reads (the LM head and the
    /// audio tower's begin/end table, which the thinker does not use).
    static func map(_ key: String, _ value: MLXArray) -> (String, MLXArray)? {
        if key == "lm_head.weight" || key.hasPrefix("audio_tower.audio_bos_eos_token") { return nil }
        if key == "visual.patch_embed.proj.weight", value.ndim == 5 { return (key, value.reshaped(value.dim(0), -1)) }
        if key.hasPrefix("audio_tower.conv"), key.hasSuffix(".weight"), value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
        return (key, value)
    }
}

/// One piece of the rendered prompt: literal text (with special tokens) or a media placeholder to expand.
enum LCOPiece: Equatable {
    case text(String)
    case image(Int)
    case video(Int)
    case audio(Int)
}

enum LCOPrompt {
    /// sentence_transformers.jinja for one user message: the fixed system turn, the parts in order, then the
    /// summary line named by the first of image, video, audio present (else text), and the generation prompt.
    static func render(_ parts: [LCOPart]) -> [LCOPiece] {
        var pieces: [LCOPiece] = [], text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        var image = 0, video = 0, audio = 0
        for part in parts {
            switch part {
            case .text(let value): text += value
            case .image:
                pieces.append(.text(text + "<|vision_bos|>")); pieces.append(.image(image)); image += 1; text = "<|vision_eos|>"
            case .video:
                pieces.append(.text(text + "<|vision_bos|>")); pieces.append(.video(video)); video += 1; text = "<|vision_eos|>"
            case .audio:
                pieces.append(.text(text + "<|audio_bos|>")); pieces.append(.audio(audio)); audio += 1; text = "<|audio_eos|>"
            }
        }
        let kind = image > 0 ? "image" : video > 0 ? "video" : audio > 0 ? "audio" : "text"
        text += "\nSummarize the above \(kind) in one word:<|im_end|>\n<|im_start|>assistant\n"
        pieces.append(.text(text))
        return pieces
    }
}

enum LCOPart: Equatable {
    case text(String)
    case image(URL)
    case video(URL)
    case audio(URL)
}

/// Token ids and 3-axis rotary positions, built as get_rope_index does for one sequence without audio in video:
/// text and audio tokens count up on all three axes; an image or video block starts at the next position and lays its
/// merged patches out as (start + t·seconds_per_grid·25, start + row, start + column); the next token follows its maximum.
struct LCOSequence {
    private(set) var ids: [Int] = []
    private(set) var positions: [[Int]] = [[], [], []]
    private(set) var media: [(kind: String, range: Range<Int>)] = []
    private var next = 0
    mutating func text(_ tokens: [Int]) {
        for token in tokens { ids.append(token); for a in 0..<3 { positions[a].append(next) }; next += 1 }
    }
    mutating func audio(token: Int, count: Int) {
        let start = ids.count
        text(Array(repeating: token, count: count))
        media.append(("audio", start..<ids.count))
    }
    mutating func vision(token: Int, grid: LCOGrid, merge: Int, temporalStep: Double, kind: String) {
        let start = ids.count, base = next, gh = grid.h / merge, gw = grid.w / merge
        var top = base
        for t in 0..<grid.t {
            let time = base + Int(Double(t) * temporalStep)
            for y in 0..<gh {
                for x in 0..<gw {
                    ids.append(token); positions[0].append(time); positions[1].append(base + y); positions[2].append(base + x)
                    top = max(top, time, base + y, base + x)
                }
            }
        }
        next = top + 1
        media.append((kind, start..<ids.count))
    }
}

enum LCOMedia {
    /// qwen_omni_utils / Qwen2-VL smart_resize, with Python's round-half-to-even.
    static func smartResize(height: Int, width: Int, factor: Int = 28, minPixels: Int, maxPixels: Int) throws -> (Int, Int) {
        guard height > 0, width > 0, Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw WorkerFailure(code: "invalid_request", message: "Image aspect ratio must be under 200:1.")
        }
        let f = Double(factor), h = Double(height), w = Double(width)
        var hBar = max(factor, Int((h / f).rounded(.toNearestOrEven)) * factor)
        var wBar = max(factor, Int((w / f).rounded(.toNearestOrEven)) * factor)
        if hBar * wBar > maxPixels {
            let beta = (h * w / Double(maxPixels)).squareRoot()
            hBar = Int(floor(h / beta / f)) * factor; wBar = Int(floor(w / beta / f)) * factor
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = Int(ceil(h * beta / f)) * factor; wBar = Int(ceil(w * beta / f)) * factor
        }
        return (hBar, wBar)
    }
    /// fetch_image bounds: 4 to 16,384 merged tokens.
    static let imageMinPixels = 4 * 28 * 28, imageMaxPixels = 16384 * 28 * 28
    /// fetch_video bounds per frame: 128 to 768 merged tokens, and the whole-clip budget over the sampled frames.
    static let videoMinPixels = 128 * 28 * 28, videoMaxPixels = 768 * 28 * 28
    static func videoMaxPixels(frames: Int) -> Int {
        let total = 128000.0 * 28 * 28 * 0.9
        return Int(max(min(Double(videoMaxPixels), total / Double(frames) * 2), Double(Int(Double(videoMinPixels) * 1.05))))
    }
    /// smart_nframes at fps 2: at least 4, at most 768 (and the clip), an even count.
    static func frameCount(total: Int, fps: Double) -> Int? {
        let minimum = 4, maximum = min(768, total) / 2 * 2
        var count = Double(total) / fps * 2
        count = min(min(max(count, Double(minimum)), Double(maximum)), Double(total))
        let frames = Int(floor(count / 2)) * 2
        return frames >= 2 && frames <= total ? frames : nil
    }
    /// torch.linspace(0, total - 1, n).round(): half to even.
    static func sampleIndices(total: Int, count: Int) -> [Int] {
        guard count > 1 else { return [0] }
        return (0..<count).map { Int((Double($0) * Double(total - 1) / Double(count - 1)).rounded(.toNearestOrEven)) }
    }
    /// Resize to the processor's size and patchify, as an MLX array.
    static func patches(_ frames: [LCORGB], height: Int, width: Int) -> (MLXArray, LCOGrid) {
        let resized = frames.map { $0.width == width && $0.height == height ? $0 : LCOResample.bicubic($0, width: width, height: height) }
        let (values, grid) = LCOPatches.make(resized)
        return (MLXArray(values, [values.count / (3 * 2 * 14 * 14), 3 * 2 * 14 * 14]), grid)
    }
    /// Special-token strings that must not appear in user text (they would become control tokens).
    static func controlToken(in text: String, specials: [String]) -> String? { specials.first { text.contains($0) } }
}

final class LCOAdapter: NativeModel {
    let architecture = "qwen2_5_omni_thinker"
    let operations: Set<String> = ["embed"]
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { graph.map { Dictionary(uniqueKeysWithValues: $0.parameters().flattened()) } ?? [:] }
    private(set) var graph: LCOGraph?
    /// Token ids of the last embedded item (lab parity checks compare them with the maker's processor).
    private(set) var lastIDs: [Int] = []
    private var tokenizer: (any Tokenizers.Tokenizer)?
    /// The tokenizer's special tokens (added_tokens_decoder entries marked special).
    private(set) var specials: [String] = []
    /// The language model's trained context (max_position_embeddings); a longer input is refused before any media work.
    var maxInputTokens = 32768

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        guard !context.randomWeights else { throw failure("LCO-Embedding Omni has no random-weight lab graph.") }
        let config = try JSONDecoder().decode(LCOConfiguration.self, from: Data(contentsOf: context.weights.appendingPathComponent("config.json")))
        let model = LCOGraph(config)
        let archives = try FileManager.default.contentsOfDirectory(at: context.weights, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }
        _ = try NativePrecisionTier(context.precision, half: context.precision == "bf16" ? .bfloat16 : .float16).load(
            model, archives: archives,
            map: { key, value in
                try cancellation.check()
                return LCOGraph.map(key, value)
            }, eligible: TextRetrievalAdapter.eligible)
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer_config.json"))) as! [NSString: Any]
        let dataJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer.json"))) as! [NSString: Any]
        tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(configJSON), tokenizerData: Config(dataJSON))
        specials = ((configJSON["added_tokens_decoder"] as? [String: [String: Any]]) ?? [:]).values.compactMap {
            $0["special"] as? Bool == true ? $0["content"] as? String : nil
        }.sorted()
        model.train(false)
        graph = model
    }

    /// Request item → ordered parts. Strings and text parts are text; image, video, audio and document parts carry a path.
    static func parts(_ item: ArsenalWire.JSONValue) throws -> [(LCOPart?, URL?)] {
        let parts = item["type"]?.string == "content" ? item["content"]?.array ?? [] : [item]
        return try parts.map { part in
            if let text = part.string ?? (part["type"]?.string == "text" ? part["text"]?.string : nil) { return (.text(text), nil) }
            let type = part["type"]?.string ?? ""
            guard let path = part[type]?["path"]?.string else {
                throw WorkerFailure(code: "invalid_request", message: "Input needs text or a local image/audio/video/document path.")
            }
            let url = URL(fileURLWithPath: path)
            switch type {
            case "image": return (.image(url), nil)
            case "video": return (.video(url), nil)
            case "audio": return (.audio(url), nil)
            case "document": return (nil, url)
            default: throw WorkerFailure(code: "invalid_request", message: "LCO-Embedding accepts text, images, PDFs, audio and video.")
            }
        }
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "embed", let graph, let tokenizer else { throw failure("LCO-Embedding is not loaded or the operation is unsupported.") }
        guard let items = input["input"]?.array, !items.isEmpty, items.count <= 2048 else { throw failure("Provide nonempty embedding inputs.") }
        guard input["dimensions"] == nil || input["dimensions"]?.number == Double(graph.config.text.hiddenSize) else {
            throw failure("LCO-Embedding has no Matryoshka projection; use its native dimension, \(graph.config.text.hiddenSize).")
        }
        var vectors: [ArsenalWire.JSONValue] = [], total = 0
        for item in items {
            try cancellation.check()
            let vector = try embed(item, graph: graph, tokenizer: tokenizer, cancellation: cancellation, tokens: &total)
            vectors.append(.array(vector.map { .number(Double($0)) }))
        }
        return ["embeddings": .array(vectors), "input_tokens": .number(Double(total))]
    }

    private func embed(_ item: ArsenalWire.JSONValue, graph: LCOGraph, tokenizer: any Tokenizers.Tokenizer, cancellation: NativeCancellation, tokens total: inout Int) throws
        -> [Float]
    {
        // Image sources in prompt order: a file, or one page of a PDF (rendered only once its tokens are admitted).
        enum Picture { case file(URL), page(CGPDFPage) }
        var parts: [LCOPart] = [], pictures: [Picture] = [], documents: [CGPDFDocument] = []
        for (part, document) in try Self.parts(item) {
            if let document {
                guard let pdf = CGPDFDocument(document as CFURL), pdf.numberOfPages > 0, pdf.numberOfPages <= 128 else { throw failure("PDF requires 1–128 pages.") }
                documents.append(pdf)
                for index in 1...pdf.numberOfPages {
                    guard let page = pdf.page(at: index) else { throw failure("Unreadable PDF page.") }
                    pictures.append(.page(page)); parts.append(.image(document))
                }
            } else if let part {
                if case .text(let text) = part, let token = LCOMedia.controlToken(in: text, specials: specials) {
                    throw failure("Text contains the control token \(token); remove it.")
                }
                if case .image(let url) = part { pictures.append(.file(url)) }
                parts.append(part)
            }
        }
        guard !parts.isEmpty else { throw failure("Embedding content is empty.") }
        var videos: [URL] = [], audios: [URL] = []
        for part in parts {
            if case .video(let url) = part { videos.append(url) }
            if case .audio(let url) = part { audios.append(url) }
        }
        var sequence = LCOSequence(), budget = 0
        func admit(_ count: Int) throws {
            budget += count
            guard budget <= maxInputTokens else {
                throw failure("This input needs more than \(maxInputTokens) tokens (\(budget) so far); embed fewer pages, a shorter clip or less text.")
            }
        }
        var features: [MLXArray] = []
        let config = graph.config, merge = config.vision.spatialMergeSize
        for piece in LCOPrompt.render(parts) {
            try cancellation.check()
            switch piece {
            case .text(let text):
                let ids = tokenizer.encode(text: text, addSpecialTokens: false)
                try admit(ids.count); sequence.text(ids)
            case .image(let index):
                // Size first: the tokens are admitted before the pixels are decoded or the page is rendered.
                let image: LCORGB, h: Int, w: Int
                switch pictures[index] {
                case .file(let url):
                    let cg = try NativeMedia.image(url, maxPixels: 64_000_000)
                    (h, w) = try LCOMedia.smartResize(height: cg.height, width: cg.width, minPixels: LCOMedia.imageMinPixels, maxPixels: LCOMedia.imageMaxPixels)
                    try admit(h * w / (28 * 28))
                    image = try LCOImageIO.rgb(cg)
                case .page(let page):
                    let (pw, ph) = Self.pageSize(page)
                    (h, w) = try LCOMedia.smartResize(height: ph, width: pw, minPixels: LCOMedia.imageMinPixels, maxPixels: LCOMedia.imageMaxPixels)
                    try admit(h * w / (28 * 28))
                    image = try LCOImageIO.rgb(try Self.render(page, width: pw, height: ph))
                }
                let (pixels, grid) = LCOMedia.patches([image], height: h, width: w)
                features.append(try graph.vision(pixels, grids: [grid], cancellation: cancellation))
                sequence.vision(token: config.imageToken, grid: grid, merge: merge, temporalStep: Double(config.positionIdPerSeconds), kind: "image")
            case .video(let index):
                let url = videos[index]
                // A file AVFoundation cannot open or decode is the caller's mistake: 422, not 503.
                let track = try NativeMedia.decoding("Video") { try LCOVideoIO.track(url, cancellation: cancellation) }
                guard let count = LCOMedia.frameCount(total: track.frames, fps: track.fps) else { throw failure("Video needs at least two frames.") }
                let (h, w) = try LCOMedia.smartResize(
                    height: track.height, width: track.width, minPixels: LCOMedia.videoMinPixels, maxPixels: LCOMedia.videoMaxPixels(frames: count))
                try admit((count + 1) / 2 * h * w / (28 * 28))
                let frames = try NativeMedia.decoding("Video") {
                    try LCOVideoIO.frames(
                        url, track: track, indices: LCOMedia.sampleIndices(total: track.frames, count: count), width: w, height: h, cancellation: cancellation)
                }
                let (pixels, grid) = LCOMedia.patches(frames, height: h, width: w)
                features.append(try graph.vision(pixels, grids: [grid], cancellation: cancellation))
                // The processor's hard-coded fps 2 gives temporal_patch_size / fps = 1 s per grid: 25 positions.
                sequence.vision(token: config.videoToken, grid: grid, merge: merge, temporalStep: Double(config.positionIdPerSeconds) * 1.0, kind: "video")
            case .audio(let index):
                let pcm = try LCOAudioIO.mono(audios[index], maximumSamples: LCOMel.maxSamples)
                let (values, frames) = LCOMel.features(pcm)
                let count = LCOAudioEncoder.outputTokens(frames: frames)
                try admit(count)
                let mel = MLXArray(values, [LCOMel.bins, frames])
                let encoded = try graph.audio(mel, cancellation: cancellation)
                guard encoded.dim(0) == count else { throw failure("Audio encoder length mismatch.") }
                features.append(encoded)
                sequence.audio(token: config.audioToken, count: count)
            }
        }
        withExtendedLifetime(documents) {} // pages borrow from their documents until rendered
        var embeds = graph.language.embedTokens(MLXArray(sequence.ids.map(Int32.init)).reshaped(1, -1))
        for (feature, block) in zip(features, sequence.media) {
            guard feature.dim(0) == block.range.count else { throw failure("Media features do not match their placeholders.") }
            embeds[0, block.range.lowerBound..<block.range.upperBound] = feature.asType(embeds.dtype)
        }
        var vector = try graph.language.lastHidden(embeds, positions: sequence.positions, cancellation: cancellation).asType(.float32)
        vector = vector / maximum(sqrt(sum(vector * vector)), MLXArray(Float(1e-12)))
        let row = vector.asArray(Float.self)
        guard row.allSatisfy(\.isFinite) else { throw failure("Non-finite LCO-Embedding vector.") }
        total += sequence.ids.count; lastIDs = sequence.ids
        return row
    }

    /// A PDF page's render size: its crop box turned by /Rotate, at most 2,048 px on the long side and 2× its point size.
    static func pageSize(_ page: CGPDFPage) -> (Int, Int) { PDFPageRaster.pixelSize(PDFPageRaster.shownSize(page)) }
    static func pageSize(_ rect: CGRect) -> (Int, Int) { PDFPageRaster.pixelSize(rect.size) }
    static func render(_ page: CGPDFPage, width: Int, height: Int) throws -> CGImage {
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw WorkerFailure(code: "invalid_request", message: "PDF page allocation failed.") }
        PDFPageRaster.draw(page, in: context, width: width, height: height)
        guard let rendered = context.makeImage() else { throw WorkerFailure(code: "invalid_request", message: "PDF render failed.") }
        return rendered
    }

    private func failure(_ message: String) -> WorkerFailure { .init(code: "invalid_request", message: message) }
    func unload() { graph = nil; tokenizer = nil }
}

// swiftlint:enable force_cast shorthand_operator
