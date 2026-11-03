// swiftlint:disable force_cast shorthand_operator
// SPDX-License-Identifier: MIT
// EmbeddingGemma 2: text, images, PDF pages, audio and video in one 768-dimension space, served as its makers document
// it (sentence-transformers 6.2 on Transformers 5.22.1): the card's task prefixes, the chat template's part order and
// placeholders, EmbeddingGemma2Processor's expansion (<|image> n×<|image|> <image|>, one block per video frame,
// <|audio> n×<|audio|> <audio|>) between <bos> and <eos>, the towers' soft tokens scattered into the text embeddings,
// the bidirectional encoder, mean pooling over every token, L2 normalization, and Matryoshka truncation to 512, 256 or 128
// followed by re-normalization.
import AVFoundation
import CoreGraphics
import Foundation
import Hub
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Tokenizers

/// The checkpoint's root config beyond the text model: towers and placeholder ids.
struct EmbeddingGemma2Root: Decodable {
    let text: EmbeddingGemma2Configuration
    let vision: EmbeddingGemma2VisionConfiguration?
    let audio: EmbeddingGemma2AudioConfiguration?
    let imageToken: Int, videoToken: Int, audioToken: Int
    let beginImage: Int, endImage: Int, beginAudio: Int, endAudio: Int
    enum CodingKeys: String, CodingKey {
        case vision = "vision_config", audio = "audio_config", imageToken = "image_token_id", videoToken = "video_token_id"
        case audioToken = "audio_token_id", beginImage = "boi_token_id", endImage = "eoi_token_id", beginAudio = "boa_token_id"
        case endAudio = "eoa_token_index"
    }
    init(from decoder: any Swift.Decoder) throws {
        text = try EmbeddingGemma2Configuration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vision = try c.decodeIfPresent(EmbeddingGemma2VisionConfiguration.self, forKey: .vision)
        audio = try c.decodeIfPresent(EmbeddingGemma2AudioConfiguration.self, forKey: .audio)
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? 258_880
        videoToken = try c.decodeIfPresent(Int.self, forKey: .videoToken) ?? 258_884
        audioToken = try c.decodeIfPresent(Int.self, forKey: .audioToken) ?? 258_881
        beginImage = try c.decodeIfPresent(Int.self, forKey: .beginImage) ?? 255_999
        endImage = try c.decodeIfPresent(Int.self, forKey: .endImage) ?? 258_882
        beginAudio = try c.decodeIfPresent(Int.self, forKey: .beginAudio) ?? 256_000
        endAudio = try c.decodeIfPresent(Int.self, forKey: .endAudio) ?? 258_883
    }
}

final class EmbeddingGemma2Graph: Module {
    @ModuleInfo(key: "language_model") var text: EmbeddingGemma2TextModel
    @ModuleInfo(key: "vision_tower") var vision: EmbeddingGemma2VisionTower?
    @ModuleInfo(key: "embed_vision") var embedVision: EmbeddingGemma2Embedder?
    @ModuleInfo(key: "audio_tower") var audio: EmbeddingGemma2AudioTower?
    @ModuleInfo(key: "embed_audio") var embedAudio: EmbeddingGemma2Embedder?
    let config: EmbeddingGemma2Root

    init(_ config: EmbeddingGemma2Root) {
        self.config = config
        _text.wrappedValue = EmbeddingGemma2TextModel(config.text)
        if let v = config.vision {
            _vision.wrappedValue = EmbeddingGemma2VisionTower(v)
            _embedVision.wrappedValue = EmbeddingGemma2Embedder(input: v.hiddenSize, output: config.text.hiddenSize, eps: v.rmsNormEps)
        }
        if let a = config.audio {
            _audio.wrappedValue = EmbeddingGemma2AudioTower(a)
            _embedAudio.wrappedValue = EmbeddingGemma2Embedder(input: a.outputDims, output: config.text.hiddenSize, eps: a.rmsNormEps)
        }
    }

    /// Scaled token embeddings (`embed_tokens(ids) · √hidden` with the scale in the weights' dtype).
    func embed(_ ids: [Int]) -> MLXArray {
        let h = text.embedTokens(MLXArray(ids.map(Int32.init)))
        return h * MLXArray(Float(config.text.hiddenSize).squareRoot()).asType(h.dtype)
    }

    /// Per-token outputs `[L, embedding_dim]` (final norm, then the 512→768 projection) for `[L, hidden]` inputs.
    func tokens(_ embeds: MLXArray, cancellation: NativeCancellation) throws -> MLXArray {
        let hidden = try text(embeds.expandedDimensions(axis: 0), attentionMask: nil, cancellation: cancellation)
        return text.embeddingProjection(hidden[0])
    }

    /// Soft tokens of one picture `[n, hidden]` through the vision tower and its embedder.
    func picture(_ patches: EG2Patches, cancellation: NativeCancellation) throws -> MLXArray {
        guard let vision, let embedVision else { throw WorkerFailure(code: "invalid_request", message: "This checkpoint has no vision tower.") }
        let encoded = try vision.encode(patches, cancellation: cancellation)
        return embedVision(vision.pool(encoded, columns: patches.columns, rows: patches.rows))
    }

    /// Soft tokens of one clip `[n, hidden]` through the audio tower and its embedder.
    func sound(_ features: (values: [Float], frames: Int, mask: [Bool]), cancellation: NativeCancellation) throws -> MLXArray {
        guard let audio, let embedAudio else { throw WorkerFailure(code: "invalid_request", message: "This checkpoint has no audio tower.") }
        return embedAudio(try audio(features.values, frames: features.frames, mask: features.mask, cancellation: cancellation))
    }
}

/// One request part in its item's order.
enum EG2Part: Equatable {
    case text(String)
    case image(URL)
    case video(URL)
    case audio(URL)
    case document(URL)
}

/// The rendered layout of one item: text runs and media placeholders, each placeholder bound to its part.
enum EG2Piece: Equatable {
    case text(String)
    case image(Int)
    case video(Int)
    case audio(Int)
}

enum EG2Prompt {
    static let image = "<|image|>", video = "<|video|>", audio = "<|audio|>"

    /// The card's task prefixes (config_sentence_transformers.json prompts).
    static func prefix(task: String, title: String?, instruction: String?) throws -> String {
        if let instruction { return "task: \(instruction) | query: " }
        switch task {
        case "document": return "title: \(title ?? "none") | text: "
        case "query": return "task: search result | query: "
        case "question_answering": return "task: question answering | query: "
        case "fact_checking": return "task: fact checking | query: "
        case "code_retrieval": return "task: code retrieval | query: "
        case "classification": return "task: classification | query: "
        case "clustering": return "task: clustering | query: "
        case "similarity": return "task: sentence similarity | query: "
        default: throw WorkerFailure(code: "invalid_request", message: "Unsupported task \(task).")
        }
    }

    /// The chat template plus the processor's placeholder rule for one item, the prefix (a system turn) first. If any text
    /// part already holds placeholders, media parts fill them in order per kind; otherwise each sits where it was given.
    static func render(_ parts: [EG2Part], prefix: String?) throws -> [EG2Piece] {
        let texts = parts.compactMap { if case .text(let t) = $0 { return t } else { return nil } }
        let manual = texts.contains { $0.contains(image) || $0.contains(video) || $0.contains(audio) }
        var flat = ""
        var order: [EG2Piece] = [] // media in part order, kinds only (indices assigned below)
        var images = 0, videos = 0, audios = 0
        for part in parts {
            switch part {
            case .text(let t): flat += t
            case .image, .document:
                order.append(.image(images)); images += 1
                if !manual { flat += image }
            case .video:
                order.append(.video(videos)); videos += 1
                if !manual { flat += video }
            case .audio:
                order.append(.audio(audios)); audios += 1
                if !manual { flat += audio }
            }
        }
        if let prefix { flat = prefix + flat }
        var pieces: [EG2Piece] = [], run = "", rest = Substring(flat)
        var seen = (image: 0, video: 0, audio: 0)
        while !rest.isEmpty {
            if rest.hasPrefix(image) {
                if !run.isEmpty { pieces.append(.text(run)); run = "" }
                pieces.append(.image(seen.image)); seen.image += 1; rest = rest.dropFirst(image.count)
            } else if rest.hasPrefix(video) {
                if !run.isEmpty { pieces.append(.text(run)); run = "" }
                pieces.append(.video(seen.video)); seen.video += 1; rest = rest.dropFirst(video.count)
            } else if rest.hasPrefix(audio) {
                if !run.isEmpty { pieces.append(.text(run)); run = "" }
                pieces.append(.audio(seen.audio)); seen.audio += 1; rest = rest.dropFirst(audio.count)
            } else {
                run.append(rest.removeFirst())
            }
        }
        if !run.isEmpty { pieces.append(.text(run)) }
        guard seen.image == images, seen.video == videos, seen.audio == audios else {
            throw WorkerFailure(
                code: "invalid_request",
                message:
                    "The text has \(seen.image) image, \(seen.video) video and \(seen.audio) audio placeholders but the input gives \(images), \(videos) and \(audios); they must match."
            )
        }
        return pieces
    }
}

final class EmbeddingGemma2Adapter: NativeModel {
    let architecture = "embedding_gemma2"
    let operations: Set<String> = ["embed"]
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { graph.map { Dictionary(uniqueKeysWithValues: $0.parameters().flattened()) } ?? [:] }
    private(set) var graph: EmbeddingGemma2Graph?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    /// Token ids of the last embedded item (lab parity compares them with the maker's processor).
    private(set) var lastIDs: [Int] = []
    /// Per-token outputs of the last embedded item, before pooling (lab parity).
    private(set) var lastTokens: MLXArray?
    static let context = 8192
    static let dimensions = [768, 512, 256, 128]
    static let tasks = ["query", "document", "similarity", "classification", "clustering", "question_answering", "fact_checking", "code_retrieval"]

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        guard !context.randomWeights else { throw failure("EmbeddingGemma 2 has no random-weight lab graph.") }
        let config = try JSONDecoder().decode(EmbeddingGemma2Root.self, from: Data(contentsOf: context.weights.appendingPathComponent("config.json")))
        let model = EmbeddingGemma2Graph(config)
        let archives = try FileManager.default.contentsOfDirectory(at: context.weights, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }
        _ = try NativePrecisionTier(context.precision, half: context.precision == "bf16" ? .bfloat16 : .float16).load(
            model, archives: archives,
            map: { key, value in
                try cancellation.check()
                return (key, value)
            }, eligible: TextRetrievalAdapter.eligible)
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer_config.json"))) as! [NSString: Any]
        let dataJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer.json"))) as! [NSString: Any]
        tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(configJSON), tokenizerData: Config(dataJSON))
        model.train(false)
        try NativeParameters.requireHalf(parameters)
        graph = model
    }

    /// Request item → ordered parts. Strings and text parts are text; image, video, audio and document parts carry a path.
    static func parts(_ item: ArsenalWire.JSONValue) throws -> [EG2Part] {
        let list = item["type"]?.string == "content" ? item["content"]?.array ?? [] : [item]
        guard !list.isEmpty, list.count <= 32 else { throw WorkerFailure(code: "invalid_request", message: "A content item holds 1 to 32 parts.") }
        return try list.map { part in
            if let text = part.string ?? (part["type"]?.string == "text" ? part["text"]?.string : nil) { return .text(text) }
            let type = part["type"]?.string ?? ""
            guard let path = part[type]?["path"]?.string, !path.isEmpty else {
                throw WorkerFailure(code: "invalid_request", message: "Each part needs text or a local image, audio, video or PDF path.")
            }
            let url = URL(fileURLWithPath: path)
            switch type {
            case "image": return .image(url)
            case "video": return .video(url)
            case "audio": return .audio(url)
            case "document": return .document(url)
            default: throw WorkerFailure(code: "invalid_request", message: "EmbeddingGemma 2 accepts text, images, PDFs, audio and video.")
            }
        }
    }

    struct Options {
        let dimension: Int
        let prefix: String
        let instruction: String?
        let title: String?
        let normalize: Bool
        let truncate: Bool
    }

    /// Request options, checked before any work: task, title, instruction, Matryoshka size, normalize, truncate.
    static func options(_ input: JSONObject, width: Int) throws -> Options {
        func invalid(_ message: String) -> WorkerFailure { .init(code: "invalid_request", message: message) }
        let task = input["task"]?.string ?? "document"
        guard tasks.contains(task) else { throw invalid("Unsupported task \(task).") }
        var dimension = width
        if let value = input["dimensions"] {
            guard let number = value.number, let exact = Int(exactly: number), dimensions.contains(exact), exact <= width else {
                throw invalid("dimensions must be 768, 512, 256 or 128.")
            }
            dimension = exact
        }
        let instruction = input["instruction"]?.string, title = input["title"]?.string
        if let instruction, instruction.isEmpty || instruction.count > 1000 { throw invalid("instruction must be 1 to 1,000 characters.") }
        if let title, title.isEmpty || title.count > 1000 { throw invalid("title must be 1 to 1,000 characters.") }
        return Options(
            dimension: dimension, prefix: try EG2Prompt.prefix(task: task, title: title, instruction: instruction), instruction: instruction, title: title,
            normalize: input["normalize"]?.bool ?? true, truncate: input["truncate"]?.bool ?? false)
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "embed", let graph, let tokenizer else { throw failure("EmbeddingGemma 2 is not loaded or the operation is unsupported.") }
        guard let items = input["input"]?.array, !items.isEmpty, items.count <= 2048 else { throw failure("Provide 1 to 2,048 inputs.") }
        let options = try Self.options(input, width: graph.config.text.embeddingDim)
        let (dimension, prefix, instruction, title) = (options.dimension, options.prefix, options.instruction, options.title)
        let normalize = options.normalize, truncate = options.truncate
        var vectors: [ArsenalWire.JSONValue] = [], total = 0
        for item in items {
            try cancellation.check()
            // Without an explicit task, only text-only items take the default document prefix: the card passes images,
            // audio, video and interleaved inputs without one.
            let explicit = input["task"] != nil || instruction != nil || title != nil
            // Media alone carries no prefix (the card); a task given explicitly also prefixes items that mix text and media.
            let kinds = try Self.parts(item).map { part -> Bool in if case .text = part { return true } else { return false } }
            let applies = kinds.contains(true) && (explicit || kinds.allSatisfy { $0 })
            let pooled = try embed(
                item, prefix: applies ? prefix : nil, truncate: truncate, graph: graph, tokenizer: tokenizer, cancellation: cancellation, tokens: &total)
            var vector = pooled[..<dimension]
            if normalize { vector = vector / maximum(sqrt(sum(vector * vector)), MLXArray(Float(1e-12))) }
            let row = vector.asArray(Float.self)
            guard row.count == dimension, row.allSatisfy(\.isFinite) else { throw failure("Non-finite EmbeddingGemma 2 vector.") }
            vectors.append(.array(row.map { .number(Double($0)) }))
        }
        return ["embeddings": .array(vectors), "input_tokens": .number(Double(total))]
    }

    private enum Media {
        case picture(EG2Patches)
        case frames([EG2Patches])
        case sound((values: [Float], frames: Int, mask: [Bool]))
    }

    /// One item's pooled (mean over every token), unnormalized 768-number vector in float32.
    func embed(
        _ item: ArsenalWire.JSONValue, prefix: String?, truncate: Bool, graph: EmbeddingGemma2Graph, tokenizer: any Tokenizers.Tokenizer,
        cancellation: NativeCancellation, tokens total: inout Int
    ) throws -> MLXArray {
        let parts = try Self.parts(item)
        // PDFs expand into one image part per page.
        var expanded: [EG2Part] = [], pages: [Int: [EG2Picture]] = [:]
        for part in parts {
            if case .document(let url) = part {
                let rendered = try Self.renderPDF(url, cancellation: cancellation)
                for page in rendered {
                    pages[expanded.count] = [page]; expanded.append(.image(URL(fileURLWithPath: "/dev/null")))
                }
            } else {
                expanded.append(part)
            }
        }
        let pieces = try EG2Prompt.render(expanded, prefix: prefix)
        var imageParts: [(Int, URL)] = [], videoParts: [URL] = [], audioParts: [URL] = []
        for (index, part) in expanded.enumerated() {
            switch part {
            case .image(let url): imageParts.append((index, url))
            case .video(let url): videoParts.append(url)
            case .audio(let url): audioParts.append(url)
            default: break
            }
        }
        let config = graph.config
        // Lay out ids and media, counting the budget before any tower runs.
        var ids: [Int] = [2], media: [(Range<Int>, Media)] = [], textOnly = true
        func admit() throws {
            guard ids.count + 1 <= Self.context || (truncate && textOnly) else {
                throw failure("This input needs \(ids.count + 1) tokens; EmbeddingGemma 2 reads at most \(Self.context). Embed fewer images or frames, shorter audio or less text.")
            }
        }
        for piece in pieces {
            try cancellation.check()
            switch piece {
            case .text(let text):
                ids += tokenizer.encode(text: text, addSpecialTokens: false)
            case .image(let index):
                textOnly = false
                let (at, url) = imageParts[index]
                let picture = try pages[at]?.first ?? EG2Picture.load(url)
                let patches = try EG2Vision.patches(picture, maxSoftTokens: EG2Vision.imageSoftTokens)
                ids.append(config.beginImage)
                let start = ids.count
                ids += Array(repeating: config.imageToken, count: patches.softTokens)
                media.append((start..<ids.count, .picture(patches)))
                ids.append(config.endImage)
            case .video(let index):
                textOnly = false
                // A file AVFoundation cannot open or decode is the caller's mistake: 422, not 503.
                let track = try NativeMedia.decoding("Video") { try EG2VideoIO.track(videoParts[index], cancellation: cancellation) }
                let indices = EG2Vision.sampleFrames(total: track.frames, fps: track.fps)
                guard !indices.isEmpty else { throw failure("The video has no frames.") }
                // Each frame becomes its small patch tensor as it is decoded; no full-size frame outlives its turn.
                let patches = try NativeMedia.decoding("Video") {
                    try EG2VideoIO.frames(videoParts[index], track: track, indices: indices, cancellation: cancellation) {
                        try EG2Vision.patches($0, maxSoftTokens: EG2Vision.videoSoftTokens)
                    }
                }
                for frame in patches {
                    ids.append(config.beginImage)
                    ids += Array(repeating: config.videoToken, count: frame.softTokens)
                    ids.append(config.endImage)
                }
                // Mark the whole video span; its features are the frames' soft tokens in order at the video ids.
                media.append((0..<0, .frames(patches)))
            case .audio(let index):
                textOnly = false
                let pcm = try EG2AudioIO.mono(audioParts[index], maximumSamples: EG2AudioIO.decodeLimit)
                guard !pcm.isEmpty else { throw failure("The audio clip is empty.") }
                guard pcm.count <= EG2Mel.maxSamples || truncate else {
                    throw failure("EmbeddingGemma 2 reads the first 30 seconds of a clip, as its processor does; pass truncate: true to embed them, or cut the clip.")
                }
                let features = EG2Mel.features(Array(pcm.prefix(EG2Mel.maxSamples)))
                let count = EG2Mel.tokens(samples: min(pcm.count, EG2Mel.maxSamples))
                guard count > 0 else { throw failure("The audio clip is too short (under 10 ms).") }
                ids.append(config.beginAudio)
                let start = ids.count
                ids += Array(repeating: config.audioToken, count: count)
                media.append((start..<ids.count, .sound(features)))
                ids.append(config.endAudio)
            }
            try admit()
        }
        if ids.count + 1 > Self.context {
            // Only text-only items reach here (admit refuses the rest): keep <bos>, the first tokens and <eos>.
            ids = Array(ids.prefix(Self.context - 1))
        }
        ids.append(1)
        // Placeholders become pad id 0 before the lookup, then the soft tokens replace them.
        let placeholders: Set<Int> = [config.imageToken, config.videoToken, config.audioToken]
        var embeds = graph.embed(ids.map { placeholders.contains($0) ? 0 : $0 })
        var videoCursor = ids.startIndex
        for (range, item) in media {
            try cancellation.check()
            switch item {
            case .picture(let patches):
                let features = try graph.picture(patches, cancellation: cancellation)
                guard features.dim(0) == range.count else { throw failure("Image features do not match their placeholders.") }
                embeds[range.lowerBound..<range.upperBound] = features.asType(embeds.dtype)
            case .frames(let frames):
                for frame in frames {
                    guard let start = ids[videoCursor...].firstIndex(of: config.videoToken) else { throw failure("Video placeholders are missing.") }
                    let features = try graph.picture(frame, cancellation: cancellation)
                    let end = start + features.dim(0)
                    guard end <= ids.count, ids[start..<end].allSatisfy({ $0 == config.videoToken }) else { throw failure("Video features do not match their placeholders.") }
                    embeds[start..<end] = features.asType(embeds.dtype)
                    videoCursor = end
                }
            case .sound(let features):
                let encoded = try graph.sound(features, cancellation: cancellation)
                guard encoded.dim(0) == range.count else { throw failure("Audio features do not match their placeholders.") }
                embeds[range.lowerBound..<range.upperBound] = encoded.asType(embeds.dtype)
            }
        }
        let tokens = try graph.tokens(embeds, cancellation: cancellation)
        lastIDs = ids; lastTokens = tokens
        total += ids.count
        return tokens.asType(.float32).mean(axis: 0)
    }

    static func renderPDF(_ url: URL, cancellation: NativeCancellation) throws -> [EG2Picture] {
        guard let document = CGPDFDocument(url as CFURL), document.numberOfPages > 0, document.numberOfPages <= 29 else {
            throw WorkerFailure(code: "invalid_request", message: "PDFs need 1 to 29 pages (280 tokens each within the 8,192-token context).")
        }
        var pages: [EG2Picture] = []
        for index in 1...document.numberOfPages {
            try cancellation.check()
            guard let page = document.page(at: index) else { throw WorkerFailure(code: "invalid_request", message: "Unreadable PDF page.") }
            let (width, height) = PDFPageRaster.pixelSize(PDFPageRaster.shownSize(page))
            var rgba = [UInt8](repeating: 255, count: width * height * 4)
            let ok = rgba.withUnsafeMutableBytes { buffer -> Bool in
                guard
                    let context = CGContext(
                        data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                else { return false }
                PDFPageRaster.draw(page, in: context, width: width, height: height)
                return true
            }
            guard ok else { throw WorkerFailure(code: "invalid_request", message: "PDF page render failed.") }
            var rgb = [UInt8](repeating: 0, count: width * height * 3)
            for i in 0..<(width * height) { rgb[3 * i] = rgba[4 * i]; rgb[3 * i + 1] = rgba[4 * i + 1]; rgb[3 * i + 2] = rgba[4 * i + 2] }
            pages.append(EG2Picture(width: width, height: height, pixels: rgb))
        }
        return pages
    }

    private func failure(_ message: String) -> WorkerFailure { .init(code: "invalid_request", message: message) }
    func unload() { graph = nil; tokenizer = nil; lastTokens = nil }
}

enum EG2AudioIO {
    /// Clips up to 10 minutes decode (so `truncate` can keep their first 30 s); longer files are refused.
    static let decodeLimit = 16_000 * 600
    /// 16 kHz mono as librosa.load(sr=16000, mono=True): every channel resampled, then their mean.
    static func mono(_ url: URL, maximumSamples: Int) throws -> [Float] {
        let channels = Int(try NativeMedia.openAudio(url).processingFormat.channelCount)
        guard (1...8).contains(channels) else { throw WorkerFailure(code: "invalid_request", message: "Audio has \(channels) channels; at most 8 are read.") }
        if channels > 2 { return try SurroundAudio.mono(url, sampleRate: EG2Mel.sampleRate, maximumSamples: maximumSamples) }
        let audio = try NativeMedia.audio(url, sampleRate: EG2Mel.sampleRate, channels: channels, maximumFrames: maximumSamples)
        guard channels > 1 else { return audio.samples }
        return stride(from: 0, to: audio.samples.count, by: channels).map { start in
            audio.samples[start..<(start + channels)].reduce(0, +) / Float(channels)
        }
    }
}

// swiftlint:enable force_cast shorthand_operator
