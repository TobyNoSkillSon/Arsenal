// SPDX-License-Identifier: MIT
// The maker's decision path (OneJev qev/engine.py DecisionEngine @81ce62f + serve.patch, Apache-2.0) over the MLX
// decoder: render every branch, cut the longest common token prefix that ends at the state block, prefill it once,
// then score each question suffix from that shared state (`_run_sequential`, the exact path) with the slot_lm readout.
import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// One branch's slot logits (float32 readout) and its suffix length.
public struct Branch: Sendable {
    public let kind: QuestionKind
    public let labels: [String]
    public let descriptions: [PyJSON?]
    public let logits: [Double]
    public init(kind: QuestionKind, labels: [String], descriptions: [PyJSON?], logits: [Double]) {
        self.kind = kind; self.labels = labels; self.descriptions = descriptions; self.logits = logits
    }
}

/// The token layout of one request: the shared prefix and each branch's suffix (qev `_prefix_and_suffix_ids`).
public struct Layout: Sendable {
    public let prefix: [Int]
    public let suffixes: [[Int]]
    public var total: Int { prefix.count + suffixes.reduce(0) { $0 + $1.count } }
    public var longest: Int { prefix.count + (suffixes.map(\.count).max() ?? 0) }
}

public struct LimitError: Error, CustomStringConvertible { public let description: String }

/// A shared prefix that holds pictures or videos: expanded token ids, grids and pixels of the images and of the videos
/// (each run through the vision tower on its own, as HF get_image_features / get_video_features), and M-RoPE positions.
public struct MediaPrefix: Sendable {
    public let prefix: [Int]
    public let grids: [THW]
    public let pixels: [Float]
    public let videoGrids: [THW]
    public let videoPixels: [Float]
    public let positions: [[Int32]]
    public let delta: Int
    public init(
        prefix: [Int], grids: [THW], pixels: [Float], videoGrids: [THW] = [], videoPixels: [Float] = [], positions: [[Int32]], delta: Int
    ) {
        self.prefix = prefix; self.grids = grids; self.pixels = pixels; self.videoGrids = videoGrids; self.videoPixels = videoPixels
        self.positions = positions; self.delta = delta
    }
}

/// One item of the makers' media list.
public enum MediaItem: Sendable {
    case image(ImageSource)
    case video(VideoSource)
    public var kind: MediaKind {
        if case .image = self { return .image }
        return .video
    }
}
public struct NonFiniteError: Error {}

public final class ElephantEngine: @unchecked Sendable {
    public let model: ElephantModel
    public let tokenizer: any Tokenizer
    /// Slot token ids, A..Z then the two-letter labels when the tokenizer holds every one as one round-trip token.
    public let slotIDs: [Int]
    public let vision: VisionConfig
    public let images: ImageProcessorConfig
    public let videos: VideoProcessorConfig
    /// Arsenal's bound on images per request; the token limits bound their size.
    public static let maxImages = 32
    public var maxSlots: Int { slotIDs.count }
    /// qev DecisionEngine defaults: max_branch_tokens and max_request_tokens.
    public static let maxBranchTokens = 32768
    public static let maxRequestTokens = 65536

    public init(weights: URL, cancelled: () -> Bool = { false }) throws {
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: weights.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
        let config = try ElephantConfig(json: configJSON)
        vision = try VisionConfig(json: configJSON)
        let processorJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: weights.appendingPathComponent("processor_config.json"))) as? [String: Any] ?? [:]
        images = try ImageProcessorConfig(json: processorJSON)
        videos = try VideoProcessorConfig(json: processorJSON)
        guard images.patch == vision.patch, images.merge == vision.merge, images.temporal == vision.temporal else {
            throw PromptError("processor_config.json and config.json disagree on the vision patch layout.")
        }
        tokenizer = try Self.tokenizer(weights)
        slotIDs = try Self.verifySlots(tokenizer)
        let model = ElephantModel(config, vision: vision)
        var parameters: [String: MLXArray] = [:]
        var head: MLXArray?
        let files = try FileManager.default.contentsOfDirectory(at: weights, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw PromptError("No safetensors in the weights folder.") }
        var dtype: DType?
        for file in files {
            if cancelled() { throw Cancelled() }
            for (key, value) in try MLX.loadArrays(url: file) {
                if key == "lm_head.weight" { head = value; continue }
                guard let (name, mapped) = ElephantModel.map(key, value) ?? VisionModel.map(key, value) else { continue }
                guard [.float16, .bfloat16].contains(mapped.dtype), dtype == nil || dtype == mapped.dtype else {
                    throw PromptError("Weights must be one 16-bit format; \(key) is \(mapped.dtype).")
                }
                dtype = mapped.dtype
                parameters[name] = mapped
            }
        }
        // slotRows is filled from lm_head below, not read from a checkpoint key.
        let expected = Set(model.parameters().flattened().map(\.0)).subtracting(["slotRows"])
        let missing = expected.subtracting(parameters.keys), unused = Set(parameters.keys).subtracting(expected)
        guard missing.isEmpty, unused.isEmpty, let head else {
            throw PromptError("Checkpoint does not match the decoder: missing \(missing.sorted().prefix(4)), unused \(unused.sorted().prefix(4)), head \(head != nil).")
        }
        for (name, value) in model.parameters().flattened() where name != "slotRows" && parameters[name]!.shape != value.shape {
            throw PromptError("Shape mismatch for \(name): \(parameters[name]!.shape) vs \(value.shape).")
        }
        model.update(parameters: ModuleParameters.unflattened(parameters))
        model.setSlotRows(head, slots: slotIDs)
        model.train(false)
        eval(model)
        self.model = model
    }

    public static func tokenizer(_ weights: URL) throws -> any Tokenizer {
        var config = try JSONSerialization.jsonObject(with: Data(contentsOf: weights.appendingPathComponent("tokenizer_config.json"))) as? [NSString: Any] ?? [:]
        // The checkpoint names Qwen3_5Tokenizer, which swift-transformers does not list; the byte-level BPE, its NFC
        // normalizer and split regex all come from tokenizer.json, which Qwen2Tokenizer reads unchanged.
        if config["tokenizer_class"] as? String == "Qwen3_5Tokenizer" { config["tokenizer_class"] = "Qwen2Tokenizer" }
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: weights.appendingPathComponent("tokenizer.json"))) as? [NSString: Any] ?? [:]
        return try AutoTokenizer.from(tokenizerConfig: Config(config), tokenizerData: Config(data))
    }

    /// qev `_verify_slots`: A..Z must be single round-trip tokens; the two-letter labels are all or nothing.
    public static func verifySlots(_ tokenizer: any Tokenizer) throws -> [Int] {
        func one(_ label: String) -> Int? {
            let ids = tokenizer.encode(text: label, addSpecialTokens: false)
            return ids.count == 1 && tokenizer.decode(tokens: ids, skipSpecialTokens: false) == label ? ids[0] : nil
        }
        let letters = Prompt.slotLabels.prefix(26).map(one)
        guard letters.allSatisfy({ $0 != nil }) else { throw PromptError("Answer slots A..Z are not single round-trip tokens.") }
        var ids = letters.map { $0! }
        let extra = Prompt.slotLabels.dropFirst(26).map(one)
        if extra.allSatisfy({ $0 != nil }) { ids += extra.map { $0! } }
        guard Set(ids).count == ids.count else { throw PromptError("Answer slot tokens collide.") }
        return ids
    }

    public func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    public func layout(_ state: PyJSON?, _ rendered: [RenderedQuestion], images: Int = 0) throws -> Layout {
        try Self.layout(tokenizer, state, rendered, media: Array(repeating: .image, count: images))
    }

    public func layout(_ state: PyJSON?, _ rendered: [RenderedQuestion], media: [MediaKind]) throws -> Layout {
        try Self.layout(tokenizer, state, rendered, media: media)
    }

    /// qev `_prefix_and_suffix_ids`.
    public static func layout(_ tokenizer: any Tokenizer, _ state: PyJSON?, _ rendered: [RenderedQuestion], images: Int = 0) throws -> Layout {
        try layout(tokenizer, state, rendered, media: Array(repeating: .image, count: images))
    }

    public static func layout(_ tokenizer: any Tokenizer, _ state: PyJSON?, _ rendered: [RenderedQuestion], media: [MediaKind]) throws -> Layout {
        func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
        if !media.isEmpty, rendered.contains(where: { Prompt.hasPlaceholder($0.suffix) }) {
            throw PromptError("media placeholders belong in the state, not in a question")
        }
        let full = try rendered.map { encode(try Prompt.chat(state: state, suffix: $0.suffix, media: media)) }
        let prefix = encode(try Prompt.prefixText(state: state, media: media))
        var n = prefix.count
        while n > 0, !full.allSatisfy({ $0.count >= n && Array($0[0..<n]) == Array(prefix[0..<n]) }) { n -= 1 }
        let suffixes = full.map { Array($0[n...]) }
        guard suffixes.allSatisfy({ !$0.isEmpty }) else { throw PromptError("a branch has an empty suffix; state block boundary mismatch") }
        return Layout(prefix: Array(prefix[0..<n]), suffixes: suffixes)
    }

    /// The branches of a request as qev `decide` builds them (debias 1): one per question, or for a choice wider than
    /// the slots, balanced chunks of at most maxSlots - 1 options plus the shared "none of these" option.
    public func branches(_ questions: [Question]) throws -> Plan { try Self.branches(questions, maxSlots: maxSlots) }

    public typealias Plan = (rendered: [RenderedQuestion], owners: [(question: Int, chunk: Int?)], plans: [Int: [[Int]]])

    public static func branches(_ questions: [Question], maxSlots: Int) throws -> Plan {
        var rendered: [RenderedQuestion] = [], owners: [(Int, Int?)] = [], plans: [Int: [[Int]]] = [:]
        for (qi, q) in questions.enumerated() {
            if q.kind == .choice, q.options.count > maxSlots {
                let nChunks = (q.options.count + maxSlots - 2) / (maxSlots - 1)
                let size = (q.options.count + nChunks - 1) / nChunks
                let chunks = stride(from: 0, to: q.options.count, by: size).map { Array($0..<Swift.min($0 + size, q.options.count)) }
                plans[qi] = chunks
                for (ci, chunk) in chunks.enumerated() {
                    let sub = Question(kind: .choice, instructions: q.instructions, options: chunk.map { q.options[$0] } + [(Prompt.noneOption, .string(Prompt.noneDescription))])
                    rendered.append(try Prompt.render(sub)); owners.append((qi, ci))
                }
                continue
            }
            rendered.append(try Prompt.render(q)); owners.append((qi, nil))
        }
        return (rendered, owners, plans)
    }

    /// The trained length limits (refused, never truncated).
    public func check(_ layout: Layout) throws {
        if layout.longest > Self.maxBranchTokens {
            throw LimitError(description: "state plus longest question is \(layout.longest) tokens, limit is \(Self.maxBranchTokens)")
        }
        if layout.total > Self.maxRequestTokens { throw LimitError(description: "request is \(layout.total) tokens, limit is \(Self.maxRequestTokens)") }
    }

    /// `_run_sequential`: one prefill of the shared prefix, then each suffix from that state.
    public func run(
        _ layout: Layout, _ rendered: [RenderedQuestion], media: MediaPrefix? = nil, probe: Probe? = nil, cancelled: () -> Bool = { false }
    ) throws -> [Branch] {
        let base: DecoderState
        if let media {
            let width = 3 * vision.temporal * vision.patch * vision.patch
            var parts: [(features: MLXArray, tokenId: Int)] = []
            var captured: [MLXArray] = []
            for (values, grids, token) in [(media.pixels, media.grids, vision.imageTokenId), (media.videoPixels, media.videoGrids, vision.videoTokenId)]
            where !grids.isEmpty {
                let pixels = MLXArray(values, [values.count / width, width])
                let features = try model.visual(pixels.asType(model.dtype), grids: grids, cancelled: cancelled)
                eval(features)
                parts.append((features, token)); captured.append(features)
            }
            probe?("vision", concatenated(captured, axis: 0))
            (_, base) = try model.forward(
                media.prefix, after: nil, vision: parts, positions: (media.positions, media.delta), probe: probe, cancelled: cancelled)
        } else {
            (_, base) = try model.forward(layout.prefix, after: nil, probe: probe, cancelled: cancelled)
        }
        eval(base.layers.compactMap { $0 }.flatMap(ElephantModel.arrays))
        Memory.clearCache()
        var out: [Branch] = []
        for (r, suffix) in zip(rendered, layout.suffixes) {
            if cancelled() { throw Cancelled() }
            let (hidden, _) = try model.forward(suffix, after: base, probe: probe, cancelled: cancelled)
            let logits = model.slotLogits(hidden, count: r.labels.count)
            eval(logits)
            let values = try Self.finite(logits.asArray(Float.self).map(Double.init))
            out.append(Branch(kind: r.kind, labels: r.labels, descriptions: r.descriptions, logits: values))
        }
        return out
    }

    /// A NaN or infinite slot logit would otherwise become a confident answer (a NaN P(yes) is clamped to 0.0).
    public static func finite(_ logits: [Double]) throws -> [Double] {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite) else { throw NonFiniteError() }
        return logits
    }

    /// A whole request: the answers object (question names in request order) and the request's token count.
    public func decide(
        state: PyJSON?, questions: [(String, Question)], images: [ImageSource], cancelled: () -> Bool = { false }
    ) throws -> (answers: [(String, PyJSON)], tokens: Int) {
        try decide(state: state, questions: questions, media: images.map { .image($0) }, cancelled: cancelled)
    }

    public func decide(
        state: PyJSON?, questions: [(String, Question)], media items: [MediaItem] = [], cancelled: () -> Bool = { false }
    ) throws -> (answers: [(String, PyJSON)], tokens: Int) {
        let plan = try branches(questions.map(\.1))
        let (layout, media) = try prepare(state, plan.rendered, media: items, cancelled: cancelled)
        let results = try run(layout, plan.rendered, media: media, cancelled: cancelled)
        return (Self.assemble(questions, plan, results), layout.total)
    }

    /// The token layout of a request and, with media, its expanded prefix, pixels and rotary positions: vision tokens
    /// counted from stored sizes and the limits checked before any pixel is decoded, then one picture or frame list at a
    /// time decoded and patchified.
    public func prepare(
        _ state: PyJSON?, _ rendered: [RenderedQuestion], media items: [MediaItem], cancelled: () -> Bool = { false }
    ) throws -> (Layout, MediaPrefix?) {
        var layout = try layout(state, rendered, media: items.map(\.kind))
        var media: MediaPrefix?
        if !items.isEmpty {
            try Self.refuseVisionTokens(layout.suffixes, vision: vision)
            // Count the vision tokens from the stored sizes and check the limits before any pixel is decoded.
            let expansion = try items.map { item -> Expansion in
                switch item {
                case .image(let source):
                    return .image(try ImageInput.tokens(height: source.height, width: source.width, config: images).grid)
                case .video(let source):
                    let grid = try VideoInput.grid(source, config: videos)
                    let labels = VideoInput.timestamps(frames: source.frames.count, fps: source.fps, temporal: videos.temporal)
                    return .video(grid, labels.map { encode(VideoInput.label($0)) })
                }
            }
            let prefix = try Self.expand(layout.prefix, items: expansion, vision: vision)
            layout = Layout(prefix: prefix, suffixes: layout.suffixes)
            try check(layout)
            let (pixels, grids, videoPixels, videoGrids) = try Self.decodeMedia(items, expansion, images: images, videos: videos, cancelled: cancelled)
            let rope = try MultimodalPositions.rope(prefix, blocks: Self.blocks(expansion, vision: vision), merge: vision.merge)
            media = MediaPrefix(
                prefix: prefix, grids: grids, pixels: pixels, videoGrids: videoGrids, videoPixels: videoPixels, positions: rope.positions, delta: rope.delta)
        } else {
            try check(layout)
        }
        return (layout, media)
    }

    /// Decode and patchify each media item, one at a time, checking that the pixels give the grid its tokens were
    /// counted for (a file can change between the size check and the decode).
    public static func decodeMedia(
        _ items: [MediaItem], _ expansion: [Expansion], images: ImageProcessorConfig, videos: VideoProcessorConfig, cancelled: () -> Bool = { false }
    ) throws -> (pixels: [Float], grids: [THW], videoPixels: [Float], videoGrids: [THW]) {
        guard items.count == expansion.count else { throw PromptError("Media items and their expansion differ in count.") }
        var pixels: [Float] = [], grids: [THW] = [], videoPixels: [Float] = [], videoGrids: [THW] = []
        for (item, counted) in zip(items, expansion) {
            if cancelled() { throw Cancelled() }
            // One picture or frame list decoded at a time; only its patches are kept. The decoded pixels must give
            // the grid the tokens were counted for (a file can change between the size check and the decode).
            switch (item, counted) {
            case (.image(let source), .image(let grid)):
                let (values, actual) = try ImageInput.pixels(try ImageInput.load(source.url), config: images)
                guard actual == grid else { throw PromptError("A picture changed size while it was read.") }
                pixels += values; grids.append(actual)
            case (.video(let source), .video(let grid, _)):
                let (values, actual) = try VideoInput.pixels(source, config: videos, cancelled: cancelled)
                guard actual == grid else { throw PromptError("A video changed size while it was read.") }
                videoPixels += values; videoGrids.append(actual)
            default: throw PromptError("A media item does not match its expansion.")
            }
        }
        return (pixels, grids, videoPixels, videoGrids)
    }

    /// How one media item expands in the prompt: an image's grid, or a video's grid and its frame-pair timestamp tokens.
    public enum Expansion {
        case image(THW)
        case video(THW, [[Int]])
    }

    /// The vision blocks of the expanded prompt in order: one per image, one per video frame pair (a [1, h, w] grid).
    public static func blocks(_ items: [Expansion], vision: VisionConfig) -> [(token: Int, grid: THW)] {
        items.flatMap { item -> [(token: Int, grid: THW)] in
            switch item {
            case .image(let g): return [(vision.imageTokenId, g)]
            case .video(let g, _): return (0..<g.t).map { _ in (vision.videoTokenId, THW(1, g.h, g.w)) }
            }
        }
    }

    /// The processor's expansion (Qwen3VLProcessor replace_image_token / replace_video_token): an image pad becomes its
    /// grid's merged-patch count of image pads; a video pad becomes, per frame pair, "<t seconds>", vision start, that
    /// pair's video pads and vision end (inside the template's own vision start and end).
    public static func expand(_ ids: [Int], items: [Expansion], vision: VisionConfig) throws -> [Int] {
        var out: [Int] = [], next = 0
        let merge = vision.merge * vision.merge
        for id in ids {
            guard id == vision.imageTokenId || id == vision.videoTokenId else { out.append(id); continue }
            guard next < items.count else { throw PromptError("More vision blocks than media items.") }
            switch (items[next], id == vision.imageTokenId) {
            case (.image(let g), true):
                out += Array(repeating: vision.imageTokenId, count: g.t * g.h * g.w / merge)
            case (.video(let g, let labels), false):
                guard labels.count == g.t else { throw PromptError("A video's timestamps do not match its frame pairs.") }
                for label in labels {
                    out += label + [vision.visionStartTokenId] + Array(repeating: vision.videoTokenId, count: g.h * g.w / merge) + [vision.visionEndTokenId]
                }
            default: throw PromptError("A vision block does not match its media item.")
            }
            next += 1
        }
        guard next == items.count else { throw PromptError("Some media items have no vision block in the prompt.") }
        return out
    }

    /// The makers' mm_engine `_prepare`: vision special tokens belong to the state, never to a question's tokens
    /// (a question that types `<|image_pad|>` would otherwise take a picture's place).
    public static func refuseVisionTokens(_ suffixes: [[Int]], vision: VisionConfig) throws {
        let ids: Set<Int> = [vision.imageTokenId, vision.videoTokenId, vision.visionStartTokenId, vision.visionEndTokenId]
        guard !suffixes.contains(where: { $0.contains(where: ids.contains) }) else {
            throw PromptError("media placeholders belong in the state, not in a question")
        }
    }

    /// The processor's expansion of each image-pad token into its grid's merged-patch count, in image order.
    public static func expand(_ ids: [Int], grids: [THW], imageTokenId: Int, merge: Int) throws -> [Int] {
        var out: [Int] = [], image = 0
        for id in ids {
            guard id == imageTokenId else { out.append(id); continue }
            guard image < grids.count else { throw PromptError("More image blocks than images.") }
            let g = grids[image]
            out += Array(repeating: imageTokenId, count: g.t * g.h * g.w / (merge * merge))
            image += 1
        }
        guard image == grids.count else { throw PromptError("Some images have no image block in the prompt.") }
        return out
    }

    /// qev `decide`'s answer assembly (temperature 1: the checkpoint ships no calibration.json).
    public static func assemble(_ questions: [(String, Question)], _ plan: Plan, _ results: [Branch]) -> [(String, PyJSON)] {
        var answers: [(String, PyJSON)] = []
        for (qi, (name, q)) in questions.enumerated() {
            let mine = zip(results, plan.owners).filter { $0.1.question == qi }.map(\.0)
            if let chunks = plan.plans[qi] {
                // qev: each option's probability within its chunk, renormalised without "none", weighted by its chunk's
                // share of the non-"none" mass.
                var scores = [Int: (Double, Double)](), weights: [Double] = []
                for (chunk, branch) in zip(chunks, mine) {
                    let p = Answers.softmax(branch.logits)
                    let keep = Swift.max(1 - p[chunk.count], 1e-9)
                    for (j, option) in chunk.enumerated() { scores[option] = (p[j] / keep, keep) }
                    weights.append(keep)
                }
                let totalWeight = weights.reduce(0, +) == 0 ? 1 : weights.reduce(0, +)
                let probs = q.options.indices.map { scores[$0]!.0 * (scores[$0]!.1 / totalWeight) }
                answers.append((name, Answers.choice(q.options.map(\.0), probs)))
                continue
            }
            let p = Answers.softmax(mine[0].logits)
            switch q.kind {
            case .noul: answers.append((name, Answers.noul(p[0])))
            case .choice: answers.append((name, Answers.choice(mine[0].labels, p)))
            case .score: answers.append((name, Answers.score(mine[0].descriptions, p)))
            }
        }
        return answers
    }

    /// One small request at load, so kernels compile before the first real request.
    public func warm() throws {
        let q = Question(kind: .noul, instructions: .string("Is this a test?"), options: [])
        _ = try decide(state: .string("warm-up"), questions: [("q", q)])
    }
}
