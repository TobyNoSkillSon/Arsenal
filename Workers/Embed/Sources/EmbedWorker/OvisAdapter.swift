// swiftlint:disable force_cast shorthand_operator
// SPDX-License-Identifier: MIT
import AVFoundation
import CoreImage
import Foundation
import Hub
import IntegrationKit
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import ArsenalWire
import Tokenizers

final class OvisAdapter: NativeModel {
    let architecture = "qwen3_5"
    let operations: Set<String> = ["embed"]
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { graph.map { Dictionary(uniqueKeysWithValues: $0.parameters().flattened()) } ?? [:] }
    private var graph: OvisGraph?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var processor: MLXVLM.Qwen3VLProcessor?
    private var template: String?
    private var random = false
    private var vocabulary = 0
    private var videoConfiguration: OvisVideoConfiguration?
    /// The image processor's pixel bounds and patch·merge factor, for counting image tokens before rendering.
    private var imageBounds = (minimum: 65536, maximum: 16_777_216, factor: 32)
    /// Arsenal bound on one input's prompt tokens; the maker publishes none. It admits every input the maker's processor
    /// configuration produces (the video processor's whole-clip budget, 234,881,024 px, is 114,688 tokens) and 68
    /// letter-size PDF pages at the 2,048-px render, and stops a request before it renders or prefills more.
    static let defaultMaxInputTokens = 131_072
    var maxInputTokens = OvisAdapter.defaultMaxInputTokens

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let data = try Data(contentsOf: context.weights.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
        let model = OvisGraph(config); graph = model; random = context.randomWeights; vocabulary = config.textConfiguration.vocabularySize
        if random {
            try NativeParameters.initializeRandom(model)
            // RMSNorm in the upstream Qwen3.5 graph uses direct scales; native HF weights add one at Load.
            if context.precision == "int8" || context.precision == "int4" {
                let file = context.weights.appendingPathComponent("random.safetensors")
                try MLX.save(arrays: parameters, url: file)
                defer { try? FileManager.default.removeItem(at: file) }
                _ = try NativePrecisionTier(context.precision).load(model, archives: [file], map: { ($0, $1) }, eligible: TextRetrievalAdapter.eligible)
            }
        } else {
            let archives = try FileManager.default.contentsOfDirectory(at: context.weights, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }
            _ = try NativePrecisionTier(context.precision).load(
                model, archives: archives,
                map: { key, value in
                    try cancellation.check()
                    var name = key.replacingOccurrences(of: "model.language_model.", with: "language_model.model.").replacingOccurrences(of: "model.visual.", with: "vision_tower.")
                    var value = value
                    if name.contains("conv1d.weight"), value.dim(-1) != 1 { value = value.movedAxis(source: 2, destination: 1) }
                    if name == "vision_tower.patch_embed.proj.weight", value.ndim == 5 { value = value.transposed(0, 2, 3, 4, 1) }
                    let norms = [".input_layernorm.weight", ".post_attention_layernorm.weight", ".model.norm.weight", ".q_norm.weight", ".k_norm.weight"]
                    if norms.contains(where: { name.hasSuffix($0) }), value.ndim == 1 { value = Self.nativeNorm(value) }
                    // DeltaNet's recurrent state is computed, never a learned inventory item.
                    name = name.replacingOccurrences(of: "language_model.model.model.", with: "language_model.model.")
                    return (name, value)
                }, eligible: TextRetrievalAdapter.eligible)
            let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer_config.json"))) as! [NSString: Any]
            let dataJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer.json"))) as! [NSString: Any]
            tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(configJSON), tokenizerData: Config(dataJSON))
            template = try String(contentsOf: context.weights.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        }
        var processorJSON: [String: Any]
        if random {
            processorJSON = [
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "merge_size": config.visionConfiguration.spatialMergeSize,
                "patch_size": config.visionConfiguration.patchSize, "temporal_patch_size": config.visionConfiguration.temporalPatchSize,
                "image_processor_type": "Qwen2VLImageProcessorFast", "min_pixels": 1024, "max_pixels": 1024
            ]
            videoConfiguration = try OvisVideoConfiguration([
                "size": ["shortest_edge": 1024, "longest_edge": 4096], "patch_size": 16, "merge_size": 2,
                "temporal_patch_size": 2, "min_frames": 4, "max_frames": 4, "fps": 2.0,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "image_processor_type": "Qwen2VLImageProcessorFast"
            ])
        } else {
            let settings = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("processor_config.json"))) as! [String: Any]
            guard let imageSettings = settings["image_processor"] as? [String: Any], let videoSettings = settings["video_processor"] as? [String: Any] else {
                throw failure("Ovis processor_config.json requires image and video settings.")
            }
            processorJSON = imageSettings
            videoConfiguration = try OvisVideoConfiguration(videoSettings)
            if let size = processorJSON["size"] as? [String: Int] {
                processorJSON["min_pixels"] = size["shortest_edge"]; processorJSON["max_pixels"] = size["longest_edge"]
            }
        }
        let args = try JSONDecoder().decode(MLXVLM.Qwen3VLProcessorConfiguration.self, from: JSONSerialization.data(withJSONObject: processorJSON))
        imageBounds = (args.minPixels, args.maxPixels, args.patchSize * args.mergeSize)
        processor = MLXVLM.Qwen3VLProcessor(args, tokenizer: OvisTokenizer(inner: tokenizer))
        model.train(false)
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "embed", let graph, let processor, let videoConfiguration else { throw failure("Ovis is not loaded or operation is unsupported.") }
        guard let items = input["input"]?.array, !items.isEmpty, items.count <= 2048 else { throw failure("Provide nonempty embedding inputs.") }
        guard input["dimensions"] == nil || input["dimensions"]?.number == Double(graph.config.textConfiguration.hiddenSize) else {
            throw failure("Ovis has no Matryoshka projection; use its native dimension.")
        }
        var vectors: [ArsenalWire.JSONValue] = []; var total = 0
        for item in items {
            try cancellation.check()
            var content: [[String: String]] = [], images: [CIImage] = [], videos: [[CIImage]] = [], videoTimes: [[Double]] = []
            // Visual tokens counted as each part is read, so an oversized input stops before more pages render.
            var visual = 0
            func count(_ tokens: Int) throws {
                visual += tokens
                guard visual <= maxInputTokens else {
                    throw failure("This input needs more than \(maxInputTokens) tokens (\(visual) so far); embed fewer pages or a shorter video.")
                }
            }
            let parts = item["type"]?.string == "content" ? item["content"]?.array ?? [] : [item]
            for part in parts {
                if let text = part.string ?? (part["type"]?.string == "text" ? part["text"]?.string : nil) { content.append(["type": "text", "text": text]); continue }
                let type = part["type"]?.string ?? ""
                guard let path = part[type]?["path"]?.string else { throw failure("Input needs text or a local image/video/document path.") }
                let url = URL(fileURLWithPath: path)
                switch type {
                case "image":
                    let image = CIImage(cgImage: try NativeMedia.image(url))
                    try count(Self.imageTokens(width: Int(image.extent.width), height: Int(image.extent.height), bounds: imageBounds))
                    images.append(image); content.append(["type": "image"])
                case "document":
                    guard let document = CGPDFDocument(url as CFURL), document.numberOfPages > 0, document.numberOfPages <= 128 else { throw failure("PDF requires 1–128 pages.") }
                    for index in 1...document.numberOfPages {
                        try cancellation.check()
                        guard let page = document.page(at: index) else { throw failure("Unreadable PDF page.") }
                        let rect = page.getBoxRect(.mediaBox), scale = min(2.0, 2048 / max(rect.width, rect.height))
                        try count(Self.imageTokens(width: max(1, Int(rect.width * scale)), height: max(1, Int(rect.height * scale)), bounds: imageBounds))
                        guard
                            let context = CGContext(
                                data: nil, width: max(1, Int(rect.width * scale)), height: max(1, Int(rect.height * scale)), bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                        else { throw failure("PDF page allocation failed.") }
                        context.setFillColor(CGColor(gray: 1, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: context.width, height: context.height))
                        context.scaleBy(x: scale, y: scale); context.translateBy(x: -rect.minX, y: -rect.minY); context.drawPDFPage(page)
                        guard let rendered = context.makeImage() else { throw failure("PDF render failed.") }
                        images.append(CIImage(cgImage: rendered)); content.append(["type": "image"])
                    }
                case "video":
                    // Count compressed samples without retaining decoded frames so uniform
                    // sampling spans the whole video, including clips above max_frames.
                    let metadata = try ovisAwait { try await OvisVideoMetadata.read(url: url, cancellation: cancellation) }
                    let selected = videoConfiguration.sampleIndices(total: metadata.0, sourceFPS: metadata.1)
                    let reader = try ovisAwait { try await NativeVideoReader(url: url) }; defer { reader.cancel() }
                    var frames: [CIImage] = [], times: [Double] = [], index = 0, selectedIndex = 0
                    while selectedIndex < selected.count, let sample = try reader.next() {
                        try cancellation.check()
                        if index == selected[selectedIndex], let pixels = CMSampleBufferGetImageBuffer(sample) {
                            frames.append(CIImage(cvPixelBuffer: pixels).transformed(by: reader.transform))
                            times.append(Double(index) / metadata.1); selectedIndex += 1
                        }
                        index += 1
                    }
                    guard !frames.isEmpty, selectedIndex == selected.count else { throw failure("Video contains no readable sampled frames.") }
                    videos.append(frames); videoTimes.append(times); content.append(["type": "video"])
                default: throw failure("Ovis accepts text, images, PDFs and videos; audio is unsupported.")
                }
            }
            guard !content.isEmpty else { throw failure("Embedding content is empty.") }
            // The maker trained with ms-swift's `qwen3_5_emb` template (args.json `"template": "qwen3_5_emb"`): default
            // system "Represent the user's input.", an empty assistant turn with the non-thinking prefix, then the
            // `<|endoftext|>` suffix, whose hidden state is pooled (`extract_embedding`, no sparse_info.json: data[-1]).
            let system = input["instruction"]?.string ?? "Represent the user's input."
            var ids: [Int]
            if let tokenizer {
                let messages: [[String: any Sendable]] = [["role": "system", "content": system], ["role": "user", "content": content]]
                ids = try tokenizer.applyChatTemplate(
                    messages: messages, chatTemplate: template.map { .literal($0) }, addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil,
                    additionalContext: ["enable_thinking": false])
                guard let end = tokenizer.convertTokenToId("<|endoftext|>") else { throw failure("Ovis tokenizer lacks the embedding suffix token.") }
                ids.append(end)
            } else {
                ids =
                    [1]
                    + content.flatMap {
                        $0["type"] == "text"
                            ? [2, 3]
                            : [graph.config.visionStartTokenId, $0["type"] == "video" ? graph.config.videoTokenId : graph.config.imageTokenId, graph.config.visionEndTokenId]
                    } + [4]
            }
            var prompt = OvisPromptExpansion(tokenizer?.decode(tokens: ids, skipSpecialTokens: false) ?? "")
            var pixels: [MLXArray] = [], frames: [THW] = [], imageFrames: [THW] = [], videoFrames: [THW] = []
            var imageIndex = 0, videoIndex = 0, markerCursor = 0
            for part in content where part["type"] != "text" {
                try cancellation.check()
                let isVideo = part["type"] == "video"
                let result: (MLXArray, THW)
                var times: [Double] = []
                if isVideo {
                    let sampled = videos[videoIndex]; times = videoTimes[videoIndex]
                    let (height, width) = try videoConfiguration.targetSize(frames: sampled.count, height: Int(sampled[0].extent.height), width: Int(sampled[0].extent.width))
                    let resized = sampled.map { MediaProcessing.resampleBicubic($0, to: CGSize(width: width, height: height)) }
                    var settings = videoConfiguration.json
                    // Resizing has already applied the total video budget. Keep patchify's
                    // image helper at exactly this size instead of applying image limits.
                    settings["min_pixels"] = height * width; settings["max_pixels"] = height * width
                    settings["image_processor_type"] = "Qwen2VLImageProcessorFast"
                    let config = try JSONDecoder().decode(MLXVLM.Qwen3VLProcessorConfiguration.self, from: JSONSerialization.data(withJSONObject: settings))
                    result = try MLXVLM.Qwen3VLProcessor(config, tokenizer: OvisTokenizer(inner: tokenizer)).preprocess(images: resized, processing: nil)
                    videoIndex += 1; videoFrames.append(result.1)
                } else {
                    result = try processor.preprocess(images: [images[imageIndex]], processing: nil)
                    imageIndex += 1; imageFrames.append(result.1)
                }
                pixels.append(result.0); frames.append(result.1)
                let marker = isVideo ? graph.config.videoTokenId : graph.config.imageTokenId
                let grid = result.1, merge = graph.config.visionConfiguration.spatialMergeSize
                if tokenizer != nil {
                    let original = isVideo ? "<|vision_start|><|video_pad|><|vision_end|>" : "<|image_pad|>"
                    let replacement =
                        isVideo
                        ? OvisVideoFraming.placeholder(times: times, grid: grid, temporalSize: videoConfiguration.temporalSize, mergeSize: merge)
                        : String(repeating: "<|image_pad|>", count: grid.t * grid.h * grid.w / (merge * merge))
                    try prompt.replace(original, with: replacement)
                    // Tokenize the full expanded prompt once below, preserving BPE
                    // boundaries around timestamps and the surrounding text.
                } else {
                    guard let position = ids[markerCursor...].firstIndex(of: marker) else { throw failure("Native template omitted a visual placeholder.") }
                    let count = grid.h * grid.w / (merge * merge)
                    if isVideo {
                        let replacement = (0..<grid.t).flatMap { _ in
                            [2, graph.config.visionStartTokenId] + Array(repeating: marker, count: count) + [graph.config.visionEndTokenId]
                        }
                        ids.replaceSubrange((position - 1)...(position + 1), with: replacement)
                        markerCursor = position - 1 + replacement.count
                    } else {
                        ids.replaceSubrange(position...position, with: Array(repeating: marker, count: count)); markerCursor = position + count
                    }
                }
            }
            if let tokenizer { ids = tokenizer.encode(text: prompt.text, addSpecialTokens: false) }
            guard ids.count <= min(maxInputTokens, graph.config.textConfiguration.maxPositionEmbeddings) else {
                throw failure("This input needs \(ids.count) tokens; Ovis takes at most \(maxInputTokens). Embed fewer pages or a shorter video.")
            }
            if random { ids = ids.map { $0 < vocabulary ? $0 : 1 } }
            var vector = try graph.embed(
                ids: ids, pixels: pixels.isEmpty ? nil : concatenated(pixels), frames: frames, imageFrames: imageFrames, videoFrames: videoFrames, cancellation: cancellation
            ).asType(.float32)
            vector = vector / maximum(sqrt(sum(vector * vector)), MLXArray(1e-12))
            let row = vector.asArray(Float.self)
            guard row.allSatisfy(\.isFinite) else { throw failure("Non-finite Ovis embedding.") }
            vectors.append(.array(row.map { .number(Double($0)) })); total += ids.count
        }
        return ["embeddings": .array(vectors), "input_tokens": .number(Double(total))]
    }
    /// Image tokens after the processor's smart_resize (image_processing_qwen2_vl.py): the pixel count rounded to the
    /// patch·merge grid and held within the processor's bounds, one token per merged patch.
    static func imageTokens(width: Int, height: Int, bounds: (minimum: Int, maximum: Int, factor: Int)) -> Int {
        let f = Double(bounds.factor), h = Double(height), w = Double(width)
        var rows = max(1, (h / f).rounded(.toNearestOrAwayFromZero)), columns = max(1, (w / f).rounded(.toNearestOrAwayFromZero))
        if rows * columns * f * f > Double(bounds.maximum) {
            let beta = (h * w / Double(bounds.maximum)).squareRoot()
            rows = max(1, (h / beta / f).rounded(.down)); columns = max(1, (w / beta / f).rounded(.down))
        } else if rows * columns * f * f < Double(bounds.minimum) {
            let beta = (Double(bounds.minimum) / (h * w)).squareRoot()
            rows = (h * beta / f).rounded(.up); columns = (w * beta / f).rounded(.up)
        }
        return Int(rows * columns)
    }
    static func nativeNorm(_ value: MLXArray) -> MLXArray {
        value.asType(.float16) + MLXArray(1, dtype: .float16)
    }

    private func failure(_ message: String) -> WorkerFailure { .init(code: "invalid_request", message: message) }
    func unload() { graph = nil; tokenizer = nil; processor = nil; videoConfiguration = nil; template = nil }
}

private struct OvisTokenizer: MLXLMCommon.Tokenizer {
    let inner: (any Tokenizers.Tokenizer)?
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { inner?.encode(text: text, addSpecialTokens: addSpecialTokens) ?? [1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { inner?.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) ?? "" }
    func convertTokenToId(_ token: String) -> Int? { inner?.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner?.convertIdToToken(id) }
    var bosToken: String? { inner?.bosToken }; var eosToken: String? { inner?.eosToken }; var unknownToken: String? { inner?.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        guard let inner else { return [1] }
        return try inner.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
    }
}
private func ovisAwait<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<T, Error>?
    Task {
        do { result = .success(try await operation()) } catch { result = .failure(error) }; semaphore.signal()
    }
    semaphore.wait()
    return try result!.get()
}

// swiftlint:enable force_cast shorthand_operator
