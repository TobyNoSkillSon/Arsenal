// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ImageIO
import AVFoundation
import MLX
import MLXNN
import IntegrationKit
import PerceptionSupport
import ArsenalWire

public final class NativeSAM31: NativeModel {
    public let architecture = "sam3.1_video"
    public let operations: Set<String> = ["segment", "segment_session", "video_session", "track_video"]
    private var model: SAM31Model?
    private var sessions: [String: FrameFeatures] = [:]
    private struct VideoState { let url: URL; let state = SessionState(); var ids: [Int: Int] = [:] }
    private var videos: [String: VideoState] = [:]
    private var side: Float = 1008
    /// Converted tensors the graph never reads. CLIP's pooled-output projection is unused in SAM 3.1's forward (the
    /// detector projects the CLIP hidden states with its own `detector_model.text_projection`). In the FP32 weights it
    /// holds two values of -9.58e18, beyond FP16's range, so casting it failed Get.
    public static let unusedTensors: Set<String> = ["detector_model.text_encoder.text_projection.weight"]
    /// v3: Meta's own checkpoint (facebook/sam3.1) mapped to the port's layout at Get (Loading/OfficialCheckpoint.swift).
    public static let recipe = "cast:sam31-fp16-v3"
    public static var conversion: NativeConversion {
        NativeConversion(recipe: recipe) { context, cancellation in
            try OfficialCheckpoint.convert(context, cancellation: cancellation, dropping: unusedTensors)
        }
    }
    public init() {}
    public var parameters: [String: MLXArray] { Dictionary(uniqueKeysWithValues: model?.root.parameters().flattened() ?? []) }
    public var peakBytes: Int { PerceptionIO.bytes(model?.root) }
    public func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let data = context.randomWeights ? try JSONEncoder().encode(context.config) : try Data(contentsOf: context.weights.appendingPathComponent("config.json"))
        var config = try JSONDecoder().decode(ModelConfig.self, from: data)
        if context.randomWeights && context.config["full_inventory"]?.bool != true { config = Self.smallConfig() }
        let root = SAM31Root(config)
        try PerceptionIO.prepare(root, context: context, map: Self.map, cancellation: cancellation)
        side = Float(config.trackerConfig.imageSize)
        let loaded = SAM31Model(nativeConfig: config, root: root)
        loaded.textTokenizer = try CLIPTokenizer.load(directory: PerceptionIO.asset(context, "clip-bpe"))
        model = loaded
    }
    public static func map(_ key: String, _ value: MLXArray) throws -> (String, MLXArray)? {
        guard !unusedTensors.contains(key) else { return nil }
        // The converted snapshot is already OHWI and uses the port's names; the remaining rewrites are idempotent.
        let sentinel = "__native.patch_embeddings.projection.weight"
        return sanitize([key: value, sentinel: MLXArray.zeros([1, 2, 2, 3])]).first { $0.key != sentinel }.map { ($0.key, $0.value) }
    }
    static func smallConfig() -> ModelConfig {
        var config = ModelConfig()
        var vision = VisionEncoderConfig()
        vision.backboneConfig.hiddenSize = 64; vision.backboneConfig.numHiddenLayers = 1
        vision.backboneConfig.numAttentionHeads = 4; vision.backboneConfig.intermediateSize = 256
        vision.backboneConfig.imageSize = 112; vision.backboneConfig.pretrainImageSize = 112
        vision.backboneConfig.windowSize = 8; vision.backboneConfig.globalAttnIndexes = [0]
        vision.backboneFeatureSizes = [[32, 32], [16, 16], [8, 8]]
        config.detectorConfig.visionConfig = vision
        config.detectorConfig.textConfig.hiddenSize = 256; config.detectorConfig.textConfig.numHiddenLayers = 1
        config.detectorConfig.textConfig.numAttentionHeads = 8; config.detectorConfig.textConfig.intermediateSize = 512
        config.detectorConfig.detrEncoderConfig.numLayers = 1
        config.detectorConfig.detrDecoderConfig.numLayers = 1
        config.detectorConfig.detrDecoderConfig.numQueries = 4
        config.trackerConfig.visionConfig = vision; config.trackerConfig.imageSize = 112
        config.trackerConfig.promptEncoderConfig.imageSize = 112
        config.trackerConfig.memoryAttentionNumLayers = 1
        config.trackerConfig.memoryAttentionRopeFeatSizes = [8, 8]
        config.trackerConfig.maskDownsamplerInputSize = 128
        config.trackerConfig.memoryFuserNumLayers = 1
        config.lowResMaskSize = 32
        return config
    }
    private func prompt(_ input: JSONObject) throws -> SegmentPrompt {
        let points = try (input["points"]?.array ?? []).map { item -> PromptPoint in
            guard let p = item.object, let x = p["x"]?.number, let y = p["y"]?.number, (0...1).contains(x), (0...1).contains(y) else {
                throw PerceptionIO.error("Points must use normalized coordinates.")
            }
            return PromptPoint(x: Float(x) * side, y: Float(y) * side, label: p["label"]?.string == "background" ? .negative : .positive)
        }
        let boxes = input["boxes"]?.array ?? []
        guard boxes.count <= 1 else { throw PerceptionIO.error("Interactive segmentation accepts one box per object.") }
        if let b = boxes.first?.object, let x0 = b["xmin"]?.number, let x1 = b["xmax"]?.number, let y0 = b["ymin"]?.number, let y1 = b["ymax"]?.number {
            guard x0 < x1, y0 < y1, x0 >= 0, y0 >= 0, x1 <= 1, y1 <= 1 else { throw PerceptionIO.error("Box corners must be ordered and normalized.") }
            let box = CGRect(x: Double(side) * x0, y: Double(side) * y0, width: Double(side) * (x1 - x0), height: Double(side) * (y1 - y0))
            return points.isEmpty ? .box(box) : .boxAndPoints(box, points)
        }
        guard !points.isEmpty else { throw PerceptionIO.error("Provide text, a box, or foreground/background points.") }
        return .points(points)
    }
    private func masks(_ features: FrameFeatures, input: JSONObject) throws -> [(Mask, Float)] {
        guard let model else { throw PerceptionIO.error("Model is unloaded.") }
        if let text = input["text"]?.string, !text.isEmpty {
            guard input["points"] == nil, input["boxes"] == nil else { throw PerceptionIO.error("Use text or interactive prompts in one request.") }
            return try model.detect(features, text: text, scoreThreshold: Float(input["threshold"]?.number ?? 0.5)).map { ($0.mask, $0.score) }
        }
        let result = try model.segment(features, prompt: prompt(input))
        return [(result.mask, result.score)]
    }
    private func write(_ mask: Mask, size: CGSize, root: URL, name: String, cancellation: NativeCancellation) throws -> JSONValue {
        try cancellation.check()
        let binary = mask.upsampled(to: size)
        guard let image = binary.cgImage() else { throw PerceptionIO.error("Mask dimensions are invalid.") }
        let file = root.appendingPathComponent(name + ".png")
        guard let output = CGImageDestinationCreateWithURL(file as CFURL, "public.png" as CFString, 1, nil) else { throw PerceptionIO.error("Cannot encode mask PNG.") }
        CGImageDestinationAddImage(output, image, nil)
        guard CGImageDestinationFinalize(output) else { throw PerceptionIO.error("Cannot finish mask PNG.") }
        return .object(try NativeMedia.file(file, root: root, mime: "image/png", id: name))
    }
    public func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard let model else { throw PerceptionIO.error("Model is unloaded.") }
        try cancellation.check()
        if operation == "segment_session" || operation == "video_session" {
            let action = input["action"]?.string ?? "create"
            if action == "close" {
                guard let id = input["session_id"]?.string else { throw PerceptionIO.error("Closing requires a session ID.") }
                let existed = operation == "segment_session" ? sessions.removeValue(forKey: id) != nil : videos.removeValue(forKey: id) != nil
                guard existed else { throw PerceptionIO.error("Session is stale or belongs to another loaded model.") }
                return ["session_id": .string(id), "closed": true]
            }
            guard action == "create", sessions.count + videos.count < 4 else { throw PerceptionIO.error("At most four image/video sessions may be open.") }
            let id = UUID().uuidString
            if operation == "segment_session" {
                let image = try NativeMedia.image(PerceptionIO.path(input, "image"))
                sessions[id] = try model.encode(image)
            } else {
                let url = try PerceptionIO.path(input, "video")
                let reader = try PerceptionIO.awaitResult { try await NativeVideoReader(url: url) }; reader.cancel()
                videos[id] = VideoState(url: url)
            }
            return ["session_id": .string(id)]
        }
        guard let out = input["out_dir"]?.string else { throw PerceptionIO.error("Missing output directory.") }
        let root = URL(fileURLWithPath: out); try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if operation == "segment" {
            let features: FrameFeatures
            if let id = input["session_id"]?.string {
                guard let existing = sessions[id] else { throw PerceptionIO.error("Image session is stale.") }; features = existing
            } else {
                features = try model.encode(NativeMedia.image(PerceptionIO.path(input, "image")))
            }
            let found = try masks(features, input: input)
            var files: [JSONValue] = []
            for (index, item) in found.prefix(256).enumerated() {
                files.append(try write(item.0, size: features.sourceSize, root: root, name: "mask-\(index)", cancellation: cancellation))
            }
            if files.isEmpty {
                let empty = try Mask(logits: [-1], width: 1, height: 1)
                files = [try write(empty, size: features.sourceSize, root: root, name: "no-matches", cancellation: cancellation)]
            }
            return ["files": .array(files), "scores": .array(found.prefix(256).map { .number(Double(max(0, min(1, $0.1)))) })]
        }
        guard operation == "track_video" else { throw PerceptionIO.error("Unsupported operation.") }
        let sessionID = input["session_id"]?.string
        var video: VideoState
        if let sessionID {
            guard let value = videos[sessionID] else { throw PerceptionIO.error("Video session is stale.") }; video = value
        } else {
            video = VideoState(url: try PerceptionIO.path(input, "video"))
        }
        let reader = try PerceptionIO.awaitResult { try await NativeVideoReader(url: video.url) }
        defer { reader.cancel(); if let sessionID { videos[sessionID] = video } }
        guard reader.transform == .identity else { throw PerceptionIO.error("Rotate the video into upright pixels before tracking.") }
        let updates = input["prompt_updates"]?.array ?? []
        let maximum = Int(input["max_frames"]?.number ?? 128)
        guard (1...128).contains(maximum) else { throw PerceptionIO.error("Tracking is bounded to 1...128 frames per call.") }
        let start = Int(input["start_frame"]?.number ?? 0)
        let end = Int(input["end_frame"]?.number ?? Double(start + maximum - 1))
        guard start >= 0, end >= start, end - start < maximum else { throw PerceptionIO.error("Frame range must fit the requested frame bound.") }
        var files: [JSONValue] = []; var index = 0
        while let sample = try reader.next() {
            try cancellation.check()
            if index < start { index += 1; continue }
            if index > end { break }
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { throw PerceptionIO.error("Video frame has no pixels.") }
            let features = try model.encode(buffer)
            var prompts = updates.compactMap(\.object).filter { $0["frame_index"]?.number == Double(index) }
            if index == start, prompts.isEmpty, input["text"] != nil {
                guard video.ids.isEmpty else { throw PerceptionIO.error("Use prompt updates to change text prompts in an existing tracking session.") }
                let found = try masks(features, input: input).sorted { $0.1 > $1.1 }.prefix(16)
                for (id, item) in found.enumerated() {
                    try cancellation.check()
                    try model.sessionAddObject(video.state, id: id, frameIndex: index, features: features, prompt: .mask(item.0))
                    video.ids[id] = id
                }
            } else if index == start, prompts.isEmpty, input["points"] != nil || input["boxes"] != nil {
                var first = input; first["object_id"] = 0; prompts = [first]
            }
            for update in prompts {
                let objectID = Int(update["object_id"]?.number ?? 0)
                let action = update["action"]?.string ?? "add"
                if action == "remove" {
                    guard let internalID = video.ids.removeValue(forKey: objectID) else { throw PerceptionIO.error("Cannot remove an unknown tracked object.") }
                    video.state.requestRemoval(internalID); continue
                }
                if action == "update" {
                    guard let internalID = video.ids[objectID] else { throw PerceptionIO.error("Cannot refine an unknown tracked object.") }
                    _ = try model.sessionFrame(video.state, features: features, frameIndex: index)
                    if update["text"] != nil {
                        guard let best = try masks(features, input: update).max(by: { $0.1 < $1.1 }) else { throw PerceptionIO.error("Text update did not locate an object.") }
                        try model.sessionRefineMask(video.state, id: internalID, frameIndex: index, features: features, mask: best.0)
                    } else {
                        _ = try model.sessionRefine(video.state, id: internalID, frameIndex: index, features: features, prompt: prompt(update))
                    }
                } else {
                    guard video.ids[objectID] == nil, video.ids.count < 16 else { throw PerceptionIO.error("Object IDs must be unique; tracking holds at most 16 objects.") }
                    let trackPrompt: TrackPrompt
                    if update["text"] != nil {
                        guard let best = try masks(features, input: update).max(by: { $0.1 < $1.1 }) else { throw PerceptionIO.error("Text prompt did not locate an object.") }
                        trackPrompt = .mask(best.0)
                    } else {
                        trackPrompt = .segment(try prompt(update))
                    }
                    video.ids[objectID] = objectID
                    try model.sessionAddObject(video.state, id: objectID, frameIndex: index, features: features, prompt: trackPrompt)
                }
            }
            if !video.ids.isEmpty {
                let tracked = try model.sessionFrame(video.state, features: features, frameIndex: index)
                for (id, object) in tracked.objects.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
                    guard files.count < 256 else { throw PerceptionIO.error("Tracking exceeds the 256-mask output bound.") }
                    files.append(try write(object.mask, size: features.sourceSize, root: root, name: "frame-\(index)-object-\(id.rawValue)", cancellation: cancellation))
                }
            }
            if video.ids.isEmpty, input["text"] != nil {
                let empty = try Mask(logits: [-1], width: 1, height: 1)
                files.append(try write(empty, size: features.sourceSize, root: root, name: "frame-\(index)-no-matches", cancellation: cancellation))
            }
            index += 1
        }
        guard !files.isEmpty else { throw PerceptionIO.error("Tracking needs at least one object prompt on a decoded frame.") }
        return ["files": .array(files)]
    }
    public func unload() { sessions.removeAll(); videos.removeAll(); model = nil }
}
