// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import IntegrationKit
import PerceptionSupport
import ArsenalWire

public final class NativeRFDETR: NativeModel {
    public let architecture = "rf_detr"
    public let operations: Set<String> = ["detect"]
    private(set) var model: RFDETRModel?
    private var resolution = 384
    private var labels: [String: JSONValue] = [:]
    public init() {}
    public var parameters: [String: MLXArray] { Dictionary(uniqueKeysWithValues: model?.parameters().flattened() ?? []) }
    public var peakBytes: Int { PerceptionIO.bytes(model) }
    public func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let config = context.randomWeights ? context.config : try PerceptionIO.json(context.weights.appendingPathComponent("config.json"))
        let shrunk = context.randomWeights && context.config["full_inventory"]?.bool != true
        let shape = Self.architectureShape(config)
        resolution = shrunk ? 64 : shape.resolution
        labels = config["id2label"]?.object ?? Self.cocoLabels
        let layout = Self.backboneLayout(config)
        let backbone = DINOv2Backbone(
            imgSize: resolution, patchSize: shape.patchSize, embedDim: shrunk ? 64 : shape.embedDim, depth: shrunk ? 4 : shape.depth,
            numHeads: shrunk ? 4 : shape.heads, numWindows: shape.windows,
            featureIndices: shrunk ? [0, 1, 2, 3] : layout.features,
            windowBlockIndices: shrunk ? [0, 1, 2] : layout.windows, numRegisterTokens: 0)
        let projector = MultiScaleProjector(scaleFactors: [1], inChannelsList: Array(repeating: shrunk ? 64 : shape.embedDim, count: 4), hiddenDim: 256)
        let configRF = RFDETRConfig(
            decLayers: shrunk ? 1 : shape.decoderLayers, dimFeedforward: shrunk ? 256 : 2048, numQueries: shrunk ? 4 : shape.queries,
            groupDetr: shrunk ? 1 : 13, numClasses: 90)
        let loaded = RFDETRModel(config: configRF, backbone: backbone, projector: projector)
        try PerceptionIO.prepare(loaded, context: context, map: Self.map, cancellation: cancellation)
        model = loaded
    }
    /// Sizes that differ between the Hugging Face RF-DETR checkpoints this runner serves (Nano: 384 px, 2 decoder
    /// layers; Large 2026: 704 px, 4 decoder layers). Defaults are Nano's, so a config without them loads as before.
    struct Shape: Equatable { var resolution, patchSize, embedDim, depth, heads, windows, decoderLayers, queries: Int }
    static func architectureShape(_ config: JSONObject) -> Shape {
        let backbone = config["backbone_config"]?.object ?? [:]
        func int(_ object: JSONObject, _ key: String, _ fallback: Int) -> Int { object[key]?.number.map { Int($0) } ?? fallback }
        return Shape(
            resolution: int(backbone, "image_size", 384), patchSize: int(backbone, "patch_size", 16),
            embedDim: int(backbone, "hidden_size", 384), depth: int(backbone, "num_hidden_layers", 12),
            heads: int(backbone, "num_attention_heads", 6), windows: int(backbone, "num_windows", 2),
            decoderLayers: int(config, "decoder_layers", 2), queries: int(config, "num_queries", 300))
    }
    static func backboneLayout(_ config: JSONObject) -> (features: [Int], windows: [Int]) {
        let backbone = config["backbone_config"]?.object ?? [:]
        let features = backbone["out_indices"]?.array?.compactMap { $0.number.map { Int($0) - 1 } } ?? [2, 5, 8, 11]
        let windows = backbone["window_block_indexes"]?.array?.compactMap { $0.number.map(Int.init) } ?? [0, 1, 2, 4, 5, 7, 8, 10, 11]
        return (features, windows)
    }
    public static func map(_ key: String, _ value: MLXArray) throws -> (String, MLXArray)? { sanitized(key: key, value: value).first }
    public func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "detect", let model else { throw PerceptionIO.error("Unsupported operation or unloaded model.") }
        let threshold = input["threshold"]?.number ?? 0.5
        guard threshold >= 0, threshold <= 1 else { throw PerceptionIO.error("Threshold must lie between zero and one.") }
        let image = try NativeMedia.image(PerceptionIO.path(input, "image"))
        try cancellation.check()
        let rgb = try RFDETRPreprocess.rgb(image)
        let pixels = RFDETRPreprocess.pixels(rgb.bytes, width: rgb.width, height: rgb.height, size: resolution)
        let raw = model(MLXArray(pixels, [1, resolution, resolution, 3]).asType(.float16))
        guard let logits = raw["pred_logits"], let boxes = raw["pred_boxes"] else { throw PerceptionIO.error("Detector omitted boxes or logits.") }
        eval(logits, boxes); try cancellation.check()
        let scores = sigmoid(logits).asType(.float32).asArray(Float.self)
        let coordinates = boxes.asType(.float32).asArray(Float.self)
        guard scores.allSatisfy(\.isFinite), coordinates.allSatisfy(\.isFinite) else { throw PerceptionIO.error("Detector produced non-finite output.") }
        let selectedLabels = Set(input["labels"]?.array?.compactMap(\.string) ?? [])
        let classCount = logits.dim(-1)
        let indices = scores.indices.filter { i in
            let label = labels[String(i % classCount)]?.string ?? "N/A"
            return label != "N/A" && Double(scores[i]) >= threshold && (selectedLabels.isEmpty || selectedLabels.contains(label))
        }.sorted { scores[$0] > scores[$1] }.prefix(300)
        let detections: [JSONValue] = indices.map { index in
            let query = index / classCount
            let cx = coordinates[query * 4], cy = coordinates[query * 4 + 1], w = coordinates[query * 4 + 2], h = coordinates[query * 4 + 3]
            func n(_ value: Float) -> JSONValue { .number(Double(max(0, min(1, value)))) }
            return .object([
                "label": labels[String(index % classCount)]!, "score": .number(Double(scores[index])),
                "box": ["xmin": n(cx - w / 2), "ymin": n(cy - h / 2), "xmax": n(cx + w / 2), "ymax": n(cy + h / 2)]
            ])
        }
        return ["detections": .array(detections), "coordinate_system": "normalized"]
    }
    public func unload() { model = nil; labels = [:] }
    public static let cocoLabels: JSONObject = COCOLabels.names
}
