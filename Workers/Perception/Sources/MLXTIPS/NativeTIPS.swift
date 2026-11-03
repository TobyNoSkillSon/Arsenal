// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ImageIO
import MLX
import MLXNN
import IntegrationKit
import PerceptionSupport
import ArsenalWire

public final class NativeTIPS: NativeModel {
    public let architecture = "tipsv2"
    public let operations: Set<String> = ["embed", "detect", "segment"]
    private var model: TIPSModel?
    private var size = 448
    public init() {}
    public var parameters: [String: MLXArray] { Dictionary(uniqueKeysWithValues: model?.parameters().flattened() ?? []) }
    public var peakBytes: Int { PerceptionIO.bytes(model) }
    public func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let config = context.randomWeights ? context.config : try PerceptionIO.json(context.weights.appendingPathComponent("config.json"))
        let large = config["variant"]?.string == "l14" || config["vision_config"]?.object?["hidden_size"]?.number == 1024
        let shrunk = context.randomWeights && context.config["full_inventory"]?.bool != true
        size = shrunk ? 28 : 448
        let loaded = Self.makeModel(large: large, shrunk: shrunk)
        // Real loads read the tokenizer the checkpoint pins in its own folder; random-weight lab loads name a fixture folder.
        let tokenizerFolder = context.randomWeights ? try PerceptionIO.asset(context, "tokenizer") : context.weights
        let tokenizerURL = tokenizerFolder.appendingPathComponent("tokenizer.model")
        loaded.tokenizer = try TIPSTokenizer(modelPath: tokenizerURL.path)
        try PerceptionIO.prepare(loaded, context: context, map: Self.map, cancellation: cancellation)
        model = loaded
    }
    static func makeModel(large: Bool, shrunk: Bool) -> TIPSModel {
        let vision = shrunk ? VisionTransformer(imgSize: 28, embedDim: 64, depth: 1, numHeads: 4) : VisionTransformer(variant: large ? .L : .B)
        let text = shrunk ? TextEncoder(dim: 64, numHeads: 4, numLayers: 1, mlpDim: 256) : TextEncoder(variant: large ? .L : .B)
        return TIPSModel(vision: vision, text: text)
    }
    public static func map(_ key: String, _ value: MLXArray) throws -> (String, MLXArray)? {
        if key == "temperature" || key == "temperature_contrastive" { return nil }
        var key = key
        key = key.replacingOccurrences(of: "transformer.resblocks.", with: "blocks.")
            .replacingOccurrences(of: "attn.in_proj_weight", with: "attn.in_proj.weight")
            .replacingOccurrences(of: "attn.in_proj_bias", with: "attn.in_proj.bias")
            .replacingOccurrences(of: "mlp.c_fc", with: "mlp_c_fc")
            .replacingOccurrences(of: "mlp.c_proj", with: "mlp_c_proj")
        return (key, key == "vision_encoder.patch_embed.proj.weight" ? value.transposed(0, 2, 3, 1) : value)
    }
    private func vector(_ value: JSONValue, cancellation: NativeCancellation) throws -> (MLXArray, Int) {
        guard let model else { throw PerceptionIO.error("Model is not loaded.") }
        try cancellation.check()
        if let text = value.string ?? value.object?["text"]?.string {
            guard !text.isEmpty, text.utf8.count <= 32768 else { throw PerceptionIO.error("Text must contain 1...32768 UTF-8 bytes.") }
            let tokens = try model.tokenizer!.tokenize(text)
            return (model.encodeText(ids: tokens.ids, paddings: tokens.paddings), Int(sum(1 - tokens.paddings).item(Int32.self)))
        }
        guard let item = value.object, item["type"]?.string == "image" else { throw PerceptionIO.error("TIPS accepts text or image inputs.") }
        let image = try NativeMedia.image(PerceptionIO.path(item, "image"))
        let output = model.encodeImage(try PerceptionIO.pixels(image, size: size))
        return (output.clsToken.reshaped(1, -1), 0)
    }
    public func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operations.contains(operation), let model else { throw PerceptionIO.error("Unsupported operation or unloaded model.") }
        if operation == "embed" {
            guard let inputs = input["input"]?.array, !inputs.isEmpty, inputs.count <= 256 else { throw PerceptionIO.error("Input must contain 1...256 text or image items.") }
            var vectors: [JSONValue] = []; var count = 0
            for item in inputs {
                let (raw, tokens) = try vector(item, cancellation: cancellation); count += tokens
                let embedding = input["normalize"]?.bool == false ? raw : l2Normalize(raw)
                eval(embedding)
                let values = embedding.asType(.float32).asArray(Float.self)
                guard values.allSatisfy(\.isFinite) else { throw PerceptionIO.error("Embedding contains non-finite values.") }
                vectors.append(.array(values.map { .number(Double($0)) }))
            }
            return ["embeddings": .array(vectors), "input_tokens": .number(Double(count))]
        }
        if operation == "segment" { return try segment(input, cancellation: cancellation) }
        guard let labels = input["labels"]?.array?.compactMap(\.string), !labels.isEmpty, labels.count <= 256 else {
            throw PerceptionIO.error("Zero-shot classification requires 1...256 labels.")
        }
        let image = try NativeMedia.image(PerceptionIO.path(input, "image"))
        let imageVector = l2Normalize(model.encodeImage(try PerceptionIO.pixels(image, size: size)).clsToken.reshaped(1, -1))
        // The model card's zero-shot classification: the label text itself against the first CLS token.
        var textVectors: [MLXArray] = []
        for label in labels {
            try cancellation.check()
            textVectors.append(l2Normalize(try model.encodeText(label)))
        }
        let logits = matmul(imageVector, concatenated(textVectors, axis: 0).T)
        let scores = clip(logits, min: -1, max: 1).asType(.float32).asArray(Float.self)
        guard scores.allSatisfy(\.isFinite) else { throw PerceptionIO.error("Classification contains non-finite scores.") }
        let threshold = input["threshold"]?.number ?? 0
        let detections: [JSONValue] = zip(labels, scores).filter { Double($0.1) >= threshold }.sorted { $0.1 > $1.1 }.map { label, score in
            .object(["label": .string(label), "score": .number(Double(score))])
        }
        return ["detections": .array(detections), "coordinate_system": "normalized"]
    }
    private func segment(_ input: JSONObject, cancellation: NativeCancellation) throws -> JSONObject {
        guard let model, let labels = input["labels"]?.array?.compactMap(\.string), (1...256).contains(labels.count), let out = input["out_dir"]?.string else {
            throw PerceptionIO.error("Segmentation requires an image, 1...256 labels and output directory.")
        }
        let image = try NativeMedia.image(PerceptionIO.path(input, "image"))
        let classes = try Self.segmentClasses(
            model, pixels: try PerceptionIO.pixels(image, size: size), labels: labels, height: image.height, width: image.width, cancellation: cancellation)
        guard classes.count == image.width * image.height else { throw PerceptionIO.error("Label map does not match the image.") }
        // One counting sort of the label map, so each mask touches only its own pixels.
        var starts = [Int](repeating: 0, count: labels.count + 1)
        for label in classes { starts[Int(label) + 1] += 1 }
        for index in labels.indices { starts[index + 1] += starts[index] }
        var cursor = starts, order = [Int](repeating: 0, count: classes.count)
        for (pixel, label) in classes.enumerated() { order[cursor[Int(label)]] = pixel; cursor[Int(label)] += 1 }
        let root = URL(fileURLWithPath: out); try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var files: [JSONValue] = []
        for index in labels.indices {
            try cancellation.check()
            var pixels = [UInt8](repeating: 0, count: image.width * image.height)
            for pixel in order[starts[index]..<starts[index + 1]] { pixels[pixel] = 255 }
            guard let provider = CGDataProvider(data: Data(pixels) as CFData),
                let mask = CGImage(
                    width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: image.width, space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
            else { throw PerceptionIO.error("Cannot render label mask.") }
            let file = root.appendingPathComponent("label-\(index).png")
            guard let target = CGImageDestinationCreateWithURL(file as CFURL, "public.png" as CFString, 1, nil) else { throw PerceptionIO.error("Cannot encode label mask.") }
            CGImageDestinationAddImage(target, mask, nil)
            guard CGImageDestinationFinalize(target) else { throw PerceptionIO.error("Cannot finish label mask.") }
            files.append(.object(try NativeMedia.file(file, root: root, mime: "image/png", name: labels[index] + ".png", id: "label-\(index)")))
        }
        return ["files": .array(files)]
    }

    /// The prompt templates of the official zero-shot segmentation recipe (TCL's nine, in
    /// `pytorch/TIPS_zeroshot_segmentation.ipynb` and the google/TIPSv2 Space).
    static let segmentTemplates = [
        "itap of a {}.", "a bad photo of a {}.", "a origami {}.", "a photo of the large {}.", "a {} in a video game.", "art of the {}.",
        "a photo of the small {}.", "a photo of many {}.", "a photo of {}s."
    ]
    /// The notebook's class embedding: each template's text vector L2-normalized, averaged, normalized again. `(1, D)` float32.
    static func segmentLabelVector(_ model: TIPSModel, _ label: String) throws -> MLXArray {
        let prompts = segmentTemplates.map { $0.replacingOccurrences(of: "{}", with: label) }
        let each = l2Normalize(try model.encodeText(prompts).asType(.float32))
        return l2Normalize(mean(each, axis: 0, keepDims: true))
    }
    /// The notebook's `predict_whole`: value-attention patch features of the last block (MaskCLIP's values trick),
    /// cosine against the ensembled label vectors, bilinearly upsampled to the image, argmax per pixel. Row-major
    /// label indices, `height * width` of them.
    static func segmentClasses(
        _ model: TIPSModel, pixels: MLXArray, labels: [String], height: Int, width: Int, cancellation: NativeCancellation
    ) throws -> [Int32] {
        let features = model.visionEncoder.encodeValueAttention(pixels)[0] // (h, w, D)
        let rows = features.dim(0), columns = features.dim(1)
        let patches = l2Normalize(features.reshaped(rows * columns, -1).asType(.float32))
        var vectors: [MLXArray] = []
        for label in labels {
            try cancellation.check()
            vectors.append(try segmentLabelVector(model, label))
        }
        let similarity = matmul(patches, concatenated(vectors, axis: 0).T).reshaped(rows, columns, labels.count)
        return try upsampledArgMax(similarity, height: height, width: width, cancellation: cancellation)
    }
    /// `F.interpolate(mode="bilinear", align_corners=False)` interpolation weights, `(output, input)`.
    static func bilinearWeights(output: Int, input: Int) -> MLXArray {
        var weights = [Float](repeating: 0, count: output * input)
        let scale = Float(input) / Float(output)
        for row in 0..<output {
            let source = max(0, (Float(row) + 0.5) * scale - 0.5)
            let low = min(Int(source), input - 1), high = min(low + 1, input - 1), fraction = source - Float(low)
            weights[row * input + low] += 1 - fraction
            weights[row * input + high] += fraction
        }
        return MLXArray(weights, [output, input])
    }
    /// Bilinear upsampling of a `(h, w, labels)` score grid to `height × width`, then argmax over labels, in tiles of at
    /// most `budget` interpolated scores so a 16-megapixel image with 256 labels stays a few tens of megabytes.
    static func upsampledArgMax(_ grid: MLXArray, height: Int, width: Int, budget: Int = 1 << 22, cancellation: NativeCancellation) throws -> [Int32] {
        let rows = grid.dim(0), columns = grid.dim(1), count = grid.dim(2)
        let vertical = bilinearWeights(output: height, input: rows), horizontal = bilinearWeights(output: width, input: columns)
        let flat = grid.asType(.float32).reshaped(rows, columns * count)
        let tileWidth = min(width, 1024), tileHeight = max(1, budget / (tileWidth * count))
        var labels = [Int32](repeating: 0, count: height * width)
        for top in stride(from: 0, to: height, by: tileHeight) {
            try cancellation.check()
            let bottom = min(height, top + tileHeight)
            let band = matmul(vertical[top..<bottom], flat).reshaped(bottom - top, columns, count)
            for left in stride(from: 0, to: width, by: tileWidth) {
                let right = min(width, left + tileWidth)
                let tile = argMax(matmul(horizontal[left..<right], band), axis: -1).asType(.int32).asArray(Int32.self)
                let span = right - left
                for row in 0..<(bottom - top) {
                    labels.replaceSubrange((top + row) * width + left..<(top + row) * width + right, with: tile[row * span..<(row + 1) * span])
                }
            }
        }
        return labels
    }
    public func unload() { model = nil }
}
