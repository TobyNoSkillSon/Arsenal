// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ModelPrecision
import ArsenalWire

/// Get-time conversion of BUT-FIT/diarizen-wavlm-large-s80-md-v2 (an FP32 PyTorch state dict, config.toml and the VBx
/// PLDA npz files) plus the separately pinned pyannote/wespeaker-voxceleb-resnet34-LM asset (a pyannote 3.1 Lightning
/// checkpoint) into one FP16 snapshot. The positional convolution's weight norm is folded (g · v / ‖v‖ over every
/// axis but the kernel's, in Float32, before the cast), convolutions move to MLX's channels-last layout, BatchNorm step
/// counters are dropped. The embedding goes to `embedding/model.safetensors`, the asset's `into` directory. 16-bit only.
enum DiariZenConversion {
    static let recipe = "cast:audio-diarizen-v1"
    static let base = NativeConversion(recipe: recipe, convert: prepare)
    static let marker = "arsenal-diarizen-layout.json"
    static let embeddingAsset = "wespeaker-resnet34-lm"
    static let embeddingDirectory = "embedding"
    static let positionPrefix = "wavlm_model.encoder.transformer.pos_conv_embed.conv."

    /// Segmentation checkpoint name → (snapshot name, axes to transpose), or nil to drop.
    static func segmentationTarget(_ name: String, shape: [Int]) throws -> (String, [Int]?)? {
        if name.hasSuffix(".num_batches_tracked") { return nil }
        guard ["wavlm_model.", "weight_sum.", "proj.", "lnorm.", "conformer.", "classifier."].contains(where: name.hasPrefix) else {
            throw PrecisionError("Unexpected DiariZen tensor: " + name)
        }
        if name.hasSuffix(".weight"), shape.count == 3 { return ("segmentation." + name, [0, 2, 1]) } // Conv1d [O, I, K] → [O, K, I]
        return ("segmentation." + name, nil)
    }

    /// WeSpeaker (pyannote 3.1) name → Community-1's embedding module name, or nil to drop.
    static func embeddingTarget(_ name: String, shape: [Int]) throws -> (String, [Int]?)? {
        if name.hasSuffix(".num_batches_tracked") { return nil }
        guard name.hasPrefix("resnet.") else { throw PrecisionError("Unexpected WeSpeaker tensor: " + name) }
        let key =
            "embedding."
            + name.dropFirst("resnet.".count).replacingOccurrences(of: ".shortcut.0.", with: ".shortcut.")
            .replacingOccurrences(of: ".shortcut.1.", with: ".shortcutBN.")
        if name.hasSuffix(".weight"), shape.count == 4 { return (key, [0, 2, 3, 1]) } // Conv2d [O, I, H, W] → [O, H, W, I]
        return (key, nil)
    }

    /// Dimensions from the checkpoint's shapes. Remaining head indices are the maker's configuration; their counts must
    /// equal the stored query widths, and pruned (absent) attentions must be exactly the empty lists.
    static func dimensions(_ shapes: [String: [Int]]) throws -> DiariZenDimensions {
        var d = DiariZenDimensions.s80
        let prefix = "wavlm_model.encoder.transformer.layers."
        let layers = Set(shapes.keys.compactMap { key -> Int? in key.hasPrefix(prefix) ? Int(key.dropFirst(prefix.count).prefix { $0 != "." }) : nil })
        guard layers == Set(0..<d.remainingHeads.count) else { throw PrecisionError("DiariZen checkpoint has \(layers.count) WavLM layers, expected 24") }
        d.convChannels = try (0..<7).map { i in
            guard let shape = shapes["wavlm_model.feature_extractor.conv_layers.\(i).conv.weight"] else { throw PrecisionError("DiariZen conv layer \(i) is missing") }
            return shape[0]
        }
        d.feedForward = try (0..<d.layers).map { i in
            guard let shape = shapes[prefix + "\(i).feed_forward.intermediate_dense.weight"] else { throw PrecisionError("DiariZen layer \(i) has no feed-forward") }
            return shape[0]
        }
        for i in 0..<d.layers {
            let width = shapes[prefix + "\(i).attention.q_proj.weight"]?[0] ?? 0
            guard width == d.remainingHeads[i].count * d.headDimension else {
                throw PrecisionError("DiariZen layer \(i) stores \(width / d.headDimension) heads, the maker's configuration lists \(d.remainingHeads[i].count)")
            }
        }
        guard let classes = shapes["classifier.weight"]?[0], let width = shapes["proj.weight"]?[0], let states = shapes["weight_sum.weight"]?[1],
            states == d.layers + 1
        else { throw PrecisionError("DiariZen checkpoint lacks its head") }
        d.classes = classes; d.conformerWidth = width
        let conformerPrefix = "conformer.conformer_layer."
        d.conformerLayers =
            Set(
                shapes.keys.compactMap { key -> Int? in
                    key.hasPrefix(conformerPrefix) ? Int(key.dropFirst(conformerPrefix.count).prefix { $0 != "." }) : nil
                }
            ).count
        if let ffn = shapes[conformerPrefix + "0.ffn1.w_1.weight"] { d.conformerFeedForward = ffn[0] }
        if let depthwise = shapes[conformerPrefix + "0.conv.depthwise_conv.weight"] { d.conformerKernel = depthwise[2] }
        return d
    }

    /// weight_norm(dim=2) folded: w[o, i, k] = g[k] · v[o, i, k] / ‖v[:, :, k]‖, in Float32. Returns [O, I, K].
    static func foldWeightNorm(g: MLXArray, v: MLXArray) -> MLXArray {
        let v32 = v.asType(.float32), g32 = g.asType(.float32)
        let norm = sqrt((v32 * v32).sum(axes: [0, 1], keepDims: true))
        return g32 * v32 / norm
    }

    static func prepare(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits == nil else { throw PrecisionError("DiariZen ships at 16-bit only") }
        guard let asset = context.assets[embeddingAsset] else { throw PrecisionError("DiariZen needs the verified WeSpeaker ResNet34 asset") }
        let segmentation = try TorchCheckpoint(context.source.appendingPathComponent("pytorch_model.bin"))
        let embedding = try TorchCheckpoint(asset.appendingPathComponent("pytorch_model.bin"))
        let segmentationTensors = try segmentation.tensors()
        let embeddingTensors = try embedding.tensors(at: ["state_dict"])
        let containers = try Dictionary(
            uniqueKeysWithValues: ["xvec_transform.npz", "plda.npz"].map { ($0, try C1Archive(context.source.appendingPathComponent("plda/" + $0))) })

        enum Source { case segmentation(TorchCheckpoint.Tensor, [Int]?), embedding(TorchCheckpoint.Tensor, [Int]?), positional, plda(String, String) }
        var model: [String: Source] = [:], embedded: [String: Source] = [:]
        var shapes: [String: [Int]] = [:], converted: [String: [Int]] = [:]
        func moved(_ shape: [Int], _ axes: [Int]?) -> [Int] { axes.map { $0.map { shape[$0] } } ?? shape }
        for (name, tensor) in segmentationTensors {
            try cancellation.check()
            shapes[name] = tensor.shape
            if name.hasPrefix(positionPrefix + "parametrizations.weight.") { continue }
            guard let (key, axes) = try segmentationTarget(name, shape: tensor.shape) else { continue }
            model[key] = .segmentation(tensor, axes); converted[key] = moved(tensor.shape, axes)
        }
        guard let g = segmentationTensors[positionPrefix + "parametrizations.weight.original0"],
            let v = segmentationTensors[positionPrefix + "parametrizations.weight.original1"], v.shape.count == 3, g.shape == [1, 1, v.shape[2]]
        else { throw PrecisionError("DiariZen positional convolution lacks its weight-norm parameters") }
        let positionalKey = "segmentation." + positionPrefix + "weight"
        model[positionalKey] = .positional; converted[positionalKey] = [v.shape[0], v.shape[2], v.shape[1]]
        for (name, tensor) in embeddingTensors {
            guard let (key, axes) = try embeddingTarget(name, shape: tensor.shape) else { continue }
            embedded[key] = .embedding(tensor, axes); converted[key] = moved(tensor.shape, axes)
        }
        for (key, container, array) in Community1Conversion.pldaArrays {
            model[key] = .plda(container, array)
            converted[key] = try containers[container]!.numpy(array).shape
        }

        let dims = try dimensions(shapes)
        let expected = Dictionary(uniqueKeysWithValues: DiariZenNetwork(dims).parameters().flattened().map { ($0.0, $0.1.shape) })
        guard Set(expected.keys) == Set(converted.keys), converted.allSatisfy({ expected[$0.key] == $0.value }) else {
            let missing = Set(expected.keys).subtracting(converted.keys).sorted().prefix(4), extra = Set(converted.keys).subtracting(expected.keys).sorted().prefix(4)
            let wrong = converted.filter { expected[$0.key] != nil && expected[$0.key] != $0.value }.keys.sorted().prefix(4)
            throw PrecisionError("DiariZen checkpoints do not match the network (missing \(Array(missing)), unexpected \(Array(extra)), shape \(Array(wrong)))")
        }
        func read(_ source: Source) throws -> MLXArray {
            try cancellation.check()
            switch source {
            case .segmentation(let tensor, let axes):
                let array = try segmentation.array(tensor)
                return axes.map { array.transposed(axes: $0) } ?? array
            case .embedding(let tensor, let axes):
                let array = try embedding.array(tensor)
                return axes.map { array.transposed(axes: $0) } ?? array
            case .positional:
                return foldWeightNorm(g: try segmentation.array(g), v: try segmentation.array(v)).transposed(0, 2, 1)
            case .plda(let container, let array):
                return try containers[container]!.numpy(array)
            }
        }
        _ = try StreamedPrecision.convert(
            keys: model.keys.sorted(), read: { try read(model[$0]!) }, to: context.destination.appendingPathComponent("model.safetensors"),
            half: context.half, eligible: { _, _ in false })
        let embeddingFolder = context.destination.appendingPathComponent(embeddingDirectory)
        try FileManager.default.createDirectory(at: embeddingFolder, withIntermediateDirectories: true)
        _ = try StreamedPrecision.convert(
            keys: embedded.keys.sorted(), read: { try read(embedded[$0]!) }, to: embeddingFolder.appendingPathComponent("model.safetensors"),
            half: context.half, eligible: { _, _ in false })
        try FileManager.default.copyItem(at: context.source.appendingPathComponent("config.toml"), to: context.destination.appendingPathComponent("config.toml"))
        var config = try JSONSerialization.jsonObject(with: JSONEncoder().encode(dims)) as? [String: Any] ?? [:]
        config["architecture"] = "diarizen"
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(to: context.destination.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: ["architecture": "diarizen", "layout_version": 1, "embedding": embeddingDirectory], options: [.sortedKeys])
            .write(to: context.destination.appendingPathComponent(marker))
        return try context.receipt(cancellation: cancellation)
    }
}
