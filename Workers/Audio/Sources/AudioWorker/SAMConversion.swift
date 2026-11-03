// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ModelPrecision
import ArsenalWire

/// SAM Audio's selected snapshot is self-contained: the separation checkpoint at its root and the
/// encoder half of the separately pinned T5-base asset in `text_encoder/`. Get prepares the half
/// snapshot once from the verified source and asset; derived tiers stream that prepared snapshot; Load never
/// reads the raw source or asset.
///
/// `cast:audio-sam-v1` takes a source already in the port's layout (an FP16 safetensors re-upload).
/// `cast:audio-sam-v2` takes Meta's original `checkpoint.pt` (FP32 `torch.save`) and writes the same layout
/// (`SAMConversion.Official`), tensor by tensor.
enum SAMConversion {
    static let textAsset = "sam-t5-base"
    static let textDirectory = "text_encoder"
    static let checkpoint = "checkpoint.pt"
    static let marker = "arsenal-sam-layout.json"
    static let legacyMarker = "mobius-sam-layout.json" // legacy-compat: read in snapshots written before the rename
    static let base = NativeConversion(recipe: "cast:audio-sam-v1", convert: prepare)
    static let official = NativeConversion(recipe: "cast:audio-sam-v2", convert: prepareOfficial)
    static let tiers = NativeConversion(recipe: "native:audio-sam-v1", convert: derive)

    /// Only T5's encoder runs. `shared.weight` is the token table; the tied encoder copy, decoder and LM head are not kept.
    static func textKey(_ name: String) -> Bool {
        name == "shared.weight" || name.hasPrefix("encoder.") && name != "encoder.embed_tokens.weight"
    }

    static func prepare(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits == nil, let asset = context.assets[textAsset] else {
            throw PrecisionError("SAM Audio base preparation needs a half tier and the verified T5-base asset")
        }
        try context.convertTree(from: context.source, cancellation: cancellation, eligible: { _, _ in false })
        try prepareText(context, asset: asset, cancellation: cancellation)
        return try context.receipt(cancellation: cancellation)
    }

    /// Meta's checkpoint and config in, the port's `model.safetensors` and config out. The plan is checked against the
    /// port's own parameter names and shapes before any tensor is read; each tensor is then read, transformed in its
    /// stored FP32, cast to the selected half and written before the next is read.
    static func prepareOfficial(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits == nil, let asset = context.assets[textAsset] else {
            throw PrecisionError("SAM Audio base preparation needs a half tier and the verified T5-base asset")
        }
        let files = try NativeConversionContext.files(in: context.source)
        guard Set(files) == [checkpoint, "config.json"] else {
            throw PrecisionError("SAM Audio's official source is exactly checkpoint.pt and config.json")
        }
        let configURL = context.source.appendingPathComponent("config.json")
        let configuration = try JSONDecoder().decode(SAMAudioConfig.self, from: Data(contentsOf: configURL))
        let torch = try TorchCheckpoint(context.source.appendingPathComponent(checkpoint))
        let tensors = try torch.tensors()
        guard tensors.values.allSatisfy({ [.float32, .float16].contains($0.dtype) }) else {
            throw PrecisionError("SAM Audio checkpoint holds a tensor that is not FP32 or FP16")
        }
        let shapes = tensors.mapValues(\.shape)
        let plan = try Official.plan(shapes)
        let expected = Dictionary(uniqueKeysWithValues: SAMAudio(config: configuration).parameters().flattened().map { ($0.0, $0.1.shape) })
        try Official.require(plan, shapes: shapes, equals: expected)
        _ = try StreamedPrecision.convert(
            keys: plan.keys.sorted(),
            read: { key in
                try cancellation.check()
                return try Official.read(plan[key]!, tensors: tensors, from: torch)
            }, to: context.destination.appendingPathComponent("model.safetensors"), half: context.half, eligible: { _, _ in false })
        try FileManager.default.copyItem(at: configURL, to: context.destination.appendingPathComponent("config.json"))
        try prepareText(context, asset: asset, cancellation: cancellation)
        return try context.receipt(cancellation: cancellation)
    }

    /// The encoder half of T5-base into `text_encoder/`, and the layout marker.
    static func prepareText(_ context: NativeConversionContext, asset: URL, cancellation: NativeCancellation) throws {
        let text = context.destination.appendingPathComponent(textDirectory)
        try FileManager.default.createDirectory(at: text, withIntermediateDirectories: true)
        for file in ["config.json", "tokenizer.json"] {
            try cancellation.check()
            try FileManager.default.copyItem(at: asset.appendingPathComponent(file), to: text.appendingPathComponent(file))
        }
        let archive = try TensorArchive(asset.appendingPathComponent("model.safetensors"))
        let keys = archive.entries.keys.filter(textKey).sorted()
        guard keys.contains("shared.weight"), keys.contains(where: { $0.hasPrefix("encoder.block.") }) else {
            throw PrecisionError("T5-base asset lacks its encoder tensors")
        }
        _ = try StreamedPrecision.convert(
            keys: keys,
            read: { key in
                try cancellation.check(); return try archive.array(key)
            }, to: text.appendingPathComponent("model.safetensors"), half: context.half, eligible: { _, _ in false })
        try JSONSerialization.data(withJSONObject: ["architecture": "sam_audio", "layout_version": 1, "text_encoder": textDirectory], options: [.sortedKeys])
            .write(to: context.destination.appendingPathComponent(marker))
    }

    static func derive(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits != nil, FileManager.default.fileExists(atPath: StoredNames.current(marker, legacy: legacyMarker, in: context.source).path) else {
            throw PrecisionError("Prepare the SAM Audio half snapshot before deriving int8/int4")
        }
        try context.convertTree(from: context.source, cancellation: cancellation, eligible: audioEligible)
        return try context.receipt(cancellation: cancellation)
    }

    /// Meta's original names and layouts mapped to the ones the port loads, following mlx-audio's `sam_audio`
    /// (`sanitize`, `_convert_weight_name` and the load-time transposes in `_load_weights`):
    /// - dropped: the vision encoder, the masked-video aligner and the rankers/span predictor the port does not run;
    /// - renamed: DAC-VAE encoder/decoder blocks, the watermark model's pre/post stages, LSTM and quantizer projections;
    /// - LSTM `bias_ih` + `bias_hh` summed into one bias (in FP32, before the half cast);
    /// - every rank-3 convolution tensor (weights, `weight_v`, Snake `alpha`) from (out, in, k) to MLX's (out, k, in),
    ///   and the decoder's ConvTranspose1d upsamplers (`block_1`, `block_3`) from (in, out, k) to (out, k, in).
    /// Weight-normalized convolutions keep `weight_g`/`weight_v`: the port normalizes them when it runs.
    enum Official {
        struct Source: Equatable {
            /// One source tensor, or the two LSTM biases that are summed.
            let names: [String]
            let permutation: [Int]?
        }
        static let dropped = ["vision_encoder.", "align_masked_video.", "text_encoder.", "span_predictor.", "visual_ranker.", "text_ranker."]

        static func plan(_ shapes: [String: [Int]]) throws -> [String: Source] {
            var plan: [String: Source] = [:]
            var biases: [String: [String: String]] = [:]
            func add(_ target: String, _ source: Source) throws {
                guard plan.updateValue(source, forKey: target) == nil else { throw PrecisionError("Two SAM Audio tensors map to " + target) }
            }
            for key in shapes.keys.sorted() {
                if dropped.contains(where: { key.hasPrefix($0) }) || key.contains("wm_rates") { continue }
                if let (base, kind, layer) = lstmBias(key) {
                    biases[base + ".combined_bias_l" + layer, default: [:]][kind] = key
                    continue
                }
                let target = name(key)
                try add(target, Source(names: [key], permutation: permutation(target, rank: shapes[key]!.count)))
            }
            for (combined, pair) in biases.sorted(by: { $0.key < $1.key }) {
                guard let ih = pair["ih"], let hh = pair["hh"], shapes[ih] == shapes[hh] else { throw PrecisionError("Unpaired LSTM bias " + combined) }
                try add(name(combined), Source(names: [ih, hh], permutation: nil))
            }
            return plan
        }

        /// The shape a planned tensor is written with.
        static func shape(_ source: Source, in shapes: [String: [Int]]) -> [Int] {
            let shape = shapes[source.names[0]] ?? []
            return source.permutation.map { $0.map { shape[$0] } } ?? shape
        }

        /// The plan must produce exactly the port's parameters (`expected`, in module names): same names, same shapes.
        /// Stored LSTM names (`Wx`, `Wh`, `bias`) become the module's at load (`audioLSTMKey`).
        static func require(_ plan: [String: Source], shapes: [String: [Int]], equals expected: [String: [Int]]) throws {
            var planned: [String: [Int]] = [:]
            for (key, source) in plan {
                guard planned.updateValue(shape(source, in: shapes), forKey: audioLSTMKey(key)) == nil else { throw PrecisionError("Two SAM Audio tensors load as " + key) }
            }
            let missing = Set(expected.keys).subtracting(planned.keys).sorted(), extra = Set(planned.keys).subtracting(expected.keys).sorted()
            let mismatched = planned.filter { entry in expected[entry.key].map { $0 != entry.value } ?? false }.keys.sorted()
            guard missing.isEmpty, extra.isEmpty, mismatched.isEmpty else {
                throw PrecisionError(
                    "SAM Audio checkpoint does not match the port (missing \(missing.prefix(3)), unexpected \(extra.prefix(3)), shape \(mismatched.prefix(3)))")
            }
        }

        /// One planned tensor in its stored precision, summed and permuted on the CPU (lazy until the half cast).
        static func read(_ source: Source, tensors: [String: TorchCheckpoint.Tensor], from checkpoint: TorchCheckpoint) throws -> MLXArray {
            var value = try checkpoint.array(tensors[source.names[0]]!)
            for name in source.names.dropFirst() { value = MLX.add(value, try checkpoint.array(tensors[name]!), stream: .cpu) }
            if let permutation = source.permutation { value = MLX.transposed(value, axes: permutation, stream: .cpu) }
            return value
        }

        static func permutation(_ name: String, rank: Int) -> [Int]? {
            guard rank == 3 else { return nil }
            let parts = name.split(separator: ".")
            if parts.count == 6, parts[0] == "audio_codec", parts[1] == "decoder", parts[2] == "blocks", Int(parts[3]) != nil,
                ["block_1", "block_3"].contains(parts[4]), ["weight", "weight_g", "weight_v"].contains(parts[5])
            {
                return [1, 2, 0]
            }
            return name.hasSuffix(".weight_g") ? nil : [0, 2, 1]
        }

        /// `(base, "ih" | "hh", layer)` for `<base>.lstm.bias_ih_l<layer>`.
        static func lstmBias(_ key: String) -> (String, String, String)? {
            for kind in ["ih", "hh"] {
                guard let range = key.range(of: ".lstm.bias_\(kind)_l", options: .backwards) else { continue }
                let layer = String(key[range.upperBound...])
                guard !layer.isEmpty, layer.allSatisfy(\.isNumber) else { continue }
                return (String(key[..<range.lowerBound]) + ".lstm", kind, layer)
            }
            return nil
        }

        /// mlx-audio's `_convert_weight_name`, step for step (four encoder and four decoder blocks).
        static func name(_ key: String) -> String {
            var result = key
            @discardableResult func swap(_ old: String, _ new: String) -> Bool {
                guard result.hasPrefix(old) else { return false }
                result = new + result.dropFirst(old.count)
                return true
            }
            // A residual unit's block.0…3 are act1, conv1, act2, conv2.
            func unit(_ prefix: String) {
                for (index, part) in ["act1", "conv1", "act2", "conv2"].enumerated() where swap(prefix + "block.\(index).", prefix + part + ".") { return }
            }
            swap("audio_codec.encoder.block.0.", "audio_codec.encoder.conv_in.")
            for block in 1...4 {
                for residual in 0..<3 {
                    let prefix = "audio_codec.encoder.blocks.\(block - 1).res\(residual + 1)."
                    if swap("audio_codec.encoder.block.\(block).block.\(residual).", prefix) {
                        unit(prefix)
                        break
                    }
                }
                swap("audio_codec.encoder.block.\(block).block.3.", "audio_codec.encoder.blocks.\(block - 1).snake.")
                swap("audio_codec.encoder.block.\(block).block.4.", "audio_codec.encoder.blocks.\(block - 1).conv.")
            }
            swap("audio_codec.encoder.block.5.", "audio_codec.encoder.snake_out.")
            swap("audio_codec.encoder.block.6.", "audio_codec.encoder.conv_out.")
            swap("audio_codec.decoder.model.0.", "audio_codec.decoder.conv_in.")
            decoder: for block in 1...4 {
                for part in [0, 1, 3, 4, 5, 6, 7, 8, 11] {
                    let prefix = "audio_codec.decoder.blocks.\(block - 1).block_\(part)."
                    if swap("audio_codec.decoder.model.\(block).block.\(part).", prefix) {
                        if (4...8).contains(part) { unit(prefix) }
                        break decoder
                    }
                }
            }
            // The decoder's last Snake and convolution live in the watermark model's first encoder stage.
            swap("audio_codec.decoder.wm_model.encoder_block.pre.0.", "audio_codec.decoder.snake_out.")
            swap("audio_codec.decoder.wm_model.encoder_block.pre.1.", "audio_codec.decoder.conv_out.")
            for stage in ["encoder_block", "decoder_block"] {
                for side in ["pre", "post"] {
                    for index in 0..<4 { result = result.replacingOccurrences(of: ".\(stage).\(side).\(index).", with: ".\(stage).\(side)_\(index).") }
                }
            }
            for (old, new) in [("weight_ih_l", "Wx"), ("weight_hh_l", "Wh"), ("combined_bias_l", "bias")] {
                guard let range = result.range(of: ".lstm." + old, options: .backwards) else { continue }
                let layer = result[range.upperBound...]
                guard !layer.isEmpty, layer.allSatisfy(\.isNumber) else { continue }
                result = String(result[..<range.lowerBound]) + ".lstm.layers.\(layer)." + new
                break
            }
            swap("audio_codec.quantizer.in_proj.", "audio_codec.quantizer_in_proj.")
            swap("audio_codec.quantizer.out_proj.", "audio_codec.quantizer_out_proj.")
            return result
        }
    }
}
