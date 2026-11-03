// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ModelPrecision
import ArsenalWire

/// Component/layout preparation is done once at Get. Derived tiers consume that complete prepared snapshot.
enum MusicConversion {
    static func eligible(_ name: String, _ shape: [Int]) -> Bool {
        name.hasSuffix(".weight") && shape.count == 2 && shape[1] % 64 == 0
    }
    static let yueBase = NativeConversion(recipe: "cast:yue2-v1", convert: prepareYuE)
    static let yueTiers = NativeConversion(recipe: "native:yue2-v1", convert: derivePrepared)
    static let layoutMarker = "arsenal-music-layout.json"
    static let legacyLayoutMarker = "mobius-music-layout.json" // legacy-compat: read in snapshots written before the rename
    static let aceBase = NativeConversion(recipe: "cast:ace-step-v1", convert: prepareACE)
    static let aceTiers = NativeConversion(recipe: "native:ace-step-v1", convert: derivePrepared)
    static let miniMaxBase = NativeConversion(recipe: "cast:minimax-music3-v1", convert: prepareMiniMax)

    static func derivePrepared(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> [String: JSONValue] {
        guard FileManager.default.fileExists(atPath: StoredNames.current(layoutMarker, legacy: legacyLayoutMarker, in: context.source).path) else {
            throw PrecisionError("Prepare the canonical half music snapshot before deriving a tier")
        }
        try context.convertTree(from: context.source, cancellation: cancellation, eligible: eligible)
        return try context.receipt(cancellation: cancellation)
    }
    static func mark(_ context: NativeConversionContext, architecture: String) throws {
        try JSONSerialization.data(withJSONObject: ["architecture": architecture, "layout_version": 1, "half": "fp16"]).write(
            to: context.destination.appendingPathComponent(layoutMarker))
    }

    /// Canonical source key → runtime key. A nil mapping is a documented unused source component/buffer.
    static func yueLMName(_ name: String) -> String {
        name.replacingOccurrences(of: "time_embedder.mlp.0.", with: "time_embedder.linear1.")
            .replacingOccurrences(of: "time_embedder.mlp.2.", with: "time_embedder.linear2.")
    }
    static func yueDecoderName(_ name: String) -> String? {
        guard name.hasPrefix("decoder.") else { return nil }
        return YuE2Pipeline.decoderName(String(name.dropFirst(8)))
    }
    static func aceName(_ name: String) -> String? {
        if name.contains("rotary_emb") { return nil }
        return name.replacingOccurrences(of: "decoder.proj_in.1.", with: "decoder.proj_in.")
            .replacingOccurrences(of: "decoder.proj_out.1.", with: "decoder.proj_out.")
    }
    static func copyFiles(_ source: URL, destination: URL, cancellation: NativeCancellation) throws {
        for path in try NativeConversionContext.files(in: source)
        where !path.hasSuffix(".safetensors") && !StoredNames.reserved.contains(URL(fileURLWithPath: path).lastPathComponent) {
            try cancellation.check()
            let target = destination.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: source.appendingPathComponent(path), to: target)
        }
    }
    static func transform(
        _ source: URL, to destination: URL, half: DType,
        name: (String) -> String?, value: (String, MLXArray) throws -> MLXArray,
        foldWeightNorm: Bool = false, cancellation: NativeCancellation
    ) throws {
        let archive = try TensorArchive(source)
        var keys: [String: String] = [:]
        for key in archive.entries.keys.sorted() {
            if foldWeightNorm && key.hasSuffix(".weight_g") { continue }
            guard var output = name(key) else { continue }
            if foldWeightNorm && key.hasSuffix(".weight_v") { output = String(output.dropLast(9)) + ".weight" }
            guard keys.updateValue(key, forKey: output) == nil else { throw PrecisionError("Duplicate transformed parameter: " + output) }
        }
        _ = try StreamedPrecision.convert(
            keys: keys.keys.sorted(),
            read: { output in
                try cancellation.check()
                let key = keys[output]!
                var tensor = try archive.array(key)
                if foldWeightNorm && key.hasSuffix(".weight_v") {
                    let gain = try archive.array(String(key.dropLast(2)) + "_g")
                    tensor = WeightNorm.fold(g: gain.asType(.float32), v: tensor.asType(.float32))
                }
                return try value(key, tensor)
            }, to: destination, half: half, eligible: { _, _ in false })
    }
    static func prepareYuE(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> [String: JSONValue] {
        guard context.bits == nil, let sourceVAE = context.assets["yue2-vae"] else { throw PrecisionError("YuE2 base preparation requires the verified VAE and a half tier") }
        try copyFiles(context.source, destination: context.destination, cancellation: cancellation)
        try transform(
            context.source.appendingPathComponent("model.safetensors"), to: context.destination.appendingPathComponent("model.safetensors"), half: context.half,
            name: { yueLMName($0) }, value: { _, x in x }, cancellation: cancellation)
        let vae = context.destination.appendingPathComponent("vae")
        try copyFiles(sourceVAE, destination: vae, cancellation: cancellation)
        try transform(
            sourceVAE.appendingPathComponent("model.safetensors"), to: vae.appendingPathComponent("model.safetensors"), half: context.half, name: yueDecoderName,
            value: { key, x in
                if key.hasSuffix(".weight_v") {
                    let base = String(yueDecoderName(key)!.dropLast(9))
                    return base.hasSuffix("conv_t1") ? x.transposed(1, 2, 0) : x.swappedAxes(1, 2)
                }
                return x
            }, foldWeightNorm: true, cancellation: cancellation)
        try mark(context, architecture: "yue2")
        return try context.receipt(cancellation: cancellation)
    }
    static func prepareACE(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> [String: JSONValue] {
        guard context.bits == nil else { throw PrecisionError("ACE-Step base preparation requires a half tier") }
        try copyFiles(context.source, destination: context.destination, cancellation: cancellation)
        for path in try NativeConversionContext.files(in: context.source) where path.hasSuffix(".safetensors") {
            let input = context.source.appendingPathComponent(path), output = context.destination.appendingPathComponent(path)
            if path.hasPrefix("vae/") {
                try transform(
                    input, to: output, half: context.half, name: { $0 },
                    value: { key, x in
                        if key.hasSuffix(".weight_v") { return key.contains("conv_t1.") ? x.transposed(1, 2, 0) : x.swappedAxes(1, 2) }
                        if key.hasSuffix(".alpha") || key.hasSuffix(".beta") { return x.reshaped(-1) }
                        return x
                    }, foldWeightNorm: true, cancellation: cancellation)
            } else if path.hasPrefix("acestep-v15-turbo/") {
                try transform(
                    input, to: output, half: context.half, name: aceName,
                    value: { key, x in
                        if key.hasPrefix("decoder.proj_in.1.") && key.hasSuffix(".weight") { return x.swappedAxes(1, 2) }
                        if key.hasPrefix("decoder.proj_out.1.") && key.hasSuffix(".weight") { return x.transposed(1, 2, 0) }
                        return x
                    }, cancellation: cancellation)
            } else {
                try transform(input, to: output, half: context.half, name: { "model." + $0 }, value: { _, x in x }, cancellation: cancellation)
            }
        }
        // Convert the small serialized silence tensor into a canonical half safetensors component.
        let path = "acestep-v15-turbo/silence_latent.pt"
        let silence = try TorchTensorFile.silenceLatent(context.source.appendingPathComponent(path)).asType(context.half)
        let target = context.destination.appendingPathComponent("acestep-v15-turbo/silence.safetensors")
        let writer = try TensorWriter(target); try writer.append("silence", silence); try writer.finish()
        try FileManager.default.removeItem(at: context.destination.appendingPathComponent(path))
        try mark(context, architecture: "acestep")
        return try context.receipt(cancellation: cancellation)
    }

    /// MiniMax Music 3: the five diffusers component directories, cast to half with MLX convolution layouts.
    /// Vocoder weight normalization is folded; snake scales become per-channel vectors. Every other key is unchanged.
    enum MiniMaxLayout: Equatable { case keep, conv, convTranspose, vector }
    static func miniMaxLayout(component: String, key: String) -> MiniMaxLayout {
        switch component {
        case "condition_encoder" where key == "proj.weight", "transformer" where key.hasSuffix("process_conv.weight"):
            return .conv
        case "vocoder":
            if key.hasSuffix(".alpha") { return .vector }
            if key.hasSuffix(".weight_v") { return key.contains("conv_t1.") ? .convTranspose : .conv }
            return key == "dec_in_proj.weight" ? .conv : .keep
        default:
            return .keep
        }
    }
    /// torch Conv1d (out, in, k) → MLX (out, k, in); ConvTranspose1d (in, out, k) → MLX (out, k, in).
    static func miniMaxValue(component: String, key: String, _ x: MLXArray) -> MLXArray {
        switch miniMaxLayout(component: component, key: key) {
        case .keep: return x
        case .conv: return x.swappedAxes(1, 2)
        case .convTranspose: return x.transposed(1, 2, 0)
        case .vector: return x.reshaped(-1)
        }
    }
    static func prepareMiniMax(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> [String: JSONValue] {
        guard context.bits == nil, context.half == .float16 else { throw PrecisionError("MiniMax Music 3 prepares a 16-bit (fp16) snapshot only") }
        try copyFiles(context.source, destination: context.destination, cancellation: cancellation)
        for path in try NativeConversionContext.files(in: context.source) where path.hasSuffix(".safetensors") {
            let component = String(path.split(separator: "/").first ?? "")
            guard MiniMaxComponents.all.contains(component) else { throw PrecisionError("Unexpected MiniMax Music 3 weight file: " + path) }
            try transform(
                context.source.appendingPathComponent(path), to: context.destination.appendingPathComponent(path), half: context.half, name: { $0 },
                value: { key, x in miniMaxValue(component: component, key: key, x) }, foldWeightNorm: component == "vocoder", cancellation: cancellation)
        }
        try mark(context, architecture: "minimax_music3")
        return try context.receipt(cancellation: cancellation)
    }
}
