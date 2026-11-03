// SPDX-License-Identifier: MIT
import Foundation
import MLX
import StableAudio3
import VireoCore
import ModelPrecision

public enum SoundEngines {
    public static func quantizedDirectory(_ source: ModelSource) -> URL? {
        source.options["quantized_directory"].map { URL(fileURLWithPath: $0) }
    }
    public static func load(_ source: ModelSource) async throws -> any AudioModel {
        switch source.id {
        case "stable-audio-3-medium", "stable-audio-3-small-sfx": return try await StableAudio3Engine.load(source)
        case "re-use": return try await ReUseEngine.load(source)
        case MossFormer2Engine.modelID: return try MossFormer2Engine.load(source)
        case UniPASEEngine.modelID: return try UniPASEEngine.load(source)
        case DeepFilterNet3Engine.modelID: return try DeepFilterNet3Engine.load(source)
        default: throw EngineError("Unknown sound or enhancement model")
        }
    }
    /// Loader key for a model ID (the ModelSource `engine`).
    public static func engine(for model: String) -> String {
        switch model {
        case "re-use": return "re-use"
        case MossFormer2Engine.modelID: return "mossformer2"
        case UniPASEEngine.modelID: return "unipase"
        case DeepFilterNet3Engine.modelID: return "deepfilternet3"
        default: return "stable-audio-3"
        }
    }
    /// Models that run in exactly one half dtype, with no quantized tiers.
    public static let halfOnlyModels = [MossFormer2Engine.modelID: MossFormer2Engine.dtype, UniPASEEngine.modelID: UniPASEEngine.dtype,
        DeepFilterNet3Engine.modelID: DeepFilterNet3Engine.dtype
    ]
    /// `cast:` recipes that turn the maker's checkpoint into that one half dtype: (dtype, convert(source, output)).
    public static let halfCastRecipes: [String: (dtype: String, convert: (URL, URL) throws -> Void)] = [
        "cast:" + MossFormer2Engine.modelID: (MossFormer2Engine.dtype, MossFormer2Engine.convert),
        "cast:" + UniPASEEngine.modelID: (UniPASEEngine.dtype, UniPASEEngine.convert),
        "cast:" + DeepFilterNet3Engine.modelID: (DeepFilterNet3Engine.dtype, DeepFilterNet3Engine.convert)
    ]
    /// Models whose no-weights contract check can load their own random initialisation.
    public static let randomWeightModels: Set<String> = [MossFormer2Engine.modelID, UniPASEEngine.modelID, DeepFilterNet3Engine.modelID]
    public static func convert(_ source: ModelSource, dtype: String, into directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if (source.id == "stable-audio-3-medium" && source.options["intake"] == "1") || source.id == "stable-audio-3-small-sfx" {
            try StableAudio3Engine.convertIntake(source, dtype: dtype, into: directory)
            return
        }
        if source.id == "re-use" {
            try ReUseEngine.convertIntake(source, dtype: dtype, into: directory)
            return
        }
        let bits = dtype == "int8" ? 8 : dtype == "int4" ? 4 : 0
        guard bits > 0, source.precision == "fp16" else { throw EngineError("Quantization requires the fp16 source and int8/int4 output") }
        let rule = source.id == "re-use" ? QuantRules.reUse : source.id == "stable-audio-3-small-sfx" ? QuantRules.stableAudio3Small : QuantRules.stableAudio3
        var report = QuantReport(rule: rule, bits: bits, groupSize: 64)
        if source.id == "re-use" { try ReUseEngine.convert(source, bits: bits, into: directory, report: &report) }
        else {
            try StableAudio3Engine.convertDiT(source, bits: bits, into: directory, report: &report)
            // A derived snapshot remains standalone: no source-directory requirement at load.
            for file in try SA3Variant(option: source.options["variant"]).officialFiles where !file.contains("dit_") {
                let target = directory.appendingPathComponent(file)
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: source.directory.appendingPathComponent(file).resolvingSymlinksInPath(), to: target)
            }
        }
        try JSONSerialization.data(withJSONObject: report.manifest, options: [.sortedKeys]).write(to: directory.appendingPathComponent("conversion.json"))
    }
}
