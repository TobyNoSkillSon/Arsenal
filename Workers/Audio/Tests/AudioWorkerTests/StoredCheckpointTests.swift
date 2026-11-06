import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import AudioWorker

extension PrecisionTests {
    @Test(.heavy) func allModelCheckpointMapsInstallStoredTiers() throws {
        guard let metallib = (ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"]) else {
            throw audioFailure("Pinned metallib is required.")
        }
        setTestMetallib(URL(fileURLWithPath: metallib))
        setenv("ARSENAL_NATIVE_LAB", "1", 1)
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let ids = [
            "qwen3-asr-1.7b", "confucius4-r2t2", "cohere-transcribe-03-2026", "granite-speech-5.0-470m-turboctc", "nemotron-3-diarization", "smart-turn-v3", "sam-audio-small",
            "sam-audio-base", "sam-audio-large", "htdemucs-ft"
        ]
        for id in ids {
            try autoreleasepool {
                let shipped = repo.appendingPathComponent("models/" + id)
                let retired = repo.appendingPathComponent("retired-models/" + id)
                let folder = FileManager.default.fileExists(atPath: shipped.appendingPathComponent("gate/no-weights.json").path) ? shipped : retired
                let fixture = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/no-weights.json")))
                var config = fixture["config"]!.object!
                if id == "nemotron-3-diarization" { config["lab_layers"] = 31 }
                let type = config["model_type"]!.string!
                let registration = AudioWorker.registrations.first { $0.modelTypes.contains(type) }!
                let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                let source = root.appendingPathComponent("source")
                try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
                defer { try? FileManager.default.removeItem(at: root); Memory.clearCache() }
                let configData = try WireJSON.encode(config)
                try configData.write(to: source.appendingPathComponent("config.json"))
                if id == "htdemucs-ft" { try configData.write(to: source.appendingPathComponent("htdemucs_ft_config.json")) }
                let tokenizer = folder.appendingPathComponent("gate/tokenizer")
                if FileManager.default.fileExists(atPath: tokenizer.path) {
                    for file in try FileManager.default.contentsOfDirectory(at: tokenizer, includingPropertiesForKeys: nil) {
                        try FileManager.default.copyItem(at: file, to: source.appendingPathComponent(file.lastPathComponent))
                    }
                }
                guard let dense = registration.make() as? AudioAdapter else { throw audioFailure("Expected an audio adapter.") }
                let randomContext = try NativeLoadContext([
                    "model_dir": .string(folder.path), "weights_dir": .string(source.path), "dtype": "fp16",
                    "config": .object(config.merging(["random_weights": true]) { _, new in new })
                ])
                try dense.load(randomContext, cancellation: NativeCancellation())
                let layout = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/header-map.json")))
                var inverse: [String: (String, [Int]?)] = [:]
                for (name, rule) in layout["tensors"]?.object ?? [:] {
                    let local = rule["name"]?.string ?? name
                    inverse[local] = (name, rule["permutation"]?.array?.compactMap { $0.number.map(Int.init) })
                }
                var tensors: [String: MLXArray] = [:]
                for (name, tensor) in dense.graph!.parameters().flattened() {
                    let (key, axes) = inverse[name] ?? (name, nil)
                    if let axes {
                        var reverse = axes
                        for (index, axis) in axes.enumerated() { reverse[axis] = index }
                        tensors[key] = tensor.transposed(axes: reverse)
                    } else {
                        tensors[key] = tensor
                    }
                }
                try save(arrays: tensors, url: source.appendingPathComponent("model.safetensors"))
                let asset = root.appendingPathComponent("text-asset")
                if let sam = dense as? SAMAdapter, let text = sam.textGraph {
                    try FileManager.default.createDirectory(at: asset, withIntermediateDirectories: true)
                    let textConfig: JSONObject = ["vocab_size": 512, "d_model": 64, "d_kv": 16, "d_ff": 128, "num_layers": 1, "num_heads": 4]
                    try WireJSON.encode(textConfig).write(to: asset.appendingPathComponent("config.json"))
                    let unigram: JSONObject = [
                        "version": "1.0", "added_tokens": [],
                        "pre_tokenizer": ["type": "Whitespace"],
                        "model": ["type": "Unigram", "unk_id": 2, "vocab": [["<pad>", 0], ["</s>", 0], ["<unk>", 0], ["voice", -1]]]
                    ]
                    try WireJSON.encode(unigram).write(to: asset.appendingPathComponent("tokenizer.json"))
                    var textWeights: [String: MLXArray] = [:]
                    for (name, value) in text.parameters().flattened() {
                        let key = name.replacingOccurrences(of: ".self_attention.", with: ".layer.0.").replacingOccurrences(of: ".ff.", with: ".layer.1.")
                        textWeights[key] = value
                    }
                    try save(arrays: textWeights, url: asset.appendingPathComponent("model.safetensors"))
                }
                let eligibleCount = dense.parameters.filter { audioEligible($0.key, $0.value.shape) }.count
                dense.unload()
                for precision in ["fp16", "int8", "int4"] {
                    let destination = root.appendingPathComponent(precision)
                    // SAM prepares its half snapshot (with the T5 encoder) once; int8/int4 derive from that snapshot.
                    let sam = dense is SAMAdapter
                    let converter = sam ? (precision == "fp16" ? SAMConversion.base : SAMConversion.tiers) : registration.conversion!
                    var request: JSONObject = [
                        "src_dir": .string(sam && precision != "fp16" ? root.appendingPathComponent("fp16").path : source.path), "dst_dir": .string(destination.path),
                        "dtype": .string(precision), "group_size": 64, "recipe": .string(converter.recipe)
                    ]
                    if sam && precision == "fp16" { request["assets"] = [SAMConversion.textAsset: .string(asset.path)] }
                    _ = try converter.run(request, cancellation: NativeCancellation())
                    let model = registration.make()
                    let context = try NativeLoadContext([
                        "model_dir": .string(folder.path), "weights_dir": .string(destination.path), "dtype": .string(precision), "config": .object(config)
                    ])
                    try model.load(context, cancellation: NativeCancellation())
                    try NativeParameters.requireHalf(model.parameters)
                    let packed = model.parameters.filter { $0.value.dtype == .uint32 }
                    #expect(packed.count == (precision == "fp16" ? 0 : eligibleCount), "\(id) \(precision)")
                    model.unload()
                }
            }
        }
    }
}
