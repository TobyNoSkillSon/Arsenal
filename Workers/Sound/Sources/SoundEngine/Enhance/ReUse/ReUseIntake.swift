// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ModelPrecision
import VireoCore

final class ReUseIntakeSession {
    var inputs: [String: MLXArray]
    let capture: Bool
    var valid = true
    var used = Set<String>()
    var outputs: [String: MLXArray] = [:]
    let recorder = ComponentCapture()
    init(inputs: [String: MLXArray], capture: Bool) { self.inputs = inputs; self.capture = capture }
    func input(_ key: String, _ value: MLXArray) -> MLXArray {
        used.insert(key)
        if capture { eval(value); inputs[key] = value }
        guard let saved = inputs[key], saved.shape == value.shape else { valid = false; return value }
        return saved.asType(value.dtype)
    }
    func output(_ key: String, _ value: MLXArray) { eval(value); outputs[key] = value }
}

extension ReUseEngine {
    /// Every g64-compatible matrix, including SSM xProj; convolutions and width-4 dtProj stay half.
    static func convertIntake(_ source: ModelSource, dtype: String, into directory: URL) throws {
        // NVIDIA's own repository ships the PyTorch checkpoint (model.safetensors); the older MLX conversion shipped
        // model_mlx.safetensors. Both become the same canonical file: MLX names and channels-last convolutions.
        let (archive, names, read) = try ReUseMakerLayout.canonicalSource(source.directory)
        let half: DType = dtype == "bf16" ? .bfloat16 : dtype == "fp16" ? .float16 : archive.entries.values.first(where: { [.float16, .bfloat16].contains($0.dtype) })?.dtype ?? .float16
        let bits = dtype == "int8" ? 8 : dtype == "int4" ? 4 : nil
        let halfModules = (source.options["intake_half_modules"] ?? "").split(separator: ",").map(String.init)
        let keepsHalf: (String) -> Bool = { key in halfModules.contains { key == $0 || key.hasPrefix($0 + ".") } }
        let coverage = try StreamedPrecision.convert(keys: names.keys.sorted(), read: read, to: directory.appendingPathComponent(weightsFile), half: half, bits: bits) { key, _ in !keepsHalf(key) }
        try JSONEncoder().encode(coverage).write(to: directory.appendingPathComponent("coverage.json"))
        for name in ["LICENSE", "NOTICE"] {
            let from = source.directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: from.path) { try FileManager.default.copyItem(at: from, to: directory.appendingPathComponent(name)) }
        }
        var packed = 0, total = 0, kept: [String: String] = [:]
        for (key, source) in names {
            let entry = archive.entries[source]!
            let count = entry.shape.reduce(1, *); total += count
            if bits != nil, !keepsHalf(key), key.hasSuffix(".weight"), entry.shape.count == 2, entry.shape[1] % 64 == 0 { packed += count }
            else { kept[key] = keepsHalf(key) ? "int8 component quality fallback" : entry.shape.count == 2 ? "input width incompatible with group 64" : "convolution, norm, bias or SSM vector" }
        }
        try JSONSerialization.data(withJSONObject: ["total_parameters": total, "packed_parameters": packed, "kept": kept], options: [.sortedKeys]).write(to: directory.appendingPathComponent("component-coverage.json"))
    }

    /// Calibration: nearest/truncated BF16 of the original FP32 file, streamed on CPU before GPU execution.
    public static func castSource(from source: URL, to target: URL, truncate: Bool) throws {
        let (_, names, read) = try ReUseMakerLayout.canonicalSource(source)
        let writer = try TensorWriter(target.appendingPathComponent(weightsFile))
        try Device.withDefaultDevice(.cpu) {
            for key in names.keys.sorted() {
                try autoreleasepool {
                    // Contiguous before the byte-level truncation below reads the buffer in order.
                    let value = contiguous(try read(key), stream: .cpu)
                    guard value.dtype == .float32 else { throw EngineError("Source calibration requires original FP32") }
                    let half: MLXArray
                    if truncate {
                        var bytes = Data(count: value.size * 2)
                        let values = value.asArray(Float.self)
                        bytes.withUnsafeMutableBytes { (out: UnsafeMutableRawBufferPointer) in
                            let bits = out.bindMemory(to: UInt16.self)
                            for i in values.indices { bits[i] = UInt16(truncatingIfNeeded: values[i].bitPattern >> 16).littleEndian }
                        }
                        half = MLXArray(bytes, value.shape, dtype: .bfloat16)
                    } else { half = value.asType(.bfloat16, stream: .cpu) }
                    eval(half); try writer.append(key, half)
                }
            }
            try writer.finish()
        }
    }
}

/// The maker's PyTorch checkpoint (nvidia/NVIDIA-NemotronLabs-RE-USE) in the port's layout: SEMamba module names in
/// camel case, `nn.Sequential` children under `layers`, and convolution weights moved from PyTorch's channels-first
/// (O, I, H, W) / (O, I, K) to MLX's channels-last (O, H, W, I) / (O, K, I). Values are untouched.
enum ReUseMakerLayout {
    /// The source folder as canonical tensors: the port's own `model_mlx.safetensors` as is, or the maker's
    /// `model.safetensors` renamed and moved to channels-last. Returns the archive, canonical → stored names and a reader.
    static func canonicalSource(_ directory: URL) throws -> (TensorArchive, [String: String], (String) throws -> MLXArray) {
        let maker = directory.appendingPathComponent(ReUseEngine.makerWeightsFile)
        let isMaker = !FileManager.default.fileExists(atPath: directory.appendingPathComponent(ReUseEngine.weightsFile).path)
            && FileManager.default.fileExists(atPath: maker.path)
        let archive = try TensorArchive(isMaker ? maker : directory.appendingPathComponent(ReUseEngine.weightsFile))
        let names = isMaker ? try names(Array(archive.entries.keys)) : Dictionary(uniqueKeysWithValues: archive.entries.keys.map { ($0, $0) })
        let read: (String) throws -> MLXArray = { key in
            guard let stored = names[key] else { throw EngineError("RE-USE source has no tensor \(key)") }
            let value = try archive.array(stored)
            return isMaker ? channelsLast(value) : value
        }
        return (archive, names, read)
    }

    /// Canonical name → maker name, for every maker tensor; refuses a name the layout does not know.
    static func names(_ maker: [String]) throws -> [String: String] {
        var out: [String: String] = [:]
        for name in maker {
            let canonical = try rename(name)
            guard out.updateValue(name, forKey: canonical) == nil else { throw EngineError("RE-USE maker layout maps two tensors to \(canonical)") }
        }
        return out
    }
    static func rename(_ name: String) throws -> String {
        var parts = name.split(separator: ".").map(String.init)
        let modules = ["dense_encoder": "denseEncoder", "mask_decoder": "maskDecoder", "phase_decoder": "phaseDecoder", "TSMamba": "tfMamba"]
        guard let first = parts.first, let module = modules[first] else { throw EngineError("Unknown RE-USE maker tensor \(name)") }
        parts[0] = module
        let simple = [
            "dense_conv_1": "conv1", "dense_conv_2": "conv2", "up_conv1": "upConv1", "up_conv2": "upConv2", "final_conv": "finalConv", "phase_conv_i": "phaseConvI",
            "phase_conv_r": "phaseConvR", "freq_mamba": "freqMamba", "time_mamba": "timeMamba", "backward_blocks": "backward", "forward_blocks": "forward",
            "dt_proj": "dtProj", "in_proj": "inProj", "out_proj": "outProj", "x_proj": "xProj", "output_proj": "outputProj",
        ]
        var out: [String] = []
        var i = 0
        while i < parts.count {
            let part = parts[i]
            if part == "dense_block", i + 3 < parts.count, parts[i + 1] == "dense_block", Int(parts[i + 2]) != nil, Int(parts[i + 3]) != nil {
                out += ["denseBlock", "layers", parts[i + 2], "layers", parts[i + 3]]; i += 4; continue
            }
            if ["dense_conv_1", "dense_conv_2", "up_conv1", "up_conv2"].contains(part), i + 1 < parts.count, Int(parts[i + 1]) != nil {
                out += [simple[part]!, "layers", parts[i + 1]]; i += 2; continue
            }
            out.append(simple[part] ?? part); i += 1
        }
        return out.joined(separator: ".")
    }
    /// The axis order that moves a PyTorch convolution weight to MLX's layout; nil for anything that is not one.
    static func channelsLastAxes(rank: Int) -> [Int]? {
        switch rank {
        case 4: return [0, 2, 3, 1]
        case 3: return [0, 2, 1]
        default: return nil
        }
    }
    static func channelsLast(_ value: MLXArray) -> MLXArray {
        channelsLastAxes(rank: value.ndim).map { value.transposed(axes: $0) } ?? value
    }
}

extension MambaEnhancer {
    func intakeComponents(fixture: URL, output: URL, capture: Bool) throws {
        let half = model.denseEncoder.conv1.parameters().flattened().first!.1.dtype
        let values = try ComponentCapture.inputs(at: fixture, create: capture) {
            // Small fixed spectral input: 161 bins (8-kHz training base), 17 frames.
            let n = 161 * 17
            return ["magnitude": MLXArray((0..<n).map { Float(0.4 + 0.2 * sin(Double($0) * 0.013)) }, [1, 161, 17]).asType(half),
                    "phase": MLXArray((0..<n).map { Float(0.7 * cos(Double($0) * 0.007)) }, [1, 161, 17]).asType(half)]
        }
        let archive = fixture.appendingPathComponent("components.safetensors")
        let session = ReUseIntakeSession(inputs: capture ? [:] : try TensorArchive(archive).arrays(), capture: capture)
        model.intake = session; defer { model.intake = nil }
        let outputs = model(values["magnitude"]!.asType(half), values["phase"]!.asType(half)); eval(outputs.0, outputs.1)
        let expected = Set(["encoder", "magnitude", "phase"] + (0..<30).flatMap { ["block-\($0)/time", "block-\($0)/frequency"] })
        guard session.valid, Set(session.inputs.keys) == session.used, Set(session.outputs.keys) == expected else { throw EngineError("RE-USE component count, shapes or keys differ") }
        if capture { try save(arrays: session.inputs, url: archive); try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: archive.path) }
        try session.recorder.write(session.outputs, to: output)
    }
}

public extension ReUseModel {
    func intakeComponents(fixture: URL, output: URL, capture: Bool) async throws {
        guard let enhancer else { throw EngineError("RE-USE unloaded") }
        try await enhancer.intakeComponents(fixture: fixture, output: output, capture: capture)
    }
}
