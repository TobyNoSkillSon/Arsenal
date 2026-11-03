// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ModelPrecision

extension KleinWeights {
    public static func storedHalf(snapshot: URL) throws -> DType {
        let archive = try TensorArchive(snapshot.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors"))
        let types = Set(archive.entries.values.map(\.dtype).filter { $0 == .float16 || $0 == .bfloat16 })
        guard types.count == 1, let dtype = types.first else { throw KleinLoadError("Snapshot must name one half compute dtype") }
        return dtype
    }

    /// The same conversion used by Get and the intake pilot. Large matrix weights include Qwen's embedding.
    public static func convert(source: URL, destination: URL, dtype: String) throws {
        guard ["fp16", "int8", "int4"].contains(dtype) else { throw KleinLoadError("Unsupported conversion precision") }
        let defaultHalf: DType = dtype == "fp16" ? .float16 : .bfloat16
        let bits = dtype == "int8" ? 8 : dtype == "int4" ? 4 : nil
        guard let walker = FileManager.default.enumerator(at: source, includingPropertiesForKeys: [.isRegularFileKey]) else { throw KleinLoadError("Missing source directory") }
        for case let file as URL in walker {
            guard try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            let relative = String(file.path.dropFirst(source.path.count + 1))
            if ["arsenal-snapshot.json", "arsenal-derived.json", "conversion.json"].contains(relative) { continue }
            let output = destination.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            if file.pathExtension == "safetensors" {
                let archive = try TensorArchive(file)
                let half = dtype == "fp16" ? defaultHalf : archive.entries.values.first(where: { $0.dtype == .float16 || $0.dtype == .bfloat16 })?.dtype ?? defaultHalf
                _ = try StreamedPrecision.convert(keys: archive.entries.keys.sorted(), read: archive.array, to: output, half: half, bits: bits) { key, shape in
                    // Convolutional VAE weights are small; its attention matrices also stay half as one decoder unit.
                    !relative.hasPrefix("vae/") && key.hasSuffix(".weight") && shape.reduce(1, *) >= 65_536
                }
            } else {
                try FileManager.default.copyItem(at: file, to: output)
            }
        }
    }
}

public extension KleinPipeline {
    func intakeComponents(fixture: URL, output: URL, capture: Bool) throws {
        let prompt = "A red ceramic teapot on a wooden table, soft daylight, product photograph."
        let inputs = try ComponentCapture.inputs(at: fixture, create: capture) {
            let (tokens, valid) = textEncoder.tokenize(prompt)
            let ids = MLXArray(tokens.map(Int32.init), [1, 512])
            let mask = KleinTextEncoder.mask(length: 512, valid: valid)
            let embeds = textEncoder.encoder(ids, mask: mask, activations: precision.encoder)[0]
            let sigmas = KleinSchedule.sigmas(seqLen: 4096, steps: 4)
            let (cos, sin) = Self.ropeTables(textLength: 512, gridH: 64, gridW: 64, dtype: precision.activations)
            var x = Self.noise(seed: 24301, width: 1024, height: 1024, dtype: precision.activations)
            var saved: [String: MLXArray] = ["ids": ids, "mask": mask, "conditioning": embeds[.newAxis], "cos": cos, "sin": sin]
            for i in 0..<4 {
                saved["latent-\(i)"] = x; saved["t-\(i)"] = MLXArray([Float16(sigmas[i] * 1000)]).asType(precision.activations)
                let v = transformer(x, encoder: embeds[.newAxis], timestep: saved["t-\(i)"]!, cos: cos, sin: sin, activations: precision.activations)
                x += MLXArray(Float16(sigmas[i + 1] - sigmas[i])).asType(precision.activations) * v
                eval(x)
            }
            saved["decoder-input"] = x
            return saved
        }
        var shapes = ["ids": [1, 512], "mask": [1, 1, 512, 512], "conditioning": [1, 512, 7680], "cos": [4608, 64], "sin": [4608, 64], "decoder-input": [1, 4096, 128]]
        for i in 0..<4 { shapes["latent-\(i)"] = [1, 4096, 128]; shapes["t-\(i)"] = [1] }
        try ComponentCapture.validate(inputs, shapes: shapes)
        let recorder = ComponentCapture()
        var outputs: [String: MLXArray] = [:]
        outputs["prompt"] = textEncoder.encoder(inputs["ids"]!, mask: inputs["mask"]!, activations: precision.encoder) { recorder.observe("prompt/" + $0, $1) }[0]
        eval(outputs["prompt"]!)
        for i in [0, 1, 3] {
            outputs["denoiser-\(i)"] = transformer(
                inputs["latent-\(i)"]!, encoder: inputs["conditioning"]!, timestep: inputs["t-\(i)"]!, cos: inputs["cos"]!.asType(precision.activations),
                sin: inputs["sin"]!.asType(precision.activations), activations: precision.activations
            ) {
                recorder.observe("denoiser-\(i)/" + $0, $1)
            }
            eval(outputs["denoiser-\(i)"]!)
        }
        vae.decoder.activationTrace = { recorder.observe("decoder/" + $0, $1) }
        defer { vae.decoder.activationTrace = nil }
        outputs["decoder"] = decode(inputs["decoder-input"]!, width: 1024, height: 1024)
        try recorder.write(outputs, to: output)
    }
}
