// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
import ModelPrecision
@testable import StableAudio3

/// Explicit lab capture only: normal test runs never load model checkpoints.
@Test(.heavy) func sameSFP32SourceHalfQualificationCapture() throws {
    let env = ProcessInfo.processInfo.environment
    guard let root = env["ARSENAL_SAMES_FORMAT_LAB"] else { return }
    if let path = env["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
    Memory.memoryLimit = 4_000_000_000
    Memory.cacheLimit = 64_000_000
    let lab = URL(fileURLWithPath: root)
    let fixture = try TensorArchive(lab.appendingPathComponent("inputs.safetensors")).arrays()
    let decoderArchive = try NPZArchive(url: lab.appendingPathComponent("same_s_decoder_f32.npz"))
    let ditArchive = try NPZArchive(url: lab.appendingPathComponent("dit_sm-sfx_f16.npz"))
    func cast(_ value: MLXArray, dtype: DType, truncate: Bool) -> MLXArray {
        if truncate {
            // Conversion only, on host: alternate BF16 mantissa rounding, no FP32 model execution.
            let values = value.asArray(Float.self).map { Float(bitPattern: $0.bitPattern & 0xffff0000) }
            return MLXArray(values).reshaped(value.shape).asType(.bfloat16)
        }
        return value.asType(dtype)
    }
    let candidates: [(String, DType, Bool)] =
        env["ARSENAL_SAMES_FORMAT_CANDIDATE"] == "1"
        ? [("fp16-candidate", .float16, false)]
        : [("bf16-reference", .bfloat16, false), ("bf16-rounding", .bfloat16, true), ("bf16-repeat", .bfloat16, false)]
    for (name, dtype, truncate) in candidates {
        try autoreleasepool {
            let decoder = SAMESDecoder(.small)
            var weights: [String: MLXArray] = [:]
            for key in decoderArchive.keys {
                weights[key] = cast(try decoderArchive.array(key), dtype: dtype, truncate: truncate)
                eval(weights[key]!)
            }
            weights = sa3SAMESDecoderWeights(weights)
            try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            #expect(decoder.parameters().flattened().allSatisfy { $0.1.dtype == dtype })
            let recorder = ComponentCapture()
            var index = 0
            decoder.activationTrace = { key, value in
                recorder.observe("decoder/\(index)/" + key, value); index += 1
            }
            let output = sa3DecodeLatents(decoder, fixture["decoder-input"]!)
            eval(output)
            let weight = cast(try ditArchive.array("cond.seconds_total_weight"), dtype: dtype, truncate: truncate)
            let bias = cast(try ditArchive.array("cond.seconds_total_bias"), dtype: dtype, truncate: truncate)
            let seconds = SecondsTotalEmbedder(weight: weight, bias: bias, halfOperands: true)(fixture["seconds-value"]!.item(Float.self))
            let padding = cast(try ditArchive.array("cond.padding_embedding"), dtype: dtype, truncate: truncate)
            try recorder.write(["decoder": output, "seconds": seconds, "padding": padding], to: lab.appendingPathComponent(name))
            #expect(output.dtype == dtype && seconds.dtype == dtype && padding.dtype == dtype)
        }
        Memory.clearCache()
    }
}
