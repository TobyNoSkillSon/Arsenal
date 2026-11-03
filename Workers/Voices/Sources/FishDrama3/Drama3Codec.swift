// SPDX-License-Identifier: MIT
// The Fish S1 DAC codec behind Drama 3's codec boundary.
import FishS1Codec
import Foundation
import MLX

extension FishS1Codec: Drama3Codec {
    public func encodeCodes(_ samples: [Float]) throws -> MLXArray {
        guard !samples.isEmpty else { throw Drama3Error("reference audio is empty") }
        return encode(samples)
    }
    public func resetStreaming() { resetStreamingState() }
    public func decodeStep(_ codes: MLXArray) -> MLXArray { streamingStep(codes) }
}

enum Drama3CodecLoader {
    /// `codec.safetensors` beside the backbone shards, cast to `dtype` in memory.
    static func load(directory: URL, dtype: DType) throws -> FishS1Codec {
        let url = directory.appendingPathComponent("codec.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { throw Drama3Error("codec.safetensors is missing from \(directory.path)") }
        return try FishS1Codec.load(url: url, dtype: dtype)
    }
}
