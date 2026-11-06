// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import Testing
@testable import FishS1Codec

/// Fish S1 DAC (Drama 3's codec) structure and streaming, on the CPU with tiny random weights. No checkpoint is read and the
/// real-size module is built only to compare parameter names and shapes, never evaluated.
@Suite(.serialized) struct FishS1CodecTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    /// A codec small enough to run in milliseconds on the CPU; the 3-frame quantizer window makes 7 frames exercise windowing.
    static let tiny: FishS1CodecConfig = {
        var c = FishS1CodecConfig()
        c.encoderDim = 2; c.encoderRates = [2, 2, 2, 2]; c.latentDim = 32
        c.decoderDim = 32; c.decoderRates = [2, 2, 2, 2]
        c.codebookSize = 16; c.codebookDim = 4; c.semanticCodebookSize = 32
        c.encoderTransformerLayers = [0, 0, 0, 1]; c.encoderWindow = 4; c.encoderBlockSize = 256; c.encoderHeadDim = 8
        c.quantizerLayers = 2; c.quantizerHeads = 4; c.quantizerHeadDim = 8; c.quantizerIntermediate = 48
        c.quantizerWindow = 3; c.quantizerBlockSize = 64
        return c
    }()

    static func checkpointShapes() throws -> [String: [Int]] {
        let url = try #require(Bundle.module.url(forResource: "fish-s1-codec-keys", withExtension: "json"))
        return try JSONDecoder().decode([String: [Int]].self, from: Data(contentsOf: url))
    }

    @Test func sanitizeMapsTheReferenceKeyForms() throws {
        let keys = [
            "decoder.model.0.conv.parametrizations.weight.original0", "decoder.model.0.conv.parametrizations.weight.original1",
            "decoder.model.0.conv.bias", "quantizer.downsample.0.1.dwconv.conv.bias", "quantizer.upsample.1.0.conv.weight",
            "quantizer.quantizer.quantizers.3.in_proj.weight_g", "encoder.block.4.block.5.freqs_cis", "x.layers.0.causal_mask"
        ]
        let prefixes = FishS1Codec.weightNormPrefixes(keys)
        #expect(prefixes == ["decoder.model.0"])
        let mapped = keys.map { FishS1Codec.sanitizedKey($0, weightNormPrefixes: prefixes) }
        #expect(
            mapped == [
                "decoder.model.0.weight_g", "decoder.model.0.weight_v", "decoder.model.0.bias", "quantizer.downsample.0.1.dwconv.conv.bias",
                "quantizer.upsample.1.0.conv.weight", "quantizer.quantizer.quantizers.3.in_proj.weight_g", "encoder.block.4.block.5.freqs_cis", nil
            ])
    }

    /// Every checkpoint tensor lands on exactly one parameter of the Drama 3 codec, with its shape, and vice versa.
    @Test func checkpointKeysAndShapesMatchTheModuleTree() throws {
        let shapes = try Self.checkpointShapes()
        #expect(shapes.count == 538)
        let prefixes = FishS1Codec.weightNormPrefixes(shapes.keys)
        var expected: [String: [Int]] = [:]
        for (key, shape) in shapes {
            guard let mapped = FishS1Codec.sanitizedKey(key, weightNormPrefixes: prefixes) else { continue }
            #expect(expected[mapped] == nil, "two checkpoint tensors map to \(mapped)")
            expected[mapped] = shape
        }
        let codec = FishS1Codec(config: .drama3) // lazy initial arrays: shapes only, nothing evaluated
        let actual = Dictionary(uniqueKeysWithValues: codec.parameters().flattened().map { ($0.0, $0.1.shape) })
        #expect(Set(actual.keys).subtracting(expected.keys).sorted() == [])
        #expect(Set(expected.keys).subtracting(actual.keys).sorted() == [])
        let mismatched = expected.filter { actual[$0.key] != nil && actual[$0.key] != $0.value }.keys.sorted()
        #expect(mismatched == [])
        #expect(codec.numCodebooks == 10 && codec.frameLength == 2048 && codec.sampleRate == 44_100)
    }

    static func randomCodes(_ codec: FishS1Codec, frames: Int) -> MLXArray {
        let semantic = MLXRandom.randInt(low: 0, high: codec.semanticCodebookSize, [1, 1, frames])
        let residual = MLXRandom.randInt(low: 0, high: codec.codebookSize, [1, codec.numCodebooks - 1, frames])
        return concatenated([semantic, residual], axis: 1).asType(.int32)
    }

    /// Concatenated streaming steps equal one whole decode, for several chunkings, and again after a reset.
    @Test func streamingStepsMatchWholeDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(7)
            let codec = FishS1Codec(config: Self.tiny)
            eval(codec.parameters())
            let codes = Self.randomCodes(codec, frames: 7)
            let whole = codec.decode(codes)
            eval(whole)
            #expect(whole.shape == [7 * codec.frameLength])
            #expect(whole.abs().max().item(Float.self) > 0)
            for chunk in [1, 2, 4, 7] {
                codec.resetStreamingState()
                var pieces: [MLXArray] = []
                var start = 0
                while start < 7 {
                    let end = min(7, start + chunk)
                    let piece = codec.streamingStep(codes[0..., 0..., start..<end])
                    eval(piece)
                    #expect(piece.shape == [(end - start) * codec.frameLength])
                    pieces.append(piece)
                    start = end
                }
                let diff = (concatenated(pieces, axis: 0) - whole).abs().max().item(Float.self)
                #expect(diff < 1e-4, "chunk \(chunk): max |diff| \(diff)")
            }
        }
    }

    /// Folding weight norm once gives the same audio as normalising on every call.
    @Test func foldedWeightNormMatchesUnfolded() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(11)
            let codec = FishS1Codec(config: Self.tiny)
            eval(codec.parameters())
            let codes = Self.randomCodes(codec, frames: 3)
            let unfolded = codec.decode(codes)
            codec.foldWeightNorm(dtype: .float32)
            let folded = codec.decode(codes)
            #expect((folded - unfolded).abs().max().item(Float.self) < 1e-5)
        }
    }

    /// Loading casts to the requested dtype, keeps the RoPE tables float32, and verifies every key.
    @Test func loadCastsWeightsAndKeepsRopeTablesFloat32() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(3)
            let source = FishS1Codec(config: Self.tiny)
            eval(source.parameters())
            let checkpoint = Dictionary(uniqueKeysWithValues: source.parameters().flattened().map { ($0.0, $0.1) })
            let codec = FishS1Codec(config: Self.tiny)
            try codec.loadWeights(checkpoint, dtype: .float16)
            #expect(codec.dtype == .float16)
            for (key, value) in codec.parameters().flattened() {
                #expect(value.dtype == (key.hasSuffix("freqs_cis") ? .float32 : .float16), "\(key)")
            }
            let codes = Self.randomCodes(codec, frames: 2)
            let half = codec.decode(codes), full = source.decode(codes)
            #expect(half.dtype == .float32)
            #expect((half - full).abs().max().item(Float.self) < 5e-2)
        }
    }

    /// Encoding yields [10, ceil(samples / frameLength)] codes inside each codebook's range.
    @Test func encodeShapesAndRanges() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(5)
            let codec = FishS1Codec(config: Self.tiny)
            eval(codec.parameters())
            let count = codec.frameLength * 3 - 5
            let samples = (0..<count).map { Float(sin(Double($0) * 0.05)) * 0.5 }
            let codes = codec.encode(samples)
            eval(codes)
            #expect(codes.shape == [10, 3])
            #expect(codes.dtype == .int32)
            #expect(codes.min().item(Int32.self) >= 0)
            #expect(codes[0].max().item(Int32.self) < Int32(codec.semanticCodebookSize))
            #expect(codes[1...].max().item(Int32.self) < Int32(codec.codebookSize))
        }
    }
}
