// SPDX-License-Identifier: MIT
import Foundation
import Testing
import SoundEngine
import VireoCore
import StableAudio3
import MLX
import MLXNN
@testable import SoundWorker

@Test func exactDisablesInexactSoundComponents() {
    let exact = FastPath(overrides: ["VIREO_EXACT": "1"])
    #expect(!exact.enabled("VIREO_SA3_HALF"))
    #expect(!exact.enabled("VIREO_REUSE_METAL"))
    #expect(exact.enabled("VIREO_SA3_SWA_VIEWS"))
    #expect(exact.enabled("VIREO_SA3_F32_WEIGHTS"))
    #expect(FastPath().enabled("VIREO_SA3_HALF"))
    #expect(FastPath().enabled("VIREO_REUSE_METAL"))
}
@Test func inputChecksBeforeInference() throws {
    let r = try soundRequest(["op": "generate_sound", "prompt": "rain", "seed": 42, "duration": 2], model: "stable-audio-3-small-sfx", maxSeconds: 120)
    #expect(r.seconds == 2 && r.seed == 42 && r.steps == 8)
    #expect(throws: (any Error).self) { try soundRequest(["op": "generate_music", "prompt": "rain"], model: "stable-audio-3-small-sfx", maxSeconds: 120) }
    #expect(throws: (any Error).self) { try soundRequest(["prompt": "rain", "steps": 2.5], model: "stable-audio-3-medium", maxSeconds: 380) }
    #expect(throws: (any Error).self) { try soundRequest(["prompt": "rain", "steps": "invalid"], model: "stable-audio-3-medium", maxSeconds: 380) }
    #expect(throws: (any Error).self) { try soundRequest(["prompt": "rain", "duration": 121], model: "stable-audio-3-small-sfx", maxSeconds: 120) }
    for model in ["stable-audio-3-small-sfx", "stable-audio-3-medium", "re-use"] { try checkDtype("int4", model: model) }
    #expect(throws: (any Error).self) { try checkDtype("fp32", model: "stable-audio-3-small-sfx") }
    #expect(throws: (any Error).self) { try PCM.encode([Float.nan]) }
}
@Test func stereoWAVRoundTrip() throws {
    let directory = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-sound/wave-" + UUID().uuidString)
    let sink = try WaveSink(directory: directory, rate: 44100, channels: 2)
    try sink.append(PCM.encode([0.5, -0.5, 1, -1])); try sink.finish()
    let clip = try PCM.input(sink.file)
    #expect(clip.channels == 2 && clip.sampleRate == 44100 && clip.frames == 2)
    #expect(abs(clip.samples[0] - 0.5) < 0.001)
    #expect(clip.samples[1] < 0)
    #expect(try PCM.digest(sink.file).count == 64)
}

@Test(.heavy) func halfDecoderKeepsResidualsAndAttentionHalfWithFloatInput() throws {
    if let library = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: library) }
    try Device.withDefaultDevice(.cpu) {
        var config = SAMELConfig()
        config.latentDim = 4; config.dim = 8; config.numHeads = 2; config.headDim = 4
        config.ropeDims = 2; config.numBlocks = 1; config.ffInner = 16; config.outChannels = 4; config.sinStartBlock = 0
        let decoder = SAMELDecoder(config)
        let weights = Dictionary(uniqueKeysWithValues: decoder.parameters().flattened().map { ($0.0, $0.1.asType(.float16)) })
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        var traceTypes: [DType] = []
        decoder.activationTrace = { _, value in traceTypes.append(value.dtype) }
        let output = decoder(MLXArray.ones([1, 4, 1], dtype: .float32), fullAttention: true)
        eval(output)
        #expect(output.dtype == .float16 && traceTypes == [.float16])
        #expect(output.shape == [1, 4, 16])
        #expect(output.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
