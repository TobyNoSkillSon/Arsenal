// SPDX-License-Identifier: MIT
import ArsenalWire
import CryptoKit
import Foundation
import MLX
import SoundEngine
import StableAudio3
import Testing
import VireoCore
@testable import SoundWorker
@testable import UniPASE

/// UniPASE: weight-free checks against the maker golden (models/unipase/gate/golden/), with MLX on its CPU device,
/// plus the opt-in component parity on the GPU against the maker's saved FP16 tensors.
private enum UPGolden {
    struct Component: Decodable {
        let count: Int?; let step: Int?; let values: [Double]?
        let tolerance_rel_l2: Double?; let tolerance_cosine_min: Double?; let tolerance_stft_magnitude_rel_l2: Double?
    }
    struct PLC: Decodable { let padded_length: Int; let mask: [Int] }
    struct Buckets: Decodable { let first: Int; let values: [Int] }
    struct Fixture: Decodable { let samples: Int; let sha256_f32le: String }
    struct File: Decodable {
        let fixture: Fixture; let components: [String: Component]; let network: [String: Component]
        let weight_free_fixtures: [String: FixtureEntry]
    }
    struct FixtureEntry: Decodable { let rate: Int?; let seconds: Double?; let samples: Int?; let sha256_f32le: String? }

    static var folder: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("models/unipase/gate/golden")
    }
    static func useTestMetallib() {
        let env = ProcessInfo.processInfo.environment
        if let library = env["ARSENAL_MLX_METALLIB"] ?? env["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: library)) }
    }
    static func data() throws -> Data { try Data(contentsOf: folder.appendingPathComponent("golden.json")) }
    static func load() throws -> File { try JSONDecoder().decode(File.self, from: data()) }
    static func raw(_ key: String) throws -> Data {
        let object = try JSONSerialization.jsonObject(with: data()) as? [String: Any]
        let components = try #require(object?["components"] as? [String: Any])
        return try JSONSerialization.data(withJSONObject: try #require(components[key]))
    }

    /// ReUseEngine.testSignal's formula at any rate.
    static func fixture(rate: Int, seconds: Double) -> [Float] {
        var state: UInt32 = 20_260_926
        return (0..<Int(Double(rate) * seconds)).map { i in
            state = state &* 1_664_525 &+ 1_013_904_223
            let t = Double(i) / Double(rate)
            let f0 = 140 + 40 * sin(2 * .pi * 1.3 * t)
            let voiced = (1...6).reduce(0.0) { $0 + sin(2 * .pi * f0 * Double($1) * t) / Double($1) }
            let envelope = 0.5 + 0.5 * sin(2 * .pi * 3 * t)
            return Float(0.2 * envelope * voiced + 0.05 * (Double(state >> 8) / Double(1 << 24) - 0.5))
        }
    }
    static func sha256(_ samples: [Float]) -> String {
        SHA256.hash(data: samples.withUnsafeBufferPointer { Data(buffer: $0) }).map { String(format: "%02x", $0) }.joined()
    }
    /// Relative L2 and cosine of `actual`, sampled at the golden's stride, against the maker values.
    static func compare(_ actual: [Float], _ c: Component) -> (relL2: Double, cosine: Double) {
        let sampled = Swift.stride(from: 0, to: actual.count, by: c.step ?? 1).map { Double(actual[$0]) }
        guard let values = c.values, sampled.count == values.count else { return (.infinity, 0) }
        var diff = 0.0, ref = 0.0, act = 0.0, dot = 0.0
        for (a, b) in zip(sampled, values) { diff += (a - b) * (a - b); ref += b * b; act += a * a; dot += a * b }
        return ((diff / max(ref, 1e-30)).squareRoot(), dot / max((ref * act).squareRoot(), 1e-30))
    }
}

@Test func unipaseFixturesMatchTheGolden() throws {
    let golden = try UPGolden.load()
    for (name, entry) in golden.weight_free_fixtures {
        guard let rate = entry.rate, let seconds = entry.seconds else { continue }
        let x = UPGolden.fixture(rate: rate, seconds: seconds)
        #expect(x.count == entry.samples, "\(name)")
        #expect(UPGolden.sha256(x) == entry.sha256_f32le, "\(name)")
    }
}

/// Every rate change the maker makes goes through torchaudio's sinc resampler; the port's must match it.
@Test(arguments: [(48_000, 16_000), (44_100, 16_000), (22_050, 16_000), (16_000, 48_000), (48_000, 44_100), (16_000, 8_000), (48_000, 24_000)])
func unipaseResamplerMatchesTorchaudio(_ from: Int, _ to: Int) throws {
    let golden = try UPGolden.load(), c = try #require(golden.components["resample_\(from)_\(to)"])
    let y = UniPASEDSP.resample(UPGolden.fixture(rate: from, seconds: 1), from: from, to: to)
    #expect(y.count == c.count)
    let delta = UPGolden.compare(y, c).relL2
    #expect(delta <= c.tolerance_rel_l2!, "resample \(from) → \(to): relative L2 \(delta)")
    print("unipase golden resample_\(from)_\(to) relative L2 \(delta)")
}

@Test func unipasePaddingAndPacketLossMaskMatchTheMaker() throws {
    let plc = try JSONDecoder().decode(UPGolden.PLC.self, from: UPGolden.raw("plc"))
    var x = UPGolden.fixture(rate: 48_000, seconds: 1)
    for i in 12_000..<16_800 { x[i] = 0 }
    for i in 33_000..<34_440 { x[i] = 0 }
    let wav = UniPASEDSP.padForEncoder(UniPASEDSP.resample(x, from: 48_000, to: 16_000))
    #expect(wav.count == plc.padded_length)
    let mask = try #require(UniPASEDSP.packetLossMask(wav))
    #expect(mask.map { $0 ? 1 : 0 } == plc.mask)
    #expect(mask.filter { $0 }.count > 3)
    // Padding: 320·⌊n/320⌋ + 80, cropping when n mod 320 > 80; no whole packet means no mask.
    #expect(UniPASEDSP.padForEncoder([Float](repeating: 1, count: 48_100)).count == 48_080)
    #expect(UniPASEDSP.padForEncoder([Float](repeating: 1, count: 48_000)).count == 48_080)
    #expect(UniPASEDSP.packetLossMask([Float](repeating: 0, count: 319)) == nil)
    // 317 near-zero samples of 320 is a lost packet; 316 is not.
    var packet = [Float](repeating: 0.5, count: 320)
    for i in 0..<316 { packet[i] = 0 }
    #expect(UniPASEDSP.packetLossMask(packet) == [false])
    packet[316] = 0
    #expect(UniPASEDSP.packetLossMask(packet) == [true])
}

@Test func unipaseRelativePositionBucketsMatchWavLM() throws {
    let buckets = try JSONDecoder().decode(UPGolden.Buckets.self, from: UPGolden.raw("buckets"))
    let ours = buckets.values.indices.map { UniPASENetwork.relativeBucket(buckets.first + $0) }
    #expect(ours == buckets.values)
}

@Test func unipaseVocoderISTFTMatchesTheMaker() throws {
    UPGolden.useTestMetallib()
    let golden = try UPGolden.load(), c = try #require(golden.components["vocos_istft"])
    try Device.withDefaultDevice(.cpu) {
        let t = 50
        var hidden = [Float](repeating: 0, count: t * 1282)
        for ti in 0..<t { for ch in 0..<1282 { hidden[ti * 1282 + ch] = Float(2 * sin(0.031 * Double(ch) + 0.17 * Double(ti)) + 0.002 * Double(ch) - 1) } }
        let y = UniPASEDSP.vocosISTFT(MLXArray(hidden, [t, 1282])).asArray(Float.self)
        #expect(y.count == t * 320 && y.count == c.count)
        let delta = UPGolden.compare(y, c).relL2
        #expect(delta <= c.tolerance_rel_l2!, "vocos ISTFT relative L2 \(delta)")
        print("unipase golden vocos_istft relative L2 \(delta)")
    }
}

/// The PostNet's processing around its network (normalise, 48 kHz STFT, band split and merge, crossfade, iSTFT,
/// resample, length, rescale), with the maker's network replaced by y = 0.3 x + 0.01 on both sides.
@Test(arguments: [48_000, 44_100, 24_000])
func unipasePostNetProcessingMatchesTheMaker(_ rateOut: Int) throws {
    UPGolden.useTestMetallib()
    let golden = try UPGolden.load(), c = try #require(golden.components["postnet_dsp_\(rateOut)"])
    try Device.withDefaultDevice(.cpu) {
        let y = UniPASEDSP.postNet(UPGolden.fixture(rate: 16_000, seconds: 1), rateOut: rateOut) { 0.3 * $0 + 0.01 }
        #expect(y.count == c.count)
        let delta = UPGolden.compare(y, c).relL2
        #expect(delta <= c.tolerance_rel_l2!, "PostNet DSP → \(rateOut): relative L2 \(delta)")
        print("unipase golden postnet_dsp_\(rateOut) relative L2 \(delta)")
    }
}

/// inference_long.py with a fake model whose output depends on the position within each segment, so a wrong
/// boundary or overlap shows.
@Test(arguments: [10.5, 12.0, 13.7, 16.5, 21.3])
func unipaseLongFormMatchesTheMaker(_ seconds: Double) throws {
    let golden = try UPGolden.load(), c = try #require(golden.components["long_\(seconds)"])
    let full = UPGolden.fixture(rate: 16_000, seconds: 21.3), clip = Array(full.prefix(Int(16_000 * seconds)))
    let parts = UniPASEDSP.longFormSegments(count: clip.count, rate: 16_000).map { range -> [Float] in
        let patch = Array(clip[range]), n = Float(patch.count)
        return patch.enumerated().map { i, v in v * (1 + Float(i) / n) + 0.001 * n / 16_000 }
    }
    let y = UniPASEDSP.matchPeak(UniPASEDSP.stitch(parts, rate: 16_000), to: clip)
    #expect(y.count == c.count)
    let delta = UPGolden.compare(y, c).relL2
    #expect(delta <= c.tolerance_rel_l2!, "long form \(seconds) s: relative L2 \(delta)")
    print("unipase golden long_\(seconds) relative L2 \(delta)")
}

/// Regression (2 Nov 2026): the maker's long-form script crashes on clips the loop leaves with no segment (under 6 s)
/// or one (6–10 s); the port then enhances the clip in one piece. Segments always cover the clip.
@Test func unipaseLongFormSegmentsCoverEveryClip() {
    for seconds in [0.5, 5.9, 6.5, 9.9, 10.0, 10.1, 12.0, 30.0, 600.0] {
        let n = Int(16_000 * seconds), segments = UniPASEDSP.longFormSegments(count: n, rate: 16_000)
        #expect(segments.first?.lowerBound == 0 && segments.last?.upperBound == n, "\(seconds) s")
        for (a, b) in zip(segments, segments.dropFirst()) { #expect(b.lowerBound == a.lowerBound + 64_000, "\(seconds) s") }
        if seconds <= 10 { #expect(segments.count == 1, "\(seconds) s") }
    }
    // At another output rate the stitch keeps that rate's seconds: 3 segments of a 16 s clip at 48 kHz out.
    let parts = [[Float](repeating: 1, count: 8 * 48_000), [Float](repeating: 2, count: 8 * 48_000), [Float](repeating: 3, count: 8 * 48_000)]
    let y = UniPASEDSP.stitch(parts, rate: 48_000)
    #expect(y.count == 16 * 48_000)
    #expect(y[6 * 48_000 - 1] == 1 && y[6 * 48_000] == 2 && y[10 * 48_000 - 1] == 2 && y[10 * 48_000] == 3)
}

@Test func unipaseUtteranceNormIsOverTimeAndChannels() throws {
    UPGolden.useTestMetallib()
    try Device.withDefaultDevice(.cpu) {
        let values: [Float] = [1, 2, 3, 4, 5, 9]
        let y = UniPASEDSP.utteranceNorm(MLXArray(values, [1, 2, 3]).asType(.float16)).asType(.float32).asArray(Float.self)
        let mean = values.reduce(0, +) / 6, std = (values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / 6 + 1e-6).squareRoot()
        for (a, v) in zip(y, values) { #expect(abs(a - (v - mean) / std) <= 2e-3) }
    }
    #expect(UniPASEDSP.outputLength(44_100, rateIn: 44_100, rateOut: 48_000) == 48_000)
    #expect(UniPASEDSP.outputLength(1_001, rateIn: 16_000, rateOut: 44_100) == 2_759)
    #expect(UniPASEDSP.matchPeak([0.5, -2], to: [0.25, 1]).map { ($0 * 1000).rounded() } == [250, -1000])
}

@Test func unipaseCheckpointMapsOntoEveryParameter() throws {
    let data = try Data(contentsOf: UPGolden.folder.appendingPathComponent("checkpoint-keys.json"))
    let source = try JSONDecoder().decode([String: [String: [Int]]].self, from: data)
    #expect(source.mapValues(\.count) == UniPASEWeights.sourceTensorCounts)
    UPGolden.useTestMetallib()
    try Device.withDefaultDevice(.cpu) {
        var mapped: [String: [Int]] = [:], handled = 0
        for (file, tensors) in source {
            for (name, shape) in tensors {
                guard let (key, layout) = UniPASEWeights.map(name, file: file) else { handled += 1; continue }
                #expect(mapped[key] == nil, "\(key) mapped twice")
                mapped[key] = UniPASEWeights.arrange(MLXArray.zeros(shape), layout: layout).shape
            }
        }
        // weight_g + weight_v fold into one weight; the encoder's unused final LayerNorm and the vocoder window drop.
        #expect(handled == 5)
        let g = try #require(source["DeWavLM-Omni.pt"]?["encoder.pos_conv.0.weight_g"]), v = try #require(source["DeWavLM-Omni.pt"]?["encoder.pos_conv.0.weight_v"])
        mapped["wavlm.encoder.pos_conv.0.weight"] = UniPASEWeights.foldWeightNorm(g: MLXArray.ones(g), v: MLXArray.ones(v)).shape
        try UniPASEWeights.check(mapped)
        #expect(mapped.count == UniPASEWeights.parameterCount)
    }
}

/// A transposed convolution at stride 1 is a convolution with the kernel reversed and k - 1 padding: the cast's
/// layout must give exactly torch's ConvTranspose1d on a tiny case (computed by hand here).
@Test func unipaseTransposedConvolutionLayout() throws {
    UPGolden.useTestMetallib()
    try Device.withDefaultDevice(.cpu) {
        // torch weight [in 1, out 1, k 2] = [1, 10]; input [1, 2, 3] → transposed conv output [1, 12, 23, 30].
        let w = UniPASEWeights.arrange(MLXArray([Float(1), 10], [1, 1, 2]), layout: .transposed1d)
        let x = padded(MLXArray([Float(1), 2, 3], [1, 3, 1]), widths: [.init(0), .init((1, 1)), .init(0)])
        #expect(conv1d(x, w).reshaped(-1).asArray(Float.self) == [1, 12, 23, 30])
        // Weight norm: g · v / ‖v‖ over all but the kernel axis.
        let folded = UniPASEWeights.foldWeightNorm(g: MLXArray([Float(2)], [1, 1, 1]), v: MLXArray([Float(3), 4], [2, 1, 1]))
        #expect(folded.reshaped(-1).asArray(Float.self) == [1.2, 1.6])
    }
}

@Test func unipaseServesFP16AndOneCastRecipe() throws {
    try checkDtype("fp16", model: "unipase")
    for dtype in ["bf16", "int8", "int4", "fp32"] { #expect(throws: (any Error).self) { try checkDtype(dtype, model: "unipase") } }
    #expect(SoundEngines.halfCastRecipes["cast:unipase"]?.dtype == "fp16")
    #expect(SoundEngines.engine(for: "unipase") == "unipase")
    #expect(SoundEngines.randomWeightModels.contains("unipase"))
}

@Test func unipaseRequestPlanFollowsTheMakerControls() throws {
    let plain = try UniPASEEngine.plan(inputRate: 44_100, mode: "denoise", options: [:])
    #expect(plain == .init(rateOut: 44_100, plc: true, onePassSeconds: 20))
    #expect(try UniPASEEngine.plan(inputRate: 16_000, mode: "restore", options: ["rate": "48000", "enable_plc": "0"]) == .init(rateOut: 48_000, plc: false, onePassSeconds: 20))
    #expect(try UniPASEEngine.plan(inputRate: 16_000, mode: "upscale", options: [:]).rateOut == 48_000)
    #expect(
        try UniPASEEngine.plan(inputRate: 48_000, mode: "denoise", options: ["rate": "8000", "one_time_decode_length": "60"])
            == .init(rateOut: 8_000, plc: true, onePassSeconds: 60))
    for (mode, options) in [
        ("upscale", ["rate": "16000"]), ("separate", [:]), ("denoise", ["rate": "12345"]), ("denoise", ["enable_plc": "yes"]),
        ("denoise", ["one_time_decode_length": "0"]), ("denoise", ["one_time_decode_length": "61"])
    ] {
        // A caller's mistake returns 422 (invalid_request), not 500 (2 Nov 2026 review).
        let error = #expect(throws: WorkerFailure.self) { try UniPASEEngine.plan(inputRate: 44_100, mode: mode, options: options) }
        #expect(error?.code == "invalid_request", "\(mode) \(options)")
    }
}

// Opt in: ARSENAL_UNIPASE_WEIGHTS (a converted snapshot) and ARSENAL_UNIPASE_REFERENCE (the maker capture with
// reference.npz, lab/evidence/unipase/ref-2026-11-02), under scripts/mlock gpulock. ARSENAL_UNIPASE_EVIDENCE: report path.
@Test(.heavy) func unipaseComponentParityAgainstMakerFP16() throws {
    let env = ProcessInfo.processInfo.environment
    guard let weights = env["ARSENAL_UNIPASE_WEIGHTS"], let reference = env["ARSENAL_UNIPASE_REFERENCE"] else { return }
    UPGolden.useTestMetallib()
    let golden = try UPGolden.load()
    let net = UniPASENetwork(weights: try UniPASEWeights.load(URL(fileURLWithPath: weights).appendingPathComponent(UniPASEWeights.fileName), dtype: .float16), dtype: .float16)
    let ref = try NPZArchive(url: URL(fileURLWithPath: reference).appendingPathComponent("reference.npz"))
    var report: [String: [String: Double]] = [:]
    func metrics(_ a: [Double], _ b: [Double]) -> (Double, Double) {
        var diff = 0.0, rr = 0.0, aa = 0.0, dot = 0.0
        for (x, y) in zip(a, b) { diff += (x - y) * (x - y); rr += y * y; aa += x * x; dot += x * y }
        return ((diff / rr).squareRoot(), dot / (rr * aa).squareRoot())
    }
    func check(_ name: String, _ actual: MLXArray, bound: Double, cosine: Double? = nil, as label: String? = nil, ctf: Bool = false) throws {
        // ctf: the maker tensor is [C, T, F] (PyTorch NCHW); this port keeps [T, F, C].
        let stored = try ref.array(name).asType(.float32), expected = ctf ? stored.transposed(1, 2, 0) : stored, value = actual.asType(.float32)
        #expect(value.shape == expected.shape, "\(label ?? name) shape \(value.shape) vs \(expected.shape)")
        guard value.shape == expected.shape else { return }
        let a = value.asArray(Float.self).map(Double.init), b = expected.asArray(Float.self).map(Double.init)
        let (rel, cos) = metrics(a, b)
        report[label ?? name] = ["rel_l2": rel, "cosine": cos, "count": Double(a.count), "finite": a.allSatisfy(\.isFinite) ? 1 : 0]
        #expect(a.allSatisfy(\.isFinite) && rel <= bound, "\(label ?? name) relative L2 \(rel) > \(bound)")
        if let cosine { #expect(cos >= cosine, "\(label ?? name) cosine \(cos)") }
    }
    func ref16(_ name: String) throws -> MLXArray { try ref.array(name).asType(.float16).expandedDimensions(axis: 0) }
    var trace: [String: MLXArray] = [:]
    let record: UniPASENetwork.Trace = { name, value in
        eval(value); trace[name] = value
    }
    // Encoder front from the maker's padded 16 kHz signal and mask.
    let wav16 = try ref.array("wav16").asArray(Float.self)
    #expect(UniPASEDSP.padForEncoder(UniPASEDSP.resample(try ref.array("fixture").asArray(Float.self), from: 48_000, to: 16_000)).count == wav16.count)
    let mask = try ref.array("plc_mask").asArray(Float.self).map { $0 > 0.5 }
    #expect(UniPASEDSP.packetLossMask(wav16) == mask)
    try check("position_bias", net.positionBias(wav16.count / 320)[0], bound: 1e-3)
    _ = net.encode(wav16, mask: mask, trace: record)
    try check("cnn", try #require(trace["cnn"]), bound: 0.01)
    try check("proj", try #require(trace["proj"]), bound: 0.01)
    // Transformer from the maker's masked projection; layer 1 and layer 24 also on their own maker inputs.
    trace = [:]
    let (l1, l24) = net.encodeLayers(try ref16("masked"), trace: record)
    try check("pos", try #require(trace["pos"]), bound: 0.01)
    try check("layer1", try #require(trace["layer1"]), bound: 0.01)
    try check("layer12", try #require(trace["layer12"]), bound: 0.03)
    try check("layer24", try #require(trace["layer24"]), bound: 0.05)
    try check("L1", l1[0], bound: 0.02)
    try check("L24", l24[0], bound: 0.05, cosine: 0.999)
    try check("layer1", net.layer(try ref16("pos"), index: 0)[0], bound: 0.01, as: "layer1_isolated")
    try check("layer24", net.layer(try ref16("layer23"), index: 23)[0], bound: 0.05, as: "layer24_isolated")
    // PLC off.
    let (o1, o24) = net.encode(wav16, mask: nil)
    try check("L1_noplc", o1[0], bound: 0.02)
    try check("L24_noplc", o24[0], bound: 0.05, cosine: 0.999)
    // Adapter from the maker's L1 and L24.
    trace = [:]
    let adapted = net.adapt(l1: try ref16("L1"), l24: try ref16("L24"), trace: record)
    try check("adapter_embed", try #require(trace["adapter_embed"]), bound: 0.01)
    try check("adapter_posnet", try #require(trace["adapter_posnet"]), bound: 0.02)
    try check("adapter_convnext11", try #require(trace["adapter_convnext11"]), bound: 0.02)
    try check("adapter", adapted[0], bound: 0.02, cosine: 0.9995)
    // Vocoder from the maker's adapter output; its ISTFT also from the maker's own hidden tensor.
    let hidden = net.vocoderHidden(try ref16("adapter"))
    try check("vocoder_hidden", hidden, bound: 0.05)
    try check("vocoder_wav", UniPASEDSP.vocosISTFT(hidden), bound: 0.1, cosine: 0.995)
    try check("vocoder_wav", UniPASEDSP.vocosISTFT(try ref.array("vocoder_hidden")), bound: 1e-4, as: "vocoder_istft_isolated")
    // PostNet from the maker's 16 kHz waveform; its network also from the maker's band-split input.
    trace = [:]
    let post = UniPASEDSP.postNet(try ref.array("vocoder_wav").asArray(Float.self), rateOut: 48_000) { bands in
        let out = net.postNetwork(bands, trace: record); eval(out); return out
    }
    try check("postnet_conv", try #require(trace["postnet_conv"]), bound: 0.01, ctf: true)
    try check("postnet_block0", try #require(trace["postnet_block0"]), bound: 0.02, ctf: true)
    try check("postnet_block4", try #require(trace["postnet_block4"]), bound: 0.05, ctf: true)
    try check("postnet_deconv", try #require(trace["postnet_deconv"]), bound: 0.05, ctf: true)
    try check("postnet_out", MLXArray(post), bound: 0.01)
    try check("postnet_deconv", net.postNetwork(try ref.array("postnet_bands").transposed(1, 2, 0)), bound: 0.05, as: "postnet_network_isolated", ctf: true)
    // End to end on the fixture: this port's DSP and networks together, then the scripts' peak match.
    let fixture = try ref.array("fixture").asArray(Float.self)
    let output = UniPASEDSP.matchPeak(try net.enhance(fixture, rateIn: 48_000, rateOut: 48_000, plc: true), to: fixture)
    #expect(output.count == fixture.count)
    let makerOut = try ref.array("output")
    let magnitude = { (x: MLXArray) in abs(UniPASEDSP.stft(x, nFFT: 1024, hop: 256)).asArray(Float.self).map(Double.init) }
    let (spectral, spectralCosine) = metrics(magnitude(MLXArray(output)), magnitude(makerOut))
    let (wave, waveCosine) = metrics(output.map(Double.init), makerOut.asArray(Float.self).map(Double.init))
    report["output"] = ["stft_magnitude_rel_l2": spectral, "stft_magnitude_cosine": spectralCosine, "wave_rel_l2": wave, "wave_cosine": waveCosine]
    let bound = try #require(golden.network["output"]?.tolerance_stft_magnitude_rel_l2)
    #expect(output.allSatisfy(\.isFinite) && spectral <= bound, "end to end STFT magnitude relative L2 \(spectral) > \(bound)")
    if let path = env["ARSENAL_UNIPASE_EVIDENCE"] {
        try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted]).write(to: URL(fileURLWithPath: path))
    }
}

/// Review (2 Nov 2026): an odd rate whose reduced ratio is large (44,099 Hz) would build a 706-million-float sinc
/// kernel. Such rates go through AVAudioConverter; every pair the family serves stays within the kernel limit.
@Test func unipaseOddRatesTakeTheConverterNotAHugeSincKernel() throws {
    #expect(UniPASEDSP.sincKernelSize(from: 44_099, to: 16_000) > UniPASEDSP.maxSincKernel)
    #expect(!UniPASEDSP.sincFits(from: 44_099, to: 16_000) && !UniPASEDSP.sincFits(from: 48_000, to: 44_099))
    let rates = [8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000]
    for a in rates { for b in [8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000] { #expect(UniPASEDSP.sincFits(from: a, to: b), "\(a) → \(b)") } }
    // The network refuses an unconverted odd rate rather than building the kernel (no weights are touched).
    let net = UniPASENetwork(weights: [:], dtype: .float16)
    #expect(throws: UniPASEError.self) { try net.enhance([Float](repeating: 0, count: 4_410), rateIn: 44_099, rateOut: 16_000, plc: true) }
    // The engine converts in and out: the fake network sees 16 kHz in and 48 kHz out, the caller gets 44,099 Hz.
    let x = UPGolden.fixture(rate: 44_099, seconds: 1)
    var seen: [(Int, Int)] = []
    let plan = try UniPASEEngine.plan(inputRate: 44_099, mode: "denoise", options: [:])
    let out = try UniPASEModel.process(AudioClip(samples: x, sampleRate: 44_099, channels: 1), plan: plan, control: GenerationControl()) { s, rin, rout, _ in
        seen.append((rin, rout)); return UniPASEDSP.resample(s, from: rin, to: rout)
    }
    #expect(seen.count == 1 && seen[0] == (16_000, 48_000))
    #expect(out.sampleRate == 44_099, "rate \(out.sampleRate)")
    #expect(out.frames == 44_099, "frames \(out.frames)")
    let finite = out.samples.allSatisfy { $0.isFinite }
    #expect(finite)
}

@Test func unipasePlanRefusesUnknownOptions() throws {
    #expect(throws: (any Error).self) { try UniPASEEngine.plan(inputRate: 48_000, mode: "denoise", options: ["decode_window": "4"]) }
    #expect(throws: (any Error).self) { try UniPASEEngine.plan(inputRate: 48_000, mode: "denoise", options: ["enable_PLC": "1"]) }
    #expect(try UniPASEEngine.plan(inputRate: 48_000, mode: "denoise", options: ["rate": "16000", "enable_plc": "1", "one_time_decode_length": "5"]).rateOut == 16_000)
}

/// The vocoder head clips log-magnitudes to [-20, 5] before exp: values past either bound give the bound's output.
@Test func unipaseVocoderClipsLogMagnitudeAtBothBounds() throws {
    UPGolden.useTestMetallib()
    try Device.withDefaultDevice(.cpu) {
        let t = 8
        func run(low: Float, high: Float) -> [Float] {
            var h = [Float](repeating: 0, count: t * 1282)
            for ti in 0..<t {
                for ch in 0..<641 { h[ti * 1282 + ch] = ch % 2 == 0 ? low : high }
                for ch in 641..<1282 { h[ti * 1282 + ch] = Float(sin(0.37 * Double(ch) + Double(ti))) * 3 }
            }
            return UniPASEDSP.vocosISTFT(MLXArray(h, [t, 1282])).asArray(Float.self)
        }
        let atBounds = run(low: -20, high: 5), past = run(low: -60, high: 40), inside = run(low: -19, high: 4)
        #expect(past == atBounds)
        #expect(past.allSatisfy(\.isFinite) && past.contains { $0 != 0 })
        #expect(inside != atBounds)
    }
}

/// The torch periodic Hann window in float32: symmetric about n/2, 0 at 0, 1 at n/2.
@Test func unipaseHannIsTorchsFloat32Window() {
    let w = UniPASEDSP.hann(1280)
    #expect(w.count == 1280 && w[0] == 0 && w[640] == 1)
    for k in 1..<640 { #expect(abs(w[k] - w[1280 - k]) <= 1e-6) }
    let step = Float(2 * Double.pi / 1280)
    #expect(w[123] == cos(Float(123) * step) * -0.5 + 0.5)
}

/// The engine's plumbing with a fake network: channels are enhanced separately and peak-matched to their own input,
/// long clips are cut into the maker's segments, stitched at the output rate, and fitted to the input's length.
@Test func unipaseEnginePlumbingWithAFakeNetwork() throws {
    let left = UPGolden.fixture(rate: 16_000, seconds: 21.3), right = left.map { $0 * 0.25 }
    var interleaved = [Float](repeating: 0, count: left.count * 2)
    for i in left.indices { interleaved[2 * i] = left[i]; interleaved[2 * i + 1] = right[i] }
    let clip = AudioClip(samples: interleaved, sampleRate: 16_000, channels: 2)
    var calls: [(count: Int, rin: Int, rout: Int, plc: Bool)] = []
    let fake: ([Float], Int, Int, Bool) throws -> [Float] = { s, rin, rout, plc in
        calls.append((s.count, rin, rout, plc))
        return UniPASEDSP.resample(s, from: rin, to: rout).map { $0 * 3 }
    }
    // Long form at 48 kHz out: 21.3 s → segments [0, 8), [4, 12), [8, 16), [12, 21.3) per channel.
    let plan = try UniPASEEngine.plan(inputRate: 16_000, mode: "upscale", options: ["enable_plc": "0", "one_time_decode_length": "20"])
    let out = try UniPASEModel.process(clip, plan: plan, control: GenerationControl(), segment: fake)
    #expect(calls.count == 8 && calls.allSatisfy { $0.rin == 16_000 && $0.rout == 48_000 && !$0.plc })
    #expect(calls.prefix(4).map(\.count) == [128_000, 128_000, 128_000, 148_800])
    #expect(out.channels == 2 && out.sampleRate == 48_000 && out.frames == left.count * 3)
    let peaks = (0..<2).map { c in stride(from: c, to: out.samples.count, by: 2).reduce(Float(0)) { max($0, abs(out.samples[$1])) } }
    let inPeaks = [left, right].map { $0.reduce(Float(0)) { max($0, abs($1)) } }
    for c in 0..<2 { #expect(abs(peaks[c] - inPeaks[c]) <= 1e-4) }
    // One pass under the limit: one call per channel, same rate and length.
    calls = []
    let short = AudioClip(samples: Array(interleaved.prefix(32_000)), sampleRate: 16_000, channels: 2)
    let one = try UniPASEModel.process(short, plan: try UniPASEEngine.plan(inputRate: 16_000, mode: "denoise", options: [:]), control: GenerationControl(), segment: fake)
    #expect(calls.count == 2 && calls.allSatisfy { $0.count == 16_000 && $0.plc } && one.frames == 16_000 && one.sampleRate == 16_000)
    // Cancellation before a segment stops the request.
    let control = GenerationControl(); control.cancel()
    #expect(throws: CancellationError.self) { try UniPASEModel.process(short, plan: plan, control: control, segment: fake) }
}

/// Review (2 Nov 2026): cancellation is checked before each network stage, so a long one-pass request stops between
/// the encoder, adapter, vocoder and PostNet. The first check comes before any weight is read.
@Test func unipaseNetworkChecksForCancellationBeforeTheEncoder() {
    let net = UniPASENetwork(weights: [:], dtype: .float16)
    var stages: [String] = []
    #expect(throws: CancellationError.self) {
        try net.enhance([Float](repeating: 0.1, count: 16_000), rateIn: 16_000, rateOut: 48_000, plc: true) {
            stages.append($0); throw CancellationError()
        }
    }
    #expect(stages == ["encoder"])
    #expect(UniPASENetwork.stages == ["encoder", "adapter", "vocoder", "postnet"])
}

// Opt in, as the parity test (ARSENAL_UNIPASE_WEIGHTS), under scripts/mlock gpulock: FP16 range on real weights
// with an awkward minute, 60 s of near-silence (1e-5 noise, so most packets are not digital silence) and a 1 s
// full-scale clipped burst at 30 s, in one pass to 48 kHz. Every traced activation must stay finite and inside FP16.
@Test(.heavy) func unipaseFP16RangeOnAMinuteOfSilenceAndABurst() throws {
    let env = ProcessInfo.processInfo.environment
    guard let weights = env["ARSENAL_UNIPASE_WEIGHTS"] else { return }
    UPGolden.useTestMetallib()
    let net = UniPASENetwork(weights: try UniPASEWeights.load(URL(fileURLWithPath: weights).appendingPathComponent(UniPASEWeights.fileName), dtype: .float16), dtype: .float16)
    var state: UInt32 = 7
    func noise() -> Float { state = state &* 1_664_525 &+ 1_013_904_223; return Float(state >> 8) / Float(1 << 24) - 0.5 }
    var x = (0..<(60 * 48_000)).map { _ in 2e-5 * noise() }
    for i in (30 * 48_000)..<(31 * 48_000) { x[i] = max(-1, min(1, 4 * noise())) }
    var peaks: [String: Float] = [:], stages: [String] = []
    let y = try net.enhance(
        x, rateIn: 48_000, rateOut: 48_000, plc: true,
        trace: { name, value in
            peaks[name] = abs(value.asType(.float32)).max().item(Float.self)
        },
        stage: { stages.append($0) })
    #expect(stages == UniPASENetwork.stages)
    #expect(y.count == x.count && y.allSatisfy(\.isFinite))
    #expect(peaks.count >= 20)
    for (name, peak) in peaks.sorted(by: { $0.key < $1.key }) { #expect(peak.isFinite && peak < 65_504, "\(name) peak \(peak)") }
    let worst = peaks.max { $0.value < $1.value }!
    print("unipase fp16 range: \(peaks.count) traced tensors, largest \(worst.key) \(worst.value)")
    if let path = env["ARSENAL_UNIPASE_RANGE_EVIDENCE"] {
        try JSONSerialization.data(withJSONObject: peaks.mapValues(Double.init), options: [.sortedKeys, .prettyPrinted]).write(to: URL(fileURLWithPath: path))
    }
}
