// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import MLX
import MLXNN
@testable import MossFormer2SE
import SoundEngine
import StableAudio3
import Testing
@testable import SoundWorker

/// MossFormer2_SE_48K: weight-free checks against the maker golden (models/mossformer2-se-48k/gate/golden/), run on
/// MLX's CPU device, plus the opt-in component parity on the GPU against the maker's saved BF16 tensors.
private enum MF2Golden {
    struct Component: Decodable {
        let shape: [Int]; let step: Int; let count: Int; let values: [Double]
        let tolerance_rel_l2: Double; let tolerance_cosine_min: Double?
    }
    struct Fixture: Decodable { let samples: Int; let sha256_f32le: String }
    struct File: Decodable { let fixture: Fixture; let components: [String: Component] }

    static var folder: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("models/mossformer2-se-48k/gate/golden")
    }
    /// MLX loads its Metal library even for CPU-device work; tests point it at the build's copy.
    static func useTestMetallib() {
        let env = ProcessInfo.processInfo.environment
        if let library = env["ARSENAL_MLX_METALLIB"] ?? env["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: library) }
    }
    static func load() throws -> File { try JSONDecoder().decode(File.self, from: Data(contentsOf: folder.appendingPathComponent("golden.json"))) }

    /// The golden's fixture: the formula of ReUseEngine.testSignal at 48 kHz, one second.
    static func fixture() -> [Float] {
        var state: UInt32 = 20_260_926
        return (0..<48_000).map { i in
            state = state &* 1_664_525 &+ 1_013_904_223
            let t = Double(i) / 48_000
            let f0 = 140 + 40 * sin(2 * .pi * 1.3 * t)
            let voiced = (1...6).reduce(0.0) { $0 + sin(2 * .pi * f0 * Double($1) * t) / Double($1) }
            let envelope = 0.5 + 0.5 * sin(2 * .pi * 3 * t)
            return Float(0.2 * envelope * voiced + 0.05 * (Double(state >> 8) / Double(1 << 24) - 0.5))
        }
    }
    static func sha256(_ samples: [Float]) -> String {
        let bytes = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
    /// Relative L2 and cosine of `actual`, sampled at the golden's stride, against the maker values.
    static func compare(_ actual: [Float], _ component: Component) -> (relL2: Double, cosine: Double) {
        let sampled = Swift.stride(from: 0, to: actual.count, by: component.step).map { Double(actual[$0]) }
        guard sampled.count == component.values.count else { return (.infinity, 0) }
        var diff = 0.0, ref = 0.0, act = 0.0, dot = 0.0
        for (a, b) in zip(sampled, component.values) { diff += (a - b) * (a - b); ref += b * b; act += a * a; dot += a * b }
        return ((diff / max(ref, 1e-30)).squareRoot(), dot / max((ref * act).squareRoot(), 1e-30))
    }
}

@Test func mossFormer2PreAndPostProcessingMatchTheMakerGolden() throws {
    MF2Golden.useTestMetallib()
    let golden = try MF2Golden.load(), x = MF2Golden.fixture(), config = MossFormer2Config()
    #expect(x.count == golden.fixture.samples)
    #expect(MF2Golden.sha256(x) == golden.fixture.sha256_f32le)
    try Device.withDefaultDevice(.cpu) {
        let scaled = MLXArray(x.map { $0 * MossFormer2DSP.maxWav })
        // Features: Kaldi fbank with deltas, counts before values.
        let features = MossFormer2DSP.features(scaled, config: config)
        let f = try #require(golden.components["features"])
        #expect(features.shape == f.shape)
        let featureDelta = MF2Golden.compare(features.asArray(Float.self), f).relL2
        #expect(featureDelta <= f.tolerance_rel_l2)
        // STFT → synthetic mask → iSTFT, the maker's post-processing without weights.
        let spectrum = MossFormer2DSP.stft(scaled, config: config)
        #expect(spectrum.shape == [121, 961])
        let t = MLXArray((0..<spectrum.shape[0]).map { Float($0) }).expandedDimensions(axis: 1)
        let k = MLXArray((0..<961).map { Float($0) }).expandedDimensions(axis: 0)
        let post = MossFormer2DSP.istft(spectrum * (0.5 + 0.4 * sin(0.013 * k + 0.29 * t)), length: x.count, config: config).map { $0 / MossFormer2DSP.maxWav }
        let p = try #require(golden.components["post_synthetic"])
        #expect(post.count == p.count)
        let postDelta = MF2Golden.compare(post, p).relL2
        #expect(postDelta <= p.tolerance_rel_l2)
        // Over 20 s the maker decodes 4 s windows at a 3 s stride; a fake mask that follows each window's features.
        let long = Array([[Float]](repeating: x, count: 21).joined())
        let segmented = try MossFormer2DSP.decode(long, config: config) { features in
            let first = features[0, 0..., 0].expandedDimensions(axis: 1)
            return (0.5 + 0.4 * sin(0.011 * k + 0.05 * first)).expandedDimensions(axis: 0)
        }
        let s = try #require(golden.components["segmented_fake"])
        #expect(segmented.count == s.count)
        let segmentedDelta = MF2Golden.compare(segmented, s).relL2
        #expect(segmentedDelta <= s.tolerance_rel_l2)
        print("mossformer2 golden relative L2: features \(featureDelta), post \(postDelta), segmented \(segmentedDelta)")
    }
}

@Test func mossFormer2CheckpointMapsOntoEveryParameter() throws {
    let data = try Data(contentsOf: MF2Golden.folder.appendingPathComponent("checkpoint-keys.json"))
    let source = try JSONDecoder().decode([String: [Int]].self, from: data)
    #expect(source.count == MossFormer2Weights.sourceTensorCount)
    MF2Golden.useTestMetallib()
    try Device.withDefaultDevice(.cpu) {
        let expected = Dictionary(uniqueKeysWithValues: MossFormer2SE().parameters().flattened().map { ($0.0, $0.1.shape) })
        var mapped: [String: [Int]] = [:], buffers = 0
        for (name, shape) in source {
            guard let (key, layout) = MossFormer2Weights.map(name) else { buffers += 1; continue }
            #expect(mapped[key] == nil, "\(key) mapped twice")
            mapped[key] = MossFormer2Weights.arrange(MLXArray.zeros(shape), layout: layout).shape
        }
        #expect(buffers == 25)
        #expect(mapped.count == MossFormer2Weights.parameterCount && expected.count == MossFormer2Weights.parameterCount)
        #expect(Set(mapped.keys) == Set(expected.keys))
        for (key, shape) in expected { #expect(mapped[key] == shape, "\(key)") }
    }
}

@Test func mossFormer2ServesOneHalfDtypeAndOneCastRecipe() throws {
    try checkDtype("bf16", model: "mossformer2-se-48k")
    for dtype in ["fp16", "int8", "int4", "fp32"] { #expect(throws: (any Error).self) { try checkDtype(dtype, model: "mossformer2-se-48k") } }
    #expect(SoundEngines.halfCastRecipes["cast:mossformer2-se-48k"]?.dtype == "bf16")
    #expect(SoundEngines.engine(for: "mossformer2-se-48k") == "mossformer2")
    #expect(SoundEngines.engine(for: "re-use") == "re-use")
}

// Opt in: ARSENAL_MOSSFORMER2_WEIGHTS (a converted snapshot) and ARSENAL_MOSSFORMER2_REFERENCE (the maker capture with
// reference.npz, lab/evidence/mossformer2-se-48k/ref-2026-11-02), under scripts/mlock gpulock.
@Test(.heavy) func mossFormer2ComponentParityAgainstMakerBF16() throws {
    let env = ProcessInfo.processInfo.environment
    guard let weights = env["ARSENAL_MOSSFORMER2_WEIGHTS"], let reference = env["ARSENAL_MOSSFORMER2_REFERENCE"] else { return }
    MF2Golden.useTestMetallib()
    let golden = try MF2Golden.load(), config = MossFormer2Config()
    let net = try MossFormer2Weights.load(URL(fileURLWithPath: weights).appendingPathComponent(MossFormer2Weights.fileName), dtype: .bfloat16)
    let ref = try NPZArchive(url: URL(fileURLWithPath: reference).appendingPathComponent("reference.npz"))
    // Every component consumes the reference's own input: the maker's features, never this port's.
    var trace: [String: MLXArray] = [:]
    let mask = net.mask(try ref.array("features").expandedDimensions(axis: 0).asType(.bfloat16), trace: { trace[$0] = $1 })
    eval(mask)
    #expect(mask.dtype == .bfloat16)
    var report: [String: [String: Double]] = [:]
    func check(_ name: String, _ actual: MLXArray, bound: Double, as label: String? = nil) throws {
        let expected = try ref.array(name).asType(.float32), value = actual.asType(.float32)
        #expect(value.shape == expected.shape, "\(name) shape")
        let a = value.asArray(Float.self).map(Double.init), b = expected.asArray(Float.self).map(Double.init)
        var diff = 0.0, rr = 0.0, aa = 0.0, dot = 0.0
        for (x, y) in zip(a, b) { diff += (x - y) * (x - y); rr += y * y; aa += x * x; dot += x * y }
        let rel = (diff / rr).squareRoot(), cosine = dot / (rr * aa).squareRoot()
        report[label ?? name] = ["rel_l2": rel, "cosine": cosine, "count": Double(a.count), "finite": a.allSatisfy(\.isFinite) ? 1 : 0]
        #expect(a.allSatisfy(\.isFinite) && rel <= bound, "\(name) relative L2 \(rel)")
    }
    try check("encoder", try #require(trace["encoder"]), bound: 0.02)
    try check("layer0_flash", try #require(trace["layer0_flash"]), bound: 0.03)
    try check("layer0_fsmn", try #require(trace["layer0_fsmn"]), bound: 0.03)
    try check("layer11_fsmn", try #require(trace["layer11_fsmn"]), bound: 0.05)
    try check("layer23_fsmn", try #require(trace["layer23_fsmn"]), bound: 0.05)
    try check("intra_mdl", try #require(trace["intra_mdl"]), bound: 0.05)
    try check("intra_norm", try #require(trace["intra_norm"]), bound: 0.05)
    try check("computation_block", try #require(trace["computation_block"]), bound: 0.05)
    try check("mask", mask, bound: 0.03)
    // Isolated blocks: each consumes the maker's own input tensor, so its error is that block's alone.
    let block = net.model.mossformer.mdl.intraMdl.mossformerM, encoder = try ref.array("encoder").asType(.bfloat16)
    let position = net.model.mossformer.posEnc(encoder.transposed(0, 2, 1)).asType(.bfloat16)
    let flashInput = encoder.transposed(0, 2, 1) + position.expandedDimensions(axis: 0)
    try check("layer0_flash", block.layers[0](flashInput), bound: 0.03, as: "layer0_flash_isolated")
    try check("layer0_fsmn", block.fsmn[0](try ref.array("layer0_flash").asType(.bfloat16)), bound: 0.03, as: "layer0_fsmn_isolated")
    #expect((report["mask"]?["cosine"] ?? 0) >= 0.999)
    // End to end on the fixture: this port's DSP and network together.
    let fixture = try ref.array("fixture").asArray(Float.self)
    let wave = try MossFormer2DSP.decode(fixture, config: config) { net.mask($0.asType(.bfloat16)).asType(.float32) }
    try check("wave", MLXArray(wave), bound: 0.05)
    #expect(MF2Golden.compare(wave, try #require(golden.components["wave"])).relL2 <= 0.05)
    if let path = env["ARSENAL_MOSSFORMER2_EVIDENCE"] {
        try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted]).write(to: URL(fileURLWithPath: path))
    }
}

@Test func mossFormer2DecodeControlsFollowTheMakerDefaultsAndBounds() throws {
    let base = MossFormer2Config()
    #expect(base.oneTimeDecodeLength == 20 && base.decodeWindow == 4 && base.sampleRate == 48_000)
    let set = try MossFormer2Model.decodeConfig(base, options: ["one_time_decode_length": "60", "decode_window": "1"])
    #expect(set.oneTimeDecodeLength == 60 && set.decodeWindow == 1)
    for bad in [["one_time_decode_length": "0"], ["one_time_decode_length": "61"], ["decode_window": "21"], ["decode_window": "x"]] {
        #expect(throws: (any Error).self) { try MossFormer2Model.decodeConfig(base, options: bad) }
    }
}

/// Regression (2 Nov 2026): the port's intra norm was a per-frame LayerNorm, while the maker's is GroupNorm(1) over
/// channels and time; and GlobalLayerNorm rounded to the half dtype before its affine step (1.1% error at the encoder).
@Test func mossFormer2NormsMatchTheMakerDefinitions() throws {
    MF2Golden.useTestMetallib()
    try Device.withDefaultDevice(.cpu) {
        let net = MossFormer2SE()
        #expect(net.model.mossformer.mdl.intraNorm.weight?.shape == [512])
        #expect(net.model.mossformer.mdl.intraMdl.norm.eps == 1e-6)
        // GroupNorm(1) on [1, 2, 3]: one mean and variance over all six values, then a per-channel affine.
        let norm = GlobalLayerNorm(dim: 2, shape: 2, eps: 1e-8)
        try norm.update(parameters: ModuleParameters.unflattened(["weight": MLXArray([Float(2), 0.5]), "bias": MLXArray([Float(1000), -3])]), verify: .all)
        let values: [Double] = [1, 2, 3, 4, 5, 9]
        let mean = values.reduce(0, +) / 6, std = (values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / 6 + 1e-8).squareRoot()
        let exact = values.enumerated().map { i, v in (i < 3 ? 2 : 0.5) * (v - mean) / std + (i < 3 ? 1000 : -3) }
        let output = norm(MLXArray(values.map { Float($0) }, [1, 2, 3]).asType(.bfloat16))
        #expect(output.dtype == .bfloat16)
        // One rounding to BF16 (8 significant bits) of the exact result, even beside a large bias.
        for (a, e) in zip(output.asType(.float32).asArray(Float.self), exact) { #expect(abs(Double(a) - e) <= abs(e) * 0.004 + 1e-6) }
    }
}

/// Regression (2 Nov 2026 review): in the two short-clip padding branches (shorter than one window, and shorter than
/// a window plus a stride) the clip could end inside the last window's dropped edge, so its tail came back silent.
/// Every branch must keep the maker's windows and cover each input sample exactly once.
private let windowPlanCases: [(String, Int, Bool)] = [
    ("shorter than a window, tail in the dropped edge", 191_000, false),
    ("shorter than a window, inside the kept part", 162_000, false),
    ("window plus less than a stride, tail in the dropped edge", 335_000, false),
    ("window plus less than a stride, inside the kept part", 193_000, false),
    ("exact fit", 480_000, true),
    ("the maker's uneven padding", 485_000, false)
]

@Test(arguments: windowPlanCases)
func mossFormer2WindowPlanCoversEverySampleOnce(_ name: String, _ count: Int, _ exactFit: Bool) {
    let window = 4 * 48_000, stride = 144_000, giveUp = (window - stride) / 2
    let plan = MossFormer2DSP.windowPlan(count: count, window: window, stride: stride)
    // The maker's own padding, before any extra stride: its windows must all still be decoded.
    let maker: Int
    if count < window {
        maker = window
    } else if count < window + stride {
        maker = window + stride
    } else if (count - window) % stride != 0 {
        maker = 2 * count - (count - window) / stride * stride
    } else {
        maker = count + stride
    }
    let makerStarts = Array(Swift.stride(from: 0, through: maker - window, by: stride))
    #expect(Array(plan.starts.prefix(makerStarts.count)) == makerStarts, "\(name)")
    #expect(plan.padded >= count && plan.starts.allSatisfy { $0 + window <= plan.padded }, "\(name)")
    #expect(exactFit == (plan.padded == count + stride), "\(name)")
    // Kept ranges tile [0, end) with no gap or overlap, and reach past the clip's last sample.
    var end = 0
    for (n, start) in plan.starts.enumerated() {
        let lower = n == 0 ? 0 : start + giveUp
        #expect(lower == end, "\(name): gap or overlap at window \(n)")
        end = start + window - giveUp
    }
    #expect(end >= count, "\(name): samples \(end)..<\(count) would be silent")
}

/// The same branches through the real decode with an all-pass mask (no weights): the clip's last 10 ms must carry
/// signal, as the rest does. 1 s windows keep the clip short; MLX runs on the CPU device.
@Test func mossFormer2WindowedDecodeLeavesNoSilentTail() throws {
    MF2Golden.useTestMetallib()
    var config = MossFormer2Config()
    config.oneTimeDecodeLength = 1
    config.decodeWindow = 1
    let window = 48_000, stride = 36_000
    try Device.withDefaultDevice(.cpu) {
        for count in [window + 4_000, window + stride - 500, window + 2 * stride] {
            let audio = (0..<count).map { Float(sin(Double($0) * 0.05)) * 0.5 }
            let out = try MossFormer2DSP.decode(audio, config: config) { features in
                MLXArray.ones([1, features.dim(1), config.outChannelsFinal])
            }
            #expect(out.count == count)
            let tail = out.suffix(480).map { abs($0) }.max() ?? 0
            #expect(tail > 0.1, "count \(count): last 10 ms peak \(tail)")
        }
    }
}
