// SPDX-License-Identifier: MIT
import CryptoKit
import DeepFilterNet3
import Foundation
import MLX
import SoundEngine
import StableAudio3
import Testing
@testable import SoundWorker

/// DeepFilterNet3: weight-free checks on MLX's CPU device against the maker golden (models/deepfilternet3/gate/golden/),
/// and the opt-in parity of the GPU path (Metal GRU and running-norm kernels included) against the maker's FP16 capture.
private enum DFGolden {
    struct Component: Decodable { let shape: [Int]; let step: Int; let count: Int; let values: [Double]; let tolerance_rel_l2: Double }
    struct Fixture: Decodable { let samples: Int; let sha256_f32le: String }
    struct Weights: Decodable { let pinned_tensors: Int; let bit_identical_tensors: Int; let max_abs_diff: Double; let parameters: Int }
    struct File: Decodable { let fixture: Fixture; let weights_check: Weights; let weight_free: [String: Component]; let network: [String: Component] }

    static var folder: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("models/deepfilternet3/gate/golden")
    }
    static func load() throws -> File { try JSONDecoder().decode(File.self, from: Data(contentsOf: folder.appendingPathComponent("golden.json"))) }
    static func useTestMetallib() {
        let env = ProcessInfo.processInfo.environment
        if let library = env["ARSENAL_MLX_METALLIB"] ?? env["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: library) }
    }
    /// The shared one-second fixture (ReUseEngine.testSignal's formula at 48 kHz).
    static func fixture() -> [Float] {
        var state: UInt32 = 20_260_926
        return (0..<48_000).map { i in
            state = state &* 1_664_525 &+ 1_013_904_223
            let t = Double(i) / 48_000, f0 = 140 + 40 * sin(2 * .pi * 1.3 * t)
            let voiced = (1...6).reduce(0.0) { $0 + sin(2 * .pi * f0 * Double($1) * t) / Double($1) }
            return Float(0.2 * (0.5 + 0.5 * sin(2 * .pi * 3 * t)) * voiced + 0.05 * (Double(state >> 8) / Double(1 << 24) - 0.5))
        }
    }
    static func relL2(_ actual: [Float], _ c: Component) -> Double {
        let sampled = Swift.stride(from: 0, to: actual.count, by: c.step).map { Double(actual[$0]) }
        guard actual.count == c.count, sampled.count == c.values.count else { return .infinity }
        var diff = 0.0, ref = 0.0
        for (a, b) in zip(sampled, c.values) { diff += (a - b) * (a - b); ref += b * b }
        return (diff / max(ref, 1e-30)).squareRoot()
    }
    /// libdf's running mean, as plain sequential Swift (the test's own reference for the GPU kernel's formula).
    static func runningMean(_ x: [Float], columns: Int, initial: [Float], alpha: Float) -> [Float] {
        var state = initial, out = x
        for t in 0..<(x.count / columns) {
            for c in 0..<columns { state[c] = x[t * columns + c] * (1 - alpha) + state[c] * alpha; out[t * columns + c] = state[c] }
        }
        return out
    }
    static func spectrumPair(_ spectrum: MLXArray) -> [Float] {
        stacked([spectrum.realPart(), spectrum.imaginaryPart()], axis: -1).asArray(Float.self)
    }
}

@Test func deepFilterNet3SignalProcessingMatchesTheMakerGolden() throws {
    DFGolden.useTestMetallib()
    let golden = try DFGolden.load(), x = DFGolden.fixture(), dsp = DeepFilterNetDSP()
    let bytes = x.withUnsafeBufferPointer { Data(buffer: $0) }
    #expect(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() == golden.fixture.sha256_f32le)
    #expect(dsp.alpha == 0.99 && dsp.erbWidths.count == 32 && dsp.erbWidths.reduce(0, +) == 481)
    try Device.withDefaultDevice(.cpu) {
        let wf = golden.weight_free
        // Counts before values: 102 frames of 481 bins for one second padded by one FFT.
        let spectrum = dsp.analysis(x)
        #expect(spectrum.shape == [102, 481])
        let spec = try #require(wf["spec"])
        #expect(DFGolden.relL2(DFGolden.spectrumPair(spectrum), spec) <= spec.tolerance_rel_l2)
        // Features: the band and magnitude math here, libdf's running normalisation in plain Swift.
        func features(_ audio: [Float]) -> (erb: [Float], complex: [Float]) {
            let s = dsp.analysis(audio)
            let db = dsp.erbDecibels(s).asArray(Float.self)
            let erb = zip(db, DFGolden.runningMean(db, columns: 32, initial: dsp.erbInitial, alpha: dsp.alpha)).map { ($0 - $1) / 40 }
            let low = s[0..., 0..<96], re = low.realPart().asArray(Float.self), im = low.imaginaryPart().asArray(Float.self)
            let state = DFGolden.runningMean(zip(re, im).map { ($0 * $0 + $1 * $1).squareRoot() }, columns: 96, initial: dsp.complexInitial, alpha: dsp.alpha)
            var complex: [Float] = []
            for i in re.indices { complex += [re[i] / state[i].squareRoot(), im[i] / state[i].squareRoot()] }
            return (erb, complex)
        }
        for (audio, prefix) in [(x, ""), (Array([[Float]](repeating: x, count: 120).joined()), "long_")] {
            let (erb, complex) = features(audio)
            let e = try #require(wf[prefix + "erb_feat"]), c = try #require(wf[prefix + "spec_feat"])
            #expect(DFGolden.relL2(erb, e) <= e.tolerance_rel_l2, "\(prefix)erb_feat")
            #expect(DFGolden.relL2(complex, c) <= c.tolerance_rel_l2, "\(prefix)spec_feat")
        }
        // Synthesis of a synthetically masked spectrum, trimmed by the maker's delay.
        let t = MLXArray((0..<102).map { Float($0) }).expandedDimensions(axis: 1), f = MLXArray((0..<481).map { Float($0) }).expandedDimensions(axis: 0)
        let post = dsp.synthesis(spectrum * (0.5 + 0.4 * sin(0.013 * f + 0.29 * t)), length: x.count)
        let p = try #require(wf["post_synthetic"])
        #expect(DFGolden.relL2(post, p) <= p.tolerance_rel_l2)
        // The deep filter on synthetic coefficients [1, order, T, 96, 2].
        let k = MLXArray((0..<5).map { Float($0) }).reshaped([5, 1, 1, 1]), tt = MLXArray((0..<102).map { Float($0) }).reshaped([1, 102, 1, 1])
        let ff = MLXArray((0..<96).map { Float($0) }).reshaped([1, 1, 96, 1]), cc = MLXArray([Float(0), 1]).reshaped([1, 1, 1, 2])
        let coefs = (0.3 * sin(0.7 * k + 0.05 * tt + 0.11 * ff + 1.3 * cc)).expandedDimensions(axis: 0)
        let specIn = stacked([spectrum.realPart(), spectrum.imaginaryPart()], axis: -1).expandedDimensions(axes: [0, 1])
        let d = try #require(wf["df_synthetic"])
        #expect(DFGolden.relL2(dsp.deepFilter(spec: specIn, coefs: coefs).asArray(Float.self), d) <= d.tolerance_rel_l2)
    }
}

/// Regression (2 Nov 2026 review): odd lengths keep their length, the STFT round trip is exact, and clips shorter than
/// the two-frame lookahead get all-zero features, as the maker's negative pad gives.
@Test func deepFilterNet3ShortClipsAndOddLengths() throws {
    DFGolden.useTestMetallib()
    let dsp = DeepFilterNetDSP()
    try Device.withDefaultDevice(.cpu) {
        let x = Array(DFGolden.fixture().prefix(47_123))
        let y = dsp.synthesis(dsp.analysis(x), length: x.count)
        #expect(y.count == x.count)
        let err = zip(x, y).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }, ref = x.reduce(0.0) { $0 + Double($1 * $1) }
        #expect((err / ref).squareRoot() < 1e-5)
        #expect(dsp.frames(samples: 100) == 2 && dsp.synthesis(dsp.analysis(Array(x.prefix(100))), length: 100).count == 100)
        let model = try DeepFilterNetModel(weights: DeepFilterNet3Tensors.shapes.mapValues { MLXArray.zeros($0, dtype: .float16) })
        let ones = MLXArray.ones([1, 1, 2, 32], dtype: .float16)
        #expect(model.applyLookahead(feature: ones, lookahead: 2).asType(.float32).sum().item(Float.self) == 0)
        let ramp = MLXArray((0..<5).map { Float($0) }).reshaped([1, 1, 5, 1])
        #expect(model.applyLookahead(feature: ramp, lookahead: 2).reshaped(-1).asArray(Float.self) == [2, 3, 4, 0, 0])
    }
}

@Test func deepFilterNet3TensorTableMatchesTheCheckedCheckpoint() throws {
    DFGolden.useTestMetallib()
    let golden = try DFGolden.load()
    // The pinned mlx-community file was compared with the maker's checkpoint tensor by tensor (golden weights_check).
    #expect(golden.weights_check.pinned_tensors == 118 && golden.weights_check.bit_identical_tensors == 118 && golden.weights_check.max_abs_diff == 0)
    #expect(DeepFilterNet3Tensors.shapes.count == DeepFilterNet3Engine.tensorCount)
    #expect(DeepFilterNet3Tensors.shapes.values.reduce(0) { $0 + $1.reduce(1, *) } == golden.weights_check.parameters)
    try Device.withDefaultDevice(.cpu) {
        let weights = DeepFilterNet3Tensors.shapes.mapValues { MLXArray.zeros($0, dtype: .float16) }
        _ = try DeepFilterNetModel(weights: weights)
        var missing = weights
        missing.removeValue(forKey: "mask.erb_inv_fb")
        #expect(throws: (any Error).self) { try DeepFilterNetModel(weights: missing) }
    }
}

@Test func deepFilterNet3RequestOptionsAndDtype() throws {
    try checkDtype("fp16", model: "deepfilternet3")
    for dtype in ["bf16", "int8", "int4"] { #expect(throws: (any Error).self) { try checkDtype(dtype, model: "deepfilternet3") } }
    #expect(SoundEngines.halfCastRecipes["cast:deepfilternet3"]?.dtype == "fp16")
    let none = try DeepFilterNet3Model.requestOptions([:])
    #expect(none.limit == nil && !none.postFilter)
    let set = try DeepFilterNet3Model.requestOptions(["atten_lim_db": "12", "post_filter": "1"])
    #expect(set.limit == 12 && set.postFilter)
    #expect(try DeepFilterNet3Model.requestOptions(["atten_lim_db": "0"]).limit == nil)
    for bad in [["atten_lim_db": "-1"], ["atten_lim_db": "101"], ["atten_lim_db": "x"], ["rate": "16000"]] {
        #expect(throws: (any Error).self) { try DeepFilterNet3Model.requestOptions(bad) }
    }
}

// Opt in: ARSENAL_DFN3_WEIGHTS (a converted FP16 snapshot) and ARSENAL_DFN3_REFERENCE (lab/evidence/deepfilternet3/ref-fp16),
// under scripts/mlock gpulock.
@Test(.heavy) func deepFilterNet3ComponentParityAgainstMakerFP16() throws {
    let env = ProcessInfo.processInfo.environment
    guard let weights = env["ARSENAL_DFN3_WEIGHTS"], let reference = env["ARSENAL_DFN3_REFERENCE"] else { return }
    DFGolden.useTestMetallib()
    let golden = try DFGolden.load()
    let model = try DeepFilterNetModel(weights: try MLX.loadArrays(url: URL(fileURLWithPath: weights).appendingPathComponent("model.safetensors")))
    let ref = try NPZArchive(url: URL(fileURLWithPath: reference).appendingPathComponent("reference.npz"))
    var report: [String: Double] = [:]
    func check(_ name: String, _ actual: MLXArray, _ component: DFGolden.Component? = nil) throws {
        let expected = try ref.array(name).asType(.float32), value = actual.asType(.float32)
        #expect(value.size == expected.size, "\(name) size")
        let a = value.reshaped(-1).asArray(Float.self).map(Double.init), b = expected.reshaped(-1).asArray(Float.self).map(Double.init)
        var diff = 0.0, rr = 0.0
        for (x, y) in zip(a, b) { diff += (x - y) * (x - y); rr += y * y }
        let rel = (diff / max(rr, 1e-30)).squareRoot()
        report[name] = rel
        let bound = (component ?? golden.network[name] ?? golden.weight_free[name])?.tolerance_rel_l2 ?? 0
        #expect(a.allSatisfy(\.isFinite) && rel <= bound, "\(name) relative L2 \(rel)")
    }
    // The GPU running-norm kernel on the fixture.
    let x = try ref.array("fixture").asArray(Float.self)
    let spectrum = model.dsp.analysis(x)
    try check("erb_feat", model.dsp.erbFeatures(spectrum))
    try check("spec_feat", model.dsp.complexFeatures(spectrum))
    // Network components on the maker's own spectrum and features.
    var trace: [String: MLXArray] = [:]
    let enhanced = try model.network(
        spec: try ref.array("spec"), featErb: try ref.array("erb_feat"), featSpec: try ref.array("spec_feat"), trace: { trace[$0] = $1 })
    eval(enhanced)
    #expect(trace["enc4"]?.dtype == .float16 && trace["df_coefs"]?.dtype == .float16)
    for name in ["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "enc6", "erb_mask", "df_coefs"] { try check(name, try #require(trace[name])) }
    try check("spec_enhanced", enhanced)
    // End to end on the fixture: plain, post filter, 12 dB attenuation limit.
    try check("wave", MLXArray(try model.enhance(x)))
    try check("wave_pf", MLXArray(try model.enhance(x, postFilter: true)))
    try check("wave_atten12", MLXArray(try model.enhance(x, attenuationLimit: 12)))
    // An odd length through the whole clean: same count, finite.
    let odd = try model.enhance(Array(x.prefix(47_123)), attenuationLimit: 6, postFilter: true)
    #expect(odd.count == 47_123 && odd.allSatisfy(\.isFinite))
    if let path = env["ARSENAL_DFN3_EVIDENCE"] {
        try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted]).write(to: URL(fileURLWithPath: path))
    }
}

/// Bug (2 Nov 2026 review): on exact digital silence the running mean decays below the smallest normal float after
/// about 78 s, the GPU kernel flushes it to zero, and 0 / 0 made every later complex feature NaN. 100 s of zeros through
/// the shipped Metal running-mean kernel must give finite, all-zero features.
@Test func deepFilterNet3HundredSecondsOfSilenceStaysFinite() throws {
    DFGolden.useTestMetallib()
    let dsp = DeepFilterNetDSP()
    let spectrum = dsp.analysis([Float](repeating: 0, count: 48_000 * 100))
    let low = spectrum[0..., 0..<96]
    let state = DeepFilterNetKernels.runningMean(sqrt(low.realPart().square() + low.imaginaryPart().square()), initial: dsp.complexInitial, alpha: dsp.alpha)
    // The path under test: the state has left the normal range by the end.
    #expect(state[-1].max().item(Float.self) < Float.leastNormalMagnitude)
    let features = dsp.complexFeatures(spectrum).asArray(Float.self)
    #expect(features.count == spectrum.dim(0) * 96 * 2)
    #expect(features.allSatisfy { $0 == 0 })
}
