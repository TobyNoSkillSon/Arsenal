// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
import Testing
@testable import GIMMNative
@testable import RestorationWorker

/// GIMM-VFI-R without inference: the stored inventory and the maker's padding geometry.
struct GIMMContractTests {
    /// 395 float tensors in the pinned `gimmvfi_r_arb.pt` state dict minus RAFT's 8 `norm3` aliases.
    @Test func inventory() {
        let shapes = GIMMWeights.shapes()
        #expect(shapes.count == 387)
        #expect(shapes.values.reduce(0) { $0 + $1.reduce(1, *) } == 19_789_065)
        #expect(shapes["flow_estimator.update_block.gru.convz1.weight"] == [128, 1, 5, 384])
        #expect(shapes["amt_final_decoder.convblock.0.0.weight"] == [256, 3, 3, 273])
        #expect(shapes["hyponet.params_dict.linear_wb0"] == [36, 128])
        #expect(!shapes.keys.contains { $0.contains(".norm3.") || $0.hasSuffix("num_batches_tracked") })
        #expect(!GIMMAdapter.eligible("amt_fproj.weight", [256, 1, 1, 256]))
        #expect(RestorationWorker.registrations.contains { $0.modelTypes == ["gimm_vfi_r"] && $0.conversion?.recipe == "cast:restoration-gimm-vfi-r-v1" })
    }

    /// InputPadder(divisor 32) as video_Nx.py: centred replicate padding; the fixture's 281×240 pads to 288×256 with
    /// (left 3, right 4, top 8, bottom 8), as the reference recorded. Larger frames run the flow stages at a lower scale.
    @Test func paddingGeometry() {
        let fixture = GIMMPadding(height: 240, width: 281)
        #expect(fixture.left == 3 && fixture.right == 4 && fixture.top == 8 && fixture.bottom == 8)
        #expect(fixture.paddedWidth == 288 && fixture.paddedHeight == 256 && fixture.scale == 1)
        #expect(GIMMPadding(height: 720, width: 1280).scale == 1)
        #expect(GIMMPadding(height: 1080, width: 1920).scale == 0.5)
        #expect(GIMMPadding(height: 1088, width: 1920).top == 0)
        #expect(GIMMPadding(height: 2160, width: 3840).scale == 0.25)
        let tiny = GIMMPadding(height: 20, width: 40)
        #expect(tiny.paddedHeight == 64 && tiny.paddedWidth == 64 && tiny.top == 22 && tiny.left == 12)
        // A very wide strip keeps its short side at 64 or more after scaling.
        let strip = GIMMPadding(height: 160, width: 8000)
        #expect(Double(strip.paddedHeight) * strip.scale >= 64)
    }
}

/// GIMM-VFI-R numerics that run on the CPU without weights.
struct GIMMNumericsTests {
    init() { NativeRuntime.prepare() }

    /// Every size the padding rule produces fits the Int32 correlation indices; an oversized volume is refused.
    @Test func correlationSizeGuard() throws {
        for (h, w) in [(240, 281), (736, 1280), (1080, 1920), (1440, 2560), (2160, 3840), (4320, 7680), (64, 64)] {
            let padding = GIMMPadding(height: h, width: w)
            try GIMMVFIR.checkCorrelationSize(height: padding.paddedHeight, width: padding.paddedWidth, scale: padding.scale)
        }
        #expect(throws: GIMMWeightError.self) { try GIMMVFIR.checkCorrelationSize(height: 2176, width: 3840, scale: 1) }
    }

    /// The flow deviation in float32 against a scalar float64 computation, for flows of 200 to 600 px (FP16 squares of
    /// those overflow) and for small flows; every weight stays finite.
    @Test func flowDeviationIsFloat32AndFiniteForLargeFlows() throws {
        try Device.withDefaultDevice(.cpu) {
            let h = 9, w = 11
            let g: [Double] = [1, 2, 1, 2, 4, 2, 1, 2, 1].map { $0 / 16 }
            for magnitude in [3.0, 200, 600] {
                var values = [Float]()
                for i in 0..<(h * w * 2) {
                    let fraction = Double((i * 37) % 23) / 22, sign: Double = i % 3 == 0 ? -1 : 1
                    values.append(Float(magnitude * (0.4 + 0.6 * fraction) * sign))
                }
                let flows = MLXArray(values).reshaped([1, h, w, 2])
                let got = GIMMVFIR.flowDeviation(flows, kernel: MLXArray(g.map { Float($0) }).reshaped([1, 1, 1, 3, 3]).asType(.float16))
                func f(_ y: Int, _ x: Int, _ c: Int) -> Double {
                    let ry = y < 0 ? -y : y >= h ? 2 * h - 2 - y : y, rx = x < 0 ? -x : x >= w ? 2 * w - 2 - x : x
                    return Double(values[(ry * w + rx) * 2 + c])
                }
                var expected = [Float]()
                for y in 0..<h {
                    for x in 0..<w {
                        var sum = 0.0
                        for c in 0..<2 {
                            var m2 = 0.0, m1 = 0.0
                            for dy in -1...1 { for dx in -1...1 { let v = f(y + dy, x + dx, c), k = g[(dy + 1) * 3 + dx + 1]; m2 += k * v * v; m1 += k * v } }
                            sum += max(m2 - m1 * m1, 1e-9).squareRoot()
                        }
                        expected.append(Float(sum / 2))
                    }
                }
                let reference = MLXArray(expected).reshaped([1, h, w, 1])
                #expect(MLX.all(MLX.isFinite(got)).item(Bool.self))
                let rel = (MLX.sqrt(MLX.sum(MLX.square(got - reference))) / MLX.sqrt(MLX.sum(MLX.square(reference)))).item(Float.self)
                #expect(rel < 1e-4, "magnitude \(magnitude): relative error \(rel)")
            }
            let model = GIMMVFIR(weights: GIMMWeights.random(dtype: .float16))
            var bigValues = [Float]()
            for i in 0..<(2 * 16 * 16 * 2) { bigValues.append(Float((i * 53) % 1201) - 600) }
            let big = MLXArray(bigValues).reshaped([2, 16, 16, 2])
            let (w1, w2) = model.splattingWeights(big[0..<1], big[1..<2])
            #expect(MLX.all(MLX.isFinite(w1)).item(Bool.self) && MLX.all(MLX.isFinite(w2)).item(Bool.self))
        }
    }
}

/// Fixed-tensor parity of the GIMM-VFI-R port against the maker's code (GSeanCDAT/GIMM-VFI on MPS, FP16 weights under
/// autocast, CuPy splat replaced by a proved PyTorch equivalent), captured once by lab/evidence/gimm-vfi-r/reference.py.
/// Heavy and opt-in: ARSENAL_GIMM_SOURCE names a directory holding the pinned `gimmvfi_r_arb.pt` as a regular file;
/// the test converts it with Get's own recipe. Each component consumes the reference's own inputs.
@Suite(.serialized) struct GIMMParityTests {
    static let env = ProcessInfo.processInfo.environment
    static let repo = (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() }
    static let evidence = repo.appendingPathComponent("lab/evidence/gimm-vfi-r")

    // Bounds, fixed before the first comparison (relative L2 unless named).
    static let componentBound: Float = 0.01
    static let raftBound: Float = 0.02
    static let synthesisMaxAbs: Float = 0.02
    // End-to-end bounds sit just under the worst measured values (67.37 dB full size; 62.28 and 64.39 dB downscaled), so a
    // regression fails them; they were 38 dB, frozen before the first comparison, until the review of 2 Nov.
    static let endToEndPSNR: Float = 67
    static let downscaledPSNR: Float = 62

    static func nhwc(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 1) }
    static func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32), y = b.asType(.float32)
        return (MLX.sqrt(MLX.sum(MLX.square(x - y))) / MLX.sqrt(MLX.sum(MLX.square(y)))).item(Float.self)
    }
    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self) }
    static func psnr(_ a: MLXArray, _ b: MLXArray) -> Float {
        let mse = MLX.mean(MLX.square(a.asType(.float32) - b.asType(.float32))).item(Float.self)
        return 10 * log10f(1 / mse)
    }

    static func load() throws -> (GIMMVFIR, [String: MLXArray]) {
        let source = URL(fileURLWithPath: env["ARSENAL_GIMM_SOURCE"]!)
        let staging = FileManager.default.temporaryDirectory.appendingPathComponent("gimm-convert-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: staging) }
        _ = try GIMMAdapter.conversion.run(
            [
                "src_dir": .string(source.path), "dst_dir": .string(staging.path), "dtype": "fp16", "group_size": 64,
                "recipe": "cast:restoration-gimm-vfi-r-v1"
            ], cancellation: NativeCancellation())
        let weights = try MLX.loadArrays(url: staging.appendingPathComponent("model.safetensors"))
        try GIMMWeights.validate(weights, half: .float16)
        let reference = try MLX.loadArrays(url: evidence.appendingPathComponent("reference.safetensors"))
        return (GIMMVFIR(weights: weights), reference)
    }

    static func record(_ name: String, _ values: [String: Float]) throws {
        let url = evidence.appendingPathComponent("swift-\(name).json")
        let data = try JSONSerialization.data(withJSONObject: values, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }

    @Test(.heavy, .enabled(if: env["ARSENAL_GIMM_SOURCE"] != nil)) func components() throws {
        NativeRuntime.prepare()
        let (model, r) = try Self.load()
        var m: [String: Float] = [:]

        // RAFT, both directions in one batch, from the reference's own 255 * frame inputs.
        let images = concatenated([Self.nhwc(r["raft.0.in0"]!), Self.nhwc(r["raft.1.in0"]!)], axis: 0)
        let raft = RAFT.run(images, model.p, cancellation: {})
        m["raft_flow01_rel"] = Self.relL2(raft.flow[0..<1], Self.nhwc(r["raft.0.flow"]!))
        m["raft_flow10_rel"] = Self.relL2(raft.flow[1..<2], Self.nhwc(r["raft.1.flow"]!))
        m["raft_flow01_max_abs_px"] = Self.maxAbs(raft.flow[0..<1], Self.nhwc(r["raft.0.flow"]!))
        m["raft_feat_quarter_rel"] = Self.relL2(raft.features[0][0..<1], Self.nhwc(r["raft.0.feat0"]!))
        m["raft_feat_eighth_rel"] = Self.relL2(raft.features[1][1..<2], Self.nhwc(r["raft.1.feat1"]!))
        m["raft_fmap_rel"] = Self.relL2(raft.fmap[0..<1], Self.nhwc(r["raft.0.fnet"]!))

        // Splatting weights, flow encoder, splat, residual refinement, hyponet.
        let f01 = Self.nhwc(r["weights.f01"]!), f10 = Self.nhwc(r["weights.f10"]!)
        let (w1, w2) = model.splattingWeights(f01, f10)
        m["weights1_rel"] = Self.relL2(w1, Self.nhwc(r["weights.w1"]!))
        m["weights2_rel"] = Self.relL2(w2, Self.nhwc(r["weights.w2"]!))
        m["cnn_encoder_rel"] = Self.relL2(model.cnnEncoder(Self.nhwc(r["cnn_encoder.0.in0"]!)), Self.nhwc(r["cnn_encoder.0.out"]!))
        for i in 0..<2 {
            let out = softsplatLinear(Self.nhwc(r["splat.\(i).in"]!), flow: Self.nhwc(r["splat.\(i).flow"]!), metric: Self.nhwc(r["splat.\(i).metric"]!))
            m["splat\(i)_rel"] = Self.relL2(out, Self.nhwc(r["splat.\(i).out"]!))
        }
        m["res_conv_rel"] = Self.relL2(model.resConv(Self.nhwc(r["res_conv.0.in0"]!)), Self.nhwc(r["res_conv.0.out"]!))
        for i in 0..<3 {
            let coord = r["hyponet.\(i).coord"]!.reshaped([1, 256, 288, 3])
            let out = model.hyponet(coordinates: coord, latent: r["hyponet.\(i).latent"]!)
            m["hyponet\(i)_rel"] = Self.relL2(out, r["hyponet.\(i).out"]!.reshaped([1, 256, 288, 2]))
        }

        // Frame synthesis on the reference's flows, features and correlation inputs.
        let xs = r["synth.img_xs"]!
        let frame0 = xs[0..., 0..., 0].transposed(0, 2, 3, 1), frame1 = xs[0..., 0..., 1].transposed(0, 2, 3, 1)
        let volume = CorrPyramid.volume(Self.nhwc(r["corr.fmap1"]!), Self.nhwc(r["corr.fmap2"]!))
        let h8 = 32, w8 = 36
        let pair = GIMMVFIR.Pair(
            frame0: frame0, frame1: frame1, full0: nil, full1: nil, raftFlow01: f01, raftFlow10: f10, scaler: MLXArray(Float(1)),
            weights0: w1, weights1: w2, latent0: frame0, latent1: frame1,
            features0: [Self.nhwc(r["synth.features0.0"]!), Self.nhwc(r["synth.features0.1"]!)],
            features1: [Self.nhwc(r["synth.features1.0"]!), Self.nhwc(r["synth.features1.1"]!)],
            corr: CorrPyramid(volume), corrT: CorrPyramid(volume.reshaped([h8, w8, h8 * w8]).transposed(2, 0, 1)), scale: 1)
        for (i, t) in [Float(0.25), 0.5, 0.75].enumerated() {
            let s = model.synthesize(pair, flowT: Self.nhwc(r["synth.\(i).flow_t"]!), t: t)
            m["synth\(i)_flowt0_quarter_rel"] = Self.relL2(s.flowT0Quarter, Self.nhwc(r["synth.\(i).flowt0_4"]!))
            let flow0 = r["synth.\(i).flowt0_1"]!.reshaped([1, 6, 256, 288]).transposed(0, 2, 3, 1)
            m["synth\(i)_flowt0_full_rel"] = Self.relL2(s.flowT0, flow0)
            m["synth\(i)_image_max_abs"] = Self.maxAbs(s.image, Self.nhwc(r["synth.\(i).imgt"]!))
            m["synth\(i)_image_psnr"] = Self.psnr(s.image, Self.nhwc(r["synth.\(i).imgt"]!))
        }
        // Correlation lookups on the reference's coordinates.
        let lookup = pair.corr.lookup(Self.nhwc(r["corr.0.coords0"]!)), lookupT = pair.corrT.lookup(Self.nhwc(r["corr.0.coords1"]!))
        m["corr_lookup_rel"] = Self.relL2(lookup, Self.nhwc(r["corr.0.out0"]!))
        m["corrT_lookup_rel"] = Self.relL2(lookupT, Self.nhwc(r["corr.0.out1"]!))
        try Self.record("components", m)

        for key in ["raft_flow01_rel", "raft_flow10_rel"] { #expect(m[key]! <= Self.raftBound, "\(key) \(m[key]!)") }
        for (key, value) in m where key.hasSuffix("_rel") && !key.hasPrefix("raft_flow") { #expect(value <= Self.componentBound, "\(key) \(value)") }
        for i in 0..<3 { #expect(m["synth\(i)_image_max_abs"]! <= Self.synthesisMaxAbs) }
    }

    /// The whole model from the padded fixture frames (the reference's input.xs) to the three frames, unpadded.
    @Test(.heavy, .enabled(if: env["ARSENAL_GIMM_SOURCE"] != nil)) func endToEnd() throws {
        NativeRuntime.prepare()
        let (model, r) = try Self.load()
        let xs = r["input.xs"]!
        let padding = GIMMPadding(height: 240, width: 281)
        let pair = try model.pair(xs[0..., 0..., 0].transposed(0, 2, 3, 1), xs[0..., 0..., 1].transposed(0, 2, 3, 1))
        var m: [String: Float] = [:]
        m["raft_flow01_rel"] = Self.relL2(pair.raftFlow01, Self.nhwc(r["raft.0.flow"]!))
        for (i, t) in [Float(0.25), 0.5, 0.75].enumerated() {
            let flow = model.flow(pair, t: t)
            m["t\(i)_flow_rel"] = Self.relL2(flow, r["final.\(i).flowt"]!.transposed(1, 2, 0).expandedDimensions(axis: 0))
            let image = padding.unpad(model.frame(pair, t: t))
            let reference = Self.nhwc(r["final.\(i).imgt"]!)
            m["t\(i)_psnr"] = Self.psnr(image, reference)
            m["t\(i)_max_abs"] = Self.maxAbs(image, reference)
            m["t\(i)_mean_abs"] = MLX.mean(MLX.abs(image - reference.asType(.float32))).item(Float.self)
        }
        try Self.record("end-to-end", m)
        for i in 0..<3 { #expect(m["t\(i)_psnr"]! >= Self.endToEndPSNR, "t\(i) psnr \(m["t\(i)_psnr"]!)") }
    }

    /// The makers' ds_factor path (flows at half or a quarter of the frame, frame synthesised at full size) on the
    /// makers' own 844x720 demo pair, against the makers' code run at the same ds_factor (reference_ds.py).
    @Test(.heavy, .enabled(if: env["ARSENAL_GIMM_SOURCE"] != nil)) func downscaledFlowPath() throws {
        NativeRuntime.prepare()
        let (model, _) = try Self.load()
        var m: [String: Float] = [:]
        for (tag, scale) in [("0p5", 0.5), ("0p25", 0.25)] {
            let r = try MLX.loadArrays(url: Self.evidence.appendingPathComponent("reference-ds\(tag).safetensors"))
            let xs = r["input.xs"]!
            let padding = GIMMPadding(height: 720, width: 844)
            let pair = try model.pair(xs[0..., 0..., 0].transposed(0, 2, 3, 1), xs[0..., 0..., 1].transposed(0, 2, 3, 1), scale: scale)
            m["ds\(tag)_raft_flow01_rel"] = Self.relL2(pair.raftFlow01, Self.nhwc(r["raft.0.flow"]!))
            m["ds\(tag)_flow_rel"] = Self.relL2(model.flow(pair, t: 0.5), r["final.0.flowt"]!.transposed(1, 2, 0).expandedDimensions(axis: 0))
            let image = padding.unpad(model.frame(pair, t: 0.5)), reference = Self.nhwc(r["final.0.imgt"]!)
            m["ds\(tag)_psnr"] = Self.psnr(image, reference)
            m["ds\(tag)_max_abs"] = Self.maxAbs(image, reference)
        }
        try Self.record("downscaled", m)
        for tag in ["0p5", "0p25"] { #expect(m["ds\(tag)_psnr"]! >= Self.downscaledPSNR, "ds \(tag) psnr \(m["ds\(tag)_psnr"]!)") }
    }

    /// The splat against a literal scalar transcription of the CuPy kernel on a small tensor with out-of-range,
    /// non-finite and exact-integer targets.
    @Test(.heavy) func splatMatchesKernelFormula() throws {
        let n = 1, h = 6, w = 7, c = 2
        var input = [Float](), flow = [Float](), metric = [Float]()
        for i in 0..<(h * w * c) { input.append(Float((i * 37) % 11) / 5 - 1) }
        for i in 0..<(h * w) {
            flow.append(Float((i * 13) % 17) / 3 - 2.7); flow.append(Float((i * 7) % 19) / 4 - 2.2); metric.append(0.2 + Float(i % 5) / 7)
        }
        flow[2 * 3] = .infinity; flow[2 * 9 + 1] = .nan; flow[0] = 2; flow[1] = 1
        var sums = [Float](repeating: 0, count: h * w * (c + 1))
        for y in 0..<h {
            for x in 0..<w {
                let p = y * w + x, fx = Float(x) + flow[2 * p], fy = Float(y) + flow[2 * p + 1]
                guard fx.isFinite, fy.isFinite else { continue }
                let x0 = Int(fx.rounded(.down)), y0 = Int(fy.rounded(.down))
                let corners = [
                    (x0, y0, (Float(x0 + 1) - fx) * (Float(y0 + 1) - fy)), (x0 + 1, y0, (fx - Float(x0)) * (Float(y0 + 1) - fy)),
                    (x0, y0 + 1, (Float(x0 + 1) - fx) * (fy - Float(y0))), (x0 + 1, y0 + 1, (fx - Float(x0)) * (fy - Float(y0)))
                ]
                for (cx, cy, weight) in corners where cx >= 0 && cx < w && cy >= 0 && cy < h {
                    for k in 0..<c { sums[(cy * w + cx) * (c + 1) + k] += input[p * c + k] * metric[p] * weight }
                    sums[(cy * w + cx) * (c + 1) + c] += metric[p] * weight
                }
            }
        }
        var expected = [Float]()
        for p in 0..<(h * w) {
            let norm = sums[p * (c + 1) + c] == 0 ? 1 : sums[p * (c + 1) + c]
            for k in 0..<c { expected.append(sums[p * (c + 1) + k] / norm) }
        }
        let out = softsplatLinear(
            MLXArray(input).reshaped([n, h, w, c]), flow: MLXArray(flow).reshaped([n, h, w, 2]), metric: MLXArray(metric).reshaped([n, h, w, 1]))
        #expect(Self.maxAbs(out, MLXArray(expected).reshaped([n, h, w, c])) < 1e-5)
    }
}
