// SPDX-License-Identifier: MIT
import Foundation
import ImageIO
@testable import RestorationWorker
import Testing
import IntegrationKit
import ArsenalWire
import MLX
import MLXNN
import FlashVSRNative
import MLXRandom

struct FlashVSRTests {
    @Test func geometryFollowsScaleAndPadsToBlocks() throws {
        let g = try FlashVSRAdapter.geometry(height: 480, width: 640, scale: 2, targetHeight: nil)
        #expect(g.height == 960 && g.width == 1280 && g.paddedHeight == 1024 && g.paddedWidth == 1280)
        let t = try FlashVSRAdapter.geometry(height: 240, width: 427, scale: 2, targetHeight: 720)
        #expect(t.height == 720 && t.width == 1280 && t.paddedHeight == 768)
        #expect(throws: WorkerFailure.self) { try FlashVSRAdapter.geometry(height: 480, width: 640, scale: 5, targetHeight: nil) }
        #expect(throws: WorkerFailure.self) { try FlashVSRAdapter.geometry(height: 480, width: 640, scale: 2, targetHeight: 240) }
        // An odd-height source at scale 1 floors to even and is accepted.
        #expect(try FlashVSRAdapter.geometry(height: 481, width: 640, scale: 1, targetHeight: nil).height == 480)
        // 1080p ×2 is 3840×2160, past the bound of about 4.2 megapixels per output frame.
        #expect(throws: WorkerFailure.self) { try FlashVSRAdapter.geometry(height: 1080, width: 1920, scale: 2, targetHeight: nil) }
    }

    @Test func streamScheduleCoversEveryFrame() {
        // The first chunk reads 25 LQ frames and yields 21; each later chunk reads 8 more and yields 8.
        #expect(FlashVSRPipeline.schedule(chunk: 0).needed == 25)
        #expect(FlashVSRPipeline.schedule(chunk: 1).needed == 33)
        #expect(FlashVSRPipeline.schedule(chunk: 1).cond == 21..<29)
        #expect(FlashVSRAdapter.conversion.recipe == "cast:restoration-flashvsr")
        #expect(!FlashVSRAdapter.conversion.eligible("blocks.0.self_attn.q.weight", [1536, 1536]))
    }

    /// Every pushed frame gets exactly one output frame, in order, across the first-chunk and later-chunk boundaries
    /// (random weights, a one-block DiT at 128×128).
    @Test(.heavy) func streamEmitsOneFramePerInput() throws {
        NativeRuntime.prepare()
        let dit = FlashVSRDiT(.tiny()), lq = LQProjector(outDim: 128, layers: 1, hidden: (64, 64)), decoder = TCDecoder(channels: [32, 32, 32, 32])
        for module in [dit as Module, lq, decoder] { try NativeParameters.initializeRandom(module) }
        dit.prepare(context: MLXArray.zeros([1, 512, FlashVSRDiTConfig.tiny().textDim], dtype: .float16))
        let pipeline = FlashVSRPipeline(dit: dit, lqProj: lq, decoder: decoder)
        for count in [1, 24, 25, 26, 33, 41] {
            let stream = FlashVSRStream(pipeline: pipeline, height: 128, width: 128, seed: 0)
            var emitted = 0
            for index in 0..<count {
                let frame = MLXRandom.uniform(low: -1, high: 1, [128, 128, 3], key: MLXRandom.key(UInt64(index)))
                let out = try stream.push(frame)
                emitted += out.count
                #expect(emitted <= index + 1)
            }
            let rest = try stream.finish()
            #expect(rest.allSatisfy { $0.shape == [128, 128, 3] })
            #expect(emitted + rest.count == count, "\(count) frames in, \(emitted + rest.count) out")
        }
    }

    /// FP16 against the BF16 release, per component on the BF16 run's own inputs (25 real frames, one first chunk).
    /// Bounds are fixed here, before any run: relative L2 at most 0.05, cosine at least 0.999, every value finite.
    /// FLASHVSR_WEIGHTS: the pinned BF16 snapshot; FLASHVSR_FRAMES: a folder of 25 PNG frames; FLASHVSR_REPORT: JSON out.
    @Test(.heavy) func fp16MatchesBF16Release() throws {
        let env = ProcessInfo.processInfo.environment
        guard let weights = env["FLASHVSR_WEIGHTS"], let framesDir = env["FLASHVSR_FRAMES"] else { return }
        NativeRuntime.prepare()
        let maxRelL2: Float = 0.05, minCosine: Float = 0.999
        let names = try FileManager.default.contentsOfDirectory(atPath: framesDir).filter { $0.hasSuffix(".png") }.sorted().prefix(25)
        #expect(names.count == 25)
        var lq: [MLXArray] = []
        var geometry: (height: Int, width: Int, paddedHeight: Int, paddedWidth: Int)?
        for name in names {
            let url = URL(fileURLWithPath: framesDir).appendingPathComponent(name)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                Issue.record("Cannot read \(name)"); return
            }
            let rgb = MLX.round(try MediaBridge.tensor(image)[0].asType(.float32) * 255)
            let g = try geometry ?? FlashVSRAdapter.geometry(height: rgb.dim(0), width: rgb.dim(1), scale: 2, targetHeight: nil)
            geometry = g
            var up = FlashVSRPreprocess.bicubic8(rgb, outH: g.height, outW: g.width)
            up = padded(up, widths: [.init((0, g.paddedHeight - g.height)), .init((0, g.paddedWidth - g.width)), 0], mode: .edge)
            lq.append(FlashVSRPreprocess.toModel(up))
        }
        guard let g = geometry else { return }
        let clip = stacked(lq, axis: 0).expandedDimensions(axis: 0)
        let noise = MLXRandom.normal([1, 6, g.paddedHeight / 8, g.paddedWidth / 8, 16], key: MLXRandom.key(0))
        let topk = FlashVSRPipeline.defaultTopkRatio(height: g.paddedHeight, width: g.paddedWidth)
        let sch = FlashVSRPipeline.schedule(chunk: 0)
        func pipeline(_ dtype: DType) throws -> FlashVSRPipeline {
            try FlashVSRPipeline.load(directory: URL(fileURLWithPath: weights), dtypes: FlashVSRDTypes(dit: dtype, lqProj: dtype, decoder: dtype))
        }
        func compare(_ a: MLXArray, _ b: MLXArray) -> (relL2: Float, cosine: Float, finite: Bool) {
            let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
            let rel = (MLX.sqrt(((x - y) * (x - y)).sum()) / MLX.sqrt((y * y).sum())).item(Float.self)
            let cos = ((x * y).sum() / (MLX.sqrt((x * x).sum()) * MLX.sqrt((y * y).sum()))).item(Float.self)
            return (rel, cos, MLX.all(MLX.isFinite(x)).item(Bool.self))
        }
        // Reference: the BF16 release run as shipped.
        let reference = try pipeline(.bfloat16)
        var refState = reference.begin()
        let refChunk = try reference.chunk(0, frames: { clip[0..., $0] }, noise: noise, topkRatio: topk, options: .init(), state: &refState)
        let refTokens = refChunk.lqTokens, refLatents = refChunk.latents, refFrames = refChunk.frames!
        eval(refTokens + [refLatents, refFrames])
        // Candidate: FP16, each component on the reference's inputs.
        let candidate = try pipeline(.float16)
        _ = candidate.begin()
        var tokens: [MLXArray]?
        for r in sch.clips {
            guard let cur = candidate.lqProj.stream(clip[0..., r]) else { continue }
            tokens = tokens.map { zip($0, cur).map { concatenated([$0, $1], axis: 1) } } ?? cur
        }
        var cacheK = [MLXArray?](repeating: nil, count: candidate.dit.blockCount), cacheV = cacheK
        let pred = candidate.dit.step(
            noise.asType(.float16), lq: refTokens.map { $0.asType(.float16) }, cacheK: &cacheK, cacheV: &cacheV, t0: 0, topkRatio: topk, kvLen: 3, localRange: 11
        ).pred
        let latents = noise.asType(pred.dtype) - pred
        let cond = clip[0..., sch.cond]
        let decoded = candidate.decoder.decode(refLatents.asType(.float16), lq: cond.asType(.float16)).asType(.float32) * 2 - 1
        let frames = FlashVSRPipeline.colorFix(decoded, lq: cond[0..., 0..<decoded.dim(1)])
        var endState = candidate.begin()
        let endToEnd = try candidate.chunk(0, frames: { clip[0..., $0] }, noise: noise, topkRatio: topk, options: .init(), state: &endState).frames!
        let rows: [(String, (relL2: Float, cosine: Float, finite: Bool), Bool)] = [
            ("lq_proj", compare(tokens![0], refTokens[0]), true), ("dit", compare(latents, refLatents), true),
            ("tcdecoder", compare(frames, refFrames), true), ("end_to_end (diagnostic)", compare(endToEnd, refFrames), false)
        ]
        var report: [[String: Any]] = []
        for (name, r, gated) in rows {
            if gated { #expect(r.finite && r.relL2 <= maxRelL2 && r.cosine >= minCosine, "\(name): \(r)") }
            report.append(["component": name, "rel_l2": r.relL2, "cosine": r.cosine, "finite": r.finite, "gated": gated])
        }
        let record: [String: Any] = [
            "reference": "BF16 release", "candidate": "FP16", "bounds": ["max_rel_l2": maxRelL2, "min_cosine": minCosine],
            "input": "\(names.count) frames ×2 → \(g.width)×\(g.height), padded \(g.paddedWidth)×\(g.paddedHeight), chunk 0, seed 0", "rows": report
        ]
        if let out = env["FLASHVSR_REPORT"] {
            try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out))
        }
    }
}
