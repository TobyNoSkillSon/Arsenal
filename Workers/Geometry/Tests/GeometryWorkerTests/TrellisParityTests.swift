// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import IntegrationKit
import MLX
import Testing
@testable import MLXTrellis

/// Stage-by-stage parity against tensors captured once from the vendor TRELLIS.2 code (MPS) on one fixed product cutout.
/// Heavy: needs the installed FP16 snapshot (TRELLIS_WEIGHTS) and the saved capture (TRELLIS_PARITY_DIR/ref.safetensors).
@Suite(.serialized)
struct TrellisParityTests {
    static let env = ProcessInfo.processInfo.environment
    static var enabled: Bool { env["TRELLIS_PARITY_DIR"] != nil && env["TRELLIS_WEIGHTS"] != nil }

    static func bank(_ component: String, _ file: String) throws -> TrellisWeights {
        let bank = TrellisWeights(shapes: TrellisShapeSpec(lab: false).components[component]!)
        try bank.load(archives: [URL(fileURLWithPath: env["TRELLIS_WEIGHTS"]!).appendingPathComponent(file)], precision: "fp16", cancellation: NativeCancellation())
        return bank
    }
    static func reference() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: URL(fileURLWithPath: env["TRELLIS_PARITY_DIR"]!).appendingPathComponent("ref.safetensors"))
    }
    static func compare(_ name: String, _ candidate: MLXArray, _ reference: MLXArray) -> (cosine: Float, maxAbs: Float) {
        let a = candidate.asType(.float32).reshaped([-1]), b = reference.asType(.float32).reshaped([-1])
        let cosine = ((a * b).sum() / (MLX.sqrt((a * a).sum()) * MLX.sqrt((b * b).sum()))).item(Float.self)
        let maxAbs = MLX.abs(a - b).max().item(Float.self)
        print(
            "PARITY \(name): cosine=\(cosine) maxAbs=\(maxAbs) candStd=\(MLX.sqrt(((a - a.mean()) * (a - a.mean())).mean()).item(Float.self)) refStd=\(MLX.sqrt(((b - b.mean()) * (b - b.mean())).mean()).item(Float.self))"
        )
        return (cosine, maxAbs)
    }

    @Test(.heavy, .enabled(if: enabled))
    func sparseStructureStagesMatchVendorCapture() throws {
        NativeRuntime.prepare()
        let ref = try Self.reference()
        if let path = Self.env["TRELLIS_PARITY_IMAGE"], let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        {
            let pixels = try TrellisImage.pixels(image, size: 512)
            #expect(Self.compare("pixels", pixels, ref["pixels"]!).cosine > 0.999)
        }
        let spec = TrellisShapeSpec(lab: false)
        let dino = DINOv3(weights: try Self.bank("dino", "components/trellis-dinov3/model.safetensors"), hidden: 1024, numHeads: 16, numLayers: 24)
        let cond = try dino(ref["pixels"]!.asType(.float16))
        #expect(cond.dtype == .float16)
        #expect(Self.compare("cond", cond, ref["cond"]!).cosine > 0.99)
        // The residual stream itself: measured peak per layer, and where an FP16 stream (the old path) stops being finite.
        let (cos, sin) = dino.ropeCosSin(32, 32)
        var wide = dino.embed(ref["pixels"]!.asType(.float16)).asType(.float32), half = wide.asType(.float16)
        var peaks: [Float] = [], firstNonFinite: Int?
        for layer in 0..<24 {
            wide = dino.block(wide, layer: layer, cos: cos, sin: sin); half = dino.block(half, layer: layer, cos: cos, sin: sin)
            peaks.append(MLX.abs(wide).max().item(Float.self))
            if firstNonFinite == nil, !MLX.all(MLX.isFinite(half)).item(Bool.self) { firstNonFinite = layer }
        }
        print(
            "PARITY dino residual peak per layer: \(peaks.map { $0.isFinite ? String(Int($0)) : "inf" }) fp16 stream first non-finite layer: \(firstNonFinite.map(String.init) ?? "none")"
        )
        #expect(peaks.allSatisfy { $0.isFinite })

        let flow = SparseStructureFlowModel(
            weights: try Self.bank("struct_flow", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
            numBlocks: spec.blocks, resolution: spec.ssResolution, numHeads: spec.heads)
        let refCond = ref["cond"]!.asType(.float16), noise = ref["noise"]!.asType(.float16)
        let phases = SSGrid.ropePhases(resolution: spec.ssResolution, headDim: spec.channels / spec.heads)
        let t = MLXArray([Float(1000)])
        let vPos = try flow(noise, t: t, cond: refCond, phases: phases)
        let vNeg = try flow(noise, t: t, cond: MLXArray.zeros(refCond.shape, dtype: .float16), phases: phases)
        #expect(Self.compare("v_pos0", vPos, ref["v_pos0"]!).cosine > 0.99)
        #expect(Self.compare("v_neg0", vNeg, ref["v_neg0"]!).cosine > 0.99)
        let zs = try FlowEulerSampler.sampleSS(
            model: flow, noise: noise, cond: refCond, negCond: MLXArray.zeros(refCond.shape, dtype: .float16), phases: phases, steps: 12)
        #expect(Self.compare("z_s", zs, ref["z_s"]!).cosine > 0.95)

        let decoder = SparseStructureDecoder(
            weights: try Self.bank("struct_dec", "components/trellis-structure-decoder/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"), channels: spec.ssDecoderChannels)
        let logits = decoder(ref["z_s"]!.asType(.float16))
        #expect(Self.compare("logits", logits, ref["logits"]!).cosine > 0.99)
        let occupied = decoder.coords(logits, resolution: 32).dim(0)
        let refOccupied = decoder.coords(ref["logits"]!, resolution: 32).dim(0)
        print("PARITY occupied32: candidate=\(occupied) reference=\(refOccupied)")
        // End of the stage chain: the port's own sampled latent, decoded, against the vendor occupancy (IoU at 32³).
        let mine = decoder(zs).squeezed(axis: 1) .> 0, theirs = ref["logits"]!.squeezed(axis: 1) .> 0
        func pooled(_ x: MLXArray) -> MLXArray { x.asType(.float32).reshaped([1, 32, 2, 32, 2, 32, 2]).max(axes: [2, 4, 6]) .> 0.5 }
        let a = pooled(mine), b = pooled(theirs)
        let iou = MLX.logicalAnd(a, b).sum().item(Float.self) / MLX.logicalOr(a, b).sum().item(Float.self)
        print("PARITY sampled occupancy32: candidate=\(a.sum().item(Int.self)) reference=\(b.sum().item(Int.self)) IoU=\(iou)")
        #expect(iou > 0.8)
        #expect(abs(occupied - refOccupied) * 50 <= refOccupied)
    }
}
