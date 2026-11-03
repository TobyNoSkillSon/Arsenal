// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXRandom
import ArsenalWire

public final class TrellisEngine {
    private let spec: TrellisShapeSpec
    private let ss: SparseStructureFlowModel
    private let ssDecoder: SparseStructureDecoder
    private let shapeDecoder: ShapeSlatDecoder
    private let textureDecoder: ShapeSlatDecoder
    private let dino: DINOv3
    private let flows: [String: SLatFlowModel]
    private let shapeMean, shapeStd, textureMean, textureStd: MLXArray

    public init(banks: [String: TrellisWeights], spec: TrellisShapeSpec, pipeline: [String: Any]) throws {
        self.spec = spec
        guard let sf = banks["struct_flow"], let sd = banks["struct_dec"], let shape = banks["shape_dec"], let tex = banks["tex_dec"], let conditioning = banks["dino"] else {
            throw WorkerFailure(code: "invalid_model", message: "TRELLIS component set is incomplete.")
        }
        ss = SparseStructureFlowModel(weights: sf, numBlocks: spec.blocks, resolution: spec.ssResolution, numHeads: spec.heads)
        ssDecoder = SparseStructureDecoder(weights: sd, channels: spec.ssDecoderChannels)
        shapeDecoder = ShapeSlatDecoder(weights: shape, channels: spec.decoderChannels, blocks: spec.decoderBlocks)
        textureDecoder = ShapeSlatDecoder(weights: tex, channels: spec.decoderChannels, blocks: spec.decoderBlocks)
        dino = DINOv3(weights: conditioning, hidden: spec.condition, numHeads: spec.lab ? 4 : 16, numLayers: spec.lab ? 1 : 24)
        var stages: [String: SLatFlowModel] = [:]
        for name in ["shape_flow_512", "shape_flow_1024", "tex_flow_512", "tex_flow_1024"] {
            guard let bank = banks[name] else { throw WorkerFailure(code: "invalid_model", message: "Missing TRELLIS flow: " + name) }
            stages[name] = SLatFlowModel(weights: bank, numBlocks: spec.blocks, numHeads: spec.heads)
        }
        flows = stages
        let args = pipeline["args"] as? [String: Any] ?? [:]
        func normalization(_ name: String, _ key: String, fallback: Float) throws -> MLXArray {
            if spec.lab { return MLXArray([Float](repeating: fallback, count: 32)).asType(.float16) }
            guard let group = args[name] as? [String: Any], let values = group[key] as? [Double], values.count == 32, values.allSatisfy(\.isFinite),
                key != "std" || values.allSatisfy({ $0 > 0 })
            else { throw WorkerFailure(code: "invalid_model", message: "Missing or invalid TRELLIS latent normalization: " + name) }
            return MLXArray(values.map(Float.init)).asType(.float16)
        }
        shapeMean = try normalization("shape_slat_normalization", "mean", fallback: 0)
        shapeStd = try normalization("shape_slat_normalization", "std", fallback: 1)
        textureMean = try normalization("tex_slat_normalization", "mean", fallback: 0)
        textureStd = try normalization("tex_slat_normalization", "std", fallback: 1)
    }

    public func generate(
        pixels: MLXArray, highResolutionPixels: MLXArray?, resolution: Int, textureSize: Int,
        targetFaces: Int, steps: Int, seed: UInt64, cancellation: NativeCancellation, log: (String) -> Void
    ) throws -> BakedMesh {
        let check = { try cancellation.check() }
        try check(); log("Encoding image")
        let condition = try dino(pixels, checkpoint: check), negative = MLXArray.zeros(condition.shape, dtype: .float16)
        eval(condition); try check()
        let r = spec.ssResolution
        log("Sampling sparse structure")
        let noise = SSGrid.noise(resolution: r, seed: seed).asType(.float16)
        let phases = SSGrid.ropePhases(resolution: r, headDim: spec.channels / spec.heads)
        let latent = try FlowEulerSampler.sampleSS(
            model: ss, noise: noise, cond: condition, negCond: negative, phases: phases,
            steps: steps, checkpoint: check)
        try check()
        let decoded = ssDecoder(latent)
        let coordinates = ssDecoder.coords(decoded, resolution: spec.lab ? 1 : 32)
        eval(coordinates); try check()
        guard coordinates.dim(0) > 0 else { throw WorkerFailure(code: "empty_geometry", message: "TRELLIS predicted no occupied voxels for this image.") }
        log("Sampling shape")
        func sample(_ model: SLatFlowModel, coords: MLXArray, cond: MLXArray, concat: MLXArray?, seed: UInt64, texture: Bool) throws -> MLXArray {
            MLXRandom.seed(seed)
            return try FlowEulerSampler.sampleSLat(
                model: model, noiseFeats: MLXRandom.normal([coords.dim(0), 32]).asType(.float16), coords: coords,
                cond: cond, negCond: MLXArray.zeros(cond.shape, dtype: .float16), concatCond: concat, steps: steps,
                guidanceStrength: texture ? 1 : 7.5, guidanceRescale: texture ? 0 : 0.5,
                guidanceInterval: texture ? (0.6, 0.9) : (0.6, 1), rescaleT: 3, checkpoint: check)
        }
        var shape = try sample(flows["shape_flow_512"]!, coords: coordinates, cond: condition, concat: nil, seed: seed, texture: false)
        var coords = coordinates
        var effectiveResolution = spec.lab ? 16 : 512
        var activeCondition = condition
        if resolution > 512 {
            log("Refining shape")
            let upsampled = try shapeDecoder(SparseTensor(feats: shape * shapeStd + shapeMean, coords: coordinates), checkpoint: check)
            // Keep the same cascade operations in lab mode with a smaller physical grid.
            let target = spec.lab ? (resolution == 1536 ? 48 : 32) : resolution
            let quantized = CascadeQuantize.requantize(
                upCoords: upsampled.coords, lrResolution: spec.lab ? 16 : 512,
                targetResolution: target, floorResolution: spec.lab ? 32 : 1024)
            coords = quantized.coords; effectiveResolution = quantized.hrResolution
            activeCondition = try dino(highResolutionPixels ?? pixels, checkpoint: check); eval(activeCondition); try check()
            shape = try sample(flows["shape_flow_1024"]!, coords: coords, cond: activeCondition, concat: nil, seed: seed &+ 2, texture: false)
        }
        log("Decoding shape")
        let (geometry, subdivisions) = try shapeDecoder.decodeCapturingSubs(SparseTensor(feats: shape * shapeStd + shapeMean, coords: coords), checkpoint: check)
        eval(geometry.feats, geometry.coords); try check()
        guard geometry.count > 0 else { throw WorkerFailure(code: "empty_geometry", message: "TRELLIS predicted an empty shape.") }
        log("Sampling materials")
        let textureLatent = try sample(
            flows[resolution > 512 ? "tex_flow_1024" : "tex_flow_512"]!, coords: coords,
            cond: activeCondition, concat: shape, seed: seed &+ 1, texture: true)
        let material = try textureDecoder(SparseTensor(feats: textureLatent * textureStd + textureMean, coords: coords), guidedMasks: subdivisions, checkpoint: check)
        eval(material.feats, material.coords); try check()
        guard geometry.coords.shape == material.coords.shape, geometry.count == material.count,
            MLX.all(geometry.coords .== material.coords).item(Bool.self), material.feats.dim(1) == 6
        else {
            throw WorkerFailure(code: "invalid_geometry", message: "Shape and PBR decoder topology do not match.")
        }
        let pbr = MLX.clip(material.feats.asType(.float32) * 0.5 + 0.5, min: 0, max: 1)
        log("Building textured mesh")
        let (mesh, _) = try MeshBake.run(
            shapeFeats: geometry.feats.asType(.float32), coords: geometry.coords,
            texBaseColor: pbr[0..., 0..<3], fineRes: Float(effectiveResolution), remeshRes: spec.lab ? 16 : (resolution == 1536 ? 384 : 256),
            targetFaces: targetFaces, atlasSize: textureSize, texMetallicRoughness: pbr[0..., 3..<6],
            cancellation: { (try? cancellation.check()) == nil }, log: { _ in })
        try check()
        guard mesh.vertices.dim(0) > 0, mesh.faces.dim(0) > 0,
            MLX.all(MLX.isFinite(mesh.vertices)).item(Bool.self), mesh.mrRGBA != nil
        else {
            throw WorkerFailure(code: "empty_geometry", message: "TRELLIS did not produce a finite textured triangle mesh.")
        }
        return mesh
    }
}
