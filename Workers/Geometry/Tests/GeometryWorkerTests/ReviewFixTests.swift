// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import GeometryWorker
@testable import MLXDA3
@testable import MLXMoGe
@testable import MLXTrellis

/// Regression tests for the geometry review findings (half activations, MoGe UV/recovery,
/// MoGe preprocessing, DA3 sky, cancellation, invalid depth, TRELLIS preview orientation).
/// Reference numbers were produced once by the pinned upstream code in PyTorch on these exact
/// inputs: microsoft/MoGe@74fbce054ebed49800de42d0ad0e83495065719a and
/// ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4.
@Suite(.serialized)
struct ReviewFixTests {
    /// The pinned MoGe-3 config.json (mlx-community/moge-3-vitl-mlx-fp32@f5ceca75); the lab graph shrinks its widths.
    static let mogeConfig =
        #"{"model_type":"moge3","encoder":{"backbone":"dinov2_vitl14","intermediate_layers":[5,11,17,23],"dim_out":1024},"neck":{"dim_in":[1026,2,2,2,2],"dim_out":null,"dim_res_blocks":[1024,256,128,64,32],"num_res_blocks":[0,2,2,2,0],"res_block_in_norm":"none","res_block_hidden_norm":"none","resamplers":["conv_transpose","conv_transpose","conv_transpose","bilinear"]},"points_head":{"dim_in":[1024,256,128,64,32],"dim_out":[null,null,null,null,3],"dim_res_blocks":[1024,256,128,64,32],"num_res_blocks":[0,1,1,1,0],"res_block_in_norm":"none","res_block_hidden_norm":"none","resamplers":["conv_transpose","conv_transpose","conv_transpose","bilinear"]},"normal_head":{"dim_in":[1024,256,128,64,32],"dim_out":[null,null,null,null,3],"dim_res_blocks":[1024,256,128,64,32],"num_res_blocks":[0,1,1,1,0],"res_block_in_norm":"none","res_block_hidden_norm":"none","resamplers":["conv_transpose","conv_transpose","conv_transpose","bilinear"]},"mask_head":{"dim_in":[1024,256,128,64,32],"dim_out":[null,null,null,null,1],"dim_res_blocks":[1024,256,128,64,32],"num_res_blocks":[0,1,1,1,0],"res_block_in_norm":"none","res_block_hidden_norm":"none","resamplers":["conv_transpose","conv_transpose","conv_transpose","bilinear"]},"scale_head":{"dims":[1024,1024,1024,1]},"num_tokens_range":[1200,3600],"refiner":{"encoder_channels":1026,"model_channels":[32,64,128,256,512],"encoder_blocks_per_level":1,"decoder_blocks_per_level":1,"bottleneck_blocks":1,"downsample_factors":[2,2,2,2],"encoder_downsample":16,"in_channels":3,"out_channels":1},"refiner_depth_resolution":256}"#

    /// Counts checkpoint calls and cancels the shared token on call `at`, like a user cancelling mid-run.
    final class CancelAt {
        let at: Int
        let token = NativeCancellation()
        private(set) var calls = 0
        init(_ at: Int) { self.at = at }
        func check() throws {
            calls += 1
            if calls == at { token.cancel() }
            try token.check()
        }
    }

    private func close(_ actual: [Float], _ expected: [Float], _ tolerance: Float) -> Bool {
        actual.count == expected.count && zip(actual, expected).allSatisfy { abs($0 - $1) <= tolerance }
    }

    private func failureCode(_ body: () throws -> Void) -> String? {
        do { try body() } catch let failure as WorkerFailure { return failure.code } catch { return "unexpected: \(error)" }
        return nil
    }

    private func mogeLab() throws -> MLXMoGe.MoGeModel {
        let config = try JSONSerialization.jsonObject(with: Data(Self.mogeConfig.utf8)) as? [String: Any] ?? [:]
        let model = try #require(DepthAdapter.makeGraph(architecture: "moge3", config: config, lab: true) as? MLXMoGe.MoGeModel)
        try NativeParameters.initializeRandom(model)
        model.train(false)
        return model
    }

    private func da3MonoLab() throws -> MLXDA3.DepthAnything3 {
        let model = try #require(DepthAdapter.makeGraph(architecture: "da3mono", config: [:], lab: true) as? MLXDA3.DepthAnything3)
        try NativeParameters.initializeRandom(model)
        model.train(false)
        return model
    }

    // MARK: 1. Interpolation keeps half activations through downstream learned layers

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func interpolationKeepsHalfActivations() throws {
        NativeRuntime.prepare()
        let x = MLXArray((0..<24).map { Float($0) / 24 }, [1, 2, 3, 4]).asType(.float16)
        #expect(MLXMoGe.bilinearResize(x, 4, 6).dtype == .float16)
        #expect(MLXDA3.bilinearInterpolate(x, size: (5, 7)).dtype == .float16)
        #expect(MLXMoGe.antialiasedBilinearResize(x, 1, 2).dtype == .float16)

        // Whole graphs: a float32 resize result used to promote every later conv and the outputs.
        let da3 = try da3MonoLab()
        let depth = da3(MLXArray.ones([1, 1, 56, 56, 3], dtype: .float16) * 0.5)
        #expect(depth["depth"]?.dtype == .float16)
        #expect(depth["sky"]?.dtype == .float16)

        let moge = try mogeLab()
        let maps = try moge(MLXArray.ones([1, 32, 32, 3], dtype: .float16) * 0.5, 4, refineSteps: 0)
        #expect(maps["points"]?.dtype == .float16)
        #expect(maps["normal"]?.dtype == .float16)
        #expect(maps["mask"]?.dtype == .float16)

        // TRELLIS has no interpolation in its learned path; its flow output stays half too.
        let bank = TrellisWeights(shapes: TrellisShapeSpec(lab: true).flow(input: 32, output: 32))
        try bank.initializeLab(precision: "fp16", seed: 7)
        let flow = SLatFlowModel(weights: bank, numBlocks: 1, numHeads: 4)
        let (feats, coords, cond) = trellisInputs()
        let out = try flow(MLXTrellis.SparseTensor(feats: feats, coords: coords), t: MLXArray([Float(500)]), cond: cond)
        #expect(out.feats.dtype == .float16)
    }

    // MARK: 2. MoGe view-plane UV and focal/shift recovery

    @Test func viewPlaneUVUsesPixelCentresAndNearestSampling() {
        #expect(close(viewPlaneAxis(count: 2, span: viewPlaneSpans(1).x), [-0.35355338, 0.35355338], 1e-6))
        let spans = viewPlaneSpans(1.5)
        #expect(close(viewPlaneAxis(count: 3, span: spans.x), [-0.5547002, 0, 0.5547002], 1e-6))
        #expect(close(viewPlaneAxis(count: 2, span: spans.y), [-0.2773501, 0.2773501], 1e-6))
        // torch F.interpolate(mode="nearest") source indices to the 64-sample recovery grid.
        let references: [Int: [Int]] = [
            100: [
                0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 39, 40, 42, 43, 45, 46, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 64, 65,
                67, 68, 70, 71, 73, 75, 76, 78, 79, 81, 82, 84, 85, 87, 89, 90, 92, 93, 95, 96, 98
            ],
            30: [
                0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21,
                21, 22, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29
            ],
            32: (0..<64).map { $0 >> 1 },
            64: Array(0..<64),
            129: (0..<64).map { $0 * 2 },
            20: [
                0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15,
                15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19
            ]
        ]
        for (size, expected) in references { #expect(nearestSourceIndices(inSize: size, outSize: 64) == expected) }
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func viewPlaneTensorMatchesReferenceProbe() {
        NativeRuntime.prepare()
        let uv = normalizedViewPlaneUV(width: 2, height: 2, aspectRatio: 1).asArray(Float.self)
        let probe: Float = 0.35355338
        #expect(close(uv, [-probe, -probe, probe, -probe, -probe, probe, probe, probe], 1e-6))
    }

    /// The same deterministic point map the reference script feeds `recover_focal_shift`.
    private func syntheticPointMap(height: Int, width: Int, focal: Float, shift: Float) -> (points: [Float], mask: [Bool]) {
        let spans = viewPlaneSpans(Float(width) / Float(height))
        let us = viewPlaneAxis(count: width, span: spans.x), vs = viewPlaneAxis(count: height, span: spans.y)
        var points = [Float](), mask = [Bool]()
        for y in 0..<height {
            for x in 0..<width {
                let fx = Float(x), fy = Float(y)
                let depth: Float = 1.5 + 0.03 * fx + 0.05 * fy
                let noise: Float = 0.01 * sin(0.7 * fx + 1.3 * fy)
                points += [us[x] * depth / focal + noise, vs[y] * depth / focal - noise, depth - shift]
                mask.append((x + y) % 7 != 0)
            }
        }
        return (points, mask)
    }

    @Test(.heavy) func focalShiftRecoveryMatchesReference() {
        // (height, width, upstream focal, upstream shift, upstream shift for fixed focal 1.25)
        let cases: [(Int, Int, Float, Float, Float)] = [
            (20, 30, 1.100100040435791, 0.40025627613067627, 0.7059314846992493),
            (90, 120, 1.0999791622161865, 0.40001988410949707, 0.9504093527793884)
        ]
        for (height, width, focal, shift, fixedShift) in cases {
            let map = syntheticPointMap(height: height, width: width, focal: 1.1, shift: 0.4)
            let aspect = Float(width) / Float(height)
            let (f, s) = MLXMoGe.MoGeModel.solveOptimalFocalShift(points: map.points, mask: map.mask, H: height, W: width, aspectRatio: aspect)
            #expect(abs(f - focal) < 1e-4)
            #expect(abs(s - shift) < 2e-4)
            let known = MLXMoGe.MoGeModel.solveOptimalShift(points: map.points, mask: map.mask, H: height, W: width, aspectRatio: aspect, focal: 1.25)
            // scipy stops at ftol=1e-3; the exact optimum differs from its result by < 1e-4 here.
            #expect(abs(known - fixedShift) < 2e-4)
        }
        // Fewer than two surviving samples: upstream returns focal 1, shift 0.
        let empty = syntheticPointMap(height: 4, width: 4, focal: 1.1, shift: 0.4)
        let (f, s) = MLXMoGe.MoGeModel.solveOptimalFocalShift(
            points: empty.points, mask: [Bool](repeating: false, count: 16), H: 4, W: 4, aspectRatio: 1)
        #expect(f == 1 && s == 0)
    }

    // MARK: 3. MoGe encoder preprocessing

    @Test(.heavy) func tokenGridRoundsLikeReference() {
        let fourThree = MLXMoGe.MoGeModel.tokenGrid(numTokens: 3600, width: 4, height: 3)
        #expect(fourThree.rows == 52 && fourThree.cols == 69)
        let portrait = MLXMoGe.MoGeModel.tokenGrid(numTokens: 3600, width: 3, height: 4)
        #expect(portrait.rows == 69 && portrait.cols == 52)
        let wide = MLXMoGe.MoGeModel.tokenGrid(numTokens: 1200, width: 1920, height: 1080)
        #expect(wide.rows == 26 && wide.cols == 46)
        let square = MLXMoGe.MoGeModel.tokenGrid(numTokens: 2133, width: 1000, height: 1000)
        #expect(square.rows == 46 && square.cols == 46)
        // Every antialias weight row is a partition of unity.
        let weights = antialiasedBilinearWeights(inSize: 37, outSize: 14)
        for row in 0..<14 { #expect(abs(weights[(row * 37)..<((row + 1) * 37)].reduce(0, +) - 1) < 1e-5) }
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func encoderResizeMatchesTorchAntialias() {
        NativeRuntime.prepare()
        let x = MLXArray((0..<20).map { Float($0) / 19 }, [1, 5, 4, 1])
        // torch F.interpolate(x, (2, 3), mode="bilinear", align_corners=False, antialias=True)
        let down = antialiasedBilinearResize(x, 2, 3).asArray(Float.self)
        #expect(close(down, [0.22631580, 0.28947368, 0.35263160, 0.64736837, 0.71052629, 0.77368420], 1e-5))
        // The previous plain bilinear result for the same probe, which the antialiased path must not return.
        #expect(!close(down, [0.16666667, 0.23684210, 0.30701753, 0.69298244, 0.76315790, 0.83333337], 1e-3))
        // Upscaling reduces to ordinary edge-clamped bilinear: torch (7, 6) antialias=True.
        let up = antialiasedBilinearResize(x, 7, 6).asArray(Float.self)
        let upReference: [Float] = [
            0.0, 0.02631579, 0.06140351, 0.09649124, 0.13157895, 0.15789473,
            0.12030075, 0.14661653, 0.18170425, 0.21679197, 0.25187969, 0.27819547,
            0.27067670, 0.29699251, 0.33208025, 0.36716795, 0.40225568, 0.42857143,
            0.42105263, 0.44736844, 0.48245618, 0.51754391, 0.55263162, 0.57894737,
            0.57142860, 0.59774441, 0.63283217, 0.66791987, 0.70300758, 0.72932333,
            0.72180450, 0.74812031, 0.78320801, 0.81829578, 0.85338348, 0.87969923,
            0.84210527, 0.86842108, 0.90350872, 0.93859649, 0.97368419, 1.0
        ]
        #expect(close(up, upReference, 1e-5))
    }

    // MARK: 4. DA3 Mono sky post-processing

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func monoSkyTakesNonSkyPercentile() {
        NativeRuntime.prepare()
        let depthValues = (0..<36).map { 1 + Float(($0 * 7) % 36) * 0.25 }
        let skyValues: [Float] = (0..<36).map { $0 == 1 ? 0.3 : ($0 % 3 == 0 ? 0.9 : 0.1) }
        let depth = MLXArray(depthValues, [1, 1, 6, 6]), sky = MLXArray(skyValues, [1, 1, 6, 6])
        let output = MLXDA3.DepthAnything3.processMonoSky(["depth": depth, "sky": sky])
        let q: Float = 9.694999694824219
        let expected: [Float] = [
            q, q, 4.5, q, 8.0, 9.75, q, 4.25, 6.0, q, 9.5, 2.25, q, 5.75, 7.5, q, 2.0, 3.75,
            q, 7.25, 9.0, q, 3.5, 5.25, q, 8.75, 1.5, q, 5.0, 6.75, q, 1.25, 3.0, q, 6.5, 8.25
        ]
        #expect(close(output["depth"]!.asArray(Float.self), expected, 1e-5))
        #expect(output["sky"]!.asArray(Float.self) == skyValues)
        #expect(MLXDA3.DepthAnything3.processMonoSky(["depth": depth.asType(.float16), "sky": sky])["depth"]!.dtype == .float16)

        // Population guards: with only ten sky pixels the depth is returned unchanged.
        let fewSky: [Float] = (0..<36).map { $0 < 10 ? 0.9 : 0.1 }
        let guarded = MLXDA3.DepthAnything3.processMonoSky(["depth": depth, "sky": MLXArray(fewSky, [1, 1, 6, 6])])
        #expect(guarded["depth"]!.asArray(Float.self) == depthValues)
    }

    // MARK: 5. Cancellation inside transformer blocks, refinement and mesh loops

    private func trellisInputs() -> (MLXArray, MLXArray, MLXArray) {
        var coordinates = [Int32]()
        for x in 0..<2 { for y in 0..<2 { for z in 0..<2 { coordinates += [0, Int32(x), Int32(y), Int32(z)] } } }
        return (
            MLXArray.ones([8, 32], dtype: .float16) * 0.1, MLXArray(coordinates, [8, 4]),
            MLXArray.ones([1, 5, 64], dtype: .float16) * 0.1
        )
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func mogeCancelsInsideBlocksAndRefinement() throws {
        NativeRuntime.prepare()
        let model = try mogeLab()
        let image = MLXArray.ones([32, 32, 3], dtype: .float16) * 0.5
        let full = CancelAt(Int.max)
        _ = try model.infer(image: image, refineSteps: 2, checkpoint: full.check)
        // Four encoder blocks plus the neck, heads, two refinement steps and their sparse U-Net levels.
        #expect(full.calls > 20)
        for point in [2, full.calls / 2, full.calls - 1] {
            let probe = CancelAt(point)
            let code = failureCode { _ = try model.infer(image: image, refineSteps: 2, checkpoint: probe.check) }
            #expect(code == "cancelled")
            #expect(probe.calls == point)
        }
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func da3CancelsInsideBackbone() throws {
        NativeRuntime.prepare()
        let model = try da3MonoLab()
        let input = MLXArray.ones([1, 1, 56, 56, 3], dtype: .float16) * 0.5
        let full = CancelAt(Int.max)
        _ = try model.predict(input, checkpoint: full.check)
        #expect(full.calls >= 7) // entry, four blocks, after the backbone and after the head
        let probe = CancelAt(3)
        let code = failureCode { _ = try model.predict(input, checkpoint: probe.check) }
        #expect(code == "cancelled")
        #expect(probe.calls == 3)
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func trellisCancelsInsideSamplingStep() throws {
        NativeRuntime.prepare()
        let bank = TrellisWeights(shapes: TrellisShapeSpec(lab: true).flow(input: 32, output: 32))
        try bank.initializeLab(precision: "fp16", seed: 7)
        let flow = SLatFlowModel(weights: bank, numBlocks: 1, numHeads: 4)
        let (feats, coords, cond) = trellisInputs()
        let full = CancelAt(Int.max)
        _ = try flow(MLXTrellis.SparseTensor(feats: feats, coords: coords), t: MLXArray([Float(500)]), cond: cond, checkpoint: full.check)
        #expect(full.calls == 2) // before the block and after it
        // Call 1 is the sampler's step boundary; call 2 is inside the model's forward pass.
        let probe = CancelAt(2)
        let code = failureCode {
            _ = try FlowEulerSampler.sampleSLat(
                model: flow, noiseFeats: feats, coords: coords, cond: cond, negCond: cond * 0, steps: 3, checkpoint: probe.check)
        }
        #expect(code == "cancelled")
        #expect(probe.calls == 2)
    }

    @Test func meshInpaintIsCancellable() {
        var rgb = [Float](repeating: 0, count: 64 * 64 * 3), filled = [Bool](repeating: false, count: 64 * 64)
        filled[0] = true; rgb[0] = 1
        let probe = CancelAt(3)
        let code = failureCode { try MeshBake.dilateInpaint(&rgb, &filled, 64, iterations: 64, checkpoint: probe.check) }
        #expect(code == "cancelled")
        #expect(probe.calls == 3)
    }

    // MARK: 6. Invalid depth never becomes an artifact

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func invalidDepthIsRejectedButMaskedInfinityIsKept() throws {
        NativeRuntime.prepare()
        let nan = MLXArray([Float](repeating: .nan, count: 56 * 56), [56, 56])
        for moge in [false, true] {
            let code = failureCode { try DepthOutputValidation.validate(["depth": nan], maskedInfinity: moge) }
            #expect(code == "invalid_model_output")
        }
        // MoGe's documented convention: masked pixels are +infinity in depth and points.
        let depth = MLXArray([1, 2, Float.infinity, 4], [2, 2])
        let mask = MLXArray([Float(1), 1, 0, 1], [2, 2])
        let points = MLXArray([0, 0, 1, 0, 0, 2, Float.infinity, .infinity, .infinity, 0, 0, 4], [2, 2, 3])
        try DepthOutputValidation.validate(["depth": depth, "mask": mask, "points": points], maskedInfinity: true)
        // The same infinity is invalid for DA3, and for MoGe where the mask says the pixel is valid.
        let relative = failureCode { try DepthOutputValidation.validate(["depth": depth], maskedInfinity: false) }
        #expect(relative == "invalid_model_output")
        let validMask = MLXArray([Float](repeating: 1, count: 4), [2, 2])
        let unmasked = failureCode { try DepthOutputValidation.validate(["depth": depth, "mask": validMask], maskedInfinity: true) }
        #expect(unmasked == "invalid_model_output")
        // Everything masked: nothing usable.
        let allMasked = MLXArray([Float](repeating: .infinity, count: 4), [2, 2])
        let empty = failureCode {
            try DepthOutputValidation.validate(["depth": allMasked, "mask": MLXArray.zeros([2, 2])], maskedInfinity: true)
        }
        #expect(empty == "invalid_model_output")
        // Companion tensors are checked too.
        let finite = MLXArray([Float(1), 2, 3, 4], [2, 2])
        let badNormals = MLXArray([Float](repeating: .nan, count: 12), [2, 2, 3])
        let normals = failureCode { try DepthOutputValidation.validate(["depth": finite, "normals": badNormals], maskedInfinity: true) }
        #expect(normals == "invalid_model_output")
        try DepthOutputValidation.validate(["depth": finite, "sky": MLXArray.zeros([2, 2])], maskedInfinity: false)
    }

    // MARK: 7. TRELLIS preview samples the texture with the bake's v orientation

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func previewUsesBakeTextureOrientation() throws {
        NativeRuntime.prepare()
        // 2×2 atlas: row 0 red, row 1 blue. The bake writes v ∈ [0, 0.5) into row 0 and glTF reads it there.
        let texture: [UInt8] = [255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255]
        #expect(TrellisPreview.texelOffset(u: 0.25, v: 0.25, atlasSize: 2) == 0)
        #expect(TrellisPreview.texelOffset(u: 0.75, v: 0.75, atlasSize: 2) == 12)
        let (pix, _, _) = try UVRasterize.rasterize(
            uvsPix: MLXArray([Float(0), 0, 2, 0, 0, 1], [3, 2]), faces: MLXArray([Int32(0), 1, 2], [1, 3]), textureSize: 2)
        #expect(pix.asArray(Int32.self) == [0, 0]) // texel (x 0, row 0) for v = 0.25

        // End to end: every UV samples v = 0.25, so every drawn preview pixel must be red.
        let mesh = BakedMesh(
            vertices: MLXArray([Float(-1), -1, 0, 1, -1, 0, 0, 1, 0], [3, 3]), faces: MLXArray([Int32(0), 1, 2], [1, 3]),
            normals: MLXArray.zeros([3, 3]), uvs: MLXArray([Float](repeating: 0.25, count: 6), [3, 2]),
            texRGBA: texture, atlasSize: 2, coverage: 1, mrRGBA: nil,
            filledBeforeInpaint: [true, true, true, true], wallFlipped: [false, false, false, false])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("trellis-preview-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        try TrellisPreview.write(mesh, to: url, cancellation: NativeCancellation(), size: 32)
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        var pixels = [UInt8](repeating: 0, count: 32 * 32 * 4)
        pixels.withUnsafeMutableBytes { buffer in
            let context = CGContext(
                data: buffer.baseAddress, width: 32, height: 32, bitsPerComponent: 8, bytesPerRow: 128,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            context?.draw(image, in: CGRect(x: 0, y: 0, width: 32, height: 32))
        }
        let drawn = stride(from: 0, to: pixels.count, by: 4).filter { pixels[$0 + 3] > 0 }
        #expect(drawn.count > 20)
        #expect(drawn.allSatisfy { pixels[$0] > 200 && pixels[$0 + 2] < 50 })
    }
}
