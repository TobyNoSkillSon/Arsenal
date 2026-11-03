// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
// Inference wrapper for MoGe-2. Mirrors `MoGeModel.infer` in the Python
// reference implementation of `infer`, including the focal/shift recovery
// that Python delegates to `scipy.optimize.least_squares` (Levenberg-Marquardt).
//
// Post-processing is performed on the CPU with `[Float]` arrays because the
// optimization loop is inherently sequential and per-batch. Results are
// re-materialized as MLXArrays before returning.

import Foundation
import MLX

extension MoGeModel {

    /// Full inference with post-processing.
    ///
    /// - Parameters:
    ///   - image: `(H, W, 3)` or `(B, H, W, 3)` NHWC float in `[0, 1]`.
    ///   - numTokens: Override token count. Defaults to the value derived
    ///     from `resolutionLevel` and `numTokensRange`.
    ///   - resolutionLevel: Integer in `[0, 9]` selecting a point inside
    ///     `numTokensRange`.
    ///   - forceProjection: Recompute points from depth + intrinsics.
    ///   - applyMask: Set invalid regions to `+infinity`.
    ///   - fovX: Known horizontal field of view in radians (optional).
    ///   - refineSteps: MoGe-3 sparse refinement steps. `nil` uses the
    ///     model's default — 3 for a MoGe-3 checkpoint, 0 for MoGe-2.
    /// - Returns: Dictionary with `points`, `depth`, `intrinsics`, optional
    ///   `mask` and `normal`.
    public func infer(
        image: MLXArray,
        numTokens: Int? = nil,
        resolutionLevel: Int = 9,
        forceProjection: Bool = true,
        applyMask: Bool = true,
        fovX: Float? = nil,
        refineSteps: Int? = nil,
        checkpoint: () throws -> Void = {}
    ) rethrows -> [String: MLXArray] {
        // Accept (H, W, 3) by prepending a batch dim.
        let omitBatch = image.ndim == 3
        let batched = omitBatch ? image.expandedDimensions(axis: 0) : image

        let B = batched.dim(0)
        let imgH = batched.dim(1)
        let imgW = batched.dim(2)
        let aspectRatio = Float(imgW) / Float(imgH)

        // Token count from resolution_level.
        let resolvedNumTokens: Int
        if let n = numTokens {
            resolvedNumTokens = n
        } else {
            let minT = numTokensRange[0]
            let maxT = numTokensRange[1]
            // `int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))` in double precision.
            resolvedNumTokens = Int(Double(minT) + Double(resolutionLevel) / 9.0 * Double(maxT - minT))
        }

        // Forward pass.
        let output = try self(batched, resolvedNumTokens, refineSteps: refineSteps, checkpoint: checkpoint)
        MLX.eval(Array(output.values))
        try checkpoint()

        // Materialize points and mask on the CPU for optimization.
        let pointsArr = output["points"]!.asType(.float32)
        MLX.eval(pointsArr)
        let pointsFlat: [Float] = pointsArr.asArray(Float.self)

        var maskFlat: [Bool]? = nil
        if let m = output["mask"] {
            let mf: [Float] = m.asType(.float32).asArray(Float.self)
            maskFlat = mf.map { $0 > 0.5 }
        }

        // Recover per-batch focal and shift.
        var focals = [Float](repeating: 1, count: B)
        var shifts = [Float](repeating: 0, count: B)
        let stride = imgH * imgW * 3
        let maskStride = imgH * imgW

        for b in 0..<B {
            try checkpoint()
            let ptsSlice = Array(pointsFlat[(b * stride)..<((b + 1) * stride)])
            let maskSlice: [Bool]? = maskFlat.map {
                Array($0[(b * maskStride)..<((b + 1) * maskStride)])
            }

            if let fx = fovX {
                let knownFocal =
                    aspectRatio / (1 + aspectRatio * aspectRatio).squareRoot()
                    / tanf(fx / 2)
                shifts[b] = Self.solveOptimalShift(
                    points: ptsSlice, mask: maskSlice,
                    H: imgH, W: imgW, aspectRatio: aspectRatio,
                    focal: knownFocal
                )
                focals[b] = knownFocal
            } else {
                let (f, s) = Self.solveOptimalFocalShift(
                    points: ptsSlice, mask: maskSlice,
                    H: imgH, W: imgW, aspectRatio: aspectRatio
                )
                focals[b] = f
                shifts[b] = s
            }
        }

        // Build depth: points_z + shift (broadcast over HxW).
        // Compose directly on CPU, then wrap as MLXArray.
        var depthFlat = [Float](repeating: 0, count: B * imgH * imgW)
        for b in 0..<B {
            let s = shifts[b]
            for i in 0..<(imgH * imgW) {
                let pz = pointsFlat[b * stride + i * 3 + 2]
                depthFlat[b * imgH * imgW + i] = pz + s
            }
        }

        // Clamp mask by positive depth.
        if var m = maskFlat {
            for i in 0..<m.count {
                if depthFlat[i] <= 0 { m[i] = false }
            }
            maskFlat = m
        }

        // Intrinsics.
        var fxArr = [Float](repeating: 0, count: B)
        var fyArr = [Float](repeating: 0, count: B)
        let sqrtTerm = (1 + aspectRatio * aspectRatio).squareRoot()
        for b in 0..<B {
            fxArr[b] = focals[b] / 2 * sqrtTerm / aspectRatio
            fyArr[b] = focals[b] / 2 * sqrtTerm
        }
        let cxArr = [Float](repeating: 0.5, count: B)
        let cyArr = [Float](repeating: 0.5, count: B)
        let intrinsicsMLX = intrinsicsFromFocalCenter(
            fx: fxArr, fy: fyArr, cx: cxArr, cy: cyArr)

        // Depth and (optionally) reprojected points as MLXArrays.
        var depthMLX = MLXArray(depthFlat, [B, imgH, imgW])
        var pointsMLX: MLXArray
        if forceProjection {
            pointsMLX = depthMapToPointMap(
                depth: depthMLX, intrinsics: intrinsicsMLX,
                height: imgH, width: imgW)
        } else {
            pointsMLX = MLXArray(pointsFlat, [B, imgH, imgW, 3])
            // Replace z channel with shifted depth.
            pointsMLX = pointsMLX.at[0..., 0..., 0..., 2...].add(
                depthMLX.expandedDimensions(axis: -1)
                    - pointsMLX[0..., 0..., 0..., 2...])
        }

        // Metric scale, if predicted.
        if let scaleArr = output["metric_scale"] {
            let scale = scaleArr.asType(.float32).reshaped([B, 1, 1, 1])
            pointsMLX = pointsMLX * scale
            depthMLX = depthMLX * scale.squeezed(axis: -1)
        }

        try checkpoint()
        // Apply mask → +inf on invalid pixels.
        if applyMask, let m = maskFlat {
            let inf = Float.infinity
            var pf: [Float] = pointsMLX.asArray(Float.self)
            var df: [Float] = depthMLX.asArray(Float.self)
            for b in 0..<B {
                for i in 0..<(imgH * imgW) {
                    if !m[b * imgH * imgW + i] {
                        pf[(b * imgH * imgW + i) * 3 + 0] = inf
                        pf[(b * imgH * imgW + i) * 3 + 1] = inf
                        pf[(b * imgH * imgW + i) * 3 + 2] = inf
                        df[b * imgH * imgW + i] = inf
                    }
                }
            }
            pointsMLX = MLXArray(pf, [B, imgH, imgW, 3])
            depthMLX = MLXArray(df, [B, imgH, imgW])
        }

        var result: [String: MLXArray] = [:]
        result["points"] = pointsMLX
        result["depth"] = depthMLX
        result["intrinsics"] = intrinsicsMLX

        if let m = maskFlat {
            let u8 = m.map { $0 ? Float(1) : Float(0) }
            result["mask"] = MLXArray(u8, [B, imgH, imgW])
        }
        if let normal = output["normal"] {
            var n = normal.asType(.float32)
            if applyMask, let m = maskFlat {
                var nf: [Float] = n.asArray(Float.self)
                for b in 0..<B {
                    for i in 0..<(imgH * imgW) {
                        if !m[b * imgH * imgW + i] {
                            nf[(b * imgH * imgW + i) * 3 + 0] = 0
                            nf[(b * imgH * imgW + i) * 3 + 1] = 0
                            nf[(b * imgH * imgW + i) * 3 + 2] = 0
                        }
                    }
                }
                n = MLXArray(nf, [B, imgH, imgW, 3])
            }
            result["normal"] = n
        }

        if omitBatch {
            for (k, v) in result {
                result[k] = v[0]
            }
        }
        return result
    }

    // MARK: - Focal/shift recovery

    /// UV/XYZ correspondences for one sample, exactly as `recover_focal_shift`
    /// (`moge/utils/geometry_torch.py`) samples them: the point map, its
    /// pixel-centre view-plane UV and the mask are each resized to 64×64 with
    /// `F.interpolate(mode="nearest")` (duplicating rows when the map is smaller),
    /// then filtered by the resized mask.
    static func gatherUVXYZ(
        points: [Float], mask: [Bool]?,
        H: Int, W: Int, aspectRatio: Float,
        downsample: (height: Int, width: Int) = (64, 64)
    ) -> (uv: [Float], xyz: [Float]) {
        let spans = viewPlaneSpans(aspectRatio)
        let us = viewPlaneAxis(count: W, span: spans.x)
        let vs = viewPlaneAxis(count: H, span: spans.y)
        let rows = nearestSourceIndices(inSize: H, outSize: downsample.height)
        let cols = nearestSourceIndices(inSize: W, outSize: downsample.width)

        var uv: [Float] = []
        var xyz: [Float] = []
        uv.reserveCapacity(rows.count * cols.count * 2)
        xyz.reserveCapacity(rows.count * cols.count * 3)
        for y in rows {
            for x in cols {
                let idx = y * W + x
                if let mask, !mask[idx] { continue }
                uv.append(us[x])
                uv.append(vs[y])
                xyz.append(points[idx * 3 + 0])
                xyz.append(points[idx * 3 + 1])
                xyz.append(points[idx * 3 + 2])
            }
        }
        return (uv, xyz)
    }

    /// Variable-projection form of `solve_optimal_focal_shift`
    /// (`moge/utils/geometry_numpy.py`): for a shift `s`, `p = xy / (z + s)` and the
    /// optimal focal is `Σ p·uv / Σ p·p`; residuals `f(s)·p - uv` are minimized over `s`
    /// alone by Levenberg-Marquardt from `s = 0`, then the focal is read back.
    static func focalForShift(uv: [Float], xyz: [Float], shift: Double) -> Double {
        var numerator = 0.0, denominator = 0.0
        for i in 0..<(uv.count / 2) {
            let d = Double(xyz[i * 3 + 2]) + shift
            let px = Double(xyz[i * 3]) / d, py = Double(xyz[i * 3 + 1]) / d
            numerator += px * Double(uv[i * 2]) + py * Double(uv[i * 2 + 1])
            denominator += px * px + py * py
        }
        return numerator / denominator
    }

    static func solveOptimalFocalShift(
        points: [Float], mask: [Bool]?,
        H: Int, W: Int, aspectRatio: Float
    ) -> (Float, Float) {
        let (uv, xyz) = gatherUVXYZ(
            points: points, mask: mask,
            H: H, W: W, aspectRatio: aspectRatio)
        // Upstream returns focal 1, shift 0 when fewer than two samples survive the mask.
        if uv.count / 2 < 2 { return (1.0, 0.0) }
        let n = uv.count / 2

        func residualsAndJacobian(_ s: Double) -> (cost: Double, jtj: Double, jtr: Double) {
            // f(s) = A / B with A = Σ p·uv, B = Σ p·p; dp/ds = -p / (z + s).
            var a = 0.0, b = 0.0, da = 0.0, db = 0.0
            for i in 0..<n {
                let d = Double(xyz[i * 3 + 2]) + s
                let px = Double(xyz[i * 3]) / d, py = Double(xyz[i * 3 + 1]) / d
                let u = Double(uv[i * 2]), v = Double(uv[i * 2 + 1])
                a += px * u + py * v
                b += px * px + py * py
                da += -(px * u + py * v) / d
                db += -2 * (px * px + py * py) / d
            }
            let f = a / b
            let df = (da * b - a * db) / (b * b)
            var cost = 0.0, jtj = 0.0, jtr = 0.0
            for i in 0..<n {
                let d = Double(xyz[i * 3 + 2]) + s
                let px = Double(xyz[i * 3]) / d, py = Double(xyz[i * 3 + 1]) / d
                let ru = f * px - Double(uv[i * 2]), rv = f * py - Double(uv[i * 2 + 1])
                let ju = df * px + f * (-px / d), jv = df * py + f * (-py / d)
                cost += ru * ru + rv * rv
                jtj += ju * ju + jv * jv
                jtr += ju * ru + jv * rv
            }
            return (cost, jtj, jtr)
        }

        let shift = levenbergMarquardt1D(residualsAndJacobian)
        let focal = focalForShift(uv: uv, xyz: xyz, shift: shift)
        guard shift.isFinite, focal.isFinite else { return (1.0, 0.0) }
        return (Float(focal), Float(shift))
    }

    /// Solve for `shift` alone given a known `focal` (`solve_optimal_shift`).
    static func solveOptimalShift(
        points: [Float], mask: [Bool]?,
        H: Int, W: Int, aspectRatio: Float, focal: Float
    ) -> Float {
        let (uv, xyz) = gatherUVXYZ(
            points: points, mask: mask,
            H: H, W: W, aspectRatio: aspectRatio)
        if uv.count / 2 < 2 { return 0 }
        let n = uv.count / 2
        let f = Double(focal)

        func residualsAndJacobian(_ s: Double) -> (cost: Double, jtj: Double, jtr: Double) {
            var cost = 0.0, jtj = 0.0, jtr = 0.0
            for i in 0..<n {
                let d = Double(xyz[i * 3 + 2]) + s
                let px = Double(xyz[i * 3]) / d, py = Double(xyz[i * 3 + 1]) / d
                let ru = f * px - Double(uv[i * 2]), rv = f * py - Double(uv[i * 2 + 1])
                let ju = -f * px / d, jv = -f * py / d
                cost += ru * ru + rv * rv
                jtj += ju * ju + jv * jv
                jtr += ju * ru + jv * rv
            }
            return (cost, jtj, jtr)
        }
        let shift = levenbergMarquardt1D(residualsAndJacobian)
        return shift.isFinite ? Float(shift) : 0
    }

    /// One-parameter Levenberg-Marquardt from zero (scipy `least_squares(x0=0, method="lm")`).
    private static func levenbergMarquardt1D(_ evaluate: (Double) -> (cost: Double, jtj: Double, jtr: Double)) -> Double {
        var shift = 0.0
        var lambda = 1e-3
        var current = evaluate(shift)
        guard current.cost.isFinite else { return 0 }
        for _ in 0..<100 {
            let m = current.jtj * (1 + lambda)
            if !(m > 1e-30) { break }
            let candidateShift = shift - current.jtr / m
            let candidate = evaluate(candidateShift)
            if candidate.cost.isFinite && candidate.cost < current.cost {
                let improvement = current.cost - candidate.cost
                shift = candidateShift
                current = candidate
                lambda = max(lambda / 3, 1e-12)
                if improvement <= 1e-12 * max(current.cost, 1e-30) { break }
            } else {
                lambda *= 5
                if lambda > 1e12 { break }
            }
        }
        return shift
    }
}

// swiftlint:enable all
