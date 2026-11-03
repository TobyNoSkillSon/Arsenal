// swiftlint:disable all
// Vendored inference/geometry source; see PORTED-FROM.md.
import Foundation
import MLX
import MLXMesh
import ArsenalWire

/// End-to-end mesh+texture stage: shape/tex decoder voxel outputs -> clean textured
/// mesh. Chains the verified ops: FlexiDualGrid head -> DualGridMesh -> mlx-swift-mesh
/// DC-remesh + simplify + UV unwrap -> UVRasterize -> GridSample3d bake -> inpaint.
public struct BakedMesh {
    public let vertices: MLXArray // [V,3]
    public let faces: MLXArray // [F,3] int32
    public let normals: MLXArray // [V,3]
    public let uvs: MLXArray // [V,2] in [0,1)
    public let texRGBA: [UInt8] // atlasSize*atlasSize*4
    public let atlasSize: Int
    public let coverage: Float // fraction of texels covered before inpaint
    /// glTF metallicRoughness atlas (G=roughness, B=metallic), when the tex
    /// decoder's metallic/roughness channels were provided to the bake.
    public let mrRGBA: [UInt8]?
    /// Diagnostics (atlasSize² each): texel rasterized before inpaint; texel's
    /// closest-point remap landed on an opposing-normal (wrong-wall) raw face.
    public let filledBeforeInpaint: [Bool]
    public let wallFlipped: [Bool]
}

public enum MeshBake {
    static func softplus(_ x: MLXArray) -> MLXArray { // stable: max(x,0)+log(1+exp(-|x|))
        MLX.maximum(x, 0) + MLX.log(1 + MLX.exp(-MLX.abs(x)))
    }

    /// shapeFeats [N,7] (raw shape-decoder out), coords [N,4] (b,x,y,z),
    /// texBaseColor [N,3] in [0,1] (tex-decoder basecolor at the same coords),
    /// fineRes = the dual-grid voxel resolution the coords live in.
    public static func run(
        shapeFeats: MLXArray, coords: MLXArray, texBaseColor: MLXArray,
        fineRes: Float, remeshRes: Int = 256, targetFaces: Int = 120_000, atlasSize: Int = 1024,
        texMetallicRoughness: MLXArray? = nil, // [N,3] metallic, roughness, opacity in [0,1]
        cancellation: @escaping () -> Bool = { false },
        log: (String) -> Void = { _ in }
    ) throws -> (BakedMesh, BakeStageMetrics) {
        // Sub-stage timers: each `-t.timeIntervalSinceNow` read is preceded by the
        // MLX.eval that forces the stage's GPU work to complete, so the delta is real
        // wall-clock (MLX is lazy). Do not move a read ahead of its barrier.
        func check() throws {
            if cancellation() { throw WorkerFailure(code: "cancelled", message: "Request cancelled.") }
        }
        try check()
        var m = BakeStageMetrics()
        var t = Date()
        let margin: Float = 0.5
        let dv = (1 + 2 * margin) * MLX.sigmoid(shapeFeats[0..., 0..<3]) - margin
        let inter = shapeFeats[0..., 3..<6] .> 0
        let ql = softplus(shapeFeats[0..., 6..<7])
        let coords3 = coords[0..., 1..<4]

        // 1) raw dual-grid mesh
        let (rawV, rawF) = DualGridMesh.extract(coords: coords3, dualVerts: dv, intersected: inter, quadLerp: ql, gridSize: fineRes)
        MLX.eval(rawV, rawF)
        try check()
        guard rawV.dim(0) > 0, rawF.dim(0) > 0 else { throw WorkerFailure(code: "empty_geometry", message: "TRELLIS predicted no mesh faces.") }
        // keep the raw dual-grid mesh (verts sit ON the PBR voxel shell) as the
        // attribute-sampling reference surface — texel positions get BVH-remapped onto
        // it before sampling (matches postprocess.py's closest-point remap).
        let origMesh = Mesh(vertices: rawV, faces: rawF)
        let origBVH = origMesh.bvh()
        var mesh = origMesh
        m.rawDualgridS = -t.timeIntervalSinceNow // includes the raw-shell BVH build (CPU, synchronous)
        m.rawFaces = mesh.faceCount
        log("  raw dual-grid: \(mesh.vertexCount) verts, \(mesh.faceCount) faces, \(mesh.numConnectedComponents) components, \(mesh.numBoundaryEdges) boundary edges")

        // 2+3) clean (DC remesh, watertight) then unwrap, per backend.
        let finalMesh: Mesh
        let finalUVs: MLXArray
        let finalNormals: MLXArray
        let chartCount: Int
        // simplify to budget, then xatlas. Unwrap MUST come after remesh+
        // simplify: xatlas on the raw dual-grid mesh is pathologically slow
        // (~2.2h); on the cleaned mesh it's ~minutes (measured, Phase 0).
        t = Date()
        mesh = try mesh.remeshDualContouring(resolution: remeshRes, checkpoint: check)
        MLX.eval(mesh.vertices, mesh.faces)
        try check()
        m.remeshS = -t.timeIntervalSinceNow; m.remeshFaces = mesh.faceCount; m.simplifyFaces = mesh.faceCount
        log("  remeshed: \(mesh.vertexCount) verts, \(mesh.faceCount) faces, \(mesh.numConnectedComponents) components, \(mesh.numBoundaryEdges) boundary edges")
        if mesh.faceCount > targetFaces {
            t = Date()
            mesh = try mesh.simplify(targetNumFaces: targetFaces, checkpoint: check)
            MLX.eval(mesh.vertices, mesh.faces)
            try check()
            m.simplifyS = -t.timeIntervalSinceNow; m.simplifyFaces = mesh.faceCount
            log("  simplified: \(mesh.vertexCount) verts, \(mesh.faceCount) faces")
        }
        // Full xatlas unwrap, with progress cancellation through its C ABI.
        t = Date()
        try check()
        let uv: UVUnwrapResult
        do { uv = try mesh.uvUnwrap(cancellation: cancellation) } catch { try check(); throw error }
        try check()
        finalMesh = uv.mesh
        finalUVs = uv.uvs
        finalNormals = finalMesh.vertexNormals()
        chartCount = uv.charts.count
        m.unwrapS = -t.timeIntervalSinceNow // CPU xatlas, synchronous
        MLX.eval(finalMesh.vertices, finalMesh.faces, finalUVs)
        m.finalFaces = finalMesh.faceCount; m.charts = chartCount
        log("  unwrapped[xatlas]: \(finalMesh.vertexCount) verts, \(chartCount) charts")

        // 4) rasterize + bake. Center-inside rasterization leaves sub-texel
        // gaps along jagged chart seams (thin slivers never cover a texel
        // center); those gaps then inpaint-fill with unrepresentative color —
        // visible dark seam "cracks" (worst for provenance charts, whose
        // boundaries are staircase-jagged). Approximate conservative
        // rasterization with 4 half-texel-jittered passes appended after the
        // main one; later duplicate writes lose (first-write-wins below), so
        // jitter only fills otherwise-empty texels.
        // Supersample 2×: bake into a double-resolution grid and box-average
        // down. Each final texel becomes the mean of up to 4 surface samples —
        // fixes the foreshortened-seam-texel footprint problem (texel covers
        // ~2× surface, single center sample unrepresentative) and dilutes
        // single-sample crevice pinning (measured: 64%/36% of provenance's
        // remaining bad samples respectively).
        t = Date()
        let ss = 2
        let ssSize = atlasSize * ss
        let uvPix = finalUVs * Float(ssSize)
        var pixParts: [MLXArray] = [], fidParts: [MLXArray] = [], baryParts: [MLXArray] = []
        for (dx, dy): (Float, Float) in [(0, 0), (0.49, 0.49), (-0.49, 0.49), (0.49, -0.49), (-0.49, -0.49)] {
            try check()
            let jittered =
                dx == 0 && dy == 0
                ? uvPix
                : MLX.stacked([uvPix[0..., 0] + dx, uvPix[0..., 1] + dy], axis: 1)
            let (p, f, b) = try UVRasterize.rasterize(uvsPix: jittered, faces: finalMesh.faces, textureSize: ssSize, checkpoint: check)
            pixParts.append(p); fidParts.append(f); baryParts.append(b)
        }
        let pix = MLX.concatenated(pixParts, axis: 0)
        let fid = MLX.concatenated(fidParts, axis: 0)
        let bary = MLX.concatenated(baryParts, axis: 0)
        MLX.eval(pix, fid, bary)
        m.rasterizeS = -t.timeIntervalSinceNow
        try check()
        let K = fid.dim(0)
        guard K > 0 else { throw WorkerFailure(code: "empty_geometry", message: "TRELLIS mesh has no covered texture pixels.") }
        m.coveredTexelWrites = K
        log("  rasterized: \(K) covered texel-writes (5-pass conservative)")

        // surface position per texel: Σ bary · verts[faces[fid]]
        t = Date()
        let faceVerts = finalMesh.faces.take(fid, axis: 0) // [K,3] vertex ids
        let v0 = finalMesh.vertices.take(faceVerts[0..., 0], axis: 0) // [K,3]
        let v1 = finalMesh.vertices.take(faceVerts[0..., 1], axis: 0)
        let v2 = finalMesh.vertices.take(faceVerts[0..., 2], axis: 0)
        let posRaw = bary[0..., 0..<1] * v0 + bary[0..., 1..<2] * v1 + bary[0..., 2..<3] * v2 // [K,3]
        // remap each texel position onto the original on-shell surface, so trilinear hits the thin PBR shell
        let cp = origBVH.closestPoints(mesh: origMesh, queries: posRaw)
        let pos = cp.points // [K,3]
        // diagnostic: does the remap land on a raw face whose normal opposes
        // the texel's final-mesh face normal? (wrong-wall snap)
        let fe1 = v1 - v0, fe2 = v2 - v0
        let fnl = MLX.stacked(
            [
                fe1[0..., 1] * fe2[0..., 2] - fe1[0..., 2] * fe2[0..., 1],
                fe1[0..., 2] * fe2[0..., 0] - fe1[0..., 0] * fe2[0..., 2],
                fe1[0..., 0] * fe2[0..., 1] - fe1[0..., 1] * fe2[0..., 0]
            ], axis: 1)
        let rawFN = origMesh.faceNormals().take(cp.faceIndices, axis: 0)
        let flipDot = (fnl * rawFN).sum(axis: 1)
        MLX.eval(flipDot)
        let flipArr = flipDot.asArray(Float.self)
        MLX.eval(pos)
        let query = ((pos + 0.5) * fineRes).reshaped([1, K, 3])
        let sampled = GridSample3d.sample(feats: texBaseColor, coords: coords, grid: query, mode: "trilinear")[0] // [K,3]
        MLX.eval(sampled)
        var mrSampled: [Float]? = nil
        if let mr = texMetallicRoughness {
            let mrS = GridSample3d.sample(feats: mr, coords: coords, grid: query, mode: "trilinear")[0] // [K,3]
            MLX.eval(mrS)
            mrSampled = MLX.clip(mrS, min: 0, max: 1).asArray(Float.self)
        }
        m.bakeSampleS = -t.timeIntervalSinceNow // BVH closest-point remap + GridSample3d trilinear
        let qmin = query.min(), qmax = query.max()
        let nonzero = (sampled.sum(axis: 1) .> 0).asType(.float32).mean().item(Float.self)
        log(
            "  bake diag: baseColor mean=\(texBaseColor.mean().item(Float.self))  query[\(qmin.item(Float.self))..\(qmax.item(Float.self))]  sampled mean=\(sampled.mean().item(Float.self))  hitFrac=\(nonzero)"
        )

        // 5) write texels + host-side dilation inpaint
        t = Date()
        let px = pix.asType(.int32).asArray(Int32.self) // [K*2]
        let col = MLX.clip(sampled, min: 0, max: 1).asArray(Float.self) // [K*3]
        // --- write at supersample resolution, first-write-wins per sub-texel ---
        var ssRgb = [Float](repeating: 0, count: ssSize * ssSize * 3)
        var ssMr = mrSampled != nil ? [Float](repeating: 0, count: ssSize * ssSize * 3) : []
        var ssFilled = [Bool](repeating: false, count: ssSize * ssSize)
        var ssFlip = [Bool](repeating: false, count: ssSize * ssSize)
        for k in 0..<K {
            if k % 65536 == 0 { try check() }
            let x = Int(px[k * 2]), y = Int(px[k * 2 + 1])
            if x < 0 || x >= ssSize || y < 0 || y >= ssSize { continue } // jitter can push 1 texel out
            let p = y * ssSize + x
            if ssFilled[p] { continue } // first write wins: center pass beats jitter passes
            ssRgb[p * 3] = col[k * 3]; ssRgb[p * 3 + 1] = col[k * 3 + 1]; ssRgb[p * 3 + 2] = col[k * 3 + 2]
            if let mrS = mrSampled { ssMr[p * 3] = mrS[k * 3]; ssMr[p * 3 + 1] = mrS[k * 3 + 1]; ssMr[p * 3 + 2] = mrS[k * 3 + 2] }
            ssFilled[p] = true
            if flipArr[k] < 0 { ssFlip[p] = true }
        }

        // --- box-average down to the final atlas ---
        var rgb = [Float](repeating: 0, count: atlasSize * atlasSize * 3)
        var mrBuf = mrSampled != nil ? [Float](repeating: 0, count: atlasSize * atlasSize * 3) : []
        var filled = [Bool](repeating: false, count: atlasSize * atlasSize)
        var wallFlipped = [Bool](repeating: false, count: atlasSize * atlasSize)
        for y in 0..<atlasSize {
            if y % 64 == 0 { try check() }
            for x in 0..<atlasSize {
                var r: Float = 0, g: Float = 0, b: Float = 0, m: Float = 0, ro: Float = 0, opacity: Float = 0
                var cnt = 0
                var anyFlip = false
                for sy in 0..<ss {
                    for sx in 0..<ss {
                        let sp = (y * ss + sy) * ssSize + (x * ss + sx)
                        if ssFilled[sp] {
                            r += ssRgb[sp * 3]; g += ssRgb[sp * 3 + 1]; b += ssRgb[sp * 3 + 2]
                            if mrSampled != nil { m += ssMr[sp * 3]; ro += ssMr[sp * 3 + 1]; opacity += ssMr[sp * 3 + 2] }
                            if ssFlip[sp] { anyFlip = true }
                            cnt += 1
                        }
                    }
                }
                guard cnt > 0 else { continue }
                let p = y * atlasSize + x
                let inv = 1 / Float(cnt)
                rgb[p * 3] = r * inv; rgb[p * 3 + 1] = g * inv; rgb[p * 3 + 2] = b * inv
                if mrSampled != nil { mrBuf[p * 3] = m * inv; mrBuf[p * 3 + 1] = ro * inv; mrBuf[p * 3 + 2] = opacity * inv }
                filled[p] = true
                wallFlipped[p] = anyFlip
            }
        }
        m.writeDownsampleS = -t.timeIntervalSinceNow
        let filledBeforeInpaint = filled
        let coverage = Float(filled.lazy.filter { $0 }.count) / Float(atlasSize * atlasSize)
        m.coverage = coverage
        // Inpaint to COMPLETION, not a fixed ring count: any unfilled texel is
        // black, and the renderer's mip chain averages small charts against it,
        // producing dark shard artifacts at distance (user-visible; worst for
        // the provenance backend's lower packing utilization). The dilation
        // loop exits as soon as nothing new fills, so completion is cheap.
        t = Date()
        var mrFilled = filled // copy BEFORE base-color inpaint mutates it
        try dilateInpaint(&rgb, &filled, atlasSize, iterations: atlasSize, checkpoint: check)
        var mrRGBA: [UInt8]? = nil
        if mrSampled != nil {
            try dilateInpaint(&mrBuf, &mrFilled, atlasSize, iterations: atlasSize, checkpoint: check)
            var out = [UInt8](repeating: 255, count: atlasSize * atlasSize * 4)
            for p in 0..<(atlasSize * atlasSize) {
                if p % 262_144 == 0 { try check() }
                out[p * 4] = 0 // R unused
                out[p * 4 + 1] = UInt8(max(0, min(255, Int(mrBuf[p * 3 + 1] * 255 + 0.5)))) // G = roughness
                out[p * 4 + 2] = UInt8(max(0, min(255, Int(mrBuf[p * 3] * 255 + 0.5)))) // B = metallic
            }
            mrRGBA = out
        }

        var rgba = [UInt8](repeating: 0, count: atlasSize * atlasSize * 4)
        for p in 0..<(atlasSize * atlasSize) {
            if p % 262_144 == 0 { try check() }
            rgba[p * 4] = UInt8(max(0, min(255, Int(rgb[p * 3] * 255 + 0.5))))
            rgba[p * 4 + 1] = UInt8(max(0, min(255, Int(rgb[p * 3 + 1] * 255 + 0.5))))
            rgba[p * 4 + 2] = UInt8(max(0, min(255, Int(rgb[p * 3 + 2] * 255 + 0.5))))
            rgba[p * 4 + 3] = mrSampled == nil ? 255 : UInt8(clamping: Int(mrBuf[p * 3 + 2] * 255 + 0.5))
        }
        m.inpaintS = -t.timeIntervalSinceNow
        let baked = BakedMesh(
            vertices: finalMesh.vertices, faces: finalMesh.faces,
            normals: finalNormals, uvs: finalUVs, texRGBA: rgba,
            atlasSize: atlasSize, coverage: coverage, mrRGBA: mrRGBA,
            filledBeforeInpaint: filledBeforeInpaint, wallFlipped: wallFlipped)
        return (baked, m)
    }

    /// Dilation inpaint; `checkpoint` runs once per ring and per 64-row band, so a
    /// 4096² atlas (up to 4096 rings) stays cancellable.
    static func dilateInpaint(
        _ rgb: inout [Float], _ filled: inout [Bool], _ S: Int, iterations: Int, checkpoint: () throws -> Void = {}
    ) rethrows {
        for _ in 0..<iterations {
            try checkpoint()
            var newlyFilled = [(Int, Float, Float, Float)]()
            for y in 0..<S {
                if y % 64 == 0 { try checkpoint() }
                for x in 0..<S where !filled[y * S + x] {
                    var r: Float = 0, g: Float = 0, b: Float = 0, c = 0
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let nx = x + dx, ny = y + dy
                        if nx >= 0 && nx < S && ny >= 0 && ny < S && filled[ny * S + nx] {
                            let q = (ny * S + nx) * 3; r += rgb[q]; g += rgb[q + 1]; b += rgb[q + 2]; c += 1
                        }
                    }
                    if c > 0 { newlyFilled.append((y * S + x, r / Float(c), g / Float(c), b / Float(c))) }
                }
            }
            if newlyFilled.isEmpty { break }
            for (p, r, g, b) in newlyFilled { rgb[p * 3] = r; rgb[p * 3 + 1] = g; rgb[p * 3 + 2] = b; filled[p] = true }
        }
    }
}

// swiftlint:enable all
