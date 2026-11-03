// SPDX-License-Identifier: MIT
// torchmcubes' CPU marching cubes (cxx/mcubes_cpu.cpp @ a9ac95a), the extractor TripoSR calls, plus a GLB writer.
// Same corner order, edge interpolation (with its 1e-4 snaps), table walk and exact-position vertex merge, so the mesh has
// the same vertices and faces for the same field. Positions come back as (i, j, k) grid units, as TripoSR's [2, 1, 0] swap gives.
import Foundation

struct Mesh { var vertices: [Float]; var faces: [UInt32] }

private struct Key: Hashable { let x: UInt32, y: UInt32, z: UInt32 }

/// field[i][j][k] flattened (i slowest), n per side; surface at 0, a corner is "below" when its value < 0.
func marchingCubes(_ field: [Float], n: Int) -> Mesh {
    let indexTable = [0, 1, 4, 5, 3, 2, 7, 6]
    let edgeCorners = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    var vertices: [Float] = [], faces: [UInt32] = []
    var unique: [Key: UInt32] = [:]
    var px = [Float](repeating: 0, count: 8), py = px, pz = px, val = px
    var vx = [Float](repeating: 0, count: 12), vy = vx, vz = vx
    field.withUnsafeBufferPointer { f in
        for z in 0..<(n - 1) {  // torchmcubes' z is the first axis (TripoSR's x)
            for y in 0..<(n - 1) {
                for x in 0..<(n - 1) {
                    var cube = 0
                    for c in 0..<8 {
                        let dx = c % 2, dy = (c / 2) % 2, dz = c / 4
                        let slot = indexTable[c]
                        let v = f[((z + dz) * n + (y + dy)) * n + (x + dx)]
                        val[slot] = v; px[slot] = Float(x + dx); py[slot] = Float(y + dy); pz[slot] = Float(z + dz)
                    }
                    for c in 0..<8 where val[c] < 0 { cube |= 1 << c }
                    let edges = mcEdgeTable[cube]
                    if edges == 0 { continue }
                    for e in 0..<12 where edges & (1 << e) != 0 {
                        let (a, b) = edgeCorners[e]
                        let va = val[a], vb = val[b]
                        if abs(0 - va) < 1e-4 { vx[e] = px[a]; vy[e] = py[a]; vz[e] = pz[a]; continue }
                        if abs(0 - vb) < 1e-4 { vx[e] = px[b]; vy[e] = py[b]; vz[e] = pz[b]; continue }
                        if abs(va - vb) < 1e-4 { vx[e] = px[a]; vy[e] = py[a]; vz[e] = pz[a]; continue }
                        let mu = (0 - va) / (vb - va)
                        vx[e] = px[a] + mu * (px[b] - px[a]); vy[e] = py[a] + mu * (py[b] - py[a]); vz[e] = pz[a] + mu * (pz[b] - pz[a])
                    }
                    let row = mcTriTable[cube]
                    var t = 0
                    while row[t] != -1 {
                        for e in row[t..<(t + 3)] {
                            let key = Key(x: vx[e].bitPattern, y: vy[e].bitPattern, z: vz[e].bitPattern)
                            if let index = unique[key] {
                                faces.append(index)
                            } else {
                                let index = UInt32(vertices.count / 3)
                                unique[key] = index
                                vertices += [vz[e], vy[e], vx[e]]  // (z, y, x) -> TripoSR's (i, j, k)
                                faces.append(index)
                            }
                        }
                        t += 3
                    }
                }
            }
        }
    }
    return Mesh(vertices: vertices, faces: faces)
}

/// A vertex-coloured GLB. TripoSR's own GLB path (gradio_app.py) turns the mesh upright with to_gradio_3d_orientation
/// (-90 degrees about x, then 90 about y): (x, y, z) -> (-y, z, -x); a further 180 degrees about y turns the object's front
/// to +Z, as glTF specifies, so the whole map is (x, y, z) -> (y, z, x). Faces are wound outward (flipped when the signed
/// volume is negative), smooth normals are area-weighted, and colours (sRGB 0...1 from the decoder) are stored linear,
/// as glTF COLOR_0 requires.
func writeGLB(positions world: [Float], colors: [Float], faces input: [UInt32], to url: URL) throws {
    let count = world.count / 3
    var pos = [Float](repeating: 0, count: world.count)
    for v in 0..<count {
        let x = world[v * 3], y = world[v * 3 + 1], z = world[v * 3 + 2]
        pos[v * 3] = y; pos[v * 3 + 1] = z; pos[v * 3 + 2] = x
    }
    var faces = input
    var volume: Double = 0
    for t in stride(from: 0, to: faces.count, by: 3) {
        let a = Int(faces[t]) * 3, b = Int(faces[t + 1]) * 3, c = Int(faces[t + 2]) * 3
        let ax = Double(pos[a]), ay = Double(pos[a + 1]), az = Double(pos[a + 2])
        let bx = Double(pos[b]), by = Double(pos[b + 1]), bz = Double(pos[b + 2])
        let cx = Double(pos[c]), cy = Double(pos[c + 1]), cz = Double(pos[c + 2])
        volume += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    }
    if volume < 0 { for t in stride(from: 0, to: faces.count, by: 3) { faces.swapAt(t + 1, t + 2) } }
    var normals = [Float](repeating: 0, count: pos.count)
    for t in stride(from: 0, to: faces.count, by: 3) {
        let ia = Int(faces[t]) * 3, ib = Int(faces[t + 1]) * 3, ic = Int(faces[t + 2]) * 3
        let e1 = (pos[ib] - pos[ia], pos[ib + 1] - pos[ia + 1], pos[ib + 2] - pos[ia + 2])
        let e2 = (pos[ic] - pos[ia], pos[ic + 1] - pos[ia + 1], pos[ic + 2] - pos[ia + 2])
        let nx = e1.1 * e2.2 - e1.2 * e2.1, ny = e1.2 * e2.0 - e1.0 * e2.2, nz = e1.0 * e2.1 - e1.1 * e2.0
        for i in [ia, ib, ic] { normals[i] += nx; normals[i + 1] += ny; normals[i + 2] += nz }
    }
    for v in 0..<count {
        let l = (normals[v * 3] * normals[v * 3] + normals[v * 3 + 1] * normals[v * 3 + 1] + normals[v * 3 + 2] * normals[v * 3 + 2]).squareRoot()
        if l > 0 { normals[v * 3] /= l; normals[v * 3 + 1] /= l; normals[v * 3 + 2] /= l } else { normals[v * 3 + 1] = 1 }
    }
    let linear = colors.map { c -> Float in
        let s = min(max(c, 0), 1)
        return s <= 0.04045 ? s / 12.92 : pow((s + 0.055) / 1.055, 2.4)
    }
    var lo: [Float] = [.infinity, .infinity, .infinity], hi: [Float] = [-.infinity, -.infinity, -.infinity]
    for v in 0..<count { for a in 0..<3 { lo[a] = min(lo[a], pos[v * 3 + a]); hi[a] = max(hi[a], pos[v * 3 + a]) } }

    var bin = Data()
    func append<T>(_ array: [T]) -> (Int, Int) {
        let offset = bin.count
        array.withUnsafeBufferPointer { bin.append(Data(buffer: $0)) }
        while bin.count % 4 != 0 { bin.append(0) }
        return (offset, bin.count - offset)
    }
    let p = append(pos), nrm = append(normals), col = append(linear), idx = append(faces)
    let json: [String: Any] = [
        "asset": ["version": "2.0", "generator": "Arsenal TripoSR (MLX)"],
        "scene": 0, "scenes": [["nodes": [0]]], "nodes": [["mesh": 0]],
        "meshes": [["primitives": [["attributes": ["POSITION": 0, "NORMAL": 1, "COLOR_0": 2], "indices": 3, "material": 0, "mode": 4]]]],
        "materials": [["pbrMetallicRoughness": ["baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1], "doubleSided": true]],
        "buffers": [["byteLength": bin.count]],
        "bufferViews": [
            ["buffer": 0, "byteOffset": p.0, "byteLength": count * 12, "target": 34962],
            ["buffer": 0, "byteOffset": nrm.0, "byteLength": count * 12, "target": 34962],
            ["buffer": 0, "byteOffset": col.0, "byteLength": count * 12, "target": 34962],
            ["buffer": 0, "byteOffset": idx.0, "byteLength": faces.count * 4, "target": 34963],
        ],
        "accessors": [
            ["bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3", "min": lo, "max": hi],
            ["bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3"],
            ["bufferView": 2, "componentType": 5126, "count": count, "type": "VEC3"],
            ["bufferView": 3, "componentType": 5125, "count": faces.count, "type": "SCALAR"],
        ],
    ]
    var jsonData = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
    while jsonData.count % 4 != 0 { jsonData.append(0x20) }
    var out = Data()
    func u32(_ v: Int) { var x = UInt32(v).littleEndian; withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
    u32(0x4654_6C67); u32(2); u32(12 + 8 + jsonData.count + 8 + bin.count)
    u32(jsonData.count); u32(0x4E4F_534A); out.append(jsonData)
    u32(bin.count); u32(0x004E_4942); out.append(bin)
    try out.write(to: url)
}
