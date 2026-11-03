// SPDX-License-Identifier: MIT
import Foundation
import MLX
import GeometryXatlas

public struct UVUnwrapResult {
    public let mesh: Mesh
    public let uvs: MLXArray
    public let vertexMap: MLXArray
    public let charts: [Int]
}
private final class AtlasCancellation {
    let check: () -> Bool
    init(_ check: @escaping () -> Bool) { self.check = check }
}
extension Mesh {
    public func uvUnwrap(resolution: Int = 1024, cancellation: @escaping () -> Bool = { false }) throws -> UVUnwrapResult {
        let positions = vertices.asType(.float32).asArray(Float.self)
        let indexValues = faces.asType(.int32).asArray(Int32.self)
        guard !positions.isEmpty, positions.allSatisfy(\.isFinite), !indexValues.isEmpty,
            indexValues.allSatisfy({ $0 >= 0 && Int($0) < vertexCount })
        else { throw NSError(domain: "GeometryAtlas", code: 1) }
        let indices = indexValues.map { UInt32($0) }
        let state = AtlasCancellation(cancellation)
        var result = GeometryAtlas()
        let status = positions.withUnsafeBufferPointer { p in
            indices.withUnsafeBufferPointer { i in
                geometry_atlas_generate(
                    p.baseAddress, UInt32(vertexCount), i.baseAddress, UInt32(indices.count), UInt32(resolution),
                    { pointer in
                        guard let pointer else { return 1 }
                        return Unmanaged<AtlasCancellation>.fromOpaque(pointer).takeUnretainedValue().check() ? 1 : 0
                    }, Unmanaged.passUnretained(state).toOpaque(), &result)
            }
        }
        defer { geometry_atlas_free(&result) }
        guard status == 0, let vertices = result.positions, let uvs = result.uvs,
            let faces = result.indices, let map = result.vertex_map
        else { throw NSError(domain: "GeometryAtlas", code: Int(status)) }
        let count = Int(result.vertex_count), indexCount = Int(result.index_count)
        return UVUnwrapResult(
            mesh: Mesh(
                vertices: MLXArray(Array(UnsafeBufferPointer(start: vertices, count: count * 3)), [count, 3]),
                faces: MLXArray(Array(UnsafeBufferPointer(start: faces, count: indexCount)).map { Int32($0) }, [indexCount / 3, 3])),
            uvs: MLXArray(Array(UnsafeBufferPointer(start: uvs, count: count * 2)), [count, 2]),
            vertexMap: MLXArray(Array(UnsafeBufferPointer(start: map, count: count)).map { Int32($0) }), charts: Array(0..<Int(result.chart_count)))
    }
}
