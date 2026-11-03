// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import IntegrationKit
import MLX

/// Deterministic textured triangle preview of the actual returned mesh, with a depth buffer.
public enum TrellisPreview {
    /// RGBA byte offset of the texel a UV samples, in the bake's convention: row `v * size`
    /// (UVRasterize writes texel rows from `v`, and glTF export keeps `v` with its top-left origin).
    static func texelOffset(u: Float, v: Float, atlasSize: Int) -> Int {
        let tx = Swift.max(0, Swift.min(atlasSize - 1, Int(u * Float(atlasSize))))
        let ty = Swift.max(0, Swift.min(atlasSize - 1, Int(v * Float(atlasSize))))
        return (ty * atlasSize + tx) * 4
    }

    public static func write(_ mesh: BakedMesh, to url: URL, cancellation: NativeCancellation, size: Int = 512) throws {
        let positions = mesh.vertices.asType(.float32).asArray(Float.self), indices = mesh.faces.asType(.int32).asArray(Int32.self)
        let uv = mesh.uvs.asType(.float32).asArray(Float.self)
        guard !positions.isEmpty, positions.allSatisfy(\.isFinite), !indices.isEmpty, uv.allSatisfy(\.isFinite) else { throw NSError(domain: "TrellisPreview", code: 1) }
        let count = positions.count / 3
        var projected = [SIMD3<Float>](); projected.reserveCapacity(count)
        for i in 0..<count {
            let x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2]
            let horizontal = 0.8253356 * x - 0.5646425 * z
            let depth = 0.5646425 * x + 0.8253356 * z
            projected.append(SIMD3(horizontal, 0.9553365 * y - 0.2955202 * depth, 0.2955202 * y + 0.9553365 * depth))
        }
        let minX = projected.map(\.x).min()!, maxX = projected.map(\.x).max()!
        let minY = projected.map(\.y).min()!, maxY = projected.map(\.y).max()!
        let scale = Float(size) * 0.85 / Swift.max(maxX - minX, maxY - minY, 1e-6)
        let centerX = (maxX + minX) * 0.5, centerY = (maxY + minY) * 0.5
        // The mesh is Y-up (glTF); image rows grow downward, so up maps to smaller row numbers.
        projected = projected.map { SIMD3(($0.x - centerX) * scale + Float(size) * 0.5, (centerY - $0.y) * scale + Float(size) * 0.5, $0.z) }
        var pixels = [UInt8](repeating: 0, count: size * size * 4), depth = [Float](repeating: -.infinity, count: size * size)
        for face in 0..<(indices.count / 3) {
            if face % 128 == 0 { try cancellation.check() }
            let ids = (0..<3).map { Int(indices[face * 3 + $0]) }
            guard ids.allSatisfy({ $0 >= 0 && $0 < count }) else { throw NSError(domain: "TrellisPreview", code: 2) }
            let a = projected[ids[0]], b = projected[ids[1]], c = projected[ids[2]]
            let denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
            if abs(denominator) < 1e-8 { continue }
            let left = Swift.max(0, Int(floor(Swift.min(a.x, b.x, c.x)))), right = Swift.min(size - 1, Int(ceil(Swift.max(a.x, b.x, c.x))))
            let top = Swift.max(0, Int(floor(Swift.min(a.y, b.y, c.y)))), bottom = Swift.min(size - 1, Int(ceil(Swift.max(a.y, b.y, c.y))))
            if left > right || top > bottom { continue }
            for y in top...bottom {
                for x in left...right {
                    let px = Float(x) + 0.5, py = Float(y) + 0.5
                    let wa = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denominator
                    let wb = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denominator
                    let wc = 1 - wa - wb
                    if wa < 0 || wb < 0 || wc < 0 { continue }
                    let z = wa * a.z + wb * b.z + wc * c.z, index = y * size + x
                    if z <= depth[index] { continue }
                    let u = wa * uv[ids[0] * 2] + wb * uv[ids[1] * 2] + wc * uv[ids[2] * 2]
                    let v = wa * uv[ids[0] * 2 + 1] + wb * uv[ids[1] * 2 + 1] + wc * uv[ids[2] * 2 + 1]
                    let texture = texelOffset(u: u, v: v, atlasSize: mesh.atlasSize)
                    if mesh.texRGBA[texture + 3] == 0 { continue }
                    depth[index] = z
                    for channel in 0..<4 { pixels[index * 4 + channel] = mesh.texRGBA[texture + channel] }
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
            let image = CGImage(
                width: size, height: size, bitsPerComponent: 8,
                bitsPerPixel: 32, bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw NSError(domain: "TrellisPreview", code: 3) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw NSError(domain: "TrellisPreview", code: 4) }
    }
}
