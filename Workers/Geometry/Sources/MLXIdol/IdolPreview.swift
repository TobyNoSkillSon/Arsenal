// SPDX-License-Identifier: MIT
// A small CPU splat rasterizer for the preview image: each Gaussian's 3D covariance is projected to the screen
// (the 3D Gaussian Splatting EWA approximation) and the splats are alpha-composited front to back.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import simd
import UniformTypeIdentifiers

public enum IdolPreview {
    /// Renders `g` (IDOL's z-up frame) seen from yaw / pitch (radians) at `radius`, vertical field of view `fov`
    /// (degrees). Returns premultiplied RGBA bytes [size × size × 4]; uncovered pixels are transparent.
    public static func render(_ g: IdolGaussians, size: Int, yaw: Double, pitch: Double, radius: Double = 2.2, fov: Double = 40) -> [UInt8] {
        eval(g.xyz, g.featuresDC, g.scaling, g.rotation, g.opacity)
        let xyz = g.xyz.asArray(Float.self), dc = g.featuresDC.asArray(Float.self), sc = g.scaling.asArray(Float.self)
        let rot = g.rotation.asArray(Float.self), op = g.opacity.asArray(Float.self)
        let n = g.count
        // camera on the orbit, looking at the origin, z up
        let c = SIMD3<Double>(radius * sin(yaw) * cos(pitch), radius * cos(yaw) * cos(pitch), radius * sin(pitch))
        let forward = simd_normalize(-c)
        var right = simd_cross(forward, SIMD3<Double>(0, 0, 1))
        if simd_length(right) < 1e-6 { right = SIMD3<Double>(1, 0, 0) }
        right = simd_normalize(right)
        let up = simd_cross(right, forward)
        let focal = Double(size) / 2 / tan(fov * .pi / 360)
        struct Splat { var depth: Double; var u: Double; var v: Double; var a: Double; var b: Double; var c: Double; var r: Int; var rgb: SIMD3<Double>; var o: Double }
        var splats: [Splat] = []
        splats.reserveCapacity(n)
        for i in 0..<n {
            let p = SIMD3<Double>(Double(xyz[i * 3]), Double(xyz[i * 3 + 1]), Double(xyz[i * 3 + 2])) - c
            let x = simd_dot(p, right), y = simd_dot(p, up), z = simd_dot(p, forward)
            guard z > 0.05 else { continue }
            // R S² Rᵀ from the unit quaternion (w, x, y, z)
            let w = Double(rot[i * 4]), qx = Double(rot[i * 4 + 1]), qy = Double(rot[i * 4 + 2]), qz = Double(rot[i * 4 + 3])
            let rm = simd_double3x3(rows: [
                SIMD3(1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy)),
                SIMD3(2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx)),
                SIMD3(2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy))
            ])
            let s = simd_double3x3(diagonal: SIMD3(Double(sc[i * 3]), Double(sc[i * 3 + 1]), Double(sc[i * 3 + 2])))
            let m = rm * s
            let sigma = m * m.transpose
            let view = simd_double3x3(rows: [right, up, forward])
            let j = simd_double3x2(rows: [SIMD3(focal / z, 0, -focal * x / (z * z)), SIMD3(0, -focal / z, focal * y / (z * z))])
            let t = j * view
            let cov = t * sigma * t.transpose
            let a = cov[0][0] + 0.3, b = cov[1][0], cc = cov[1][1] + 0.3
            let det = a * cc - b * b
            guard det > 0 else { continue }
            let mid = (a + cc) / 2, lambda = mid + (max(0.1, mid * mid - det)).squareRoot()
            let rgb = SIMD3<Double>(Double(dc[i * 3]), Double(dc[i * 3 + 1]), Double(dc[i * 3 + 2])) * 0.28209479177387814 + 0.5
            splats.append(
                Splat(
                    depth: z, u: Double(size) / 2 + focal * x / z, v: Double(size) / 2 - focal * y / z,
                    a: cc / det, b: -b / det, c: a / det, r: Int(ceil(3 * lambda.squareRoot())),
                    rgb: simd_clamp(rgb, SIMD3(repeating: 0), SIMD3(repeating: 1)), o: Double(op[i])))
        }
        splats.sort { $0.depth < $1.depth }
        var color = [SIMD3<Double>](repeating: .zero, count: size * size)
        var transmittance = [Double](repeating: 1, count: size * size)
        for s in splats {
            let x0 = max(0, Int(s.u) - s.r), x1 = min(size - 1, Int(s.u) + s.r)
            let y0 = max(0, Int(s.v) - s.r), y1 = min(size - 1, Int(s.v) + s.r)
            guard x0 <= x1, y0 <= y1 else { continue }
            for py in y0...y1 {
                for px in x0...x1 {
                    let k = py * size + px
                    guard transmittance[k] > 1e-4 else { continue }
                    let dx = Double(px) + 0.5 - s.u, dy = Double(py) + 0.5 - s.v
                    let power = -0.5 * (s.a * dx * dx + 2 * s.b * dx * dy + s.c * dy * dy)
                    guard power <= 0 else { continue }
                    let alpha = min(0.99, s.o * exp(power))
                    guard alpha >= 1.0 / 255 else { continue }
                    color[k] += transmittance[k] * alpha * s.rgb
                    transmittance[k] *= 1 - alpha
                }
            }
        }
        var out = [UInt8](repeating: 0, count: size * size * 4)
        for k in 0..<size * size {
            let rgb = simd_clamp(color[k], SIMD3(repeating: 0), SIMD3(repeating: 1))
            out[k * 4] = UInt8((rgb.x * 255).rounded()); out[k * 4 + 1] = UInt8((rgb.y * 255).rounded()); out[k * 4 + 2] = UInt8((rgb.z * 255).rounded())
            out[k * 4 + 3] = UInt8(((1 - transmittance[k]) * 255).rounded())
        }
        return out
    }

    /// The camera distance at which the object's bounding sphere (the 99th percentile of the opaque Gaussians' distance
    /// from the origin) fills `fill` of the frame at vertical field of view `fov` (degrees).
    public static func fittingRadius(_ g: IdolGaussians, fov: Double, fill: Double = 0.85) -> Double {
        let mask = (g.opacity .> 0.5)[0..., 0].asArray(Bool.self)
        let norms = MLX.sqrt((g.xyz * g.xyz).sum(axis: 1)).asArray(Float.self)
        let kept = zip(norms, mask).filter(\.1).map(\.0).sorted()
        let r = Double(kept.isEmpty ? 0.5 : kept[kept.count * 99 / 100])
        return max(0.2, r / sin(fov * .pi / 360) / fill)
    }

    /// Writes premultiplied RGBA (channels 4) or RGB (channels 3) bytes as a PNG.
    public static func writePNG(_ bytes: [UInt8], width: Int, height: Int, channels: Int, to url: URL) throws {
        var rgba = bytes
        if channels == 3 {
            rgba = [UInt8](repeating: 255, count: width * height * 4)
            for i in 0..<width * height { for c in 0..<3 { rgba[i * 4 + c] = bytes[i * 3 + c] } }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
            let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw IdolError("Cannot encode the preview PNG.") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw IdolError("Cannot write the preview PNG.") }
    }
}
