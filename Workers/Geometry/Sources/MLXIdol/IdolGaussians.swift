// SPDX-License-Identifier: MIT
// The object's Gaussians (idol/gaussian.py) and the standard 3D Gaussian Splatting PLY writer.
import Foundation
import MLX
import MLXNN

/// Activated Gaussians: xyz [N, 3] (z-up, inside [-0.5, 0.5]^3), colour as degree-0 spherical harmonics [N, 3],
/// scaling [N, 3], unit quaternions (w, x, y, z) [N, 4], opacity [N, 1]. All float32.
public struct IdolGaussians {
    public let xyz: MLXArray, featuresDC: MLXArray, scaling: MLXArray, rotation: MLXArray, opacity: MLXArray

    public init(rawXYZ: MLXArray, featuresDC: MLXArray, rawScaling: MLXArray, rawRotation: MLXArray, rawOpacity: MLXArray, representation r: IdolRepresentation) {
        func inverseSoftplus(_ x: Float) -> Float { x + Float(log(-expm1(-Double(x)))) }
        func inverseSigmoid(_ x: Float) -> Float { Float(log(Double(x) / (1 - Double(x)))) }
        xyz = rawXYZ - 0.5
        self.featuresDC = featuresDC
        let s = softplus(rawScaling + inverseSoftplus(r.scalingBias))
        scaling = MLX.sqrt(s * s + r.filterKernel * r.filterKernel)
        let q = rawRotation + MLXArray([Float(1), 0, 0, 0]).reshaped([1, 4])
        rotation = q / MLX.maximum(MLX.sqrt((q * q).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        opacity = sigmoid(rawOpacity + inverseSigmoid(r.opacityBias))
    }

    public var count: Int { xyz.dim(0) }

    /// The PLY columns in file order: x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3, turned from IDOL's z-up
    /// to the y-up convention viewers use ((x, y, z) -> (x, -z, y), the rotation pre-multiplied by +90° about x), with
    /// opacity and scale stored as logit and log after clamping so no value is infinite. [N, 17] float32.
    public func plyColumns() -> MLXArray {
        let eps: Float = 1e-6
        let x = xyz[0..., 0..<1], y = xyz[0..., 1..<2], z = xyz[0..., 2..<3]
        let o = MLX.clip(opacity, min: eps, max: 1 - eps)
        let logit = MLX.log(o / (1 - o))
        let logScale = MLX.log(MLX.maximum(scaling, MLXArray(eps)))
        let h = Float(0.5).squareRoot()
        let w = rotation[0..., 0..<1], qx = rotation[0..., 1..<2], qy = rotation[0..., 2..<3], qz = rotation[0..., 3..<4]
        // (h, h, 0, 0) * (w, x, y, z), Hamilton product
        let rot = MLX.concatenated([h * w - h * qx, h * qx + h * w, h * qy - h * qz, h * qz + h * qy], axis: 1)
        return MLX.concatenated([x, -z, y, MLXArray.zeros([count, 3]), featuresDC, logit, logScale, rot], axis: 1).asType(.float32)
    }

    public static let plyProperties = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
    ]

    /// Binary little-endian PLY with one float vertex element, as plyfile writes it.
    public func writePLY(to url: URL) throws {
        let columns = plyColumns()
        eval(columns)
        let values = columns.asArray(Float.self)
        guard values.allSatisfy({ $0.isFinite }) else { throw IdolError("The Gaussians contain non-finite values.") }
        var data = Data("ply\nformat binary_little_endian 1.0\nelement vertex \(count)\n".utf8)
        for name in Self.plyProperties { data.append(Data("property float \(name)\n".utf8)) }
        data.append(Data("end_header\n".utf8))
        values.withUnsafeBufferPointer { buffer in
            data.append(UnsafeBufferPointer(start: UnsafeRawPointer(buffer.baseAddress!).assumingMemoryBound(to: UInt8.self), count: buffer.count * 4))
        }
        try data.write(to: url, options: .atomic)
    }
}

/// Predicted input camera: yaw and pitch (radians), radius, vertical field of view (degrees). camera.py decode_camera.
public struct IdolCamera: Equatable, Sendable {
    public let yaw: Double, pitch: Double, radius: Double, fov: Double
    public init(latent v: [Float]) {
        let norm = (Double(v[0]) * Double(v[0]) + Double(v[1]) * Double(v[1]) + Double(v[2]) * Double(v[2])).squareRoot()
        let (dx, dy, dz) = norm > 1e-6 ? (Double(v[0]) / norm, Double(v[1]) / norm, Double(v[2]) / norm) : (0, 0, 1)
        let inverseDistance = Double(v[3]), halfWidth = Double(v[4])
        if inverseDistance < 1e-3 {
            radius = 1000; fov = 2 * atan(halfWidth / 1000) * 180 / .pi
        } else {
            radius = 1 / inverseDistance; fov = 2 * atan(halfWidth * inverseDistance) * 180 / .pi
        }
        yaw = atan2(dx, dy); pitch = asin(min(1, max(-1, dz)))
    }
}

public struct IdolError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}
