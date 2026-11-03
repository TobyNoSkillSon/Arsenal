// SPDX-License-Identifier: MIT
// Host-side schedule, positions and RoPE tables. Pure Swift (no MLX), so they are unit-tested without a GPU.
// mflux reference: flux3_flow_map_scheduler.py, flux3_latent_creator.py (prepare_ids), flux3_pos_embed.py.
import Foundation

public enum Flux3Schedule {
    public static let alpha = 6.93
    public static let defaultSteps = 16

    /// mflux `Flux3FlowMapScheduler`: t = linspace(1, 0, N + 1) in float32 (MLX's (1 − i/N)·1 + (i/N)·0), then the
    /// flow-map warp α·t / (1 + (α − 1)·t), every operation in float32 as on the GPU.
    public static func timesteps(steps: Int) -> [Float] {
        precondition(steps >= 1)
        let a = Float(alpha), am1 = Float(alpha - 1)
        return (0...steps).map { i in
            let s = Float(i) / Float(steps)
            let t = (1 - s) * 1 + s * 0
            return a * t / (1 + am1 * t)
        }
    }
}

public enum Flux3Positions {
    /// The DiT's RoPE scale for image rows and columns (mflux IMAGE_ROPE_SCALE).
    public static let imageScale: Float = 0.5

    /// (t, row, column, layer) per latent token, row-major; layer 0 is the target, −1, −2, … the references in order.
    public static func imageIDs(gridH: Int, gridW: Int, layer: Int = 0) -> [[Float]] {
        var out: [[Float]] = []; out.reserveCapacity(gridH * gridW)
        for r in 0..<gridH { for c in 0..<gridW { out.append([0, Float(r) * imageScale, Float(c) * imageScale, Float(layer)]) } }
        return out
    }
}

public enum Flux3Rope {
    /// cos and sin tables [N, Σ axes / 2] for 4-axis ids: angles in float64 on the host (as mflux and the vendor
    /// do), stored as float32. Axis a covers columns [Σ_{<a} d/2, Σ_{≤a} d/2).
    public static func tables(_ ids: [[Float]], axes: [Int], theta: Double) -> (cos: [Float], sin: [Float]) {
        let half = axes.reduce(0, +) / 2
        let omegas = axes.map { d in stride(from: 0, to: d, by: 2).map { 1.0 / Foundation.pow(theta, Double($0) / Double(d)) } }
        var cos = [Float](repeating: 0, count: ids.count * half), sin = cos
        for (n, id) in ids.enumerated() {
            var column = n * half
            for (axis, omega) in omegas.enumerated() {
                let p = Double(id[axis])
                for w in omega {
                    let angle = p * w
                    cos[column] = Float(Foundation.cos(angle)); sin[column] = Float(Foundation.sin(angle)); column += 1
                }
            }
        }
        return (cos, sin)
    }
}

/// Reference-image preparation, the geometry half (mflux `Flux3ReferenceImages`, mirroring the vendor's pipeline).
public enum Flux3Canvas {
    public static let ratio = 16
    public static let matchMaxPixels = 4 * 1024 * 1024
    public static let minSide = 64, maxSide = 4096
    public static let maxReferences = 10
    public static let multiReferenceMaxPixels = 1024 * 1024

    public struct Fit: Equatable, Sendable {
        public var fit: (width: Int, height: Int)
        public var canvas: (width: Int, height: Int)
        public static func == (a: Fit, b: Fit) -> Bool { a.fit == b.fit && a.canvas == b.canvas }
    }

    static func ceil16(_ v: Int) -> Int { (v + ratio - 1) / ratio * ratio }

    /// A single reference: scale to at most 4 MP (padded canvas included) and 4096 per side; pad up to multiples of 16.
    public static func single(width: Int, height: Int) -> Fit {
        var scale = Swift.min(1.0, (Double(matchMaxPixels) / Double(width * height)).squareRoot(), Double(maxSide) / Double(Swift.max(width, height)))
        while true {
            let fw = Swift.max(1, Int((Double(width) * scale).rounded(.toNearestOrEven)))
            let fh = Swift.max(1, Int((Double(height) * scale).rounded(.toNearestOrEven)))
            let cw = Swift.min(maxSide, Swift.max(minSide, ceil16(fw))), ch = Swift.min(maxSide, Swift.max(minSide, ceil16(fh)))
            if cw * ch <= matchMaxPixels { return Fit(fit: (fw, fh), canvas: (cw, ch)) }
            scale *= 0.995
        }
    }

    /// One of several references: scale to at most 1 MP (floored), then centre-crop to multiples of 16.
    /// Returns the scaled size and the crop rectangle (x, y, width, height) inside it.
    public static func multi(width: Int, height: Int) throws -> (scaled: (width: Int, height: Int), crop: (x: Int, y: Int, width: Int, height: Int)) {
        var w = width, h = height
        if w * h > multiReferenceMaxPixels {
            let s = (Double(multiReferenceMaxPixels) / Double(w * h)).squareRoot()
            w = Int(Double(w) * s); h = Int(Double(h) * s)
        }
        let cw = w / ratio * ratio, ch = h / ratio * ratio
        guard cw > 0, ch > 0 else { throw Flux3Error("A reference image is too small to condition on: it needs at least 16 pixels per side.") }
        return ((w, h), ((w - cw) / 2, (h - ch) / 2, cw, ch))
    }

    /// numpy `np.pad` source index for a padded position: "reflect" (edge not repeated) or, for sides of 16 pixels or
    /// fewer, "symmetric" (edge repeated), with numpy's repeated reflection when the pad exceeds the side.
    public static func padIndex(_ i: Int, size n: Int, symmetric: Bool) -> Int {
        if n == 1 { return 0 }
        let period = symmetric ? 2 * n : 2 * (n - 1)
        var k = i % period; if k < 0 { k += period }
        if symmetric { return k < n ? k : period - 1 - k }
        return k < n ? k : period - k
    }

    /// The output size and crop for an edit (mflux `resolve_canvas`): with no size, or a size equal to the single
    /// reference's fitted box, generate on the reference canvas and crop back to the box.
    public static func resolve(canvas: (width: Int, height: Int), box: (x: Int, y: Int, width: Int, height: Int)?, width: Int?, height: Int?)
        -> (width: Int, height: Int, crop: (x: Int, y: Int, width: Int, height: Int)?)
    {
        if let box, (width == nil && height == nil) || (width == box.width && height == box.height) {
            return (canvas.width, canvas.height, box)
        }
        return (width ?? canvas.width, height ?? canvas.height, nil)
    }
}
