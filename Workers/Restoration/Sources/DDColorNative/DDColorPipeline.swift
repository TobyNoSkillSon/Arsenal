// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// piddnad/DDColor `ddcolor/pipeline.py` ColorizationPipeline.process, on 8-bit RGB pixels:
/// float image, original L at full size, cv2 bilinear resize to the input size, L only through Lab with a=b=0 back to
/// RGB, the model, nearest resize of ab to full size, Lab(orig L, ab) to RGB, round to 8 bits. The colour math runs in
/// float32 (image processing around the model, as the provider does); the network runs in the model's half dtype.
public enum DDColorPipeline {
    /// `rgb` is H*W*3 bytes. Returns H*W*3 bytes.
    public static func colorize(_ model: DDColorModel, rgb: [UInt8], width: Int, height: Int, inputSize: Int = 512) -> [UInt8] {
        let image = MLXArray(rgb.map { Float($0) / 255 }, [height, width, 3])
        let bgr = flip(image)
        let origL = DDColorLab.bgrToLab(bgr)[0..., 0..., 0..<1]
        let small = bilinear(bgr, height: inputSize, width: inputSize)
        let gray = DDColorLab.grayRGBInput(fromBGR: small)
        let ab = model(gray).asType(.float32)
        let abFull = nearest(ab[0], height: height, width: width)
        let out = flip(DDColorLab.recombine(originalL: origL, modelAB: abFull.expandedDimensions(axis: 0)))
        let bytes = MLX.round(MLX.clip(out, min: 0, max: 1) * 255).asType(.uint8)
        eval(bytes)
        return bytes.asArray(UInt8.self)
    }

    /// RGB <-> BGR on the last axis.
    static func flip(_ x: MLXArray) -> MLXArray { MLX.take(x, MLXArray([2, 1, 0] as [Int32]), axis: -1) }

    /// cv2.resize INTER_LINEAR on float images: half-pixel centres, borders clamped, no antialiasing.
    public static func bilinear(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        func axis(_ n: Int, _ out: Int) -> (MLXArray, MLXArray, MLXArray) {
            let scale = Double(n) / Double(out)
            var i0 = [Int32](), i1 = [Int32](), f = [Float]()
            for d in 0..<out {
                var s = (Double(d) + 0.5) * scale - 0.5
                if s < 0 { s = 0 }
                var lo = Int(s.rounded(.down)), frac = Float(s - Double(lo))
                if lo >= n - 1 { lo = n - 1; frac = 0 }
                i0.append(Int32(lo)); i1.append(Int32(min(lo + 1, n - 1))); f.append(frac)
            }
            return (MLXArray(i0), MLXArray(i1), MLXArray(f))
        }
        let (y0, y1, fy) = axis(x.dim(0), height), (x0, x1, fx) = axis(x.dim(1), width)
        let top = MLX.take(x, y0, axis: 0), bottom = MLX.take(x, y1, axis: 0)
        let rows = top + (bottom - top) * fy.reshaped([height, 1, 1])
        let left = MLX.take(rows, x0, axis: 1), right = MLX.take(rows, x1, axis: 1)
        return left + (right - left) * fx.reshaped([1, width, 1])
    }

    /// torch F.interpolate(mode="nearest"): source index floor(dst * in / out), clamped.
    public static func nearest(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        func idx(_ n: Int, _ out: Int) -> MLXArray {
            let scale = Float(n) / Float(out)
            return MLXArray((0..<out).map { Int32(min(Int((Float($0) * scale).rounded(.down)), n - 1)) })
        }
        return MLX.take(MLX.take(x, idx(x.dim(0), height), axis: 0), idx(x.dim(1), width), axis: 1)
    }
}
