// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// advimman/lama `bin/predict.py` on 8-bit pixels: image /255, mask /255 > 0, both padded to a multiple of 8 with
/// np.pad(mode="symmetric"), generator on concat[image*(1-mask), mask], inpainted = mask*pred + (1-mask)*image,
/// cropped, then np.clip(x*255, 0, 255).astype(uint8) (truncation). The composite runs in float32 on the CPU so
/// unmasked pixels come back unchanged.
public enum LaMaPipeline {
    public static func inpaint(_ model: LaMaModel, rgb: [UInt8], mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        let ph = (8 - height % 8) % 8, pw = (8 - width % 8) % 8
        let image = MLXArray(rgb.map { Float($0) / 255 }, [height, width, 3])
        let hole = MLXArray(mask.map { $0 > 0 ? Float(1) : 0 }, [height, width, 1])
        let paddedImage = symmetricPad(image, ph, pw), paddedHole = symmetricPad(hole, ph, pw)
        let input = MLX.concatenated([paddedImage * (1 - paddedHole), paddedHole], axis: -1).expandedDimensions(axis: 0)
        let predicted = model(input)[0, 0..<height, 0..<width, 0...].asType(.float32)
        eval(predicted)
        let pred = predicted.asArray(Float.self)
        var out = [UInt8](repeating: 0, count: rgb.count)
        for i in 0..<(width * height) {
            let m: Float = mask[i] > 0 ? 1 : 0
            for c in 0..<3 {
                let j = 3 * i + c
                let v = (m * pred[j] + (1 - m) * (Float(rgb[j]) / 255)) * 255
                out[j] = UInt8(Swift.max(0, Swift.min(255, v)).rounded(.towardZero))
            }
        }
        return out
    }

    /// np.pad(mode="symmetric") on the bottom and right: the edge row/column repeats, then the mirror continues.
    static func symmetricPad(_ x: MLXArray, _ ph: Int, _ pw: Int) -> MLXArray {
        func idx(_ n: Int, _ p: Int) -> MLXArray {
            MLXArray((0..<n).map(Int32.init) + (0..<p).map { Int32(Swift.max(0, n - 1 - $0 % n)) })
        }
        var y = x
        if ph > 0 { y = MLX.take(y, idx(x.dim(0), ph), axis: 0) }
        if pw > 0 { y = MLX.take(y, idx(x.dim(1), pw), axis: 1) }
        return y
    }
}
