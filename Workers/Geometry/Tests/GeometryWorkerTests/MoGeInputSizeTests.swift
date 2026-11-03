// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import Testing
@testable import GeometryWorker

/// MoGe's input bound follows the maker's demo (`moge/scripts/app.py`): larger side above max_size (Arsenal: the demo's
/// upper limit, 4096) is scaled down with cv2.INTER_AREA; anything within it runs at full resolution.
@Suite(.serialized)
struct MoGeInputSizeTests {
    @Test func onlyImagesBeyond4096PixelsAreScaled() {
        #expect(MoGeInputSize.capped(width: 4032, height: 3024) == (4032, 3024), "a 12 MP photo keeps full-resolution maps")
        #expect(MoGeInputSize.capped(width: 4096, height: 4096) == (4096, 4096))
        #expect(MoGeInputSize.capped(width: 8064, height: 6048) == (4096, 3072))
        #expect(MoGeInputSize.capped(width: 1000, height: 16000) == (256, 4096))
        #expect(MoGeInputSize.capped(width: 4097, height: 10) == (4096, 10))
        #expect(MoGeInputSize.capped(width: 40000, height: 1) == (4096, 1))
    }

    /// INTER_AREA written as the 2-D pixel-overlap average it defines.
    private func reference(_ image: [Float], height: Int, width: Int, outHeight: Int, outWidth: Int) -> [Float] {
        let sy = Double(height) / Double(outHeight), sx = Double(width) / Double(outWidth)
        var result: [Float] = []
        for oy in 0..<outHeight {
            for ox in 0..<outWidth {
                let y0 = Double(oy) * sy, y1 = min(Double(oy + 1) * sy, Double(height))
                let x0 = Double(ox) * sx, x1 = min(Double(ox + 1) * sx, Double(width))
                for channel in 0..<3 {
                    var total = 0.0
                    for y in 0..<height {
                        let wy = max(0, min(y1, Double(y + 1)) - max(y0, Double(y)))
                        for x in 0..<width where wy > 0 {
                            let wx = max(0, min(x1, Double(x + 1)) - max(x0, Double(x)))
                            total += wy * wx * Double(image[(y * width + x) * 3 + channel])
                        }
                    }
                    result.append(Float(total / ((y1 - y0) * (x1 - x0))))
                }
            }
        }
        return result
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func areaResizeIsThePixelOverlapAverage() throws {
        NativeRuntime.prepare()
        var state: UInt64 = 0x1234_5678_9ABC_DEF1
        for (height, width, outHeight, outWidth) in [(7, 10, 3, 4), (8, 6, 4, 3), (5, 13, 5, 4), (9, 9, 1, 1)] {
            let values = (0..<(height * width * 3)).map { _ -> Float in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Float(state >> 40) / Float(1 << 24)
            }
            let resized = MoGeInputSize.areaResize(MLXArray(values, [height, width, 3]), height: outHeight, width: outWidth)
            #expect(resized.shape == [outHeight, outWidth, 3])
            let expected = reference(values, height: height, width: width, outHeight: outHeight, outWidth: outWidth)
            let delta = zip(resized.asArray(Float.self), expected).map { abs($0 - $1) }.max() ?? 1
            #expect(delta < 1e-5, "\(height)x\(width) → \(outHeight)x\(outWidth): \(delta)")
        }
        // Every weight row is a partition of unity, including the clipped last cell.
        let weights = MoGeInputSize.areaWeights(output: 3, input: 7, rows: 0..<3)
        for row in 0..<3 { #expect(abs(weights[row * 7..<(row + 1) * 7].reduce(0, +) - 1) < 1e-6) }
    }
}
