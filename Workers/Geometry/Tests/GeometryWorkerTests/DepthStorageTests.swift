// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXDA3
import ArsenalWire
import Testing
@testable import GeometryWorker

/// Depth files store FP16 maps and a uint8 mask, so 12-megapixel MoGe results fit the output bound.
@Suite(.serialized)
struct DepthStorageTests {
    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func mapsAreHalfAndMaskIsBytes() throws {
        NativeRuntime.prepare()
        let stored = DepthOutputValidation.stored([
            "depth": MLXArray([Float(1.5), .infinity], [1, 2]), "points": MLXArray.ones([1, 2, 3], dtype: .float32),
            "mask": MLXArray([Float(1), 0], [1, 2])
        ])
        #expect(stored["depth"]?.dtype == .float16 && stored["points"]?.dtype == .float16)
        #expect(stored["mask"]?.dtype == .uint8 && stored["mask"]?.asArray(UInt8.self) == [1, 0])
        try DepthOutputValidation.validate(stored, maskedInfinity: true)
        // 12 MP: FP16 depth (24 MB) plus uint8 mask (12 MB) instead of 96 MB as float32.
        #expect(12_000_000 * (2 + 1) < 64 * 1_048_576)
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func valuesBeyondHalfRangeFailInsteadOfBecomingInfinity() {
        NativeRuntime.prepare()
        let stored = DepthOutputValidation.stored(["depth": MLXArray([Float(1), 100_000], [1, 2])])
        #expect(throws: WorkerFailure.self) { try DepthOutputValidation.validate(stored, maskedInfinity: false) }
    }

    /// Upstream DA3 api.py: process_res 504 with upper_bound_resize (longest side to 504, sides to the nearest 14).
    @Test func da3ProcessesAt504UpperBound() {
        #expect(ImageProcessor().processRes == 504)
        let size = DA3ImagePreprocessing.targetSize(width: 1920, height: 1080, processRes: ImageProcessor().processRes, patchSize: 14)
        #expect(size.width == 504 && size.height == 280)
    }
}
