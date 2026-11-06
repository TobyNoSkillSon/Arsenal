// SPDX-License-Identifier: MIT
import Foundation
import MLX
import Testing
@testable import MLXBiRefNet
@testable import BiRefNetWorker

struct PortTests {
    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
    }
    @Test func pinnedVendorDecoderUsesChannelProportionalInjection() {
        let decoder = BiRefNetDecoder(config: .swinLargeDefault)
        #expect(decoder.iptOut == [384, 192, 96, 48])
        #expect(decoder.decoderBlock4.convIn.weight.shape == [64, 3, 3, 3456])
        #expect(decoder.iptBlk5.convOut.weight.shape == [384, 3, 3, 64])
        #expect(decoder.convOut1.weight.shape == [1, 1, 1, 240])
    }
    @Test(.heavy) func resizeRetainsHalfAtModelBoundaries() {
        let x = MLXArray([Float16(0), 2, 4, 6], [1, 2, 2, 1])
        let y = bilinearResize(x, 3, 3)
        #expect(y.dtype == .float16)
        #expect(y.asArray(Float16.self) == [0, 1, 2, 2, 3, 4, 4, 5, 6])
        let singleton = bilinearResize(MLXArray([Float16(2)], [1, 1, 1, 1]), 4, 4)
        #expect(singleton.dtype == .float16 && singleton.asArray(Float16.self) == Array(repeating: 2, count: 16))
    }
    @Test(.heavy) func patchInjectionPreservesVendorChannelTileOrder() {
        let x = MLXArray((0..<16).map { Float16($0) }, [1, 4, 4, 1])
        let y = image2patches(x, gridH: 2, gridW: 2)
        #expect(y.shape == [1, 2, 2, 4])
        #expect(y.asArray(Float16.self) == [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15])
    }
    @Test func checkpointRejectsOtherPrecisionBeforeReadingSource() {
        let model = BiRefNet()
        #expect(throws: BiRefNet.LoadError.self) {
            try model.loadWeights(url: URL(fileURLWithPath: "/absent/checkpoint"), dtype: .bfloat16)
        }
    }
    @Test func outputRetainsExistingTransparency() {
        var rgba: [UInt8] = [100, 80, 60, 255, 40, 20, 0, 128, 0, 0, 0, 0]
        applyAlphaMask(&rgba, mask: [255, 128, 255])
        #expect(rgba == [100, 80, 60, 255, 20, 10, 0, 64, 0, 0, 0, 0])
    }
}
