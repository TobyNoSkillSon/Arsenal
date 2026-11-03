// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing
@testable import MLXBiRefNet
@testable import BiRefNetWorker

struct ImageBoundaryTests {
    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
    }
    private var fixtures: URL { Bundle.module.resourceURL!.appendingPathComponent("Fixtures") }
    private struct Fixture: Decodable {
        struct ImageCase: Decodable {
            let name: String
            let width, height, targetWidth, targetHeight, channels: Int
            let filter: String
            let input, expected: [UInt8]
        }
        struct Mask: Decodable { let input: [Double]; let quantized, expected: [UInt8] }
        let cases: [ImageCase]
        let mask: Mask
        let partialRgb: [UInt8]
    }
    private func oracle() throws -> Fixture {
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Fixture.self, from: Data(contentsOf: fixtures.appendingPathComponent("boundaries.json")))
    }
    @Test(.heavy) func partialPNGUsesStraightColorsThroughActualExtractionAndNormalization() throws {
        let image = try #require(loadCGImage(url: fixtures.appendingPathComponent("partial-alpha.png")))
        let expected = try oracle().partialRgb
        #expect(BiRefNetPipeline.resizeToRGBPixels(image: image, width: 4, height: 1) == expected)
        #expect(expected == [255, 0, 0, 0, 255, 0, 12, 40, 200, 25, 100, 240])
        let tensor = try BiRefNet.preprocess(image: image, targetW: 4, targetH: 1, norm: .imageNet)
        let values = tensor.asArray(Float.self)
        #expect(abs(values[0] - (1 - 0.485) / 0.229) < 0.000001)
        #expect(abs(values[4] - (1 - 0.456) / 0.224) < 0.000001)
    }
    @Test func imageResizeMatchesPillowExactlyOnFixedEdgesAndCheckerboards() throws {
        let cases = try oracle().cases
        #expect(cases.count == 5)
        for item in cases {
            let name = item.name
            let width = item.width, height = item.height
            let tw = item.targetWidth, th = item.targetHeight
            let expected = item.expected
            if item.filter == "bilinear" {
                let image = try #require(loadCGImage(url: fixtures.appendingPathComponent(name + ".png")))
                #expect(BiRefNetPipeline.resizeToRGBPixels(image: image, width: tw, height: th) == expected, "\(name)")
            } else {
                let input = item.input
                #expect(ImageBoundary.resize(input, width: width, height: height, channels: 1, targetWidth: tw, targetHeight: th, filter: .bicubic) == expected, "\(name)")
            }
        }
    }
    @Test(.heavy) func halfMaskQuantizesBeforeNonsquareBicubicResizeAndRendersExactBytes() throws {
        let item = try oracle().mask
        let input = item.input.map { Float16($0) }
        let mask = MLXArray(input, [1, 3, 4, 1])
        #expect(BiRefNetPipeline.maskBytes(mask) == item.quantized)
        let expected = item.expected
        let actual = BiRefNetPipeline.sourceMaskBytes(mask, width: 7, height: 5)
        #expect(actual == expected)
        let prediction = BiRefNetPrediction(outputs: ["mask_source_bytes": MLXArray(actual, [1, 5, 7, 1])], sourceHeight: 5, sourceWidth: 7)
        let rendered = try prediction.maskCGImage()
        let data = try #require(rendered.dataProvider?.data) as Data
        #expect(Array(data) == expected)
    }
    @Test func failedPNGConsumerThrowsInsteadOfReturningAFileReceipt() throws {
        let image = try #require(loadCGImage(url: fixtures.appendingPathComponent("partial-alpha.png")))
        var callbacks = CGDataConsumerCallbacks(putBytes: { _, _, _ in 0 }, releaseConsumer: nil)
        let consumer = try #require(CGDataConsumer(info: nil, cbks: &callbacks))
        let destination = try #require(CGImageDestinationCreateWithDataConsumer(consumer, "public.png" as CFString, 1, nil))
        #expect(throws: BiRefNetPipelineError.self) { try finalizeImage(image, destination: destination) }
        #expect(throws: (any Error).self) { try saveCGImage(image, url: fixtures) }
    }

}
