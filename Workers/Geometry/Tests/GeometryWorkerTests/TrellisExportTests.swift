// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import IntegrationKit
import MLX
import Testing
@testable import MLXTrellis
import GeometryWorker

/// GLB export follows upstream o_voxel postprocess.to_glb: Y-up axes and an OPAQUE material.
@Suite(.serialized)
struct TrellisExportTests {
    private func mesh(alpha: UInt8) -> BakedMesh {
        let red: [UInt8] = [255, 0, 0, alpha]
        return BakedMesh(
            vertices: MLXArray([Float(-1), -1, 0, 1, -1, 0, 0, 1, 0], [3, 3]), faces: MLXArray([Int32(0), 1, 2], [1, 3]),
            normals: MLXArray([Float(0), 0, 1, 0, 0, 1, 0, 0, 1], [3, 3]), uvs: MLXArray([Float](repeating: 0.25, count: 6), [3, 2]),
            texRGBA: red + red + red + red, atlasSize: 2, coverage: 1, mrRGBA: nil,
            filledBeforeInpaint: [true, true, true, true], wallFlipped: [false, false, false, false])
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func yUpSwapsAxesLikeUpstream() {
        NativeRuntime.prepare()
        let converted = BakedMesh(
            vertices: MLXArray([Float(1), 2, 3], [1, 3]), faces: MLXArray([Int32(0), 0, 0], [1, 3]), normals: MLXArray([Float(4), 5, 6], [1, 3]),
            uvs: MLXArray([Float(0.1), 0.2], [1, 2]), texRGBA: [0, 0, 0, 255], atlasSize: 1, coverage: 1, mrRGBA: nil,
            filledBeforeInpaint: [true], wallFlipped: [false]
        ).yUp()
        #expect(converted.vertices.asArray(Float.self) == [1, 3, -2])
        #expect(converted.normals.asArray(Float.self) == [4, 6, -5])
        #expect(converted.uvs.asArray(Float.self) == [0.1, 0.2])
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func materialIsOpaqueEvenWithTranslucentTexels() throws {
        NativeRuntime.prepare()
        let mesh = mesh(alpha: 128)
        let glb = try GLTFExport.glbData(
            positions: mesh.vertices, indices: mesh.faces, normals: mesh.normals, uvs: mesh.uvs,
            baseColorRGBA: (mesh.texRGBA, mesh.atlasSize, mesh.atlasSize))
        let length = glb.subdata(in: 12..<16).withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))) }
        let json = try #require(try JSONSerialization.jsonObject(with: glb.subdata(in: 20..<(20 + length))) as? [String: Any])
        let material = try #require((json["materials"] as? [[String: Any]])?.first)
        #expect(material["alphaMode"] as? String == "OPAQUE")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func previewDrawsPlusYAtTheTop() throws {
        NativeRuntime.prepare()
        // Apex at +Y: the top rows of the preview hold the narrow end of the triangle.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("trellis-up-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        try TrellisPreview.write(mesh(alpha: 255), to: url, cancellation: NativeCancellation(), size: 32)
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        var pixels = [UInt8](repeating: 0, count: 32 * 32 * 4)
        pixels.withUnsafeMutableBytes { buffer in
            let context = CGContext(
                data: buffer.baseAddress, width: 32, height: 32, bitsPerComponent: 8, bytesPerRow: 128,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            context?.draw(image, in: CGRect(x: 0, y: 0, width: 32, height: 32))
        }
        // CGContext memory starts at the image's top row.
        func drawn(_ rows: Range<Int>) -> Int { rows.reduce(0) { sum, row in sum + (0..<32).filter { pixels[(row * 32 + $0) * 4 + 3] > 0 }.count } }
        #expect(drawn(4..<12) < drawn(20..<28))
    }

    /// Get converts every component (the FP32 assets included) into the half base, so Load casts nothing and refuses FP32.
    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func loadRefusesAnUnconvertedFP32Component() throws {
        NativeRuntime.prepare()
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("trellis-fp32-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: file) }
        try MLX.save(arrays: ["projection.weight": MLXArray.ones([2, 64]), "projection.bias": MLXArray([Float(3), 7])], url: file)
        let bank = TrellisWeights(shapes: ["projection.weight": [2, 64], "projection.bias": [2]])
        #expect(throws: (any Error).self) { try bank.load(archives: [file], precision: "fp16", cancellation: NativeCancellation()) }
        #expect(TrellisAdapter.components.count == 2)
    }
}
