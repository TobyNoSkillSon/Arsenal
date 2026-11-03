// SPDX-License-Identifier: MIT
import Foundation
import ImageIO
import CoreGraphics

extension RealModelsService {
    func prepareOperationImage(id: String, preparation: OperationImagePreparation, request: JSONObject, directory: URL) async throws -> JSONObject {
        guard preparation.model != id else { throw WorkerFailure(code: "invalid_dependency", message: "Image preparation must use another model.") }
        guard loadable(preparation.model) else {
            throw WorkerFailure(
                code: "dependency_not_installed",
                message: "Image preparation requires installed model \(preparation.model) and operation \(preparation.operation). Add and Get that model first.")
        }
        let (_, dependency) = try await requestOperation(id: preparation.model, op: preparation.operation)
        guard let dependency, dependency.schemaVersion == 3, dependency.prepareImage == nil, ["file", "image"].contains(dependency.output) else {
            throw WorkerFailure(code: "invalid_dependency", message: "Image preparation requires a schema-3 file/image operation without another pre-step.")
        }
        guard let image = request["image"], image["path"]?.string != nil else {
            throw WorkerFailure(code: "invalid_request", message: "Image preparation needs image.path.")
        }
        do { _ = try await admitRequest(id: preparation.model, op: preparation.operation) } catch let failure as WorkerFailure {
            if ["not_found", "not_installed", "integrity", "weights_missing"].contains(failure.code) {
                throw WorkerFailure(
                    code: "dependency_not_installed", message: "Image preparation requires installed weights for \(preparation.model). Get its selected precision first.")
            }
            throw failure
        }
        try Task.checkCancellation()
        try TransientModelJobs.prepare(directory, support: paths.support)
        // Nested run inherits cancellation, trust and force-load admission, with its own Activity entry.
        let result = try await run(id: preparation.model, op: preparation.operation, params: ["image": image, "out_dir": .string(directory.path)], sink: nil)
        try Task.checkCancellation()
        guard let file = result["files"]?.array?.first(where: { $0["id"]?.string == preparation.fileID }),
            file["mime"]?.string == "image/png", let path = file["path"]?.string
        else { throw WorkerFailure(code: "invalid_dependency_output", message: "Image preparation returned no PNG with file ID \(preparation.fileID).") }
        let url = try OperationArtifacts.confined(path, roots: [directory], code: "invalid_dependency_output")
        try PreparedImage.requireCutout(url)
        var prepared = request; prepared["image"] = ["path": .string(url.path)]
        return prepared
    }
}

/// Alpha is a structural input requirement, not a claim about the quality of the removed background.
enum PreparedImage {
    static func requireCutout(_ url: URL) throws {
        func invalid() -> WorkerFailure {
            WorkerFailure(code: "invalid_dependency_output", message: "Image preparation must return an RGBA PNG cutout with visible foreground and transparent background.")
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetType(source) as String? == "public.png",
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil), image.width > 0, image.height > 0,
            image.width <= 16_777_216 / image.height, image.colorSpace?.numberOfComponents == 3,
            [.first, .last, .premultipliedFirst, .premultipliedLast].contains(image.alphaInfo)
        else { throw invalid() }
        var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let rendered = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height)); return true
        }
        guard rendered else { throw invalid() }
        var visible = false, transparent = false
        for index in stride(from: 3, to: pixels.count, by: 4) { visible = visible || pixels[index] > 0; transparent = transparent || pixels[index] < 255 }
        guard visible && transparent else { throw invalid() }
    }
}
