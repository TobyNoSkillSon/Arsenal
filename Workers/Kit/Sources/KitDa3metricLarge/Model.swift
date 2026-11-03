// SPDX-License-Identifier: MIT
// Depth Anything 3 Large 1.1 and DA3 Metric Large (depth-anything/DA3-LARGE-1.1, depth-anything/DA3METRIC-LARGE).
// The network is Arsenal's vendored MLXDA3 (DA3-*.swift, from mnmly/mlx-swift-da3, as in Workers/Geometry), which
// ports ByteDance-Seed/Depth-Anything-3 `model/da3.py`; this file only loads, runs and writes files, as the bundled
// geometry worker's DepthAdapter does for DA3 Small.
// Input: one photo (both) or up to eight views with matching processed sizes (Large). Each view is resized with the
// maker's upper_bound_resize to 504 on its long side (multiples of 14).
// Output per view at the processed resolution: FP16 safetensors (depth, plus confidence for Large and sky for Metric)
// and a 16-bit depth preview PNG. Metric gives metres when a focal length is known (focal_length in the photo's pixels,
// else the photo's EXIF 35 mm equivalent) and mode is not "relative": metric = focal_px * output / 300, with focal_px
// the mean of fx and fy at the processed size (utils/alignment.py); otherwise relative.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import UniformTypeIdentifiers

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "perception"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["estimate_depth"]

private let processRes = 504
private let maxViews = 8
private let maxPixels = 64 * 1024 * 1024

final class Model {
    let net: DepthAnything3
    let metric: Bool

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        let name = config["model_name"] as? String ?? ""
        switch name {
        case "da3-large": metric = false; net = buildModel(config: .da3Large)
        case "da3metric-large": metric = true; net = buildModel(config: .da3monoLarge)
        default: throw RunnerError.failed("Unsupported DA3 checkpoint '\(name)'.")
        }
        var mapped: [String: MLXArray] = [:]
        for (key, value) in try Weights.load(weightsDir) {
            guard let name = remapKey(key) else { continue }  // learned camera encoder and decoder are not ported
            if name.hasSuffix(".weight"), value.ndim == 4 {
                mapped[name] = value.transposed(axes: isConvTranspose(key: name) ? [1, 2, 3, 0] : [0, 2, 3, 1])
            } else {
                mapped[name] = value
            }
        }
        try Weights.apply(mapped, to: net)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        var urls: [URL] = []
        if params.raw["image"] != nil, params.raw["images"] == nil {
            urls = [try params.file("image")]
        } else if let images = params.raw["images"] as? [Any], params.raw["image"] == nil {
            urls = try images.map { try Params(["image": $0]).file("image") }
        } else {
            throw RunnerError.invalid("Send exactly one of image or images.")
        }
        let limit = metric ? 1 : maxViews
        guard !urls.isEmpty, urls.count <= limit else {
            throw RunnerError.invalid(metric ? "This model estimates depth for one photo at a time." : "Send one to \(maxViews) views.")
        }
        if params.bool("points", default: false) || params.bool("intrinsics", default: false) {
            throw RunnerError.invalid("This model gives depth maps only; camera intrinsics and points are not ported.")
        }
        if let mode = params.raw["mode"] as? String, mode == "metric", !metric {
            throw RunnerError.invalid("This model gives relative depth; DA3 Metric Large gives metric depth.")
        }

        var batch: [MLXArray] = [], sizes: [(Int, Int)] = [], focals: [Double?] = []
        for url in urls {
            try params.checkCancelled()
            let (image, focal35) = try loadImage(url)
            sizes.append((image.width, image.height))
            focals.append(focal35.map { $0 / 36 * Double(max(image.width, image.height)) })
            let processed = ImageProcessor(processRes: processRes)
            let target = processed.targetSize(for: image)
            guard target.width >= 14, target.height >= 14 else {
                throw RunnerError.invalid("The image is too thin: its short side would be under 14 pixels after resizing.")
            }
            batch.append(processed(image).asType(.float16))
        }
        guard Set(batch.map(\.shape)).count == 1 else {
            throw RunnerError.invalid("All views must give the same processed size; use photos with the same aspect ratio.")
        }
        let result = try net.predict(MLX.stacked(batch, axis: 1), checkpoint: { try params.checkCancelled() })

        // Metric: the photo's focal length in pixels (request, else EXIF), scaled to the processed width.
        var unit = "relative"
        var scale: Float? = nil
        if metric, (params.raw["mode"] as? String) != "relative" {
            let given = (params.raw["focal_length"] as? NSNumber)?.doubleValue
            if let focal = given ?? focals[0] {
                // fx and fy scale with each side's resize; the maker averages them (utils/alignment.py).
                let sx = Double(batch[0].dim(2)) / Double(sizes[0].0), sy = Double(batch[0].dim(1)) / Double(sizes[0].1)
                scale = Float(focal * (sx + sy) / 2 / 300)
                unit = "meters"
            } else if (params.raw["mode"] as? String) == "metric" {
                throw RunnerError.invalid("Metric depth needs the camera's focal length: pass focal_length in pixels, or use a photo with EXIF focal length.")
            }
        }

        let root = try outputFolder(params)
        var files: [[String: Any]] = [], views: [[String: Any]] = []
        var outputs: [[String: MLXArray]] = []
        for index in urls.indices {
            guard var depth = result["depth"]?[0, index].asType(.float32), depth.ndim == 2 else {
                throw RunnerError.failed("The depth graph returned an invalid shape.")
            }
            if let scale { depth = depth * scale }
            var arrays: [String: MLXArray] = ["depth": depth.asType(.float16)]
            if let confidence = result["depth_conf"]?[0, index] { arrays["confidence"] = confidence.asType(.float16) }
            if let sky = result["sky"]?[0, index] { arrays["sky"] = sky.asType(.float16) }
            eval(Array(arrays.values))
            // Checked as stored: metres beyond FP16's range (a huge focal_length) would be written as infinity.
            guard arrays["depth"]!.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite) else {
                throw RunnerError.invalid("The depth values do not fit in FP16; check focal_length.")
            }
            outputs.append(arrays)
        }
        for (index, arrays) in outputs.enumerated() {
            try params.checkCancelled()
            let tensor = root.appendingPathComponent("view-\(index).safetensors")
            try save(arrays: arrays, url: tensor)
            let preview = root.appendingPathComponent("view-\(index)-depth.png")
            let depth = arrays["depth"]!.asType(.float32)
            // The network gives depth (larger is farther); the preview shows nearer as brighter.
            try writePreview(depth, to: preview)
            files.append(try fileReceipt(preview, id: "view-\(index)-depth", mime: "image/png", label: urls.count > 1 ? "Depth map \(index + 1)" : "Depth map",
                                         order: 2 * index, size: (depth.dim(1), depth.dim(0))))
            files.append(try fileReceipt(tensor, id: "view-\(index)", mime: "application/octet-stream", label: "Depth values (FP16 safetensors)", order: 2 * index + 1))
            views.append(["index": index, "file_ids": ["view-\(index)", "view-\(index)-depth"]])
        }
        return ["files": files, "mode": unit == "meters" ? "metric" : "relative", "depth_unit": unit, "views": views]
    }
}

// MARK: - Files

/// The photo (EXIF orientation not applied, as the geometry worker) and its EXIF 35 mm equivalent focal length.
private func loadImage(_ url: URL) throws -> (CGImage, Double?) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
        let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int,
        width > 0, height > 0
    else { throw RunnerError.invalid("The image could not be read.") }
    guard width * height <= maxPixels else { throw RunnerError.invalid("The image is \(width)x\(height); at most 64 megapixels.") }
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
        throw RunnerError.invalid("The image could not be decoded.")
    }
    let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any]
    let focal35 = (exif?[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber)?.doubleValue
    return (image, focal35.flatMap { $0 > 0 ? $0 : nil })
}

/// Arsenal passes out_dir (a job folder); `kit try` does not, so fall back to a job folder under ARSENAL_SUPPORT_DIR.
private func outputFolder(_ params: Params) throws -> URL {
    let root: URL
    if let dir = params.raw["out_dir"] as? String, !dir.isEmpty {
        root = URL(fileURLWithPath: dir)
    } else {
        let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory
        root = support.appendingPathComponent("jobs").appendingPathComponent(UUID().uuidString)
    }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func fileReceipt(_ url: URL, id: String, mime: String, label: String, order: Int, size: (Int, Int)? = nil) throws -> [String: Any] {
    var receipt: [String: Any] = [
        "id": id, "path": url.path, "name": url.lastPathComponent, "mime": mime, "label": label, "order": order,
        "bytes": try Files.size(url), "sha256": try Files.sha256(url),
    ]
    if let size { receipt["width"] = size.0; receipt["height"] = size.1 }
    return receipt
}

/// A 16-bit grayscale PNG with nearer pixels brighter: the map's positive finite range stretched to 65535...0.
private func writePreview(_ depth: MLXArray, to url: URL) throws {
    let values = depth.asArray(Float.self)
    let finite = values.filter { $0.isFinite && $0 > 0 }
    let lower = finite.min() ?? 0, upper = finite.max() ?? 1
    let span = max(upper - lower, Float.ulpOfOne)
    var bytes = [UInt8]()
    bytes.reserveCapacity(values.count * 2)
    for value in values {
        let normalized = value.isFinite && value > 0 ? 1 - min(1, max(0, (value - lower) / span)) : 0
        let sample = UInt16(normalized * 65535)
        bytes.append(UInt8(sample >> 8)); bytes.append(UInt8(sample & 255))
    }
    let h = depth.dim(0), w = depth.dim(1)
    guard let provider = CGDataProvider(data: Data(bytes) as CFData),
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: w * 2, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageByteOrderInfo.order16Big.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent),
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw RunnerError.failed("Could not encode the depth preview.") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw RunnerError.failed("Could not write the depth preview.") }
}
