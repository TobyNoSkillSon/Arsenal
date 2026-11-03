// SPDX-License-Identifier: MIT
// EoMT COCO panoptic (tue-mps/coco_panoptic_eomt_large_640, MIT): a photo in, one mask per object or region with its
// COCO label out, as Arsenal's perception segment. Network in Net.swift. Pre- and post-processing follow transformers
// EomtImageProcessorFast (resize the longest edge to 640, pad to 640x640) and its post_process_panoptic_segmentation
// (threshold 0.8, mask threshold 0.5, overlap 0.8, no stuff merging).
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import UniformTypeIdentifiers

let modelKind = "perception"
let modelOps = ["segment"]

final class Model {
    let net: Eomt
    let config: EomtConfig
    /// Panoptic maps are computed at the photo's size up to this many pixels, and at a scaled size above it.
    let maxMapPixels = 4_194_304

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = try EomtConfig(try Files.json(weightsDir))
        net = Eomt(config)
        try Weights.apply(Eomt.prepare(try Weights.load(weightsDir)), to: net)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "segment" else { throw RunnerError.invalid("Unknown operation \(op).") }
        for key in ["points", "boxes", "text", "visual_exemplars", "session_id"] where params.raw[key] != nil {
            throw RunnerError.invalid("This model segments the whole photo by itself; it takes no \(key).")
        }
        let url = try params.file("image")
        _ = try checkImageHeader(url)
        let (bytes, width, height) = try decodeRGB(url)
        let side = config.imageSize
        // The maker's rounding can give a side of 0 (2000x1) or one past the grid (641x300); keep both inside 1...640.
        let (rawH, rawW) = Model.resizedSize(height: height, width: width, size: side)
        let rh = min(side, max(1, rawH)), rw = min(side, max(1, rawW))
        // Resize, zero-pad bottom and right in 8-bit space, then normalize (the padding becomes -mean/std, as in the maker's order).
        let resized = pilBilinear(bytes, width: width, height: height, outWidth: rw, outHeight: rh)
        var padded = [Float](repeating: 0, count: side * side * 3)
        for y in 0..<rh {
            for x in 0..<rw {
                for ch in 0..<3 { padded[(y * side + x) * 3 + ch] = Float(resized[(y * rw + x) * 3 + ch]) / 255 }
            }
        }
        let pixels = Images.normalize(MLXArray(padded, [side, side, 3]), mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
            .expandedDimensions(axis: 0).asType(.float16)

        let out = try net(pixels, cancelled: { try params.checkCancelled() })
        let maskLogits = out.masks.asType(.float32)  // [Q, 160, 160]
        let probs = softmax(out.classes.asType(.float32), axis: -1)
        let scores = probs.max(axis: -1), labels = argMax(probs, axis: -1)
        eval(maskLogits, scores, labels)

        // remove_low_and_no_objects
        let scoreList = scores.asArray(Float.self), labelList = labels.asArray(Int32.self)
        let noObject = Int32(config.labels.count)
        let kept = (0..<config.queries).filter { labelList[$0] != noObject && scoreList[$0] > 0.8 }

        // The map is computed at the photo's size (scaled down only for very large photos).
        let scale = min(1, (Double(maxMapPixels) / Double(width * height)).squareRoot())
        let mh = max(1, Int((Double(height) * scale).rounded())), mw = max(1, Int((Double(width) * scale).rounded()))
        var segments: [(label: Int, score: Float, mask: MLXArray)] = []
        if !kept.isEmpty {
            try params.checkCancelled()
            // 160 -> 640 bilinear, crop to the resized area, then to the map size (unpad_image), as one matrix per axis.
            let grid = maskLogits.dim(1)
            let ay = bilinearMatrix(out: mh, input: rh).matmul(bilinearMatrix(out: side, input: grid)[0 ..< rh, 0...])
            let ax = bilinearMatrix(out: mw, input: rw).matmul(bilinearMatrix(out: side, input: grid)[0 ..< rw, 0...])
            let selected = take(maskLogits, MLXArray(kept.map { Int32($0) }), axis: 0)
            let maskProbs = sigmoid(ay.expandedDimensions(axis: 0).matmul(selected).matmul(ax.transposed().expandedDimensions(axis: 0)))
            let keptScores = MLXArray(kept.map { scoreList[$0] })
            let owner = argMax(keptScores.reshaped([-1, 1, 1]) * maskProbs, axis: 0)  // [mh, mw]
            // check_segment_validity for every kept query at once.
            let ids = MLXArray(Array(0..<Int32(kept.count))).reshaped([-1, 1, 1])
            let mine = owner.expandedDimensions(axis: 0) .== ids
            let original = maskProbs .>= 0.5
            let final = mine .&& original
            let mineArea = mine.asType(.float32).sum(axes: [1, 2])
            let originalArea = original.asType(.float32).sum(axes: [1, 2])
            let finalArea = final.asType(.float32).sum(axes: [1, 2])
            eval(final, mineArea, originalArea, finalArea)
            let a = mineArea.asArray(Float.self), o = originalArea.asArray(Float.self), f = finalArea.asArray(Float.self)
            for k in 0..<kept.count where a[k] > 0 && o[k] > 0 && f[k] > 0 && a[k] / o[k] > 0.8 {
                segments.append((Int(labelList[kept[k]]), scoreList[kept[k]], final[k]))
            }
        }

        let dir = try outputDirectory(params)
        var files: [[String: Any]] = []
        var outScores: [Double] = []
        let names = segments.map { Model.readable(config.labels[$0.label]) }
        for (order, segment) in segments.enumerated() {
            try params.checkCancelled()
            var mask = segment.mask
            if mh != height || mw != width {  // nearest-neighbour back to the photo's size
                let ys = MLXArray((0..<height).map { Int32(min(mh - 1, $0 * mh / height)) })
                let xs = MLXArray((0..<width).map { Int32(min(mw - 1, $0 * mw / width)) })
                mask = take(take(mask, ys, axis: 0), xs, axis: 1)
            }
            let name = names[order]
            let count = names.filter { $0 == name }.count
            let index = names[..<order].filter { $0 == name }.count + 1
            let label = count > 1 ? "\(name) \(index)" : name
            let id = "segment-\(order)"
            let file = dir.appendingPathComponent(id + ".png")
            try writeGrey((mask.asType(.uint8) * 255).reshaped([-1]).asArray(UInt8.self), width: width, height: height, to: file)
            files.append([
                "id": id, "path": file.path, "name": "\(id)-\(label.replacingOccurrences(of: " ", with: "-")).png", "mime": "image/png",
                "label": label, "bytes": try Files.size(file), "sha256": try Files.sha256(file), "width": width, "height": height, "order": order,
            ])
            outScores.append(Double(segment.score))
        }
        if files.isEmpty {
            // The kind needs at least one file: an empty mask, scored 0, when nothing passes the threshold.
            let file = dir.appendingPathComponent("no-segments.png")
            try writeGrey([UInt8](repeating: 0, count: width * height), width: width, height: height, to: file)
            files = [["id": "no-segments", "path": file.path, "name": "no-segments.png", "mime": "image/png", "label": "Nothing found",
                      "bytes": try Files.size(file), "sha256": try Files.sha256(file), "width": width, "height": height, "order": 0]]
            outScores = [0]
        }
        Memory.clearCache()
        return ["files": files, "scores": outScores]
    }

    /// transformers get_size_with_aspect_ratio(size: shortest 640, max_size: longest 640), Python's round included.
    static func resizedSize(height: Int, width: Int, size target: Int) -> (Int, Int) {
        var size = target
        var raw: Double? = nil
        let small = Double(min(height, width)), large = Double(max(height, width))
        if large / small * Double(size) > Double(target) {
            raw = Double(target) * small / large
            size = Int(raw!.rounded(.toNearestOrEven))
        }
        if (height <= width && height == size) || (width <= height && width == size) { return (height, width) }
        if width < height {
            return (Int((raw ?? Double(size)) * Double(height) / Double(width)), size)
        }
        return (size, Int((raw ?? Double(size)) * Double(width) / Double(height)))
    }

    /// COCO panoptic names without their dataset suffixes: "wall-other-merged" becomes "wall".
    static func readable(_ name: String) -> String {
        var n = name
        for suffix in ["-merged", "-other", "-stuff"] { n = n.replacingOccurrences(of: suffix, with: "") }
        return n.replacingOccurrences(of: "-", with: " ")
    }
}

/// Bilinear resize as a matrix (align_corners=False, no antialias), so that Ry * map * Rx^T matches F.interpolate.
func bilinearMatrix(out: Int, input: Int) -> MLXArray {
    var m = [Float](repeating: 0, count: out * input)
    let scale = Double(input) / Double(out)
    for o in 0..<out {
        let src = max(0, (Double(o) + 0.5) * scale - 0.5)
        let i0 = min(Int(src.rounded(.down)), input - 1)
        let i1 = min(i0 + 1, input - 1)
        let lambda = Float(src - Double(i0))
        m[o * input + i0] += 1 - lambda
        m[o * input + i1] += lambda
    }
    return MLXArray(m, [out, input])
}

/// Arsenal passes `out_dir` (a job folder under <support>/jobs/); outside Arsenal, a new folder under $ARSENAL_SUPPORT_DIR/jobs.
private func outputDirectory(_ params: Params) throws -> URL {
    let dir: URL
    if let given = params.raw["out_dir"] as? String, !given.isEmpty {
        dir = URL(fileURLWithPath: given)
    } else {
        let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] ?? NSTemporaryDirectory()
        dir = URL(fileURLWithPath: support).appendingPathComponent("jobs").appendingPathComponent(UUID().uuidString)
    }
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// An 8-bit greyscale PNG (white is the segment).
private func writeGrey(_ pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
    var bytes = pixels
    guard let context = CGContext(
        data: &bytes, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
        space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue),
        let image = context.makeImage(),
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw RunnerError.failed("Could not write \(url.lastPathComponent).") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw RunnerError.failed("Could not write \(url.lastPathComponent).") }
}

/// Reads the image's pixel size from its header, before decoding, and refuses a side over `maxSide` pixels
/// (so a huge or extreme-aspect file is refused without allocating its pixels).
private func checkImageHeader(_ url: URL, maxSide: Int = 8192) throws -> (width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
          let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue
    else { throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.") }
    guard width >= 1, height >= 1, width <= maxSide, height <= maxSide else {
        throw RunnerError.invalid("The image is \(width)x\(height); each side must be 1 to \(maxSide) pixels.")
    }
    return (width, height)
}
