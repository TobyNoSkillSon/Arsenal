// SPDX-License-Identifier: MIT
// SAM 2.1 (facebook/sam2.1-hiera-*, Apache-2.0) image segmentation from clicks and boxes: Arsenal's perception segment.
// Network in Hiera.swift and Decoder.swift. Pre- and post-processing follow transformers Sam2Processor /
// Sam2ImageProcessorFast and the maker's SAM2ImagePredictor (one image, one prompt set or a batch of boxes).
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import UniformTypeIdentifiers

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "perception"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["segment"]

final class Model {
    let net: Sam2Image

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = Sam2Image(try HieraConfig(try Files.json(weightsDir)))
        try Weights.apply(Sam2Image.prepare(try Weights.load(weightsDir)), to: net)
        net.prepare()
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "segment" else { throw RunnerError.invalid("Unknown operation \(op).") }
        let url = try params.file("image")
        let sets = try Model.promptSets(params)
        // One prompt set: the maker's default of three alternatives; a batch of boxes: one mask per box.
        let multimask = sets.count == 1 && params.bool("multimask", default: true)

        _ = try checkImageHeader(url)
        let (bytes, width, height) = try decodeRGB(url)
        let resized = MLXArray(pilBilinear(bytes, width: width, height: height, outWidth: 1024, outHeight: 1024).map { Float($0) / 255 }, [1024, 1024, 3])
        let pixels = Images.normalize(resized, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
            .expandedDimensions(axis: 0).asType(.float16)

        let image = net.encode(pixels)
        try params.checkCancelled()
        let sparse = concatenated(sets.map { net.promptEncoder.sparse($0) }, axis: 0)
        let (masks, iouPred) = net.maskDecoder(image: image, dense: net.dense(), sparse: sparse, pe: net.imagePE!)
        let logits = masks.asType(.float32)
        let iou = iouPred.asType(.float32)
        eval(logits, iou)
        guard iou.asArray(Float.self).allSatisfy({ $0.isFinite }) else {
            throw RunnerError.failed("The model returned a predicted IoU that is not a finite number.")
        }

        // (prompt set, mask index) pairs to return.
        var picks: [(set: Int, index: Int)] = []
        for s in 0..<sets.count {
            let scores = iou[s].asArray(Float.self)
            if multimask {
                picks += [1, 2, 3].map { (s, $0) }
            } else {
                picks.append((s, Model.singleMaskIndex(logits[s, 0], iou: scores)))
            }
        }

        let ry = bilinearMatrix(out: height, input: 256)
        let rx = bilinearMatrix(out: width, input: 256).transposed()
        let dir = try outputDirectory(params)
        var files: [[String: Any]] = []
        var scores: [Double] = []
        for (order, pick) in picks.enumerated() {
            try params.checkCancelled()
            let full = ry.matmul(logits[pick.set, pick.index]).matmul(rx)
            let mask = (full .> 0).asType(.uint8) * 255
            let name = sets.count > 1 ? "mask-box\(pick.set + 1)" : "mask-\(pick.index)"
            let file = dir.appendingPathComponent(name + ".png")
            try writeGrey(mask.reshaped([-1]).asArray(UInt8.self), width: width, height: height, to: file)
            let label = sets.count > 1 ? "Box \(pick.set + 1)" : (multimask ? "Mask \(order + 1)" : "Mask")
            files.append([
                "id": name, "path": file.path, "name": name + ".png", "mime": "image/png", "label": label,
                "bytes": try Files.size(file), "sha256": try Files.sha256(file), "width": width, "height": height, "order": order,
            ])
            scores.append(Double(max(0, min(1, iou[pick.set, pick.index].item(Float.self)))))
        }
        Memory.clearCache()
        return ["files": files, "scores": scores]
    }

    /// The maker's single-mask choice (Sam2MaskDecoder._dynamic_multimask_via_stability, delta 0.05, threshold 0.98):
    /// mask 0 when it is stable, else the multimask output with the highest predicted IoU.
    static func singleMaskIndex(_ logits: MLXArray, iou: [Float]) -> Int {
        let inner = (logits .> 0.05).asType(.float32).sum().item(Float.self)
        let outer = (logits .> -0.05).asType(.float32).sum().item(Float.self)
        if (outer > 0 ? inner / outer : 1) >= 0.98 { return 0 }
        return [1, 2, 3].max { iou[$0] < iou[$1] } ?? 1
    }

    /// One set (an optional box, corners first as the maker's predictor does, then the points), or one set per box
    /// when several boxes come without points.
    static func promptSets(_ params: Params) throws -> [[Prompt]] {
        let boxes = try (params.raw["boxes"] as? [[String: Any]] ?? []).map { box -> [Prompt] in
            let x0 = try unit(box, "xmin"), y0 = try unit(box, "ymin"), x1 = try unit(box, "xmax"), y1 = try unit(box, "ymax")
            guard x1 > x0, y1 > y0 else { throw RunnerError.invalid("A box needs xmax > xmin and ymax > ymin.") }
            return [(x0, y0, 2), (x1, y1, 3)]
        }
        let points = try (params.raw["points"] as? [[String: Any]] ?? []).map { point -> Prompt in
            (try unit(point, "x"), try unit(point, "y"), (point["label"] as? String) == "background" ? 0 : 1)
        }
        if params.raw["text"] != nil || params.raw["visual_exemplars"] != nil || params.raw["session_id"] != nil {
            throw RunnerError.invalid("This model takes clicks and boxes on one image, not text, exemplars or sessions.")
        }
        guard !boxes.isEmpty || !points.isEmpty else { throw RunnerError.invalid("Give at least one point or box.") }
        if boxes.count > 1 {
            guard points.isEmpty else { throw RunnerError.invalid("Several boxes give one mask each; send points with at most one box.") }
            guard boxes.count <= 64 else { throw RunnerError.invalid("At most 64 boxes per request.") }
            return boxes
        }
        guard points.count <= 64 else { throw RunnerError.invalid("At most 64 points per request.") }
        return [(boxes.first ?? []) + points]
    }

    private static func unit(_ object: [String: Any], _ key: String) throws -> Float {
        guard let n = object[key] as? NSNumber, n.doubleValue >= 0, n.doubleValue <= 1 else {
            throw RunnerError.invalid("\(key) must be a number from 0 to 1.")
        }
        return Float(n.doubleValue)
    }
}

/// Bilinear resize as a matrix (align_corners=False, no antialias), so that Ry * mask * Rx^T matches F.interpolate.
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

/// An 8-bit greyscale PNG (white is the object).
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
