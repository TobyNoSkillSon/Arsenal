// SPDX-License-Identifier: MIT
// EdgeTAM (facebook/EdgeTAM, Apache-2.0) image segmentation: points and boxes in, mask PNG files and their IoU scores out.
// The network is in Net.swift. Preprocessing and mask post-processing follow sam2/sam2_image_predictor.py and sam2/utils/transforms.py.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.segmentation"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["segment"]

final class Model {
    let net: EdgeTAMImage

    /// Called once by Load. `weightsDir` holds the pinned files (model.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = EdgeTAMImage()
        try Weights.apply(try Weights.load(weightsDir), to: net, rename: EdgeTAMImage.keepsForImagePath)
    }

    /// One request. Returns the mask files (PNG, one per candidate) with their predicted IoU.
    func run(op: String, params: Params) throws -> [String: Any] {
        let imageURL = try params.file("image")
        let prompts = try Model.prompts(params)
        guard !prompts.isEmpty else { throw RunnerError.invalid("Give at least one point or box.") }
        let multimask = params.bool("multimask_output", default: true)

        _ = try checkImageHeader(imageURL)
        let (bytes, width, height) = try decodeRGB(imageURL)
        let resized = MLXArray(pilBilinear(bytes, width: width, height: height, outWidth: 1024, outHeight: 1024).map { Float($0) / 255 }, [1024, 1024, 3])
        let pixels = Images.normalize(resized, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
            .expandedDimensions(axis: 0).asType(.float16)

        let encoded = net.encode(pixels)
        let size = PromptEncoder.embeddingSize
        let masks = net.maskDecoder.predict(
            embedding: encoded.embedding, highRes0: encoded.highRes0, highRes1: encoded.highRes1,
            dense: net.promptEncoder.denseNoMask(),
            sparse: net.promptEncoder.sparse(prompts),
            pe: net.promptEncoder.peLayer.denseGrid(h: size, w: size).asType(.float16))

        let iou = masks.iou.asType(.float32)[0].asArray(Float.self)
        let indices = multimask ? [1, 2, 3] : [Model.singleMaskIndex(masks.masks[0].asType(.float32), iou: iou)]
        let ry = bilinearMatrix(out: height, input: 256)
        let rx = bilinearMatrix(out: width, input: 256).transposed()
        let jobDir = try Model.jobDirectory(params)

        var receipts: [[String: Any]] = []
        for index in indices {
            let logits = masks.masks[0, index].asType(.float32)
            let full = ry.matmul(logits).matmul(rx)
            let binary = (full .> 0).asType(.float32)
            let url = jobDir.appendingPathComponent("mask-\(index).png")
            try Images.savePNG(binary, to: url)
            receipts.append([
                "path": url.path, "width": width, "height": height,
                "bytes": try Files.size(url), "sha256": try Files.sha256(url),
                "score": Double(iou[index]), "index": index,
            ])
        }
        let best = indices.max { iou[$0] < iou[$1] } ?? indices[0]
        Memory.clearCache()
        return ["masks": receipts, "best": best]
    }

    /// The maker's single-mask choice (MaskDecoder._dynamic_multimask_via_stability, delta 0.05, threshold 0.98): the
    /// single-mask output when it is stable, else the multimask output with the highest predicted IoU.
    static func singleMaskIndex(_ logits: MLXArray, iou: [Float]) -> Int {
        let inner = (logits[0] .> 0.05).asType(.float32).sum().item(Float.self)
        let outer = (logits[0] .> -0.05).asType(.float32).sum().item(Float.self)
        let stability: Float = outer > 0 ? inner / outer : 1
        if stability >= 0.98 { return 0 }
        return [1, 2, 3].max { iou[$0] < iou[$1] } ?? 1
    }

    /// Box corners (label 2 and 3) first, then points, as the maker's predictor concatenates them.
    private static func prompts(_ params: Params) throws -> [(x: Float, y: Float, label: Int32)] {
        var out: [(x: Float, y: Float, label: Int32)] = []
        for box in params.raw["boxes"] as? [[String: Any]] ?? [] {
            let xmin = try unit(box, "xmin"), ymin = try unit(box, "ymin")
            let xmax = try unit(box, "xmax"), ymax = try unit(box, "ymax")
            out.append((xmin, ymin, 2))
            out.append((xmax, ymax, 3))
        }
        for point in params.raw["points"] as? [[String: Any]] ?? [] {
            let label: Int32 = (point["label"] as? String) == "background" ? 0 : 1
            out.append((try unit(point, "x"), try unit(point, "y"), label))
        }
        return out
    }

    private static func unit(_ object: [String: Any], _ key: String) throws -> Float {
        guard let number = object[key] as? NSNumber, number.doubleValue >= 0, number.doubleValue <= 1 else {
            throw RunnerError.invalid("\(key) must be a number from 0 to 1.")
        }
        return Float(number.doubleValue)
    }

    /// The supervisor's job folder when it gives one, else a fresh folder under the support directory.
    private static func jobDirectory(_ params: Params) throws -> URL {
        let base = (ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] ?? NSTemporaryDirectory())
        let path = params.string("out_dir", default: base + "/jobs/edgetam-" + UUID().uuidString.lowercased())
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
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
