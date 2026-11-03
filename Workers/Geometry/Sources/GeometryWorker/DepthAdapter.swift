// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import IntegrationKit
import MLX
import MLXNN
import MLXMoGe
import MLXDA3
import ArsenalWire

/// Checkpoint names stay unchanged in Get's streamed conversion; layout changes happen once at Load.
public enum GeometryWeights {
    public static func eligible(_ name: String, _ shape: [Int]) -> Bool {
        name.hasSuffix(".weight") && shape.count == 2 && shape[1] % 64 == 0
    }
    public static func mappedName(_ name: String, moge: Bool) -> String? {
        moge ? MoGeWeightLoader.remapWeightKey(name) : MLXDA3.remapKey(name)
    }
    public static func permutation(_ name: String, shape: [Int], moge: Bool) -> [Int]? {
        guard !moge, name.hasSuffix(".weight"), shape.count == 4 else { return nil }
        return MLXDA3.isConvTranspose(key: name) ? [1, 2, 3, 0] : [0, 2, 3, 1]
    }
    /// Get-time eligibility on checkpoint names: a matrix Load drops (DA3's unported cam_enc/cam_dec/gs heads, the
    /// metric branch) stays half, because a stored pack's scales/biases cannot be dropped with it at Load.
    public static func storageEligible(moge: Bool) -> (String, [Int]) -> Bool {
        { name, shape in Self.eligible(name, shape) && Self.mappedName(name, moge: moge) != nil }
    }
    /// Load-time checkpoint mapping: rename to the MLX graph and reorder convolution kernels; nil drops the tensor.
    public static func load(_ key: String, _ value: MLXArray, moge: Bool) -> (String, MLXArray)? {
        guard let name = mappedName(key, moge: moge) else { return nil }
        let axes = Self.permutation(name, shape: value.shape, moge: moge)
        return (name, axes.map { value.transposed(axes: $0) } ?? value)
    }
}

/// Numeric acceptance of depth outputs before any artifact is written.
enum DepthOutputValidation {
    /// - `maskedInfinity`: MoGe's documented convention, where masked-out pixels carry +infinity
    ///   in depth and points (`mask == 0`). Every other value of every tensor must be finite, and
    ///   at least one depth value must be finite and positive.
    static func validate(_ tensors: [String: MLXArray], maskedInfinity: Bool) throws {
        guard let depth = tensors["depth"] else { throw invalid("The depth model returned no depth map.") }
        let pixels = depth.size
        let mask: [Float]? = maskedInfinity ? tensors["mask"].map { $0.asType(.float32).asArray(Float.self) } : nil
        if let mask, mask.count != pixels { throw invalid("The depth model returned a mask that does not match its depth map.") }
        func masked(_ pixel: Int) -> Bool { mask.map { $0[pixel] == 0 } ?? false }
        var usable = 0, rejected = 0
        for (pixel, value) in depth.asType(.float32).asArray(Float.self).enumerated() {
            if value.isFinite {
                if value > 0 { usable += 1 }
            } else if !(value == .infinity && masked(pixel)) {
                rejected += 1
            }
        }
        guard rejected == 0 else { throw invalid("The depth model returned \(rejected) non-finite depth values of \(pixels).") }
        guard usable > 0 else { throw invalid("The depth model returned no finite positive depth (0 of \(pixels) pixels).") }
        for (name, tensor) in tensors.sorted(by: { $0.key < $1.key }) where name != "depth" {
            let values = tensor.asType(.float32).asArray(Float.self)
            let channels = name == "points" && pixels > 0 && values.count % pixels == 0 ? values.count / pixels : 0
            for (index, value) in values.enumerated() where !value.isFinite {
                if channels > 0, value == .infinity, masked(index / channels) { continue }
                throw invalid("The depth model returned non-finite \(name) values.")
            }
        }
    }
    private static func invalid(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_model_output", message: message) }
    /// File storage types: maps as FP16 and the mask as uint8 0/1, so a 12-megapixel depth map with its mask is about
    /// 36 MB instead of 96 MB. Validation runs on these stored values, so a value beyond FP16's range fails as
    /// non-finite rather than being written as infinity.
    static func stored(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (name, tensor) in tensors {
            if name == "mask" {
                result[name] = (tensor.asType(.float32) .> Float(0)).asType(.uint8)
            } else {
                result[name] = tensor.dtype.isFloatingPoint ? tensor.asType(.float16) : tensor
            }
        }
        return result
    }
}

/// MoGe's input bound, from the maker's demo (`moge/scripts/app.py`, `run`): an image whose larger side exceeds the
/// "Maximum Image Size" is scaled by `max_size / larger_size` with `cv2.INTER_AREA` before inference, and the maps come
/// back at that size. Arsenal fixes `max_size` at the demo's upper limit, 4096, so photos up to 4096 px keep
/// full-resolution maps (the library and CLI default: no resize) and nothing larger reaches the graph or the outputs.
enum MoGeInputSize {
    static let maximumSide = 4096
    static func capped(width: Int, height: Int) -> (width: Int, height: Int) {
        let larger = max(width, height)
        guard larger > maximumSide else { return (width, height) }
        let scale = Double(maximumSide) / Double(larger)
        // cv2.resize with dsize (0, 0): each side is round(side * f).
        return (max(1, Int((Double(width) * scale).rounded())), max(1, Int((Double(height) * scale).rounded())))
    }
    /// OpenCV INTER_AREA downscaling weights for output rows `rows`, `(rows.count, input)`: each output cell averages the
    /// source interval `[o * s, (o + 1) * s)` (s = input / output), partial pixels by their overlap, the last cell
    /// clipped to the image.
    static func areaWeights(output: Int, input: Int, rows: Range<Int>) -> [Float] {
        var weights = [Float](repeating: 0, count: rows.count * input)
        let scale = Double(input) / Double(output)
        for row in rows {
            let start = Double(row) * scale, end = min(Double(row + 1) * scale, Double(input))
            var column = Int(start.rounded(.down))
            while Double(column) < end, column < input {
                let overlap = min(end, Double(column + 1)) - max(start, Double(column))
                if overlap > 0 { weights[(row - rows.lowerBound) * input + column] = Float(overlap / (end - start)) }
                column += 1
            }
        }
        return weights
    }
    /// Area-downscale an `(H, W, 3)` image: rows, then columns, each in bands whose weight matrix holds at most
    /// 2^22 entries (16 MB), so a 16-megapixel panorama never builds a full 4096 × 16000 matrix.
    static func areaResize(_ image: MLXArray, height: Int, width: Int) -> MLXArray {
        func leading(_ x: MLXArray, to output: Int) -> MLXArray {
            let input = x.dim(0), rest = Array(x.shape.dropFirst())
            let flat = x.reshaped(input, -1), step = max(1, (1 << 22) / input)
            var bands: [MLXArray] = []
            for top in stride(from: 0, to: output, by: step) {
                let rows = top..<min(output, top + step)
                let band = matmul(MLXArray(areaWeights(output: output, input: input, rows: rows), [rows.count, input]), flat)
                eval(band)
                bands.append(band)
            }
            return concatenated(bands, axis: 0).reshaped([output] + rest)
        }
        let rows = leading(image.asType(.float32), to: height)
        return leading(rows.transposed(1, 0, 2), to: width).transposed(1, 0, 2)
    }
}

public final class DepthAdapter: NativeModel {
    public let architecture: String
    public let operations: Set<String> = ["estimate_depth"]
    private var graph: Module?
    private var moge: Bool { architecture == "moge3" }
    private var lab = false
    public var peakBytes: Int { Memory.peakMemory }
    public var parameters: [String: MLXArray] { graph.map { Dictionary(uniqueKeysWithValues: $0.parameters().flattened()) } ?? [:] }
    public init(_ architecture: String) { self.architecture = architecture }

    /// Builds the untrained graph for an architecture; `lab` shrinks widths for wiring checks.
    static func makeGraph(architecture: String, config: [String: Any], lab: Bool) -> Module {
        let module: Module
        if architecture == "moge3" {
            var cfg = config
            if lab {
                var encoder = cfg["encoder"] as? [String: Any] ?? [:]
                encoder["embed_dim"] = 64; encoder["depth"] = 4; encoder["num_heads"] = 4
                encoder["dim_out"] = 64; encoder["intermediate_layers"] = [0, 1, 2, 3]
                cfg["encoder"] = encoder
                let widths = [64, 32, 16, 8, 8]
                for name in ["neck", "points_head", "normal_head", "mask_head"] {
                    var part = cfg[name] as? [String: Any] ?? [:]
                    part["dim_res_blocks"] = widths
                    part["dim_in"] = name == "neck" ? [66, 2, 2, 2, 2] : widths
                    cfg[name] = part
                }
                cfg["scale_head"] = ["dims": [64, 64, 64, 1]]
                var refiner = cfg["refiner"] as? [String: Any] ?? [:]
                refiner["encoder_channels"] = 66; refiner["model_channels"] = [8, 16, 32, 32, 64]
                cfg["refiner"] = refiner
            }
            module = MoGeModel(
                encoder: cfg["encoder"] as? [String: Any] ?? [:], neck: cfg["neck"] as? [String: Any] ?? [:],
                pointsHead: cfg["points_head"] as? [String: Any], normalHead: cfg["normal_head"] as? [String: Any], maskHead: cfg["mask_head"] as? [String: Any],
                scaleHead: cfg["scale_head"] as? [String: Any], refiner: cfg["refiner"] as? [String: Any],
                refinerDepthResolution: (cfg["refiner_depth_resolution"] as? NSNumber)?.floatValue ?? 256,
                remapOutput: "exp", numTokensRange: lab ? [4, 4] : cfg["num_tokens_range"] as? [Int] ?? [1200, 3600])
        } else {
            let base = architecture == "da3mono" ? DA3Config.da3monoLarge : DA3Config.da3Small
            let cfg =
                lab
                ? DA3Config(
                    name: base.name, backbone: BackboneConfig(embedDim: 64, depth: 4, numHeads: 4), outLayers: [0, 1, 2, 3],
                    altStart: base.altStart < 0 ? -1 : 1, qknormStart: base.qknormStart < 0 ? -1 : 1, ropeStart: base.ropeStart < 0 ? -1 : 1,
                    catToken: base.catToken, headType: base.headType,
                    head: HeadConfig(
                        dimIn: base.catToken ? 128 : 64, features: 16,
                        outChannels: [16, 32, 64, 64], outputDim: base.head.outputDim, posEmbed: base.head.posEmbed,
                        useSkyHead: base.head.useSkyHead, normType: base.head.normType)) : base
            module = MLXDA3.buildModel(config: cfg)
        }
        return module
    }

    public func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        lab = context.randomWeights && context.config["full_inventory"]?.bool != true
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
        let module = Self.makeGraph(architecture: architecture, config: config, lab: lab)
        graph = module
        try NativeParameters.initializeRandom(module)
        let tier = try NativePrecisionTier(context.precision)
        if context.randomWeights && tier.bits != nil {
            // Use exactly the stored/derived-tier installer used by trained checkpoints, on this graph.
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-geometry-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: temporary) }
            let file = temporary.appendingPathComponent("random.safetensors")
            try MLX.save(arrays: parameters, url: file)
            _ = try tier.load(module, archives: [file], map: { ($0, $1) }, eligible: GeometryWeights.eligible)
        } else if !context.randomWeights {
            let files = try FileManager.default.contentsOfDirectory(at: context.weights, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }
            guard !files.isEmpty else { throw failure("The verified snapshot contains no safetensors checkpoint.") }
            _ = try tier.load(module, archives: files, map: { [moge] key, value in GeometryWeights.load(key, value, moge: moge) }, eligible: GeometryWeights.eligible)
        }
        try cancellation.check(); module.train(false)
    }

    public func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "estimate_depth", let graph else { throw failure("Load a depth model before estimating depth.") }
        try cancellation.check()
        let paths: [String]
        if let path = input["image"]?.object?["path"]?.string, input["images"] == nil {
            paths = [path]
        } else if let images = input["images"]?.array, input["image"] == nil {
            paths = images.compactMap { $0.object?["path"]?.string }; guard paths.count == images.count else { throw failure("Each view needs an image path.") }
        } else {
            throw failure("Supply exactly one image or images array.")
        }
        guard !paths.isEmpty, paths.count <= (architecture == "da3small" ? 8 : 1), let output = input["out_dir"]?.string else {
            throw failure("Invalid view count or missing output directory.")
        }
        let mode = moge ? "metric" : "relative"
        guard input["mode"] == nil || input["mode"]?.string == mode else { throw failure("This checkpoint supports only \(mode) depth.") }
        let root = URL(fileURLWithPath: output)
        var files: [JSONValue] = [], views: [JSONValue] = []
        var tensors: [[String: MLXArray]] = []
        var originals: [(Int, Int)] = []
        if let model = graph as? MoGeModel {
            let image = try NativeMedia.image(URL(fileURLWithPath: paths[0]))
            var rgb = try rgbArray(image)
            let capped = MoGeInputSize.capped(width: image.width, height: image.height)
            if capped.width != image.width || capped.height != image.height {
                rgb = MoGeInputSize.areaResize(rgb, height: capped.height, width: capped.width)
            }
            let fov = input["focal_length"]?.number.map { Float(2 * atan(Double(image.width) / (2 * $0))) }
            let result = try model.infer(
                image: rgb.asType(.float16), resolutionLevel: Int(input["resolution_level"]?.number ?? 9), applyMask: input["apply_mask"]?.bool ?? true, fovX: fov,
                refineSteps: input["refine_steps"]?.number.map(Int.init), checkpoint: { try cancellation.check() })
            tensors = [result]; originals = [(image.width, image.height)]
        } else if let model = graph as? DepthAnything3 {
            var batch: [MLXArray] = []
            for path in paths {
                try cancellation.check()
                let image = try NativeMedia.image(URL(fileURLWithPath: path))
                originals.append((image.width, image.height))
                batch.append(ImageProcessor(processRes: lab ? 56 : 504)(image).asType(.float16))
            }
            guard Set(batch.map { $0.shape }).count == 1 else { throw failure("Multi-view images must produce the same processed dimensions; use matching aspect ratios.") }
            let result = try model.predict(MLX.stacked(batch, axis: 1), checkpoint: { try cancellation.check() })
            for index in paths.indices { tensors.append(result.mapValues { $0[0, index] }) }
            if architecture == "da3small", input["intrinsics"]?.bool == true || input["poses"]?.bool == true || input["points"]?.bool == true,
                let rays = result["ray"], let confidence = result["ray_conf"]
            {
                try cancellation.check()
                let cameras = RayPose.cameraInfoFromRays(ray: rays[0], rayConf: confidence[0], imageHeight: batch[0].dim(1), imageWidth: batch[0].dim(2))
                for index in paths.indices {
                    tensors[index]["intrinsics"] = cameras.intrinsics[index]
                    tensors[index]["extrinsics"] = MLX.concatenated([cameras.extrinsicsW2C[index], MLXArray([Float(0), 0, 0, 1], [1, 4])], axis: 0)
                    if input["points"]?.bool == true, let depth = tensors[index]["depth"] {
                        let h = depth.dim(0), w = depth.dim(1), k = cameras.intrinsics[index]
                        let xs = MLX.broadcast(MLX.arange(w).asType(.float32).reshaped([1, w]), to: [h, w])
                        let ys = MLX.broadcast(MLX.arange(h).asType(.float32).reshaped([h, 1]), to: [h, w])
                        tensors[index]["points"] = MLX.stacked([(xs - k[0, 2]) / k[0, 0] * depth, (ys - k[1, 2]) / k[1, 1] * depth, depth], axis: -1)
                    }
                }
            }
        }
        try cancellation.check()
        // Every view is validated before any file is written, so a failed view leaves no partial artifacts.
        var selections: [[String: MLXArray]] = []
        for index in tensors.indices {
            try cancellation.check()
            let maps = tensors[index]
            guard let depth = maps["depth"], depth.ndim == 2 else { throw failure("The depth graph returned an invalid shape.") }
            var selected: [String: MLXArray] = ["depth": depth]
            if input["points"]?.bool == true, let points = maps["points"] { selected["points"] = points }
            if input["normals"]?.bool == true, let normal = maps["normal"] { selected["normals"] = normal }
            if let mask = maps["mask"] { selected["mask"] = mask }
            if let confidence = maps["depth_conf"] { selected["confidence"] = confidence }
            if let sky = maps["sky"] { selected["sky"] = sky }
            // Ray maps preserve the multi-view model's geometric output even when pose recovery is unavailable.
            if let ray = maps["ray"] { selected["rays"] = ray }
            selected = DepthOutputValidation.stored(selected)
            try DepthOutputValidation.validate(selected, maskedInfinity: moge)
            if input["poses"]?.bool == true, let extrinsics = maps["extrinsics"], !extrinsics.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite) {
                throw failure("Camera recovery produced non-finite extrinsics.")
            }
            if input["intrinsics"]?.bool == true, let intrinsics = maps["intrinsics"], !intrinsics.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite) {
                throw failure("Camera recovery produced non-finite intrinsics.")
            }
            selections.append(selected)
        }
        for index in tensors.indices {
            let maps = tensors[index], selected = selections[index], depth = selected["depth"]!
            let file = root.appendingPathComponent("view-\(index).safetensors")
            try cancellation.check(); try MLX.save(arrays: selected, url: file)
            let identifier = "view-\(index)"
            files.append(.object(try NativeMedia.file(file, root: root, mime: "application/octet-stream", id: identifier)))
            let preview = root.appendingPathComponent("view-\(index)-depth.png")
            try writeDepthPreview(depth, to: preview, cancellation: cancellation)
            let previewID = identifier + "-depth"
            files.append(.object(try NativeMedia.file(preview, root: root, mime: "image/png", id: previewID)))
            var view: JSONObject = [
                "index": .number(Double(index)), "file_ids": [.string(identifier), .string(previewID)],
                "image_size": [.number(Double(originals[index].0)), .number(Double(originals[index].1))],
                "depth_size": [.number(Double(depth.dim(1))), .number(Double(depth.dim(0)))]
            ]
            if selected["points"] != nil { view["point_frame"] = "camera_x_right_y_down_z_forward" }
            if input["poses"]?.bool == true, let extrinsics = maps["extrinsics"] {
                let values = extrinsics.asType(.float32).asArray(Float.self)
                guard values.allSatisfy(\.isFinite) else { throw failure("Camera recovery produced non-finite extrinsics.") }
                view["pose_convention"] = "world_to_camera"
                view["extrinsics"] = .array((0..<4).map { row in .array((0..<4).map { .number(Double(values[row * 4 + $0])) }) })
            }
            if input["intrinsics"]?.bool == true, let intrinsics = maps["intrinsics"] {
                let values = intrinsics.asType(.float32).asArray(Float.self)
                guard values.allSatisfy(\.isFinite) else { throw failure("Camera recovery produced non-finite intrinsics.") }
                view["intrinsics_unit"] = .string(moge ? "normalized" : "pixels")
                view["intrinsics"] = .array((0..<3).map { row in .array((0..<3).map { .number(Double(values[row * 3 + $0])) }) })
            }
            views.append(.object(view))
        }
        return ["files": .array(files), "mode": .string(mode), "depth_unit": .string(moge ? "meters" : "relative"), "views": .array(views)]
    }
    public func unload() { graph = nil; Memory.clearCache() }
    private func failure(_ text: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: text) }
    private func writeDepthPreview(_ depth: MLXArray, to url: URL, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let values = depth.asType(.float32).asArray(Float.self)
        let finite = values.filter { $0.isFinite && $0 > 0 }
        let lower = finite.min() ?? 0, upper = finite.max() ?? 1
        let span = Swift.max(upper - lower, Float.ulpOfOne)
        var bytes = [UInt8](); bytes.reserveCapacity(values.count * 2)
        for (index, value) in values.enumerated() {
            if index % 65536 == 0 { try cancellation.check() }
            let normalized = value.isFinite && value > 0 ? Swift.min(1, Swift.max(0, (value - lower) / span)) : 0
            let sample = UInt16(normalized * 65535)
            bytes.append(UInt8(sample >> 8)); bytes.append(UInt8(sample & 255))
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
            let image = CGImage(
                width: depth.dim(1), height: depth.dim(0), bitsPerComponent: 16, bitsPerPixel: 16,
                bytesPerRow: depth.dim(1) * 2, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: .byteOrder16Big,
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw failure("Cannot encode the depth preview.") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw failure("Cannot write the depth preview.") }
    }
    private func rgbArray(_ image: CGImage) throws -> MLXArray {
        var rgba = [UInt8](repeating: 0, count: image.width * image.height * 4)
        guard
            let context = CGContext(
                data: &rgba, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw failure("Cannot decode RGB pixels.") }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        var values = [Float](); values.reserveCapacity(image.width * image.height * 3)
        for index in 0..<(image.width * image.height) {
            for channel in 0..<3 { values.append(Float(rgba[index * 4 + channel]) / 255) }
        }
        return MLXArray(values, [image.height, image.width, 3])
    }
}
