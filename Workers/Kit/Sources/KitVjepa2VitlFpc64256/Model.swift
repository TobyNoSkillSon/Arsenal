// SPDX-License-Identifier: MIT
// V-JEPA 2 ViT-L/16 video encoder (facebook/vjepa2-vitl-fpc64-256, weights from the mlx-community FP16 build),
// ported from transformers' modeling_vjepa2.py (VJEPA2Model.get_vision_features: VJEPA2Encoder). The checkpoint's
// JEPA predictor is not used. Preprocessing follows the maker's VJEPA2VideoProcessor: frames as 8-bit RGB, resized so
// the shorter side is 292 (bilinear with antialiasing, rounded to 8 bits per pass as torchvision does for uint8),
// centre-cropped to 256 x 256, scaled to 0...1 and normalised with ImageNet mean and std. Frames go in pairs
// (tubelets of 2 x 16 x 16 pixels), so a clip of T frames gives T/2 x 16 x 16 tokens.
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN

let modelKind = "local.video"
let modelOps = ["embed"]

final class Model {
    let net: VJEPA2Encoder
    let crop: Int, resizeShort: Int, tubelet: Int, patch: Int, maxFrames: Int
    let mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        let hidden = config["hidden_size"] as? Int ?? 1024
        crop = config["crop_size"] as? Int ?? 256
        resizeShort = Int(Double(crop) * 256 / 224)  // VJEPA2VideoProcessor: shortest_edge = crop * 256 / 224 = 292
        tubelet = config["tubelet_size"] as? Int ?? 2
        patch = config["patch_size"] as? Int ?? 16
        maxFrames = config["frames_per_clip"] as? Int ?? 64
        net = VJEPA2Encoder(
            hidden: hidden, layers: config["num_hidden_layers"] as? Int ?? 24, heads: config["num_attention_heads"] as? Int ?? 16,
            mlpRatio: config["mlp_ratio"] as? Int ?? 4, patch: patch, tubelet: tubelet, channels: config["in_chans"] as? Int ?? 3,
            eps: Float(config["layer_norm_eps"] as? Double ?? 1e-6), grid: crop / patch)
        try Weights.apply(try Weights.load(weightsDir), to: net) { key in
            guard key.hasPrefix("encoder.") else { return nil }  // the JEPA predictor is not served
            return String(key.dropFirst("encoder.".count))
        }
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let count = params.int("num_frames", default: maxFrames)  // the card samples 64 frames for this model
        guard count >= 1, count <= maxFrames else { throw RunnerError.invalid("num_frames must be 1-\(maxFrames).") }
        var frames: [MLXArray] = []  // each [H, W, 3] float 0...255 (whole numbers)
        var times: [Double] = []
        var still: MLXArray? = nil  // the image path: one preprocessed picture, repeated below
        if params.raw["video"] != nil {
            (frames, times) = try decodeVideo(try params.file("video"), count: count,
                                              start: params.raw["start"] as? NSNumber, end: params.raw["end"] as? NSNumber)
        } else if params.raw["frames"] != nil {
            guard let list = params.raw["frames"] as? [Any], !list.isEmpty, list.count <= maxFrames else {
                throw RunnerError.invalid("frames must be a list of 1-\(maxFrames) image files, in time order.")
            }
            for (i, item) in list.enumerated() {
                let path = (item as? [String: Any])?["path"] as? String ?? item as? String
                guard let path, FileManager.default.fileExists(atPath: path) else { throw RunnerError.invalid("frames[\(i)] must be an existing image file.") }
                frames.append(try preprocess(try imagePixels(URL(fileURLWithPath: path))))
                eval(frames[frames.count - 1])
            }
        } else if params.raw["image"] != nil {
            // The card's image example: one picture repeated as a still clip.
            // Preprocessed once and repeated, so the graph holds one resize instead of one per frame.
            let one = try preprocess(try imagePixels(try params.file("image")))
            eval(one)
            still = one
        } else {
            throw RunnerError.invalid("Give video (a clip), frames (image files in time order) or image.")
        }
        // Tubelets need an even frame count: repeat the last frame (the encoder itself repeats a single frame).
        var frameCount = still == nil ? frames.count : count
        if frameCount % tubelet != 0 { frameCount += 1 }
        let pixels: MLXArray  // [T, 256, 256, 3]
        if let still {
            pixels = broadcast(still.expandedDimensions(axis: 0), to: [frameCount] + still.shape)
        } else {
            if frames.count % tubelet != 0 { frames.append(frames[frames.count - 1]) }
            pixels = stacked(frames, axis: 0)
        }
        let hidden = try net(pixels.asType(.float16), params: params)  // [N, hidden]
        let h = hidden.asType(.float32)
        let embedding = h.mean(axis: 0)
        let tokensPerStep = (crop / patch) * (crop / patch)
        eval(embedding)
        var out: [String: Any] = [
            "embedding": list(embedding), "dimensions": embedding.dim(0), "frames": frameCount, "tokens": h.dim(0),
        ]
        if !times.isEmpty { out["frame_times"] = times }
        if params.bool("per_step", default: false) {
            let steps = h.reshaped([h.dim(0) / tokensPerStep, tokensPerStep, h.dim(1)]).mean(axis: 1)
            let rows = steps.asArray(Float.self), width = h.dim(1)
            out["per_step"] = (0 ..< steps.dim(0)).map { s in rows[(s * width) ..< ((s + 1) * width)].map(Double.init) }
        }
        return out
    }

    // MARK: frames

    /// Any image as [H, W, 3] floats 0...255, at its own size.
    func imagePixels(_ url: URL) throws -> MLXArray {
        _ = try checkImageHeader(url)
        let (pixels, _, _) = try Images.load(url)
        return (pixels * 255).round()
    }

    func cgPixels(_ image: CGImage) throws -> MLXArray {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        guard let context = CGContext(
            data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw RunnerError.failed("Could not draw a video frame.") }
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return MLXArray(bytes, [h, w, 4])[0..., 0..., ..<3].asType(.float32)
    }

    /// `count` frames evenly spaced over [start, end) of the clip (the whole clip by default), each at the centre of
    /// its share of the window, decoded exactly at that time.
    func decodeVideo(_ url: URL, count: Int, start: NSNumber?, end: NSNumber?) throws -> ([MLXArray], [Double]) {
        let asset = AVURLAsset(url: url)
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var duration = 0.0
        nonisolated(unsafe) var hasVideo = false
        Task {
            duration = (try? await asset.load(.duration).seconds) ?? 0
            hasVideo = !((try? await asset.loadTracks(withMediaType: .video)) ?? []).isEmpty
            semaphore.signal()
        }
        semaphore.wait()
        guard hasVideo, duration.isFinite, duration > 0 else { throw RunnerError.invalid("\(url.lastPathComponent) is not a readable video.") }
        // The frame size before decoding: the same 8192-pixel side and 20:1 aspect caps as images.
        let probe = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var natural = CGSize.zero
        Task {
            if let track = try? await asset.loadTracks(withMediaType: .video).first,
               let size = try? await track.load(.naturalSize), let transform = try? await track.load(.preferredTransform) {
                natural = size.applying(transform)
            }
            probe.signal()
        }
        probe.wait()
        if natural != .zero {  // unknown here: each decoded frame is still checked in preprocess
            try checkFrameSize(width: Int(abs(natural.width).rounded()), height: Int(abs(natural.height).rounded()))
        }
        let from = max(0, start?.doubleValue ?? 0), to = min(duration, end?.doubleValue ?? duration)
        guard to > from else { throw RunnerError.invalid("start must be before end, inside the clip's \(String(format: "%.2f", duration)) s.") }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [MLXArray] = [], times: [Double] = []
        for i in 0 ..< count {
            let t = from + (Double(i) + 0.5) * (to - from) / Double(count)
            let time = CMTime(seconds: t, preferredTimescale: 600_000)
            nonisolated(unsafe) var image: CGImage?
            nonisolated(unsafe) var actual = t
            let done = DispatchSemaphore(value: 0)
            generator.generateCGImageAsynchronously(for: time) { cg, when, _ in
                image = cg; actual = when.seconds; done.signal()
            }
            done.wait()
            guard let image else { throw RunnerError.invalid("Could not decode \(url.lastPathComponent) at \(String(format: "%.2f", t)) s.") }
            let frame = try preprocess(try cgPixels(image))
            eval(frame)
            frames.append(frame); times.append(actual)
        }
        return (frames, times)
    }

    /// The resize uses dense [out, in] matrices, so a side over 8192 pixels or an aspect beyond 20:1 is refused.
    func checkFrameSize(width: Int, height: Int) throws {
        guard width >= 1, height >= 1, width <= 8192, height <= 8192 else {
            throw RunnerError.invalid("The picture is \(width)x\(height); each side must be 1 to 8192 pixels.")
        }
        guard max(width, height) <= 20 * min(width, height) else {
            throw RunnerError.invalid("The picture is \(width)x\(height); its aspect must be at most 20:1.")
        }
    }

    /// [H, W, 3] 0...255 -> [256, 256, 3] normalised, as VJEPA2VideoProcessor does it.
    func preprocess(_ frame: MLXArray) throws -> MLXArray {
        let h = frame.dim(0), w = frame.dim(1)
        guard h >= 2, w >= 2 else { throw RunnerError.invalid("Frames must be at least 2 x 2 pixels.") }
        try checkFrameSize(width: w, height: h)
        // get_resize_output_image_size(shortest_edge): the short side becomes 292, the long side int(292 * long / short).
        let (newH, newW) = h <= w ? (resizeShort, Int(Double(resizeShort) * Double(w) / Double(h)))
                                  : (Int(Double(resizeShort) * Double(h) / Double(w)), resizeShort)
        var x = frame
        // torchvision's uint8 antialiased bilinear resize: horizontal pass, round, vertical pass, round.
        if newW != w {
            x = matmul(x.transposed(0, 2, 1), MLXArray(Resize.weights(w, newW), [newW, w]).transposed()).transposed(0, 2, 1)
            x = clip(floor(x + 0.5), min: 0, max: 255)
        }
        if newH != h {
            x = matmul(MLXArray(Resize.weights(h, newH), [newH, h]), x.reshaped([h, -1])).reshaped([newH, newW, 3])
            x = clip(floor(x + 0.5), min: 0, max: 255)
        }
        // The processor's centre crop takes the floor of (size - crop) / 2 (checked against it on a 519-wide frame).
        let top = (newH - crop) / 2, left = (newW - crop) / 2
        x = x[top ..< (top + crop), left ..< (left + crop)]
        return (x / 255 - MLXArray(mean)) / MLXArray(std)
    }
}

enum Resize {
    /// Row-major [out, in] weights of PyTorch's antialiased bilinear interpolation
    /// (aten UpSampleKernel _compute_indices_min_size_weights_aa, align_corners=False).
    static func weights(_ inSize: Int, _ outSize: Int) -> [Float] {
        let scale = Double(inSize) / Double(outSize)
        let support = scale >= 1 ? scale : 1.0
        let invscale = scale >= 1 ? 1 / scale : 1.0
        var w = [Float](repeating: 0, count: outSize * inSize)
        for i in 0 ..< outSize {
            let center = scale * (Double(i) + 0.5)
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize)
            var taps: [Double] = []
            for j in xmin ..< max(xmin, xmax) { taps.append(max(0, 1 - abs((Double(j) - center + 0.5) * invscale))) }
            let total = taps.reduce(0, +)
            for (k, tap) in taps.enumerated() where total > 0 { w[i * inSize + xmin + k] = Float(tap / total) }
        }
        return w
    }
}

// MARK: network (names follow the checkpoint, prefix "encoder." removed)

final class VJEPA2Encoder: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embeddings
    @ModuleInfo(key: "layer") var layer: [VJEPA2Layer]
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm
    let heads: Int, grid: Int, tubelet: Int, patch: Int

    init(hidden: Int, layers: Int, heads: Int, mlpRatio: Int, patch: Int, tubelet: Int, channels: Int, eps: Float, grid: Int) {
        self.heads = heads; self.grid = grid; self.tubelet = tubelet; self.patch = patch
        _embeddings.wrappedValue = Embeddings(hidden: hidden, patch: patch, tubelet: tubelet, channels: channels)
        _layer.wrappedValue = (0 ..< layers).map { _ in VJEPA2Layer(hidden: hidden, heads: heads, mlpRatio: mlpRatio, eps: eps) }
        _layernorm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }

    /// video [T, 256, 256, 3] (T even) -> last hidden state [T/2 * 16 * 16, hidden].
    func callAsFunction(_ video: MLXArray, params: Params) throws -> MLXArray {
        let t = video.dim(0) / tubelet, g = grid, c = video.dim(3)
        // Conv3d with stride = kernel as a matmul over tubelets; tokens in (time, row, column) order.
        let patches = video.reshaped([t, tubelet, g, patch, g, patch, c]).transposed(0, 2, 4, 1, 3, 5, 6)
            .reshaped([t * g * g, tubelet * patch * patch * c])
        let proj = embeddings.patchEmbeddings.proj
        let w = proj.weight.reshaped([proj.weight.dim(0), -1])
        var x = (matmul(patches, w.transposed()) + proj.bias).expandedDimensions(axis: 0)
        let rope = Rope(tokens: t * g * g, grid: g, headDim: x.dim(2) / heads, dtype: x.dtype)
        for block in layer {
            x = block(x, rope: rope)
            try params.checkCancelled()
        }
        return layernorm(x)[0]
    }
}

final class Embeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: PatchEmbeddings
    init(hidden: Int, patch: Int, tubelet: Int, channels: Int) {
        _patchEmbeddings.wrappedValue = PatchEmbeddings(hidden: hidden, patch: patch, tubelet: tubelet, channels: channels)
    }
}

final class PatchEmbeddings: Module {
    @ModuleInfo(key: "proj") var proj: Conv3dWeights
    init(hidden: Int, patch: Int, tubelet: Int, channels: Int) {
        _proj.wrappedValue = Conv3dWeights(out: hidden, kernel: [tubelet, patch, patch], channels: channels)
    }
}

/// The Conv3d's weight [out, kT, kH, kW, in] (the MLX build stores it channels-last) and bias.
final class Conv3dWeights: Module {
    var weight: MLXArray
    var bias: MLXArray
    init(out: Int, kernel: [Int], channels: Int) {
        weight = MLXArray.zeros([out] + kernel + [channels])
        bias = MLXArray.zeros([out])
    }
}

/// VJEPA2RopeAttention's 3-D rotary embedding: the head's first 3 x 20 channels rotate by frame, row and column
/// position (the rest pass through). As rotate_queries_or_keys does, channel pairs (2i, 2i+1) rotate together while
/// the angle table is tiled ([f0...f9, f0...f9]), so channel k uses frequency k mod 10.
struct Rope {
    let part: Int
    let cos: [MLXArray], sin: [MLXArray]  // frame, row, column: each [N, part]
    init(tokens: Int, grid: Int, headDim: Int, dtype: DType) {
        part = 2 * ((headDim / 3) / 2)
        let half = part / 2
        let omega = (0 ..< half).map { 1 / pow(10000, Double($0) / Double(half)) }
        let ids = Array(0 ..< tokens)
        let positions = [ids.map { $0 / (grid * grid) }, ids.map { ($0 % (grid * grid)) / grid }, ids.map { $0 % grid }]
        var c: [MLXArray] = [], s: [MLXArray] = []
        for pos in positions {
            var angles = [Float](repeating: 0, count: tokens * part)
            for (n, p) in pos.enumerated() {
                for k in 0 ..< part { angles[n * part + k] = Float(Double(p) * omega[k % half]) }
            }
            let a = MLXArray(angles, [tokens, part])
            c.append(MLX.cos(a).asType(dtype)); s.append(MLX.sin(a).asType(dtype))
        }
        cos = c; sin = s
    }

    func apply(_ x: MLXArray) -> MLXArray {
        var pieces: [MLXArray] = []
        for i in 0 ..< 3 {
            let seg = x[.ellipsis, (i * part) ..< ((i + 1) * part)]
            var shape = seg.shape
            let pairs = seg.reshaped(Array(shape.dropLast()) + [part / 2, 2])
            let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1)
            shape = seg.shape
            pieces.append(seg * cos[i] + rotated.reshaped(shape) * sin[i])
        }
        if 3 * part < x.dim(-1) { pieces.append(x[.ellipsis, (3 * part)...]) }
        return concatenated(pieces, axis: -1)
    }
}

final class Attention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let heads: Int
    init(hidden: Int, heads: Int) {
        self.heads = heads
        _query.wrappedValue = Linear(hidden, hidden)
        _key.wrappedValue = Linear(hidden, hidden)
        _value.wrappedValue = Linear(hidden, hidden)
        _proj.wrappedValue = Linear(hidden, hidden)
    }
    func callAsFunction(_ x: MLXArray, rope: Rope) -> MLXArray {
        let (b, n, d) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = d / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3) }
        let q = rope.apply(split(query(x))), k = rope.apply(split(key(x))), v = split(value(x))
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: Float(1 / Double(headDim).squareRoot()), mask: nil)
        return proj(o.transposed(0, 2, 1, 3).reshaped([b, n, d]))
    }
}

final class MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(hidden: Int, ratio: Int) {
        _fc1.wrappedValue = Linear(hidden, hidden * ratio)
        _fc2.wrappedValue = Linear(hidden * ratio, hidden)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }  // ACT2FN["gelu"]: exact erf GELU
}

final class VJEPA2Layer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: Attention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    init(hidden: Int, heads: Int, mlpRatio: Int, eps: Float) {
        _norm1.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
        _attention.wrappedValue = Attention(hidden: hidden, heads: heads)
        _norm2.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
        _mlp.wrappedValue = MLP(hidden: hidden, ratio: mlpRatio)
    }
    func callAsFunction(_ x: MLXArray, rope: Rope) -> MLXArray {
        let a = attention(norm1(x), rope: rope) + x
        return mlp(norm2(a)) + a
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
