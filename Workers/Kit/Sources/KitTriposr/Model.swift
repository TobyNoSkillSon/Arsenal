// SPDX-License-Identifier: MIT
// TripoSR (stabilityai/TripoSR, model.safetensors from refs/pr/11), ported from VAST-AI-Research/TripoSR @ 107cefd:
// tsr/system.py, tsr/models/{tokenizers,transformer,network_utils,nerf_renderer,isosurface}.py and run.py's preprocessing.
// config.yaml (not pinned by the kit) is fixed here: cond_image_size 512, DINO ViT-B/16 image tokenizer, triplane 3x32x32x1024,
// Transformer1D 16 layers x 16 heads x 64, ConvTranspose2d up to 3x40x64x64, NeRF MLP 9x64 SiLU, radius 0.87, exp density bias -1.
// The mesh step is torchmcubes' CPU marching cubes (MarchingCubes.swift) at density 25, written as a vertex-coloured GLB.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXFast
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "geometry"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["image_to_3d"]

private let condImageSize = 512
private let radius: Float = 0.87
private let densityThreshold: Float = 25
private let foregroundRatio = 0.85

final class Model {
    let net: TSR

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        Memory.cacheLimit = 1 << 30
        net = TSR()
        var weights = try Weights.load(weightsDir)
        weights = weights.filter { !$0.key.hasPrefix("image_tokenizer.model.pooler.") }  // pooler_output is unused
        try Weights.apply(weights, to: net) { key in
            var k = key
            // decoder.layers: nn.Sequential with SiLU at the odd indices; Linear i*2 -> i.
            if k.hasPrefix("decoder.layers."), let n = Int(k.split(separator: ".")[2]) {
                k = "decoder.layers.\(n / 2)." + k.split(separator: ".").dropFirst(3).joined(separator: ".")
            }
            k = k.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj.").replacingOccurrences(of: ".ff.net.2.", with: ".ff.out.")
            return k
        }
        net.image_tokenizer.model.embeddings.preparePositions(grid: condImageSize / 16)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "image_to_3d" else { throw RunnerError.invalid("Unknown op \(op).") }
        let url = try params.file("image")
        let resolution = params.int("resolution", default: 256)
        // The 3D kind allows 256 to 1536; this port's marching cubes runs on the CPU and stops at 512.
        guard resolution <= 512 else {
            throw RunnerError.invalid("resolution \(resolution) is over 512, this port's limit (marching-cubes grid per side; default 256).")
        }
        guard resolution >= 32 else {
            throw RunnerError.invalid("resolution must be 256 to 512 (marching-cubes grid per side; default 256).")
        }
        let image = try preprocess(url)  // [512, 512, 3] in 0...1
        try params.checkCancelled()
        let planes = net.sceneCode(image)  // [3, 64, 64, 40]
        eval(planes)
        if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
            try dumpF32(planes, URL(fileURLWithPath: dump).appendingPathComponent("triplane.f32"))
            try dumpF32(image, URL(fileURLWithPath: dump).appendingPathComponent("input512.f32"))
        }
        try params.checkCancelled()

        // Density on the (resolution)^3 grid, linspace(0, 1) per axis scaled to (-radius, radius), ij order (x slowest).
        let n = resolution
        var density = [Float](repeating: 0, count: n * n * n)
        let axis = (0..<n).map { Float($0) / Float(n - 1) * 2 * radius - radius }
        let slab = max(1, (1 << 18) / (n * n))  // x-slices per chunk, about 262k points
        let yz = MLXArray((0..<(n * n)).flatMap { [axis[$0 / n], axis[$0 % n]] }, [n * n, 2])
        for x0 in stride(from: 0, to: n, by: slab) {
            try params.checkCancelled()
            let x1 = min(x0 + slab, n)
            let xs = MLXArray(Array(axis[x0..<x1])).reshaped([x1 - x0, 1, 1])
            let points = concatenated(
                [broadcast(xs, to: [x1 - x0, n * n, 1]), broadcast(yz.reshaped([1, n * n, 2]), to: [x1 - x0, n * n, 2])], axis: -1
            ).reshaped([-1, 3])
            let d = net.query(planes, points).density.asArray(Float.self)
            density.replaceSubrange((x0 * n * n)..<(x1 * n * n), with: d)
            Memory.clearCache()
        }
        if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
            try density.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: dump).appendingPathComponent("density\(n).f32"))
        }
        try params.checkCancelled()

        // isosurface: torchmcubes on (density - 25) at level 0, vertices in grid units (x, y, z), then to world.
        let field = density.map { $0 - densityThreshold }
        let mesh = marchingCubes(field, n: n)
        guard !mesh.faces.isEmpty else {
            throw RunnerError.invalid("No surface was found: the model saw no solid object. Use a cut-out object photo (PNG with a transparent background).")
        }
        let world = mesh.vertices.map { $0 / Float(n - 1) * 2 * radius - radius }
        // Vertex colours: the decoder's colour at each vertex.
        var colors = [Float](); colors.reserveCapacity(world.count)
        let count = world.count / 3
        for start in stride(from: 0, to: count, by: 1 << 18) {
            let end = min(start + (1 << 18), count)
            let pts = MLXArray(Array(world[(start * 3)..<(end * 3)]), [end - start, 3])
            colors += net.query(planes, pts).color.asArray(Float.self)
        }
        if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
            try "\(count) \(mesh.faces.count / 3)\n".write(to: URL(fileURLWithPath: dump).appendingPathComponent("mesh-counts.txt"), atomically: true, encoding: .utf8)
        }
        let dir = try outputDirectory(params)
        let file = dir.appendingPathComponent("mesh.glb")
        try writeGLB(positions: world, colors: colors, faces: mesh.faces, to: file)
        return ["files": [[
            "id": "mesh", "path": file.path, "name": "mesh.glb", "mime": "model/gltf-binary", "label": "3D model",
            "bytes": try Files.size(file), "sha256": try Files.sha256(file),
        ]]]
    }
}

private func dumpF32(_ a: MLXArray, _ url: URL) throws {
    let values = a.asType(.float32).asArray(Float.self)
    try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: url)
}

// MARK: - Preprocessing (run.py and tsr/utils.py)

/// RGBA image: crop to the alpha>0 box (the maker's slice drops the last row and column), pad to a square, pad to
/// size/0.85, composite on 0.5 grey, truncate to 8 bits, then resize to 512 with PyTorch's antialiased bilinear.
/// An image with no transparency is used as it is (the maker's --no-remove-bg path; Arsenal has no rembg here).
private func preprocess(_ url: URL) throws -> MLXArray {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.")
    }
    // Each side 8...8192, read from the header before the pixels are decoded.
    let sideError = RunnerError.invalid("The image must be 8 to 8192 pixels per side.")
    if let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
       let pw = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
       let ph = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
       !((8...8192).contains(pw) && (8...8192).contains(ph)) { throw sideError }
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.")
    }
    let w = image.width, h = image.height
    guard (8...8192).contains(w), (8...8192).contains(h) else { throw sideError }
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let context = CGContext(
        data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { throw RunnerError.failed("Could not draw the image.") }
    context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

    var hasAlpha = false
    var y1 = Int.max, y2 = -1, x1 = Int.max, x2 = -1
    for y in 0..<h {
        for x in 0..<w {
            let a = rgba[(y * w + x) * 4 + 3]
            if a < 255 { hasAlpha = true }
            if a > 0 { y1 = min(y1, y); y2 = max(y2, y); x1 = min(x1, x); x2 = max(x2, x) }
        }
    }
    var pixels: [Float]  // RGB 0...1 after the 8-bit truncation
    var side: (Int, Int)
    if hasAlpha {
        guard y2 > y1, x2 > x1 else { throw RunnerError.invalid("The image is fully transparent; nothing to reconstruct.") }
        let fh = y2 - y1, fw = x2 - x1  // fg = image[y1:y2, x1:x2]
        let size = max(fh, fw)
        let ph0 = (size - fh) / 2, pw0 = (size - fw) / 2
        let newSize = Int(Double(size) / foregroundRatio)
        let qh0 = (newSize - size) / 2, qw0 = (newSize - size) / 2
        pixels = [Float](repeating: 0, count: newSize * newSize * 3)
        for oy in 0..<newSize {
            for ox in 0..<newSize {
                let sy = oy - qh0 - ph0, sx = ox - qw0 - pw0
                var r: Float = 0, g: Float = 0, b: Float = 0, a: Float = 0
                if sy >= 0, sy < fh, sx >= 0, sx < fw {
                    let i = ((sy + y1) * w + (sx + x1)) * 4
                    r = Float(rgba[i]) / 255; g = Float(rgba[i + 1]) / 255; b = Float(rgba[i + 2]) / 255; a = Float(rgba[i + 3]) / 255
                }
                // premultiplied colour + (1 - a) * 0.5, then (x * 255).astype(uint8)
                let o = (oy * newSize + ox) * 3
                pixels[o] = ((r + (1 - a) * 0.5) * 255).rounded(.down) / 255
                pixels[o + 1] = ((g + (1 - a) * 0.5) * 255).rounded(.down) / 255
                pixels[o + 2] = ((b + (1 - a) * 0.5) * 255).rounded(.down) / 255
            }
        }
        side = (newSize, newSize)
    } else {
        pixels = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) { for c in 0..<3 { pixels[i * 3 + c] = Float(rgba[i * 4 + c]) / 255 } }
        side = (h, w)
    }
    let resized = resizeAA(pixels, height: side.0, width: side.1, to: condImageSize)
    return MLXArray(resized, [condImageSize, condImageSize, 3])
}

/// F.interpolate(mode="bilinear", align_corners=False, antialias=True) weights for one axis (PIL-style triangle filter).
private func aaWeights(input: Int, output: Int) -> [(start: Int, weights: [Float])] {
    let scale = Double(input) / Double(output)
    let support = scale >= 1 ? scale : 1.0
    let invscale = scale >= 1 ? 1.0 / scale : 1.0
    return (0..<output).map { i in
        let center = scale * (Double(i) + 0.5)
        let xmin = max(Int(center - support + 0.5), 0)
        let xmax = min(Int(center + support + 0.5), input)
        var ws = (xmin..<xmax).map { j -> Double in
            let x = (Double(j) - center + 0.5) * invscale
            return max(0, 1 - abs(x))
        }
        let total = ws.reduce(0, +)
        if total > 0 { ws = ws.map { $0 / total } }
        return (xmin, ws.map(Float.init))
    }
}

private func resizeAA(_ src: [Float], height: Int, width: Int, to size: Int) -> [Float] {
    let wx = aaWeights(input: width, output: size), wy = aaWeights(input: height, output: size)
    var tmp = [Float](repeating: 0, count: height * size * 3)
    for y in 0..<height {
        for (ox, f) in wx.enumerated() {
            var acc: (Float, Float, Float) = (0, 0, 0)
            for (k, wgt) in f.weights.enumerated() {
                let i = (y * width + f.start + k) * 3
                acc.0 += src[i] * wgt; acc.1 += src[i + 1] * wgt; acc.2 += src[i + 2] * wgt
            }
            let o = (y * size + ox) * 3
            tmp[o] = acc.0; tmp[o + 1] = acc.1; tmp[o + 2] = acc.2
        }
    }
    var out = [Float](repeating: 0, count: size * size * 3)
    for (oy, f) in wy.enumerated() {
        for x in 0..<size {
            var acc: (Float, Float, Float) = (0, 0, 0)
            for (k, wgt) in f.weights.enumerated() {
                let i = ((f.start + k) * size + x) * 3
                acc.0 += tmp[i] * wgt; acc.1 += tmp[i + 1] * wgt; acc.2 += tmp[i + 2] * wgt
            }
            let o = (oy * size + x) * 3
            out[o] = acc.0; out[o + 1] = acc.1; out[o + 2] = acc.2
        }
    }
    return out
}

// MARK: - Network

/// DINO ViT-B/16 (transformers ViTModel; facebook/dino-vitb16 config: 12 layers, 12 heads, 768, GELU, LayerNorm eps 1e-12).
final class ViTEmbeddings: Module {
    var cls_token: MLXArray
    var position_embeddings: MLXArray
    var patch_embeddings: PatchEmbeddings
    private let cache = PositionCache()

    override init() {
        cls_token = MLXArray.zeros([1, 1, 768]); position_embeddings = MLXArray.zeros([1, 197, 768]); patch_embeddings = PatchEmbeddings()
    }

    /// transformers 4.35 (TripoSR's pin) interpolate_pos_encoding: bicubic with scale_factor (grid + 0.1) / 14, align_corners False.
    func preparePositions(grid: Int) {
        let old = 14
        let scaleInv = Double(old) / (Double(grid) + 0.1)
        var matrix = [Float](repeating: 0, count: grid * old)
        let a = -0.75
        for o in 0..<grid {
            let src = (Double(o) + 0.5) * scaleInv - 0.5
            let f = src.rounded(.down), t = src - f
            let c0 = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
            let c1 = ((a + 2) * t - (a + 3)) * t * t + 1
            let c2 = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
            let c3 = 1 - c0 - c1 - c2
            for (k, c) in [c0, c1, c2, c3].enumerated() {
                let idx = min(max(Int(f) - 1 + k, 0), old - 1)
                matrix[o * old + idx] += Float(c)
            }
        }
        let m = MLXArray(matrix, [grid, old])
        let patch = position_embeddings[0, 1...].asType(.float32).reshaped([old, old, 768])
        // rows then columns: out[i, j] = sum m[i, a] m[j, b] patch[a, b]
        let rows = matmul(m, patch.reshaped([old, old * 768])).reshaped([grid, old, 768])
        let both = matmul(m, rows.transposed(1, 0, 2).reshaped([old, grid * 768])).reshaped([grid, grid, 768]).transposed(1, 0, 2)
        cache.value = concatenated([position_embeddings[0, ..<1].asType(.float32), both.reshaped([grid * grid, 768])], axis: 0)
            .asType(.float16).expandedDimensions(axis: 0)
        eval(cache.value!)
    }

    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let patches = patch_embeddings(pixels)  // [1, N, 768]
        let x = concatenated([cls_token.asType(patches.dtype), patches], axis: 1)
        return x + cache.value!
    }
}

/// Holds the interpolated position table outside the Module's parameters.
final class PositionCache { var value: MLXArray? }

final class PatchEmbeddings: Module {
    var projection: PatchConv
    override init() { projection = PatchConv() }
    /// [H, W, 3] normalised pixels to [1, (H/16)*(W/16), 768].
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let h = pixels.dim(0) / 16, w = pixels.dim(1) / 16
        let tiles = pixels.reshaped([h, 16, w, 16, 3]).transposed(0, 2, 4, 1, 3).reshaped([h * w, 3 * 16 * 16])
        return (matmul(tiles, projection.weight.reshaped([768, -1]).transposed()) + projection.bias).expandedDimensions(axis: 0)
    }
}

final class PatchConv: Module {
    var weight = MLXArray.zeros([768, 3, 16, 16])
    var bias = MLXArray.zeros([768])
}

final class ViTSelfAttention: Module {
    var query = Linear(768, 768), key = Linear(768, 768), value = Linear(768, 768)
}
final class Dense: Module {
    var dense: Linear
    init(_ i: Int, _ o: Int) { dense = Linear(i, o) }
}
final class ViTAttention: Module {
    var attention = ViTSelfAttention()
    var output = Dense(768, 768)
}
final class ViTLayer: Module {
    var attention = ViTAttention()
    var intermediate = Dense(768, 3072)
    var output = Dense(3072, 768)
    var layernorm_before = LayerNorm(dimensions: 768, eps: 1e-12)
    var layernorm_after = LayerNorm(dimensions: 768, eps: 1e-12)

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = layernorm_before(x)
        let a = attention.attention
        let attended = attend(a.query(h), a.key(h), a.value(h), heads: 12)
        let y = attention.output.dense(attended) + x
        let m = gelu(intermediate.dense(layernorm_after(y)).asType(.float32)).asType(y.dtype)
        return output.dense(m) + y
    }
}
final class ViTEncoder: Module {
    var layer: [ViTLayer] = (0..<12).map { _ in ViTLayer() }
}
final class ViT: Module {
    var embeddings = ViTEmbeddings()
    var encoder = ViTEncoder()
    var layernorm = LayerNorm(dimensions: 768, eps: 1e-12)
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var x = embeddings(pixels)
        for l in encoder.layer { x = l(x) }
        return layernorm(x)
    }
}
final class ImageTokenizer: Module {
    var model = ViT()
}

/// Multi-head attention on [1, L, H*D] projections.
func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, heads: Int) -> MLXArray {
    let b = q.dim(0), lq = q.dim(1), lk = k.dim(1), d = q.dim(2) / heads
    let qh = q.reshaped([b, lq, heads, d]).transposed(0, 2, 1, 3)
    let kh = k.reshaped([b, lk, heads, d]).transposed(0, 2, 1, 3)
    let vh = v.reshaped([b, lk, heads, d]).transposed(0, 2, 1, 3)
    let o = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: 1 / Float(d).squareRoot(), mask: nil)
    return o.transposed(0, 2, 1, 3).reshaped([b, lq, heads * d])
}

final class Attention: Module {
    var to_q: Linear, to_k: Linear, to_v: Linear
    var to_out: [Linear]
    init(dim: Int, context: Int) {
        to_q = Linear(dim, dim, bias: false); to_k = Linear(context, dim, bias: false); to_v = Linear(context, dim, bias: false)
        to_out = [Linear(dim, dim)]
    }
    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        to_out[0](attend(to_q(x), to_k(context), to_v(context), heads: 16))
    }
}

final class GEGLUFeedForward: Module {
    var proj = Linear(1024, 8192)
    var out = Linear(4096, 1024)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = proj(x)
        let hidden = p[0..., 0..., ..<4096], gate = p[0..., 0..., 4096...]
        return out(hidden * gelu(gate.asType(.float32)).asType(hidden.dtype))
    }
}

final class BasicTransformerBlock: Module {
    var norm1 = LayerNorm(dimensions: 1024), norm2 = LayerNorm(dimensions: 1024), norm3 = LayerNorm(dimensions: 1024)
    var attn1 = Attention(dim: 1024, context: 1024)
    var attn2 = Attention(dim: 1024, context: 768)
    var ff = GEGLUFeedForward()
    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let n1 = norm1(x)
        var h = attn1(n1, context: n1) + x
        h = attn2(norm2(h), context: context) + h
        return ff(norm3(h)) + h
    }
}

/// GroupNorm(32, 1024, eps 1e-6) over [1, L, C] channels-last, statistics in float32.
final class GroupNorm32: Module {
    var weight = MLXArray.ones([1024]), bias = MLXArray.zeros([1024])
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let l = x.dim(1)
        let g = x.asType(.float32).reshaped([1, l, 32, 32])
        let mean = g.mean(axes: [1, 3], keepDims: true)
        let variance = ((g - mean) * (g - mean)).mean(axes: [1, 3], keepDims: true)
        let y = ((g - mean) * rsqrt(variance + 1e-6)).reshaped([1, l, 1024])
        return (y * weight.asType(.float32) + bias.asType(.float32)).asType(x.dtype)
    }
}

final class Transformer1D: Module {
    var norm = GroupNorm32()
    var proj_in = Linear(1024, 1024)
    var transformer_blocks: [BasicTransformerBlock] = (0..<16).map { _ in BasicTransformerBlock() }
    var proj_out = Linear(1024, 1024)
    /// [1, L, C] tokens (channels-last), image tokens [1, 1025, 768].
    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var h = proj_in(norm(x))
        for block in transformer_blocks {
            h = block(h, context: context)
            eval(h)
        }
        return proj_out(h) + x
    }
}

final class TriplaneTokenizer: Module {
    var embeddings = MLXArray.zeros([3, 1024, 32, 32])
}

/// ConvTranspose2d(1024, 40, kernel 2, stride 2): non-overlapping, so one matmul per position.
final class Upsample: Module {
    var weight = MLXArray.zeros([1024, 40, 2, 2])
    var bias = MLXArray.zeros([40])
    /// [3, 32, 32, 1024] to [3, 64, 64, 40].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = matmul(x.reshaped([-1, 1024]), weight.reshaped([1024, 160])).reshaped([3, 32, 32, 40, 2, 2])
        return y.transposed(0, 1, 4, 2, 5, 3).reshaped([3, 64, 64, 40]) + bias
    }
}
final class PostProcessor: Module {
    var upsample = Upsample()
}

final class NeRFMLP: Module {
    var layers: [Linear] = [Linear(120, 64)] + (0..<8).map { _ in Linear(64, 64) } + [Linear(64, 4)]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for (i, l) in layers.enumerated() { h = l(h); if i < layers.count - 1 { h = silu(h) } }
        return h
    }
}

final class TSR: Module {
    var image_tokenizer = ImageTokenizer()
    var tokenizer = TriplaneTokenizer()
    var backbone = Transformer1D()
    var post_processor = PostProcessor()
    var decoder = NeRFMLP()

    /// [512, 512, 3] RGB 0...1 to triplanes [3, 64, 64, 40] (float16).
    func sceneCode(_ image: MLXArray) -> MLXArray {
        let normed = Images.normalize(image, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]).asType(.float16)
        let imageTokens = image_tokenizer.model(normed)  // [1, 1025, 768]
        eval(imageTokens)
        // "Np Ct Hp Wp -> Ct (Np Hp Wp)", here channels-last [1, 3*32*32, 1024]
        let tokens = tokenizer.embeddings.asType(.float16).transposed(0, 2, 3, 1).reshaped([1, 3 * 32 * 32, 1024])
        let out = backbone(tokens, context: imageTokens)
        return post_processor.upsample(out.reshaped([3, 32, 32, 1024]))
    }

    /// query_triplane: points [N, 3] in (-radius, radius) to density_act (float32 [N]) and colour (float32 [N, 3]).
    func query(_ planes: MLXArray, _ points: MLXArray) -> (density: MLXArray, color: MLXArray) {
        let p = points.asType(.float32) / radius  // scale_tensor to (-1, 1)
        let px = p[0..., 0], py = p[0..., 1], pz = p[0..., 2]
        let pairs = [(px, py), (px, pz), (py, pz)]
        var features: [MLXArray] = []
        for (i, (u, v)) in pairs.enumerated() { features.append(gridSample(planes[i], u, v)) }
        let out = decoder(concatenated(features, axis: -1)).asType(.float32)
        let density = exp(out[0..., 0] - 1)
        let color = sigmoid(out[0..., 1...])
        eval(density, color)
        return (density, color)
    }
}

/// F.grid_sample(bilinear, align_corners=False, zeros padding) of plane [H, W, C] at u (x, width) and v (y, height) in -1...1.
func gridSample(_ plane: MLXArray, _ u: MLXArray, _ v: MLXArray) -> MLXArray {
    let h = plane.dim(0), w = plane.dim(1), c = plane.dim(2)
    let flat = plane.reshaped([h * w, c])
    let ix = ((u + 1) * Float(w) - 1) / 2, iy = ((v + 1) * Float(h) - 1) / 2
    let x0 = floor(ix), y0 = floor(iy)
    let fx = ix - x0, fy = iy - y0
    var acc = MLXArray.zeros([u.dim(0), c], dtype: .float32)
    for (dx, dy, wgt) in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)] {
        let xi = x0 + Float(dx), yi = y0 + Float(dy)
        let valid = (xi .>= 0) .&& (xi .<= Float(w - 1)) .&& (yi .>= 0) .&& (yi .<= Float(h - 1))
        let index = (clip(yi, min: 0, max: Float(h - 1)) * Float(w) + clip(xi, min: 0, max: Float(w - 1))).asType(.int32)
        let tap = flat.take(index, axis: 0).asType(.float32)
        acc = acc + tap * (wgt * valid.asType(.float32)).expandedDimensions(axis: -1)
    }
    return acc.asType(plane.dtype)
}

// MARK: - Files

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
