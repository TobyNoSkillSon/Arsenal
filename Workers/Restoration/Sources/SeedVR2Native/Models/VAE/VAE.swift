// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae/{encoder,decoder,vae}.py.
import MLX
import MLXNN

public final class Encoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock3D]
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(
        inChannels: Int = 3, outChannels: Int = 16,
        blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, temporalDownBlocks: Int = 2
    ) {
        self._convIn.wrappedValue = CausalConv3d(inChannels, blockOut[0])
        var blocks: [DownBlock3D] = []
        var outCh = blockOut[0]
        let n = blockOut.count
        for (i, ch) in blockOut.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalDown = (i >= n - temporalDownBlocks - 1) && !isFinal
            blocks.append(DownBlock3D(inCh, outCh, numLayers: layersPerBlock, addDownsample: !isFinal, temporalDown: temporalDown))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = MidBlock3D(blockOut[n - 1])
        self._convNormOut.wrappedValue = groupNorm32(blockOut[n - 1])
        self._convOut.wrappedValue = CausalConv3d(blockOut[n - 1], 2 * outChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        h = silu(vaeGroupNorm(h, convNormOut))
        return convOut(h)
    }
}

public final class Decoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock3D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(
        inChannels: Int = 16, outChannels: Int = 3,
        blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 3, temporalUpBlocks: Int = 2
    ) {
        let rev = Array(blockOut.reversed())
        self._convIn.wrappedValue = CausalConv3d(inChannels, rev[0])
        self._midBlock.wrappedValue = MidBlock3D(rev[0])
        var blocks: [UpBlock3D] = []
        var outCh = rev[0]
        let n = rev.count
        for (i, ch) in rev.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalUp = i < temporalUpBlocks
            blocks.append(UpBlock3D(inCh, outCh, numLayers: layersPerBlock, addUpsample: !isFinal, temporalUp: temporalUp))
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = groupNorm32(rev[n - 1])
        self._convOut.wrappedValue = CausalConv3d(rev[n - 1], outChannels)
        super.init()
    }

    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = midBlock(h)
        for b in upBlocks { h = b(h) }
        h = silu(vaeGroupNorm(h, convNormOut))
        return convOut(h)
    }
}

public final class SeedVR2VAE: Module {
    @ModuleInfo(key: "encoder") var encoder: Encoder3D
    @ModuleInfo(key: "decoder") var decoder: Decoder3D
    let scalingFactor: Float = 0.9152
    let latentChannels = 16

    public init(blockOut: [Int] = [128, 256, 512, 512]) {
        self._encoder.wrappedValue = Encoder3D(blockOut: blockOut)
        self._decoder.wrappedValue = Decoder3D(blockOut: blockOut)
        super.init()
    }

    /// x [B,3,T,H,W] (or [B,3,H,W]) -> latent [B,16,(T-1)/4+1,H/8,W/8]. `tile` (pixels, a multiple of 64) bounds activation
    /// memory by encoding overlapping spatial tiles; nil or a frame within one tile encodes the whole frame as upstream does.
    public func encode(_ xIn: MLXArray, tile: Int? = nil) -> MLXArray {
        let x = xIn.ndim == 4 ? xIn.expandedDimensions(axis: 2) : xIn
        let body: (MLXArray) -> MLXArray = { part in self.encoder(part)[0..., 0..<self.latentChannels] * self.scalingFactor }
        guard let tile else { return body(x) }
        return Self.tiled(x, tile: tile, overlap: 64, up: 1, down: 8, body: body)
    }

    /// z [B,16,T,H,W] -> image [B,3,4(T-1)+1,H*8,W*8]. `tile` is in output pixels (a multiple of 64).
    public func decode(_ zIn: MLXArray, tile: Int? = nil) -> MLXArray {
        let z = zIn.ndim == 4 ? zIn.expandedDimensions(axis: 2) : zIn
        let body: (MLXArray) -> MLXArray = { part in self.decoder(part / self.scalingFactor) }
        guard let tile else { return body(z) }
        return Self.tiled(z, tile: tile / 8, overlap: 8, up: 8, down: 1, body: body)
    }

    /// Frames the causal decoder returns for `latentFrames` latent frames (upstream: 4n+1 in, 4n+1 out).
    public static func decodedFrames(latentFrames: Int) -> Int { 4 * (latentFrames - 1) + 1 }

    /// Runs `body` over overlapping spatial tiles of x [B,C,T,H,W] (sizes in input units) and blends the outputs, whose
    /// spatial size is input * up / down, with linear ramps across each interior overlap. Each tile is evaluated before
    /// the next, so peak activation memory follows the tile size. Tiles see their own group-norm and mid-block
    /// attention context; the ramps hide the small seam differences. Upstream SeedVR inference does not tile (it runs on
    /// large GPUs); this bound is Arsenal's, unqualified on trained weights.
    public static func tiled(_ x: MLXArray, tile: Int, overlap: Int, up: Int, down: Int, body: (MLXArray) -> MLXArray) -> MLXArray {
        let height = x.dim(3), width = x.dim(4)
        if (height <= tile && width <= tile) || tile <= overlap { return body(x) }
        let ys = tileStarts(height, tile: tile, overlap: overlap), xs = tileStarts(width, tile: tile, overlap: overlap)
        let ramp = overlap * up / down
        var sum: MLXArray?, weight: MLXArray?
        for y0 in ys {
            for x0 in xs {
                let th = Swift.min(tile, height), tw = Swift.min(tile, width)
                let part = body(x[0..., 0..., 0..., y0..<(y0 + th), x0..<(x0 + tw)]).asType(.float32)
                let oy = y0 * up / down, ox = x0 * up / down, oh = part.dim(3), ow = part.dim(4)
                let wy = edgeRamp(oh, ramp: ramp, leading: y0 > 0, trailing: y0 + th < height)
                let wx = edgeRamp(ow, ramp: ramp, leading: x0 > 0, trailing: x0 + tw < width)
                let mask = MLXArray(wy).reshaped([1, 1, 1, oh, 1]) * MLXArray(wx).reshaped([1, 1, 1, 1, ow])
                if sum == nil {
                    var shape = part.shape
                    shape[3] = height * up / down; shape[4] = width * up / down
                    sum = MLXArray.zeros(shape, dtype: .float32)
                    weight = MLXArray.zeros([1, 1, 1, shape[3], shape[4]], dtype: .float32)
                }
                sum![0..., 0..., 0..., oy..<(oy + oh), ox..<(ox + ow)] += part * mask
                weight![0..., 0..., 0..., oy..<(oy + oh), ox..<(ox + ow)] += mask
                eval(sum!, weight!)
            }
        }
        return (sum! / weight!).asType(VAEPrecision.dtype)
    }

    /// Tile origins covering `length`: stride tile - overlap, the last tile flush with the end.
    public static func tileStarts(_ length: Int, tile: Int, overlap: Int) -> [Int] {
        guard length > tile else { return [0] }
        var starts: [Int] = [], start = 0
        while start + tile < length { starts.append(start); start += tile - overlap }
        starts.append(length - tile)
        return starts
    }

    /// Blend weights for one tile axis: ramps up over the leading overlap and down over the trailing one, never zero.
    public static func edgeRamp(_ count: Int, ramp: Int, leading: Bool, trailing: Bool) -> [Float] {
        (0..<count).map { i in
            var w: Float = 1
            if leading && i < ramp { w = Swift.min(w, Float(i + 1) / Float(ramp + 1)) }
            if trailing && count - 1 - i < ramp { w = Swift.min(w, Float(count - i) / Float(ramp + 1)) }
            return w
        }
    }
}
