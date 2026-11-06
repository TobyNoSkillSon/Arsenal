// SPDX-License-Identifier: MIT
// Per-component parity against mflux (9669af51) on the tiny random-initialised checkpoint in Fixtures/tiny: the same
// weights loaded by both implementations, fixed inputs, outputs captured once (lab/flux3/scripts/capture_tiny.py).
// Each check isolates one component by feeding mflux's captured input, then the chain runs end to end. Bounds are BF16
// rounding floors measured on M5 Max (log/002-parity.md in the add-on folder); FLUX3_PARITY_OUT=<file> appends one JSON
// line per check for the parity table.
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import Flux2Klein
@testable import Flux3Klein

func useTestMetallib() {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
}

enum ParityLog {
    static let lock = NSLock()
    static func record(_ row: [String: Any]) {
        guard let path = ProcessInfo.processInfo.environment["FLUX3_PARITY_OUT"],
            let line = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        else { return }
        lock.withLock {
            if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
            guard let handle = FileHandle(forWritingAtPath: path) else { return }
            handle.seekToEndOfFile(); handle.write(line + Data("\n".utf8)); try? handle.close()
        }
    }
}

/// max |a − b| (in float32) ≤ bound, recorded for the table. `bound` 0 means exact.
func expectClose(_ name: String, _ got: MLXArray, _ want: MLXArray, bound: Float, sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(got.shape == want.shape, "\(name): shape \(got.shape) vs \(want.shape)", sourceLocation: sourceLocation)
    guard got.shape == want.shape else { return }
    let a = got.asType(.float32), b = want.asType(.float32)
    let diff = abs(a - b).max().item(Float.self), scale = abs(b).max().item(Float.self)
    let finite = all(isFinite(a)).item(Bool.self)
    ParityLog.record([
        "component": name, "shape": got.shape, "max_abs_diff": Double(diff), "ref_max_abs": Double(scale), "dtype": String(describing: got.dtype),
        "ref_dtype": String(describing: want.dtype), "bound": Double(bound), "pass": finite && diff <= bound
    ])
    #expect(finite && diff <= bound, "\(name): max |diff| \(diff) > \(bound) (ref max \(scale))", sourceLocation: sourceLocation)
}

struct Tiny {
    let fixture: [String: MLXArray]
    let transformer: Flux3Transformer
    let text: Flux3TextEncoder
    let encoder: Flux3VAEEncoderHalf
    let decoder: Flux2VAEDecoder
    init() throws {
        useTestMetallib()
        fixture = try loadArrays(url: TinyFixture.components)
        transformer = try Flux3Weights.transformer(directory: TinyFixture.snapshot.appendingPathComponent("transformer"))
        text = try Flux3Weights.textEncoder(directory: TinyFixture.snapshot.appendingPathComponent("text_encoder"))
        (encoder, decoder) = try Flux3Weights.vae(directory: TinyFixture.snapshot.appendingPathComponent("vae"))
    }
    subscript(_ name: String) -> MLXArray { fixture[name]! }
    var rope: Flux3RopeTables {
        Flux3RopeTables(
            image: Flux3Positions.imageIDs(gridH: 4, gridW: 4),
            cond: Flux3Positions.imageIDs(gridH: 2, gridW: 4, layer: -1) + Flux3Positions.imageIDs(gridH: 4, gridW: 2, layer: -2),
            text: Flux3Prompt.textIDs(80), config: transformer.config, dtype: .bfloat16)
    }
    func mod(_ stage: String, _ mode: String) -> Flux3Mod { (self["\(stage)_\(mode)_shift"], self["\(stage)_\(mode)_scale"], self["\(stage)_\(mode)_gate"]) }
    /// NCHW float32 in [-1, 1] (u8 / 127.5 − 1) back to its 8-bit image.
    func rgb(_ name: String) -> RGBImage {
        let x = self[name][0].transposed(1, 2, 0)
        let u8 = MLX.round((x + 1) * 127.5).asType(.uint8)
        return RGBImage(width: x.dim(1), height: x.dim(0), pixels: u8.asArray(UInt8.self))
    }
}

@Suite(.serialized) struct Flux3Parity {
    @Test(.heavy) func loadsTheVendorLayoutStrictly() throws {
        let tiny = try Tiny()
        #expect(tiny.transformer.config.numLayers == 2 && tiny.transformer.config.numSingleLayers == 2 && tiny.transformer.config.heads == 2)
        #expect(tiny.transformer.config.headDim == 32 && tiny.transformer.config.jointDim == 512)
        #expect(tiny.text.layers.count == 32 && tiny.text.config.layers == 34)
        let dtypes = Set(
            (tiny.transformer.parameters().flattened() + tiny.text.parameters().flattened() + tiny.encoder.parameters().flattened()
                + tiny.decoder.parameters().flattened()).map { $0.1.dtype })
        #expect(dtypes == [.bfloat16]) // the float32 VAE file is cast; nothing stays float32
    }

    @Test(.heavy) func noiseMatchesMfluxSeeds() throws {
        let tiny = try Tiny()
        expectClose("noise seed 7 (64x64)", Flux3Pipeline.noise(seed: 7, gridH: 4, gridW: 4), tiny["noise_seed7_64"], bound: 0)
        expectClose("noise seed 42 (48x32)", Flux3Pipeline.noise(seed: 42, gridH: 2, gridW: 3), tiny["noise_seed42_32x48"], bound: 0)
    }

    @Test(.heavy) func timeEmbeddingAndModulation() throws {
        let tiny = try Tiny(), t = tiny.transformer
        expectClose("timestep sinusoid", Flux3TimestepEmbedder.sinusoid(tiny["t"], channels: 256), tiny["t_sinusoid"], bound: 0)
        expectClose("timestep sinusoid t=0", Flux3TimestepEmbedder.sinusoid(MLXArray.zeros([1]), channels: 256), tiny["t_sinusoid_zero"], bound: 0)
        let vec = t.timeVectors(t: tiny["t"], next: tiny["t_next"])
        expectClose("time vector image", vec.image, tiny["vec_image"], bound: 4e-3)
        expectClose("time vector context", vec.context, tiny["vec_context"], bound: 4e-3)
        for (stage, set) in [("early", t.early), ("single", t.single)] {
            for (mode, module, v) in [("image", set.image, tiny["vec_image"]), ("image_cond", set.imageCond, tiny["vec_context"]), ("txt", set.txt, tiny["vec_context"])] {
                let m = module(v)
                expectClose(
                    "\(stage) modulation \(mode)", concatenated([m.shift, m.scale, m.gate], axis: -1),
                    concatenated([tiny["\(stage)_\(mode)_shift"], tiny["\(stage)_\(mode)_scale"], tiny["\(stage)_\(mode)_gate"]], axis: -1), bound: 2e-3)
            }
        }
    }

    @Test(.heavy) func textEncoderLayerAndTaps() throws {
        let tiny = try Tiny()
        let (cos, sin) = tiny.text.rope(positions: Flux3Prompt.positions(length: 80, valid: 37), dtype: .bfloat16)
        expectClose("text RoPE cos", cos[0], tiny["te_rope_cos"], bound: 0)
        expectClose("text RoPE sin", sin[0], tiny["te_rope_sin"], bound: 0)
        let mask = Flux3TextEncoder.mask(length: 80, valid: 37, dtype: .bfloat16)
        expectClose("text encoder layer 0", tiny.text.layers[0](tiny["te_layer_in"], mask: mask, cos: cos, sin: sin), tiny["te_layer_out"], bound: 0.04)
        // 32 random layers amplify rounding: mflux's own float32 attention or float32 RMSNorm, swapped alone for the
        // BF16 choice this port (and HF transformers) makes, moves its output by 0.156 and 0.219 (numerics_probe.py).
        expectClose("text encoder taps 4..32", tiny.text(tiny["te_ids"], valid: 37), tiny["te_embeds"], bound: 0.25)
    }

    @Test(.heavy) func transformerBlocks() throws {
        let tiny = try Tiny(), t = tiny.transformer, rope = tiny.rope
        expectClose("embed image", t.xEmbedders.image(tiny["noise_seed7_64"].asType(.bfloat16)), tiny["embed_image"], bound: 0.02)
        expectClose("embed references", t.xEmbedders.imageCond(tiny["cond_tokens"]), tiny["embed_image_cond"], bound: 0.02)
        expectClose("embed text", t.contextEmbedder(tiny["te_embeds"]), tiny["embed_txt"], bound: 0.02)
        let tables: [String: (MLXArray, MLXArray)] = ["image": rope.image, "image_cond": rope.cond!, "txt": rope.text]
        for d in 0..<2 {
            for m in ["image", "image_cond", "txt"] {
                let block = m == "txt" ? t.txtBlocks[d] : m == "image" ? t.modeBlocks.image[d] : t.modeBlocks.imageCond[d]
                let input = d == 0 ? tiny["embed_\(m)"] : tiny["mode0_\(m)"]
                expectClose("mode block \(d) \(m)", block(input, cos: tables[m]!.0, sin: tables[m]!.1, mod: tiny.mod("early", m)), tiny["mode\(d)_\(m)"], bound: 0.02)
            }
        }
        let lengths = [("image", 16), ("image_cond", 16), ("txt", 80)]
        func perToken(_ part: String) -> MLXArray {
            concatenated(lengths.map { broadcast(tiny["single_\($0.0)_\(part)"], to: [1, $0.1, 64]) }, axis: 1)
        }
        let mod: Flux3Mod = (perToken("shift"), perToken("scale"), perToken("gate"))
        let h0 = concatenated(["image", "image_cond", "txt"].map { tiny["mode1_\($0)"] }, axis: 1)
        expectClose("single block 0", t.singleBlocks[0](h0, cos: rope.all.cos, sin: rope.all.sin, mod: mod), tiny["single0"], bound: 0.02)
        expectClose("single block 1", t.singleBlocks[1](tiny["single0"], cos: rope.all.cos, sin: rope.all.sin, mod: mod), tiny["single1"], bound: 0.02)
    }

    @Test(.heavy) func velocityAndOneStep() throws {
        let tiny = try Tiny(), t = tiny.transformer
        let edit = t(tiny["noise_seed7_64"], text: tiny["te_embeds"], cond: tiny["cond_tokens"], t: tiny["t"], next: tiny["t_next"], rope: tiny.rope)
        expectClose("velocity, edit (2 references)", edit, tiny["velocity_edit"], bound: 0.03)
        let rope = Flux3RopeTables(image: Flux3Positions.imageIDs(gridH: 4, gridW: 4), cond: [], text: Flux3Prompt.textIDs(80), config: t.config, dtype: .bfloat16)
        let t2i = t(tiny["noise_seed7_64"], text: tiny["te_embeds"], cond: nil, t: tiny["t"], next: tiny["t_next"], rope: rope)
        expectClose("velocity, text to image", t2i, tiny["velocity_t2i"], bound: 0.03)
        // The Euler jump alone, from mflux's own velocity: exact.
        let times = Flux3Schedule.timesteps(steps: 4)
        let v = tiny["velocity_t2i"]
        expectClose("denoise step (Euler jump)", tiny["noise_seed7_64"] + MLXArray(times[2] - times[1]).asType(v.dtype) * v, tiny["latents_step"], bound: 0)
    }

    @Test(.heavy) func vaeEncodeDecodeAndPatchify() throws {
        let tiny = try Tiny()
        let packed = Flux3VAEEncoderHalf.patchify(tiny["patchify_in"].transposed(0, 2, 3, 1))
        expectClose("patchify + pack", packed, tiny["patchify_out"], bound: 0)
        let image = tiny["vae_image"].transposed(0, 2, 3, 1)
        expectClose("VAE encode → tokens", tiny.encoder.tokens(image, statistics: tiny.decoder.statistics), tiny["vae_tokens"], bound: 0.05)
        let decoded = tiny.decoder.decode(tiny.decoder.latents(state: tiny["vae_decode_in"], gridH: 4, gridW: 4, dtype: .bfloat16), dtype: .bfloat16)
        // mflux's two decoders (row strips with float32 GroupNorm, the default; full image) differ by 0.111 here.
        expectClose("VAE tokens → decode", decoded, tiny["vae_decoded"].transposed(0, 2, 3, 1), bound: 0.12)
    }

    /// Query-chunked VAE attention equals the one-call attention (shared with FLUX.2's decoder).
    @Test(.heavy) func chunkedVAEAttentionMatchesFull() throws {
        let tiny = try Tiny()
        let x = MLXRandom.normal([1, 6, 7, 32], key: MLXRandom.key(5)).asType(.bfloat16)
        let attention = try #require(tiny.decoder.decoder.mid.attentions.first)
        expectClose("VAE attention, 42 pixels in chunks of 5 vs one call", attention(x, queryChunk: 5), attention(x), bound: 0)
    }

    @Test(.heavy) func tinyEditPipeline() throws {
        let tiny = try Tiny()
        let pipeline = Flux3Pipeline(transformer: tiny.transformer, textEncoder: tiny.text, tokenizer: nil, encoder: tiny.encoder, decoder: tiny.decoder)
        let refs = [tiny.rgb("ref_a"), tiny.rgb("ref_b")]
        #expect(refs.map { [$0.width, $0.height] } == [[64, 32], [32, 64]])
        let cond = pipeline.referenceTokens(refs)
        #expect(cond.ids == Flux3Positions.imageIDs(gridH: 2, gridW: 4, layer: -1) + Flux3Positions.imageIDs(gridH: 4, gridW: 2, layer: -2))
        expectClose("pipeline reference tokens", cond.tokens, tiny["pipe_cond_tokens"], bound: 0.05)
        // Denoise from mflux's own reference tokens, so the latent chain is checked on its own.
        let latents = try pipeline.denoise(
            noise: Flux3Pipeline.noise(seed: 7, gridH: 4, gridW: 4), text: tiny["te_embeds"], cond: (tiny["pipe_cond_tokens"], cond.ids), gridH: 4, gridW: 4, steps: 4)
        #expect(latents.dtype == .float32)
        expectClose("pipeline latents after 4 steps", latents, tiny["pipe_latents"], bound: 0.03)
        let image = pipeline.decode(tiny["pipe_latents"], gridH: 4, gridW: 4)
        // mflux's own strip and full-image decoders differ by 0.078 (about 10 levels) on this image.
        expectClose("pipeline decode", image, tiny["pipe_image"].transposed(0, 2, 3, 1), bound: 0.12)
        let pixels = MLXArray(Flux3Pipeline.pixels(image), [64, 64, 3]).asType(.int32)
        expectClose("pipeline pixels (8-bit levels)", pixels, tiny["pipe_pixels"].asType(.int32), bound: 12)
    }
}
