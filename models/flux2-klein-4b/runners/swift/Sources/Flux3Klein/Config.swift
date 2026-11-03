// SPDX-License-Identifier: MIT
// FLUX 3 Image Klein (distilled, flow map): a 5.6B single/multi-mode DiT over Qwen3-VL-4B-Instruct prompt features and
// the FLUX.2 VAE. Ported from mflux (src/mflux/models/flux3, commit 9669af51) as the reference; see PORTED-FROM.md.
import Foundation

public struct Flux3Error: LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// The DiT's dimensions. Defaults are the release's (mflux `Flux3Transformer` defaults); `inferred(from:)` reads them
/// from the checkpoint's tensor shapes, so no transformer config file is needed.
public struct Flux3TransformerConfig: Sendable, Equatable {
    public var inChannels = 128, condInChannels = 128
    public var numLayers = 5, numSingleLayers = 28
    public var heads = 24, headDim = 128
    public var jointDim = 20480
    public var timestepChannels = 256
    public var mlpHidden = 9216
    public var ropeTheta: Double = 10000
    public var inner: Int { heads * headDim }
    /// Four RoPE axes (t, row, column, layer-or-text-position), each a quarter of the head.
    public var axes: [Int] { Array(repeating: headDim / 4, count: 4) }
    public init() {}

    /// Dimensions from checkpoint shapes (names after the `.attn.to_out.0.` → `.attn.to_out.` rename).
    public static func inferred(from shapes: [String: [Int]]) throws -> Flux3TransformerConfig {
        func shape(_ key: String) throws -> [Int] {
            guard let s = shapes[key] else { throw Flux3Error("The transformer checkpoint has no \(key).") }
            return s
        }
        var c = Flux3TransformerConfig()
        guard shapes["target_timestep_embedder.linear_1.weight"] != nil else {
            throw Flux3Error("This checkpoint has no target time embedding: it is FLUX 3 Klein Base (50 steps with guidance), which this runner does not serve.")
        }
        let x = try shape("x_embedders.image.weight"), cond = try shape("x_embedders.image_cond.weight")
        let context = try shape("context_embedder.weight"), norm = try shape("single_transformer_blocks.0.attn.norm_q.weight")
        let time = try shape("timestep_embedder.linear_1.weight"), ff = try shape("single_transformer_blocks.0.ff.linear_out.weight")
        guard x.count == 2, cond.count == 2, context.count == 2, norm.count == 1, time.count == 2, ff.count == 2 else {
            throw Flux3Error("The transformer checkpoint has unexpected tensor ranks.")
        }
        c.headDim = norm[0]
        guard c.headDim % 8 == 0, x[0] % c.headDim == 0 else { throw Flux3Error("The transformer head size \(c.headDim) does not divide its width \(x[0]).") }
        c.heads = x[0] / c.headDim
        c.inChannels = x[1]; c.condInChannels = cond[1]; c.jointDim = context[1]; c.timestepChannels = time[1]; c.mlpHidden = ff[1]
        func count(_ prefix: String) -> Int {
            Set(
                shapes.keys.compactMap { key -> Int? in
                    guard key.hasPrefix(prefix) else { return nil }
                    return Int(key.dropFirst(prefix.count).prefix { $0 != "." })
                }
            ).count
        }
        c.numLayers = count("txt_mode_blocks."); c.numSingleLayers = count("single_transformer_blocks.")
        guard c.numLayers > 0, c.numSingleLayers > 0, count("mode_blocks.image.") == c.numLayers, count("mode_blocks.image_cond.") == c.numLayers else {
            throw Flux3Error("The transformer checkpoint's block counts disagree.")
        }
        return c
    }
}

/// The Qwen3-VL language tower's text configuration (`text_encoder/config.json` → `text_config`).
public struct Flux3TextConfig: Sendable, Equatable {
    public var vocabulary = 151936, hidden = 2560, layers = 36, heads = 32, kvHeads = 8, headDim = 128, intermediate = 9728
    public var eps: Float = 1e-6
    public var ropeTheta: Double = 5_000_000
    public init() {}

    public init(json data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw Flux3Error("text_encoder/config.json is not an object.") }
        let text = root["text_config"] as? [String: Any] ?? root
        func int(_ key: String) throws -> Int {
            guard let v = text[key] as? NSNumber else { throw Flux3Error("text_encoder/config.json has no \(key).") }
            return v.intValue
        }
        vocabulary = try int("vocab_size"); hidden = try int("hidden_size"); layers = try int("num_hidden_layers")
        heads = try int("num_attention_heads"); kvHeads = try int("num_key_value_heads"); intermediate = try int("intermediate_size")
        headDim = (text["head_dim"] as? NSNumber)?.intValue ?? hidden / heads
        eps = (text["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        ropeTheta = (text["rope_theta"] as? NSNumber)?.doubleValue ?? 5_000_000
        if let bias = text["attention_bias"] as? Bool, bias { throw Flux3Error("A text encoder with attention biases is not Qwen3-VL-4B.") }
        guard layers >= Flux3Prompt.taps.max()! else { throw Flux3Error("The text encoder has \(layers) layers; FLUX 3 reads layer \(Flux3Prompt.taps.max()!).") }
    }
}

/// The FLUX.2 VAE (identical weights in FLUX 3): `block_out_channels`, read from the encoder's tensor shapes.
public struct Flux3VAEConfig: Sendable, Equatable {
    public var blockOut = [128, 256, 512, 512]
    public init() {}
    public static func inferred(from shapes: [String: [Int]]) throws -> Flux3VAEConfig {
        var c = Flux3VAEConfig()
        c.blockOut = try (0..<4).map { i in
            guard let s = shapes["encoder.down_blocks.\(i).resnets.0.conv1.weight"], s.count == 4 else {
                throw Flux3Error("The VAE checkpoint has no encoder.down_blocks.\(i).resnets.0.conv1.weight.")
            }
            return s[0]
        }
        guard c.blockOut.allSatisfy({ $0 % 32 == 0 }) else { throw Flux3Error("VAE channels must be multiples of 32 (GroupNorm groups).") }
        return c
    }
}
