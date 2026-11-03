import Foundation

// Adapted from soniqo/speech-swift `Configuration.swift` (Apache-2.0, commit 2af9c6c).
// The DiT dimensions moved to `DiTMediumConfig`; the aufklarer variant enum became
// `SA3Checkpoint`, which also covers Stability's official F16 MLX export; `SA3Variant` names the
// official DiT/codec pairs (medium, small SFX).

/// The Stable Audio 3 checkpoints Vireo runs from Stability's official MLX export (`sa3_mlx.py` `DIT_CHOICES` /
/// `DECODER_CHOICES`). Each DiT family has a fixed codec: medium → SAME-L, the small DiTs → SAME-S. Encoders
/// (`same_*_encoder_f32.npz`) are only needed for audio-to-audio and inpainting, which Vireo does not offer.
public enum SA3Variant: String, Sendable, CaseIterable {
    case medium
    /// sa3-sm-sfx: the small DiT post-trained for sound effects.
    case smallSFX = "sm-sfx"

    /// Catalog `engine_options.variant`; absent means medium.
    public init(option: String?) throws {
        guard let v = SA3Variant(rawValue: option ?? SA3Variant.medium.rawValue) else {
            throw StableAudio3Error.precision("unknown Stable Audio 3 variant \(option ?? ""); known: " + SA3Variant.allCases.map(\.rawValue).joined(separator: ", "))
        }
        self = v
    }

    public var ditFile: String { self == .medium ? "dit_medium_f16.npz" : "dit_\(rawValue)_f16.npz" }
    public var decoderFile: String { self == .medium ? "same_l_decoder_f32.npz" : "same_s_decoder_f32.npz" }
    public var dit: DiTMediumConfig { self == .medium ? .medium : .small }
    /// Report/component name of the DiT (`dit_medium`, `dit_sm-sfx`).
    public var ditComponent: String { "dit_" + rawValue }
    /// Longest render: medium keeps the conditioner's 380 s (model card 6 min 20 s); the small DiTs ~2 min (Stability's
    /// model overview, `docs/guides/model-overview.md`).
    public var maxSeconds: Double { self == .medium ? SA3Audio.maxSeconds : 120 }
    /// Files Vireo needs from `SA3Checkpoint.officialRepo`.
    public var officialFiles: [String] {
        ["MLX/" + ditFile, "MLX/" + decoderFile, "MLX/t5gemma_f16.npz", "LICENSE.md", "LICENSE_GEMMA.md", "NOTICE"]
    }
}

/// Where the three components of a Stable Audio 3 checkpoint live.
public struct SA3Checkpoint: Sendable, Equatable {
    public var dit: URL
    public var decoder: URL
    public var t5gemma: URL

    public init(dit: URL, decoder: URL, t5gemma: URL) {
        self.dit = dit; self.decoder = decoder; self.t5gemma = t5gemma
    }

    /// Repo holding the official 16-bit MLX export (ungated; Stability AI Community
    /// License + Gemma Terms of Use for T5Gemma).
    public static let officialRepo = "stabilityai/stable-audio-3-optimized"
    /// Files Vireo needs from `officialRepo` for the medium model (≈5.2 GB). The rest of
    /// the repo (ONNX, TensorRT, TFLite, CPU-AMX, encoders) is not needed.
    public static let officialMediumFiles = SA3Variant.medium.officialFiles

    /// Official layout: `<snapshot>/MLX/{dit_medium_f16,same_l_decoder_f32,t5gemma_f16}.npz` (medium),
    /// `<snapshot>/MLX/{dit_sm-sfx_f16,same_s_decoder_f32,t5gemma_f16}.npz` (small SFX).
    public static func official(snapshot: URL, variant: SA3Variant = .medium) -> SA3Checkpoint {
        let mlx = snapshot.appendingPathComponent("MLX", isDirectory: true)
        return SA3Checkpoint(dit: mlx.appendingPathComponent(variant.ditFile),
                             decoder: mlx.appendingPathComponent(variant.decoderFile),
                             t5gemma: mlx.appendingPathComponent("t5gemma_f16.npz"))
    }

    /// aufklarer bundle layout (`aufklarer/Stable-Audio-3-DiT-Medium-MLX-8bit`):
    /// `<bundle>/{dit_medium,same_l_decoder,t5gemma}/model.safetensors`.
    public static func aufklarerBundle(_ root: URL) -> SA3Checkpoint {
        SA3Checkpoint(dit: root.appendingPathComponent("dit_medium", isDirectory: true),
                      decoder: root.appendingPathComponent("same_l_decoder", isDirectory: true),
                      t5gemma: root.appendingPathComponent("t5gemma", isDirectory: true))
    }

    /// Standard Hugging Face hub cache snapshot for `officialRepo` at `refs/main`.
    public static func officialInHubCache(env: [String: String] = ProcessInfo.processInfo.environment) -> SA3Checkpoint? {
        let hub = env["HF_HUB_CACHE"] ?? (env["HF_HOME"].map { $0 + "/hub" } ?? NSHomeDirectory() + "/.cache/huggingface/hub")
        let root = URL(fileURLWithPath: hub).appendingPathComponent("models--" + officialRepo.replacingOccurrences(of: "/", with: "--"))
        guard let sha = try? String(contentsOf: root.appendingPathComponent("refs/main"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !sha.isEmpty else { return nil }
        return official(snapshot: root.appendingPathComponent("snapshots/\(sha)", isDirectory: true))
    }
}

/// SAME-L decoder constants (mirror same_l_decoder.py).
public enum SAMELDims {
    public static let latentDim = 256
    public static let dim = 1536
    public static let numHeads = 24
    public static let headDim = 64
    public static let ropeDims = 32
    public static let numBlocks = 12
    public static let ffInner = 4608
    public static let sinStartBlock = 5
    public static let outChannels = 512
    public static let stride = 16
    public static let subChunkSize = stride + 1     // 17
    public static let sinPerPos = subChunkSize - 1  // 16
    public static let ropeBase: Float = 10000.0
}

/// T5Gemma encoder dims (t5gemma-b-b-ul2 encoder: 12 layers / 768 / 12 heads).
public enum T5GemmaDims {
    public static let hiddenSize = 768
    public static let numLayers = 12
    public static let numAttentionHeads = 12
    public static let numKeyValueHeads = 12
    public static let headDim = 64
    public static let intermediateSize = 2048
    public static let vocabSize = 256000
    public static let maxPositionEmbeddings = 8192
    public static let ropeTheta: Float = 10000.0
    public static let rmsNormEps: Float = 1e-6
    public static let attnLogitSoftcapping: Float = 50.0
    public static let queryPreAttnScalar: Int = 64
    public static let padTokenId: Int32 = 0
}

/// Audio pipeline constants.
public enum SA3Audio {
    public static let sampleRate = 44100
    public static let samplesPerLatent = 4096
    public static let channels = 2          // stereo
    public static let patchSize = 256       // PatchedPretransform patch
    /// Seconds conditioner range (NumberConditioner max_val) — the model card's 6 m 20 s.
    public static let maxSeconds: Double = 380
}
