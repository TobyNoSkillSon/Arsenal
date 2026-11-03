import Foundation

// The engine boundary: what every model in Vireo implements, one protocol per kind. No MLX types
// cross it, so the helper's queue, job pipeline and tests run on stub models without a GPU.
// Implementations live in VireoEngine (MLX) and register a loader in VireoEngine/Engines.swift.
//
// Contract for implementers:
// - The helper serialises GPU work: at most one call into any model at a time. No locking needed.
// - Check `control.isCancelled` between steps (tokens, diffusion steps, chunks) and return early.
// - Throw `EngineError` (or any error) on failure; never return non-finite samples.
// - `unload()` must drop every reference to weights, including compiled closures and thread-local
//   compile caches that capture them (Vella's RuntimeCaches pattern). The helper calls
//   `Memory.clearCache()` afterwards and a baseline test checks active memory returns to where it was.

public enum AudioKind: String, Codable, CaseIterable, Sendable {
    case speech, sfx, music, enhance = "enhancement", isolation
    /// Accepts the catalog/status spelling and the short route names ("enhance", "isolate").
    public static func parse(_ raw: String) -> AudioKind? {
        switch raw { case "enhance": .enhance; case "isolate": .isolation; default: AudioKind(rawValue: raw) }
    }
    /// The catalog heading and CLI noun.
    public var title: String {
        switch self {
        case .speech: "Speech"; case .sfx: "Sound effects"; case .music: "Music"; case .enhance: "Enhancement"; case .isolation: "Isolation"
        }
    }
}

/// PCM audio in Float32, interleaved frames when `channels > 1`.
public struct AudioClip: Sendable, Equatable {
    public var samples: [Float]
    public var sampleRate: Int
    public var channels: Int
    public init(samples: [Float], sampleRate: Int, channels: Int = 1) {
        self.samples = samples; self.sampleRate = sampleRate; self.channels = max(1, channels)
    }
    public var frames: Int { samples.count / channels }
    public var seconds: Double { sampleRate > 0 ? Double(frames) / Double(sampleRate) : 0 }
}

/// Cancellation and progress for one request. Thread-safe.
public final class GenerationControl: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var fraction = 0.0
    private let onProgress: (@Sendable (Double) -> Void)?
    private let onStep: (@Sendable (Double, Int, Int) -> Void)?
    public init(onProgress: (@Sendable (Double) -> Void)? = nil, onStep: (@Sendable (Double, Int, Int) -> Void)? = nil) { self.onProgress = onProgress; self.onStep = onStep }
    public func report(step: Int, total: Int) { if step >= 0, total > 0, step <= total { report(progress: Double(step) / Double(total), step: step, total: total) } }
    public var isCancelled: Bool { lock.withLock { cancelled } }
    public func cancel() { lock.withLock { cancelled = true } }
    public var progress: Double { lock.withLock { fraction } }
    /// 0…1, monotonic; the helper shows it in `GET /jobs/{id}` and status.
    public func report(progress: Double, step: Int? = nil, total: Int? = nil) {
        let value = min(1, max(0, progress))
        lock.withLock { fraction = max(fraction, value) }
        if let step, let total, step >= 0, total > 0, step <= total, let onStep { onStep(value, step, total) } else { onProgress?(value) }
    }
}

public struct EngineError: LocalizedError, Equatable, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// Everything a loader gets from the catalog. Weights are already downloaded and pinned.
public struct ModelSource: Sendable {
    public var id: String
    /// Loader key from the catalog `engine` field (e.g. "stable-audio-3", "re-use").
    public var engine: String
    /// Pinned snapshot directory in the Hugging Face cache.
    public var directory: URL
    /// The tier's dtype label: "bf16"/"fp16" for tier 16, "8bit"/"4bit" for the derived tiers.
    public var precision: String
    /// Catalog asset id → pinned snapshot directory (codecs, text encoders, lexicons).
    public var assets: [String: URL]
    /// Free-form `engine_options` from the catalog.
    public var options: [String: String]
    /// Model-path switches this load sets over the process environment (the selected cell:
    /// `FastPathComponents.overrides`). Engines build their configuration from both, per load.
    public var overrides: [String: String]
    public init(id: String, engine: String, directory: URL, precision: String, assets: [String: URL] = [:], options: [String: String] = [:],
                overrides: [String: String] = [:]) {
        self.id = id; self.engine = engine; self.directory = directory; self.precision = precision; self.assets = assets; self.options = options
        self.overrides = overrides
    }
}

public protocol AudioModel: AnyObject {
    var id: String { get }
    /// Output sample rate (speech/sound) or native processing rate (enhance).
    var sampleRate: Int { get }
    /// Drop weights and every cache that captures them. The object is not used afterwards.
    func unload() async
    /// /status `engine`: "optimized" while a gated fast path (custom kernels) serves, else "mlx".
    /// Declared here, not only in the extension, so `any AudioModel` dispatches to the model's own value.
    var engineLabel: String { get }
    /// /status `engine_reason`: which path runs and why (self-test verdict, fallback).
    var engineReason: String { get }
}

public extension AudioModel {
    var engineLabel: String { "mlx" }
    var engineReason: String { "stock MLX; no optimized kernels yet" }
}

public struct SpeechRequest: Sendable, Equatable {
    public var text: String
    /// Model preset voice name (Kokoro/Pocket voices).
    public var preset: String?
    /// Described voice ("tired British narrator") for voice-design models.
    public var description: String?
    /// Delivery instruction; models without instruction support ignore it (the helper warns).
    public var style: String?
    /// Reference clip + transcript for cloning a saved voice.
    public var reference: AudioClip?
    public var referenceText: String?
    public var language: String?
    public var seed: UInt64?
    public var maxTokens: Int?
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?
    public var repetitionPenalty: Float?
    public var guidance: Float?
    public init(text: String, preset: String? = nil, description: String? = nil, style: String? = nil, reference: AudioClip? = nil, referenceText: String? = nil, language: String? = nil, seed: UInt64? = nil, maxTokens: Int? = nil, temperature: Float? = nil, topP: Float? = nil, topK: Int? = nil, repetitionPenalty: Float? = nil, guidance: Float? = nil) {
        self.text = text; self.preset = preset; self.description = description; self.style = style
        self.reference = reference; self.referenceText = referenceText; self.language = language; self.seed = seed
        self.maxTokens = maxTokens; self.temperature = temperature; self.topP = topP; self.topK = topK
        self.repetitionPenalty = repetitionPenalty; self.guidance = guidance
    }
}

public protocol SpeechModel: AudioModel {
    /// Streams mono chunks at `sampleRate`, in order. Text arrives already cleaned and chunked
    /// (one sentence or paragraph per call).
    func speak(_ request: SpeechRequest, control: GenerationControl, emit: @escaping ([Float]) async throws -> Void) async throws
}

public typealias ModelLoader = @Sendable (ModelSource) async throws -> any AudioModel
