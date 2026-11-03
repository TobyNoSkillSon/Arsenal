import Foundation

/// One Models-table cell: a tier (16/8/4), a flavour (Standard: stock MLX; Optimized: our gated components) and, for
/// Optimized, a mode (Exact: only components bit-identical to stock; Fast: every gate-passing component).
/// The helper persists one per model (`<support>/selection.json`, `POST /select`); a load builds its engine
/// configuration from it (`FastPathComponents.overrides`).
public struct ModelCell: Codable, Equatable, Hashable, Sendable {
    public var tier: String
    public var flavour: String
    public var mode: String

    public static let tiers = ["16", "8", "4"]
    public static let flavours = ["standard", "optimized"]
    public static let modes = ["exact", "fast"]
    /// Standard 16: what a model without an Optimized 16 · Fast cell loads when nothing was selected or loaded.
    public static let fallback = ModelCell(tier: "16", flavour: "standard", mode: "exact")
    /// Optimized 16 · Fast: the default of every model that has it (TABLE-INTERFACE.md Change 2).
    public static let optimizedFast16 = ModelCell(tier: "16", flavour: "optimized", mode: "fast")

    public init(tier: String, flavour: String, mode: String) { self.tier = tier; self.flavour = flavour; self.mode = mode }

    public var isStandard: Bool { flavour == "standard" }
    /// "16 · Standard", "8 · Optimized · Fast".
    public var label: String { isStandard ? "\(tier) · Standard" : "\(tier) · Optimized · \(mode == "fast" ? "Fast" : "Exact")" }
    /// benchmarks.json v2 cell key.
    public var benchmarkKey: String { isStandard ? "standard" : "optimized_\(mode)" }
    public var object: [String: Any] { ["tier": tier, "flavour": flavour, "mode": mode] }

    /// Parses `{tier, flavour, mode}`; nil values keep `base`'s. Throws a one-line reason.
    public static func parse(tier: String?, flavour: String?, mode: String?, base: ModelCell = .fallback) throws -> ModelCell {
        var cell = base
        if let tier { cell.tier = tier }
        if let flavour { cell.flavour = flavour }
        if let mode { cell.mode = mode }
        guard tiers.contains(cell.tier) else { throw CellError("tier must be 16, 8 or 4") }
        guard flavours.contains(cell.flavour) else { throw CellError("flavour must be standard or optimized") }
        guard modes.contains(cell.mode) else { throw CellError("mode must be exact or fast") }
        return cell
    }

    /// A legacy precision label (catalog before tiers, config.json `precision`, `--precision`) as a tier.
    /// FP32, BF16, F16 and "native" are 16; nil for anything unknown.
    public static func tier(legacyPrecision label: String) -> String? {
        switch label.lowercased() {
        case "16", "fp32", "f32", "bf16", "fp16", "f16", "native": return "16"
        case "8", "8bit", "8-bit": return "8"
        case "4", "4bit", "4-bit": return "4"
        default: return nil
        }
    }

    public struct CellError: LocalizedError {
        public let message: String
        public init(_ message: String) { self.message = message }
        public var errorDescription: String? { message }
    }
}

/// Every gated fast-path component, its switch and whether its output is bit-identical to stock
/// (evidence: lab/notes/COMPONENTS.md). `metadata/models.json` `optimized {exact, fast}` lists these ids per model;
/// a contract test keeps the two in step.
public enum FastPathComponents {
    public struct Component: Equatable, Sendable {
        public let id: String
        /// The model-path switch that turns it off with "0".
        public let switchName: String
        public let exact: Bool
    }

    public static let all: [Component] = [
        .init(id: "qwen-fused-rope", switchName: "VIREO_QWEN_FAST_ROPE", exact: true),
        .init(id: "qwen-predictor-tables", switchName: "VIREO_QWEN_FAST_TABLES", exact: true),
        .init(id: "qwen-fused-sampler", switchName: "VIREO_QWEN_FAST_SAMPLER", exact: true),
        .init(id: "breeze-lazy-frames", switchName: "VIREO_BREEZE_LAZY", exact: true),
        .init(id: "breeze-streaming-decode", switchName: "VIREO_BREEZE_STREAM", exact: false),
        .init(id: "breeze-depth-kv-cache", switchName: "VIREO_BREEZE_DEPTH_CACHE", exact: false),
        .init(id: "breeze-fused-depth-step", switchName: "VIREO_BREEZE_FUSED", exact: false),
        .init(id: "kokoro-host-alignment", switchName: "VIREO_KOKORO_ALIGN", exact: true),
        .init(id: "kokoro-conv-weight-cache", switchName: "VIREO_KOKORO_CONV_CACHE", exact: true),
        .init(id: "kokoro-fused-lstm", switchName: "VIREO_KOKORO_LSTM", exact: true),
        .init(id: "kokoro-fp16-decoder", switchName: "VIREO_KOKORO_HALF", exact: false),
    ]
    /// Lab candidates that are off unless set to "1"; never part of a shipped recipe (forced "0" in Exact).
    public static let labOptIns = ["VIREO_QWEN_FAST_PIPELINE", "VIREO_BREEZE_CFG_BATCH", "VIREO_BREEZE_SMALLM"]
    /// Set in Exact: every exact component's self-test demands bit-identity (some allow a tolerance otherwise).
    public static let exactSwitch = "VIREO_EXACT"

    public static func component(_ id: String) -> Component? { all.first { $0.id == id } }

    /// Model-path switches a load of `cell` sets over the process environment: Standard forces stock, Exact turns
    /// every inexact component and lab candidate off and demands identity, Fast adds nothing.
    public static func overrides(_ cell: ModelCell) -> [String: String] {
        if cell.isStandard { return ["VIREO_FORCE_STOCK": "1"] }
        guard cell.mode == "exact" else { return [:] }
        var o = [exactSwitch: "1"]
        for c in all where !c.exact { o[c.switchName] = "0" }
        for name in labOptIns { o[name] = "0" }
        return o
    }
}
