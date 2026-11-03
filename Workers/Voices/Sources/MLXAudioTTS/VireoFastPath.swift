import Foundation

/// Vireo: thrown by an optimized generation path whose runtime check failed (non-finite logits or samples). The
/// engine adapter records the failure (stock becomes sticky for the gate key) and serves stock.
public struct FastPathFailure: Error, LocalizedError {
    public var detail: String
    public init(_ detail: String) { self.detail = detail }
    public var errorDescription: String? { detail }
}

/// Vireo: lab switches of the vendored models. The engine adapter installs them from its model-path configuration
/// (VireoEngine `FastPath`, which also forms the self-test verdict identity) before a model loads; vendored code never
/// reads the environment for them. Defaults are the shipping path.
public enum VireoModelSwitches {
    /// Streaming codec decode adds the transposed-conv bias once at chunk boundaries (false: upstream, lab A/B).
    nonisolated(unsafe) public static var streamBiasFix = true
    /// Qwen3-TTS keeps the speaker embedding in fp32 (lab A/B of the old path).
    nonisolated(unsafe) public static var qwenF32Speaker = false
    /// Codec frames per streamed Breeze decode (lab); nil keeps the streaming interval's frame count.
    nonisolated(unsafe) public static var breezeStreamFrames: Int?
}
