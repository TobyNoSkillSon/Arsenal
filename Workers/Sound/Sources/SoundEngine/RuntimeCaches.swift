// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// MLX process state: buffer-cache limit, memory readings, and retirement after an unload.
///
/// Retirement follows Vella (`Worker/Sources/MLXAudioSTT/RuntimeCaches.swift`, `4690c12`): MLX's
/// compile cache is `static thread_local`, so a compiled closure that captured weights keeps them
/// alive after the model is released. Rules for every engine in Vireo:
/// 1. pass weights to `compile` as explicit inputs (`compile(inputs: [module])`), never as captured constants;
/// 2. keep compiled closures and any model-dependent global cache inside the model object, or register a
///    retirement hook here so `retire()` can drop it;
/// 3. load quantized layers directly from checkpoint arrays (quantizing random placeholders leaks per load).
/// Proof: `lab/tests/load-unload-baseline.sh` (≥3 load/unload cycles back to the same active memory).
public enum RuntimeCaches {
    nonisolated(unsafe) private static var hooks: [String: () -> Void] = [:]
    private static let lock = NSLock()

    /// Register a model-independent cache drop (e.g. a filterbank table) under a stable key.
    public static func register(_ key: String, _ drop: @escaping () -> Void) { lock.withLock { hooks[key] = drop } }

    /// After the last reference to an unloaded model is gone: finish GPU work, drop registered caches and
    /// the MLX buffer cache.
    public static func retire() {
        Stream.gpu.synchronize()
        let drops = lock.withLock { Array(hooks.values) }
        drops.forEach { $0() }
        Memory.clearCache()
    }

    /// A signed .app keeps its Metal library in Contents/Resources/mlx.metallib; set the path before MLX starts.
    /// `VIREO_METALLIB` overrides (development runs of a bare helper binary); otherwise MLX's own lookup applies.
    public static func useBundledMetallib() {
        if let override = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { GPU.metallib = URL(fileURLWithPath: override); return }
        if let override = FastPath.current.value("VIREO_METALLIB") { GPU.metallib = URL(fileURLWithPath: override); return }
        guard let executable = Bundle.main.executableURL?.resolvingSymlinksInPath() else { return }
        let library = executable.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources/mlx.metallib")
        if FileManager.default.fileExists(atPath: library.path) { GPU.metallib = library }
    }
    private static var configured = false
    /// First model load only: an idle helper never initialises Metal.
    public static func prepare(cacheLimitMB: Int) {
        guard !configured else { return }
        configured = true
        configure(cacheLimitMB: cacheLimitMB)
    }

    /// Keep freed buffers bounded (Verdict saw 21 GB held without a cap).
    public static func configure(cacheLimitMB: Int) { Memory.cacheLimit = cacheLimitMB * 1024 * 1024 }
    /// Drop the buffer cache only (memory-pressure warning, after long renders).
    public static func trim() { Memory.clearCache() }

    public static var activeMB: Double { Double(Memory.activeMemory) / 1_048_576 }
    public static var cacheMB: Double { Double(Memory.cacheMemory) / 1_048_576 }
    public static var peakMB: Double { Double(Memory.peakMemory) / 1_048_576 }
    public static func resetPeak() { Memory.peakMemory = 0 }
}
