// SPDX-License-Identifier: MIT
import Foundation
import MLX

public enum NativeRuntime {
    /// The MLX buffer-cache cap every native family worker starts with.
    public static let cacheLimitBytes = 256 * 1024 * 1024
    /// Combined workers resolve resources beside their family symlink, before any MLX memory call initializes the device.
    public static func prepare() {
        setenv("MLX_ENABLE_TF32", "0", 1)
        let library =
            ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        if FileManager.default.fileExists(atPath: library.path) { setMetallib(library) }
        // MLX keeps freed buffers for reuse up to its cache limit, by default most of the Mac's memory. Buffers that
        // grow every step (KV caches, per-step activations) are freed and never reused at the next size, so an uncapped
        // cache grew a TeleOCR worker by tens of gigabytes over one generation (20 Oct). Every family starts capped; a
        // runner that measured a need for more sets its own limit at load.
        Memory.cacheLimit = cacheLimitBytes
        // Lab-only allocator ceiling. Ordinary serving retains the user's Load Anyway / macOS swap contract.
        if let bytes = labBudget(ProcessInfo.processInfo.environment) {
            Memory.memoryLimit = bytes
            Memory.cacheLimit = min(cacheLimitBytes, bytes / 8)
        }
    }
    public static func labBudget(_ environment: [String: String]) -> Int? {
        guard environment["ARSENAL_NATIVE_LAB"] == "1", let text = environment["ARSENAL_LAB_MAX_BYTES"],
            let bytes = Int(text), bytes >= 64 * 1024 * 1024
        else { return nil }
        return bytes
    }

}
