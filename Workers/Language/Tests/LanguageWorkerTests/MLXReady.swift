// SPDX-License-Identifier: MIT
import Foundation
import MLX
import IntegrationKit

/// mlx-swift creates its global streams lazily, under its eval lock; a compiled function (compiledGeluApproximate) runs
/// holding that lock and asks for the default stream. When two tests make the process's first MLX calls at the same
/// time, one waits in the stream initializer for the lock while the other holds the lock waiting for the initializer
/// (seen 18 Oct in GemmaProcessorTests and VisionActivationTests). Every suite here creates the streams once, before
/// any MLX work of its own, with the no-fast-math test metallib every other native suite uses.
///
/// MLX reads MLX_ENABLE_TF32 once per process, at its first matmul of any dtype, and float32 matmuls on M5 take the
/// Neural Accelerator path, with about 3e-4 relative error, while it is on. NativeRuntime.prepare turns it off, but only
/// in the suites that call it, so a suite that ran matmuls first (DecodePositionTests) made LFMProcessorTests' resize
/// flaky (20 Oct: 0.0698 off in the full run, exact in every filtered run). This gate runs prepare, as the worker does at
/// start-up, before any suite's MLX work, so the process-wide setting is fixed before the first matmul.
enum MLXReady {
    private static let streams: Void = {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
        NativeRuntime.prepare()
        _ = Stream.defaultStream
    }()
    static func ensure() { _ = streams }
}
