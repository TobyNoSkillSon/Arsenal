// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// MLX keeps the metallib path in one unguarded C++ string, and suites set it from test inits that run in parallel: two
/// writes at once corrupted the heap (6 Nov 2026: IntegrationKitTests and Flux2KleinTests crashed in malloc). Every test
/// sets it here, under one process-wide lock (shared by every test module in the process), and only when it changes.
func setTestMetallib(_ url: URL) {
    objc_sync_enter(ProcessInfo.processInfo)
    defer { objc_sync_exit(ProcessInfo.processInfo) }
    if GPU.metallib?.path != url.path { GPU.metallib = url }
}
