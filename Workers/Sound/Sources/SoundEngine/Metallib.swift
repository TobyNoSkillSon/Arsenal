// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// MLX keeps the metallib path in one unguarded C++ string. Workers set it once at start, but tests start engines from
/// parallel suites, and two writes at once corrupted the heap (6 Nov 2026). Every write takes one process-wide lock, the
/// same one the test modules' setTestMetallib takes, and happens only when the path changes.
func setMetallib(_ url: URL) {
    objc_sync_enter(ProcessInfo.processInfo)
    defer { objc_sync_exit(ProcessInfo.processInfo) }
    if GPU.metallib?.path != url.path { GPU.metallib = url }
}
