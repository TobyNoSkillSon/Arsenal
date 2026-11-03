// SPDX-License-Identifier: MIT
import Foundation
import Darwin

/// One queue drainer per support directory. Kernel locks release on crash, so stale files do not stall.
public final class BenchmarkRunLease {
    private let descriptor: Int32
    public init(support: URL) throws {
        let directory = support.appendingPathComponent("benchmarks")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let fd = Darwin.open(directory.appendingPathComponent("run.lock").path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { throw ManifestError.invalid("Cannot open benchmark run lease") }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { Darwin.close(fd); throw ManifestError.invalid("Another agent is draining this benchmark queue") }
        descriptor = fd
    }
    deinit { flock(descriptor, LOCK_UN); Darwin.close(descriptor) }
}
