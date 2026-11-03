// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import ArsenalCore

/// The Mac's memory as the MCP texts state it, in decimal GB (the wire unit; texts convert to the app's GB).
struct MCPMacMemory: Sendable, Equatable {
    var inUseGB: Double
    var limitGB: Double
    var totalGB: Double
    /// Room under the Mac-wide line now.
    var roomGB: Double { max(0, limitGB - inUseGB) }
    static func live() -> MCPMacMemory {
        let probe = MemoryProbe()
        let limit = MacMemoryGuard.safeInUseMB(totalMB: probe.totalMB)
        let room = MacMemoryGuard.roomUnderLineMB(totalMB: probe.totalMB, noSwapAvailableMB: probe.noSwapAvailableMB(loadedMB: 0))
        return .init(inUseGB: (limit - room) / 1000, limitGB: limit / 1000, totalGB: probe.totalMB / 1000)
    }
}

/// Everything time, disk and memory dependent the MCP surface reads, injectable so tests run a 10-minute download in
/// milliseconds (`scaled`). Production uses `live`.
struct MCPEnvironment: Sendable {
    var now: @Sendable () -> Date = { Date() }
    var sleep: @Sendable (Double) async throws -> Void = { seconds in try await Task.sleep(for: .seconds(max(0, seconds))) }
    var freeDiskBytes: @Sendable (URL) -> Int64? = { url in
        (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage).map { Int64($0) }
    }
    var macMemory: @Sendable () -> MCPMacMemory = { MCPMacMemory.live() }
    /// Clones one open file (APFS) to a path; returns 0 or the errno. Tests force the copy path with EXDEV.
    var cloneFile: @Sendable (Int32, String) -> Int32 = { descriptor, destination in fclonefileat(descriptor, AT_FDCWD, destination, 0) == 0 ? 0 : errno }
    /// How long reading an input file may block (a privacy prompt, a sleeping disk) before the call is refused.
    var pathDeadlineSeconds = 10.0
    /// A download is reported as started at its first bytes, or after this many seconds.
    var confirmSeconds = 5.0
    /// Progress notifications go out at most every `tickSeconds` and at least every `heartbeatSeconds`.
    var tickSeconds = 5.0
    var heartbeatSeconds = 20.0
    /// `arsenal_wait`'s default budget: under Codex's 60 s tool timeout, for every client.
    var waitSeconds = 50.0
    static let live = MCPEnvironment()
    /// Simulated time running `factor` times faster than the wall clock.
    static func scaled(_ factor: Double, base: MCPEnvironment = .init()) -> MCPEnvironment {
        let start = Date(), origin = ContinuousClock.now
        var value = base
        value.now = {
            let elapsed = (ContinuousClock.now - origin).components
            return start.addingTimeInterval((Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18) * factor)
        }
        value.sleep = { seconds in try await Task.sleep(for: .seconds(max(0, seconds) / factor)) }
        return value
    }
}

/// Sustained download rates measured on this Mac, per source host: the median of the last five Gets that moved at
/// least 100 MB. Kept in the support directory so the first answer after a restart still has a basis.
struct MCPRateHistory: Sendable {
    let file: URL
    static let minimumBytes: Int64 = 100_000_000
    func median(host: String) -> Double? {
        let values = (read()[host] ?? []).sorted()
        guard !values.isEmpty else { return nil }
        return values.count % 2 == 1 ? values[values.count / 2] : (values[values.count / 2 - 1] + values[values.count / 2]) / 2
    }
    func record(host: String, bytes: Int64, seconds: Double) {
        guard bytes >= Self.minimumBytes, seconds > 0 else { return }
        var all = read()
        all[host] = Array(((all[host] ?? []) + [Double(bytes) / seconds]).suffix(5))
        guard let data = try? JSONEncoder().encode(all) else { return }
        try? data.write(to: file, options: .atomic)
    }
    private func read() -> [String: [Double]] { (try? JSONDecoder().decode([String: [Double]].self, from: Data(contentsOf: file))) ?? [:] }
}

/// A sliding window of (time, bytes) samples: the rate over the last 15 s, shown once 3 s and 1 MB have been seen.
struct MCPRateWindow: Sendable {
    static let window = 15.0
    private(set) var samples: [(Date, Int64)] = []
    private(set) var first: (Date, Int64)?
    private(set) var lastChange: Date?
    mutating func add(_ time: Date, _ bytes: Int64) {
        if first == nil { first = (time, bytes); lastChange = time }
        if let last = samples.last, bytes > last.1 { lastChange = time }
        samples.append((time, bytes))
        samples.removeAll { time.timeIntervalSince($0.0) > Self.window * 2 }
    }
    /// Bytes per second over the last 15 s of samples; nil until 3 s and 1 MB of data. A stall shows in `stalled`.
    func rate(at now: Date) -> Double? {
        guard let first, let last = samples.last, now.timeIntervalSince(first.0) >= 3, last.1 - first.1 >= 1_000_000 else { return nil }
        let start = last.0.addingTimeInterval(-Self.window)
        let base = samples.last { $0.0 <= start } ?? (samples.first.map { $0.0 < first.0 ? first : $0 } ?? first)
        let span = last.0.timeIntervalSince(base.0)
        guard span > 0 else { return nil }
        return Double(max(0, last.1 - base.1)) / span
    }
    /// Seconds since the byte count last grew.
    func stalled(at now: Date) -> Double { lastChange.map { now.timeIntervalSince($0) } ?? 0 }
}
