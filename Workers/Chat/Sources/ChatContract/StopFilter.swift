// SPDX-License-Identifier: MIT
import Foundation

/// Holds back only suffixes that could become a stop string, so streamed output never leaks a split stop.
public struct StopFilter: Sendable {
    public private(set) var stopped = false
    private var pending = ""
    private let stops: [String]
    public init(stops: [String]) { self.stops = stops }
    public mutating func append(_ text: String) -> String {
        guard !stopped else { return "" }
        pending += text
        let ranges = stops.compactMap { pending.range(of: $0) }
        if let first = ranges.min(by: { $0.lowerBound < $1.lowerBound }) {
            let output = String(pending[..<first.lowerBound]); pending = ""; stopped = true; return output
        }
        var held = 0
        for stop in stops {
            for n in 1...stop.count where pending.hasSuffix(String(stop.prefix(n))) { held = max(held, n) }
        }
        let output = String(pending.dropLast(held)); pending = String(pending.suffix(held)); return output
    }
    public mutating func finish() -> String {
        defer { pending = "" }
        return stopped ? "" : pending
    }
}

public final class Cancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    public init() {}
    public func cancel() { lock.withLock { value = true } }
    public var isCancelled: Bool { lock.withLock { value } }
}
