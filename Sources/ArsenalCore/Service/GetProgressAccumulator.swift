// SPDX-License-Identifier: MIT
import Foundation

struct PendingDownloadFile: Sendable {
    let contentID: String
    let file: String
    let bytes: Int64
}
struct GetProgressAccumulator: Sendable {
    private let files: [String: PendingDownloadFile]
    private var completed: [String: Int64] = [:]
    private var phases: [String: String] = [:]
    /// Files counted before the latest conversion step began. While converting, their late callbacks still count
    /// bytes, but only a file that starts afterwards (the next source shard) puts the row back to downloading.
    private var seenBeforeDeriving: Set<String>?
    private let started: Date
    let totalBytes: Int64
    var doneBytes: Int64 { completed.values.reduce(0, +) }
    var fraction: Double { totalBytes == 0 ? 1 : Double(doneBytes) / Double(totalBytes) }
    init(files pending: [PendingDownloadFile], started: Date = Date()) throws {
        var unique: [String: PendingDownloadFile] = [:]
        for file in pending {
            guard file.bytes >= 0, unique[file.contentID]?.bytes == nil || unique[file.contentID]?.bytes == file.bytes else {
                throw WorkerFailure(code: "download_size", message: "Conflicting pinned file sizes.")
            }
            unique[file.contentID] = file
        }
        var sum: Int64 = 0
        for file in unique.values {
            let next = sum.addingReportingOverflow(file.bytes)
            guard !next.overflow else { throw WorkerFailure(code: "download_size", message: "Pinned download size is too large.") }; sum = next.partialValue
        }
        files = unique; totalBytes = sum; self.started = started
    }
    @discardableResult mutating func record(contentID: String?, file: String, completedBytes: Int64) -> Bool {
        let key =
            contentID
            ?? {
                let matches = files.values.filter { $0.file == file }; return matches.count == 1 ? matches[0].contentID : nil
            }()
        guard let key, let expected = files[key] else { return false }
        let value = min(expected.bytes, max(0, completedBytes))
        if let previous = completed[key], value <= previous { return false }
        completed[key] = value; return true
    }
    mutating func record(_ value: DownloadProgress) -> PrecisionState? {
        if value.contentID == GetProgressDelivery.derivingContentID {
            seenBeforeDeriving = Set(phases.keys).union(completed.keys)
            return .init(state: .deriving)
        }
        let key =
            value.contentID
            ?? {
                let matches = files.values.filter { $0.file == value.file }; return matches.count == 1 ? matches[0].contentID : nil
            }()
        guard let key, files[key] != nil else { return nil }
        if value.resetsProgress == true { completed.removeValue(forKey: key) }
        let changed = record(contentID: key, file: value.file, completedBytes: value.completedBytes)
        let phase = value.phase ?? "downloading"
        let phaseChanged = phases[key] != phase; phases[key] = phase
        guard changed || phaseChanged || value.resetsProgress == true else { return nil }
        if let seen = seenBeforeDeriving {
            guard !seen.contains(key) else { return nil }
            seenBeforeDeriving = nil
        }
        return .init(state: .downloading, progress: fraction, bytes: doneBytes, totalBytes: totalBytes, bytesPerSecond: rate(), phase: value.phase ?? "downloading")
    }
    mutating func finish() { completed = files.mapValues(\.bytes) }
    func rate(now: Date = Date()) -> Double { Double(doneBytes) / max(0.001, now.timeIntervalSince(started)) }
}
