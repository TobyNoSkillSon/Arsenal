// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

public final class WorkerChannel: @unchecked Sendable {
    private let lock = NSLock()
    private let output: FileHandle
    private struct Timing { var accepted: Double; var scheduled: Double?; var phase: String?; var peak: (@Sendable () -> Double)? }
    private var timings: [Double: Timing] = [:]
    public func accepted(_ id: JSONValue) throws {
        guard let number = id.number else { return }
        lock.withLock { timings[number] = Timing(accepted: ProcessInfo.processInfo.systemUptime) }
        try send(["id": id, "event": "progress", "phase": "queued"])
    }
    public func scheduled(_ id: JSONValue, phase: String, peak: @escaping @Sendable () -> Double) throws {
        guard let number = id.number else { return }
        lock.withLock {
            timings[number]?.scheduled = ProcessInfo.processInfo.systemUptime; timings[number]?.phase = phase; timings[number]?.peak = peak
        }
        try send(["id": id, "event": "progress", "phase": .string(phase)])
    }
    public init(output: FileHandle) { self.output = output }
    public func send(_ object: JSONObject) throws { try lock.withLock { try FrameIO.write(.json(object), to: output) } }
    public func audio(stream: UInt32, pcm: Data) throws {
        // Even a long Kokoro sentence cannot exceed the wire's maximum frame.
        let size = WireFrame.maximumLength - 5
        try lock.withLock {
            for offset in stride(from: 0, to: pcm.count, by: size) {
                try FrameIO.write(.binary(stream: stream, bytes: pcm.subdata(in: offset..<min(pcm.count, offset + size))), to: output)
            }
        }
    }
    public func reply(_ id: JSONValue, _ fields: JSONObject = [:]) throws {
        var reply = fields
        let timing = lock.withLock { id.number.flatMap { timings.removeValue(forKey: $0) } }
        if let timing, let scheduled = timing.scheduled {
            var fields = RequestTiming(
                queue_s: max(0, scheduled - timing.accepted),
                load_s: timing.phase == "loading" ? ProcessInfo.processInfo.systemUptime - scheduled : nil,
                peak_mb: timing.peak?()
            ).json
            fields.merge(reply["timing"]?.object ?? [:]) { _, value in value }
            reply["timing"] = .object(fields)
            try send(["id": id, "event": "progress", "phase": "done", "timing": .object(fields)])
        }
        reply["id"] = id; reply["ok"] = true; try send(reply)
    }
    public func failure(_ id: JSONValue, code: String, message: String) throws {
        _ = lock.withLock { id.number.flatMap { timings.removeValue(forKey: $0) } }
        let sentence = message.split(whereSeparator: { $0.isNewline }).joined(separator: " ")
        try send(["id": id, "ok": false, "error": ["code": .string(code), "message": .string(sentence)]])
    }
}
