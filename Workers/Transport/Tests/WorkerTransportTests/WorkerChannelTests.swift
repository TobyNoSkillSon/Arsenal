// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalWire
import WorkerTransport

private func capture(_ body: (WorkerChannel) throws -> Void) throws -> [WireFrame] {
    let directory = URL(fileURLWithPath: "/tmp/arsenal-mb-pol-workers/transport-tests")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = directory.appendingPathComponent(UUID().uuidString)
    #expect(FileManager.default.createFile(atPath: file.path, contents: nil))
    let output = try FileHandle(forWritingTo: file)
    try body(WorkerChannel(output: output))
    try output.close()
    let input = try FileHandle(forReadingFrom: file)
    defer { try? input.close() }
    var frames: [WireFrame] = []
    while let frame = try FrameIO.read(from: input) { frames.append(frame) }
    return frames
}

@Test func repliesAndErrorsKeepTheirWireShape() throws {
    let frames = try capture { channel in
        try channel.reply(7, ["loaded": true])
        try channel.failure(8, code: "engine_error", message: "One line\nsecond line\rthird line")
    }
    #expect(
        frames == [
            .json(["id": 7, "ok": true, "loaded": true]),
            .json(["id": 8, "ok": false, "error": ["code": "engine_error", "message": "One line second line third line"]])
        ])
}

@Test func audioSplitsOnlyAtTheMaximumFrameBoundary() throws {
    let limit = WireFrame.maximumLength - 5
    let pcm = Data(repeating: 0x45, count: limit + 13)
    let frames = try capture { try $0.audio(stream: 42, pcm: pcm) }
    #expect(
        frames == [
            .binary(stream: 42, bytes: Data(pcm.prefix(limit))),
            .binary(stream: 42, bytes: Data(pcm.suffix(13)))
        ])
    #expect(try capture { try $0.audio(stream: 42, pcm: Data()) }.isEmpty)
}

@Test func concurrentRepliesCannotInterleaveFrames() throws {
    let frames = try capture { channel in
        DispatchQueue.concurrentPerform(iterations: 100) { index in
            do { try channel.reply(.number(Double(index))) } catch { Issue.record("reply failed: \(error)") }
        }
    }
    let ids = frames.compactMap { frame -> Double? in
        guard case .json(let value) = frame, value["ok"] == true else { return nil }
        return value["id"]?.number
    }
    #expect(ids.count == 100)
    #expect(Set(ids) == Set((0..<100).map(Double.init)))
}

@Test func blockingInputUsesAnOwnedNamedThreadAndFinishesOnEOF() async throws {
    let pipe = Pipe()
    let (stream, continuation) = AsyncStream<String>.makeStream()
    WorkerThread.start(name: "arsenal.test.stdin") {
        continuation.yield(Thread.current.name ?? "")
        do {
            #expect(try FrameIO.read(from: pipe.fileHandleForReading) == nil)
            continuation.yield("eof")
        } catch { Issue.record(error) }
        continuation.finish()
    }
    var values = stream.makeAsyncIterator()
    #expect(await values.next() == "arsenal.test.stdin")
    // Awaiting the reader must not monopolize the executor needed to close its input.
    await Task.yield()
    try pipe.fileHandleForWriting.close()
    #expect(await values.next() == "eof")
    #expect(await values.next() == nil)
}

@Test func progressAndFinalTimingShareTheRequestIdentity() throws {
    let frames = try capture { channel in
        try channel.accepted(9)
        try channel.scheduled(9, phase: "step", peak: { 123 })
        try channel.reply(9, ["duration_s": 1])
    }
    let objects = frames.compactMap { frame -> JSONObject? in
        if case .json(let object) = frame { return object }; return nil
    }
    #expect(objects.map { $0["id"] } == [9, 9, 9, 9])
    #expect(objects.prefix(3).map { $0["phase"] } == ["queued", "step", "done"])
    #expect(objects.last?["timing"]?["peak_mb"] == 123)
    #expect(objects.last?["timing"]?["ttft_s"] == nil)
    #expect(objects.last?["duration_s"] == 1 && objects.last?["ok"] == true)
}
