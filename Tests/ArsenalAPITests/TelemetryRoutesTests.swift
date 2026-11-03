// SPDX-License-Identifier: MIT
import Foundation
import Network
import Testing
import ArsenalCore
@testable import ArsenalAPI

private actor TelemetryRouteService: ModelsService {
    var requester = ""
    var recordedRequest: RequestActivity?
    let aggregator = ActivityAggregator(sample: { .init(thermal: "nominal") })
    let live: Bool
    let burst: Int
    let failAfterProgress: Bool
    init(live: Bool = false, burst: Int = 1, failAfterProgress: Bool = false) { self.live = live; self.burst = burst; self.failAfterProgress = failAfterProgress }
    func subscribers() async -> Int { await aggregator.subscriberCount }
    func operations(id: String) async -> [ModelOperation] {
        [.init(name: "chat", title: "Chat", inputSchema: [:], output: "text-stream", stream: true)]
    }
    func rows() async -> [ModelRow] { [ModelRow(id: "chat", name: "Chat", kind: "chat", states: ["bf16": .init(state: .loaded)])] }
    func kinds() async -> [KindSpec] { [] }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        requester = ActivityRequestContext.requester
        let requestID = aggregator.begin(model: id, op: op, requester: requester)
        recordedRequest = await aggregator.snapshot(rows: rows(), memory: .init()).models.first?.requests.first
        defer { aggregator.finish(requestID) }
        try await sink?(.json(["event": "progress", "phase": "queued"]))
        if failAfterProgress { throw WorkerFailure(code: "context_length_exceeded", message: "Prompt exceeds context.") }
        try await sink?(.json(["event": "token", "text": "hello"]))
        return ["text": "hello", "usage": ["prompt_tokens": 10, "completion_tokens": 2], "timing": ["ttft_s": 0.01, "peak_mb": 42]]
    }
    func activity() async -> ActivitySnapshot { await aggregator.snapshot(rows: rows(), memory: .init(models: ["chat": .init(footprintMB: 42)])) }
    func activityStream(machine: Bool) async -> AsyncStream<ActivitySnapshot> {
        if live { return await aggregator.stream(machine: machine) { [weak self] in await self?.activity() ?? .init() } }
        let snapshot = await activity(), burst = burst
        return AsyncStream { continuation in
            for index in 0..<burst { var value = snapshot; value.models[0].queued = index; continuation.yield(value) }
            continuation.finish()
        }
    }
}

@Test(arguments: [0, 1, 2]) func telemetryChatTimingIsOnJSONOrFinalSSEChunk(_ mode: Int) async throws {
    let service = TelemetryRouteService()
    let support = URL(fileURLWithPath: "/tmp/arsenal-telemetry-api-" + UUID().uuidString)
    let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
    let port = try await server.start(); defer { server.stop() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
    request.httpMethod = "POST"; request.timeoutInterval = 10
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if mode == 0 { request.setValue("Fixture MCP", forHTTPHeaderField: "X-Arsenal-Client") }
    if mode == 1 { request.setValue("sdk-product/3.0 transport/1", forHTTPHeaderField: "User-Agent") }
    if mode == 2 { request.setValue("", forHTTPHeaderField: "User-Agent") }
    var body: JSONObject = ["model": "chat", "messages": [["role": "user", "content": "Hi"]], "stream": .bool(mode > 0)]
    if mode == 2 { body["stream_options"] = ["include_usage": true] }
    request.httpBody = try WireJSON.encode(body)
    let (data, response) = try await URLSession.shared.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 200)
    let expectedRequester = mode == 0 ? "Fixture MCP" : "API"
    #expect(await service.recordedRequest?.requester == expectedRequester)
    let final: JSONObject
    if mode == 0 {
        final = try WireJSON.decode(data)
    } else {
        let frames = String(decoding: data, as: UTF8.self).components(separatedBy: "\n").filter { $0.hasPrefix("data: {") }
        final = try WireJSON.decode(Data(try #require(frames.last).dropFirst(6).utf8))
        #expect(String(decoding: data, as: UTF8.self).contains("data: [DONE]"))
    }
    #expect(final["arsenal_timing"]?["ttft_s"] == 0.01 && final["arsenal_timing"]?["peak_mb"] == 42)
    #expect(final["arsenal_timing"]?["decode_tps"] == nil)
    if mode != 1 { #expect(final["usage"]?["prompt_tokens"] == 10) }
}

@Test(arguments: [false, true]) func telemetryActivityRoutesEncodeSnapshotAndSSE(_ stream: Bool) async throws {
    let server = ArsenalAPIServer(service: TelemetryRouteService(), supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-activity-api-" + UUID().uuidString), port: 0)
    let port = try await server.start(); defer { server.stop() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/activity" + (stream ? "/stream" : ""))!)
    request.timeoutInterval = 10
    let (data, response) = try await URLSession.shared.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 200)
    let snapshot: JSONObject
    if stream {
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("event: activity"))
        let line = try #require(text.components(separatedBy: "\n").first { $0.hasPrefix("data: ") })
        snapshot = try WireJSON.decode(Data(line.dropFirst(6).utf8))
    } else {
        snapshot = try WireJSON.decode(data)
    }
    #expect(snapshot["models"]?.array?.first?["memory_mb"] == 42)
    #expect(snapshot["machine"]?.object?.isEmpty == true)
}

@Test func telemetryUniversalProgressDoesNotTurnContextFailureIntoHTTP200() async throws {
    let server = ArsenalAPIServer(
        service: TelemetryRouteService(failAfterProgress: true), supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-progress-api-" + UUID().uuidString), port: 0)
    let port = try await server.start(); defer { server.stop() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/chat/ops/chat")!)
    request.httpMethod = "POST"; request.httpBody = Data("{}".utf8); request.timeoutInterval = 10
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    let (data, response) = try await URLSession.shared.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 400)
    #expect(!String(decoding: data, as: UTF8.self).contains("event: progress"))
}

@Test func telemetryActivitySSEThrottlesBurstsToTenPerSecond() async throws {
    let server = ArsenalAPIServer(service: TelemetryRouteService(burst: 11), supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-rate-api-" + UUID().uuidString), port: 0)
    let port = try await server.start(); defer { server.stop() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/activity/stream")!); request.timeoutInterval = 10
    let clock = ContinuousClock(), start = clock.now
    let (data, _) = try await URLSession.shared.data(for: request)
    #expect(String(decoding: data, as: UTF8.self).components(separatedBy: "event: activity").count - 1 == 11)
    #expect(start.duration(to: clock.now) >= .seconds(1))
}

private actor ActivityHTTPProbe {
    var received = false
    func mark() { received = true }
}
@Test func telemetryActivityHTTPDisconnectStopsSubscriber() async throws {
    let service = TelemetryRouteService(live: true), probe = ActivityHTTPProbe()
    let server = ArsenalAPIServer(service: service, supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-disconnect-api-" + UUID().uuidString), port: 0)
    let port = try await server.start(); defer { server.stop() }
    let session = URLSession(configuration: .ephemeral)
    let consumer = Task {
        let (bytes, _) = try await session.bytes(from: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/activity/stream")!)
        for try await line in bytes.lines where line.hasPrefix("data: ") { await probe.mark() }
    }
    for _ in 0..<100 { if await probe.received { break }; try await Task.sleep(for: .milliseconds(20)) }
    #expect(await probe.received)
    #expect(await service.subscribers() == 1)
    consumer.cancel(); session.invalidateAndCancel(); _ = try? await consumer.value
    for _ in 0..<100 { if await service.subscribers() == 0 { break }; try await Task.sleep(for: .milliseconds(20)) }
    #expect(await service.subscribers() == 0)
    #expect(await service.activity().machine.thermal == nil)
}

/// `?machine=0` (a tool window following its own run) subscribes to requests only: the machine is never read for it.
@Test func activityStreamWithoutMachineNeverReadsTheMachine() async throws {
    let service = TelemetryRouteService(live: true), probe = ActivityHTTPProbe()
    let server = ArsenalAPIServer(service: service, supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-machine-api-" + UUID().uuidString), port: 0)
    let port = try await server.start(); defer { server.stop() }
    let session = URLSession(configuration: .ephemeral)
    defer { session.invalidateAndCancel() }
    func subscribe(_ query: String) -> Task<Void, any Error> {
        Task {
            let (bytes, _) = try await session.bytes(from: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/activity/stream" + query)!)
            for try await line in bytes.lines where line.hasPrefix("data: ") { await probe.mark() }
        }
    }
    let tool = subscribe("?machine=0")
    for _ in 0..<100 { if await probe.received { break }; try await Task.sleep(for: .milliseconds(20)) }
    #expect(await service.subscribers() == 1)
    try await Task.sleep(for: .milliseconds(300))
    #expect(await service.aggregator.sampleCount == 0)
    let panel = subscribe("")
    for _ in 0..<100 { if await service.aggregator.sampleCount > 0 { break }; try await Task.sleep(for: .milliseconds(20)) }
    #expect(await service.aggregator.sampleCount > 0)
    tool.cancel(); panel.cancel()
}
