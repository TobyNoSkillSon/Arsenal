// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func requesterIdentityAndTaskInheritance() async throws {
    #expect(ActivityRequestContext.identity(headers: ["x-arsenal-client": "  Claude Code ", "user-agent": "openai/1.0"]) == "Claude Code")
    #expect(ActivityRequestContext.identity(headers: ["user-agent": "openai-python/2.6 other/1"]) == "API")
    #expect(ActivityRequestContext.identity(headers: [:]) == "API")
    await ActivityRequestContext.$requester.withValue("MCP fixture") {
        #expect(await Task { ActivityRequestContext.requester }.value == "MCP fixture")
    }
    #expect(ActivityRequestContext.requester == "Arsenal")
}

@Test func activityTracksQueuePhasesAndDiscardsTerminalRequests() async {
    let aggregator = ActivityAggregator()
    let row = ModelRow(id: "chat", name: "Chat", kind: "chat", states: ["bf16": .init(state: .loaded)])
    let id = aggregator.begin(model: "chat", op: "chat", requester: "Arsenal")
    var snapshot = await aggregator.snapshot(rows: [row], memory: .init(models: ["chat": .init(footprintMB: 512)]))
    #expect(snapshot.models.first?.queued == 1)
    #expect(snapshot.models.first?.requests.first?.requester == "Arsenal")
    #expect(snapshot.models.first?.memoryMB == 512 && snapshot.machine.gpuBusyPct == nil)
    aggregator.receive(
        id,
        .json([
            "event": "progress", "phase": "decode", "prompt_tokens": 100, "cached_tokens": 50,
            "generated_tokens": 3, "batch": 1, "timing": ["decode_tps": 25]
        ]))
    snapshot = await aggregator.snapshot(rows: [row], memory: .init())
    #expect(snapshot.models.first?.state == "decode" && snapshot.models.first?.queued == 0)
    #expect(snapshot.models.first?.requests.first?.tps == 25)
    #expect(snapshot.models.first?.requests.first?.cachedTokens == 50)
    aggregator.receive(id, .json(["event": "progress", "phase": "done"]))
    let completed = await aggregator.snapshot(rows: [row], memory: .init())
    #expect(completed.models.first?.state == "done" && completed.models.first?.queued == 0)
    aggregator.finish(id)
    #expect(await aggregator.snapshot(rows: [row], memory: .init()).models.first?.requests.isEmpty == true)
}

@Test func activitySamplerExistsOnlyDuringSubscription() async throws {
    let aggregator = ActivityAggregator(sample: { .init(thermal: "nominal") }, machineInterval: .milliseconds(100))
    try await Task.sleep(for: .milliseconds(300))
    #expect(await aggregator.sampleCount == 0)
    let stream = await aggregator.stream { await aggregator.snapshot(rows: [], memory: .init()) }
    let consumer = Task { for await _ in stream { await Task.yield() } }
    for _ in 0..<100 { if await aggregator.sampleCount >= 2 { break }; try await Task.sleep(for: .milliseconds(20)) }
    #expect(await aggregator.subscriberCount == 1)
    #expect(await aggregator.sampleCount >= 2)
    consumer.cancel(); await consumer.value
    for _ in 0..<100 { if await aggregator.subscriberCount == 0 { break }; try await Task.sleep(for: .milliseconds(20)) }
    let count = await aggregator.sampleCount
    #expect(await aggregator.subscriberCount == 0)
    try await Task.sleep(for: .milliseconds(300))
    #expect(await aggregator.sampleCount == count)
    #expect(await aggregator.snapshot(rows: [], memory: .init()).machine.gpuBusyPct == nil)
}

@Test func rawDecisionTimingIsTransportMetadata() async throws {
    let frame = Data(#"{"id":1,"ok":true,"answers":{"yes":0.1234567890123456789},"timing":{"peak_mb":42}}"#.utf8)
    let body = try RawJSONObject.removing(["id", "ok", "timing"], from: frame)
    #expect(String(decoding: body, as: UTF8.self) == #"{"answers":{"yes":0.1234567890123456789}}"#)
}

private struct TelemetryFixtureArtifact: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        var launch = runtimeFixtureLaunch()
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let code = """
            import sys,time
            sys.path.insert(0, \(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers").path)))
            from arsenal_worker import Worker
            w=Worker('chat',runner='native')
            w.op('load')(lambda r: {'memory':{'footprint_mb':256,'mlx_active_mb':0,'mlx_cache_mb':0}})
            @w.op('chat')
            def chat(r):
                w.emit(r,'progress',phase='decode',prompt_tokens=10,cached_tokens=0,generated_tokens=2,timing={'decode_tps':20})
                time.sleep(.5)
                return {'text':'fixture','timing':{'ttft_s':.01,'decode_tps':20}}
            w.run()
            """
        launch.arguments = ["-I", "-c", code]; return launch
    }
}

@Test func activityServiceAttributesARealFixtureWorkerAndCleansUp() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: TelemetryFixtureArtifact())
    defer { fixture.stop() }
    try await fixture.service.get(id: "fixture-chat", precision: "bf16")
    try await fixture.service.load(id: "fixture-chat")
    let run = Task {
        try await ActivityRequestContext.$requester.withValue("Fixture MCP") {
            try await fixture.service.run(id: "fixture-chat", op: "chat", params: ["messages": [["role": "user", "content": "Hi"]]])
        }
    }
    var observed: RequestActivity?
    for _ in 0..<40 {
        let value = await fixture.service.activity()
        if let request = value.models.first?.requests.first, request.phase == "decode" { observed = request; break }
        try await Task.sleep(for: .milliseconds(20))
    }
    #expect(observed?.requester == "Fixture MCP")
    #expect(observed?.generatedTokens == 2 && observed?.tps == 20)
    let result = try await run.value
    #expect(result["timing"]?["ttft_s"]?.number != nil)
    #expect(await fixture.service.activity().models.first?.requests.isEmpty == true)
    await fixture.service.shutdown()
    #expect(await fixture.service.activity().models.isEmpty == true)
}

@Test func activityRunTimingIncludesAdmissionButSeparatesLoading() {
    let clock = ActivityRunTracking(id: UUID(), accepted: 10)
    clock.startLoading(at: 12); clock.finishLoading(at: 15); clock.schedule(at: 16)
    let result = clock.decorate(["_arsenal_first_token_at": 20, "timing": ["queue_s": 0.5, "ttft_s": 1, "decode_tps": 30]])
    #expect(result["timing"]?["queue_s"] == 3.5)
    #expect(result["timing"]?["load_s"] == 3)
    #expect(result["timing"]?["ttft_s"] == 10)
    #expect(result["timing"]?["decode_tps"] == 30)
    #expect(result["_arsenal_first_token_at"] == nil)
    #expect(clock.decorate(["timing": ["ttft_s": 1]])["timing"]?["ttft_s"] == 7)
}

private func activityWaitForSampler(_ signal: DispatchSemaphore) -> Bool { signal.wait(timeout: .now() + 15) == .success }

@Test(.timeLimit(.minutes(1))) func activitySlowMachineSamplerDoesNotBlockRequestState() async throws {
    let started = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
    let aggregator = ActivityAggregator(sample: {
        // Hold the sampler until the assertions release it; a two-second auto-release raced
        // the test task under load and published the machine reading before the snapshot.
        started.signal(); _ = release.wait(timeout: .now() + 60)
        return .init(thermal: "nominal")
    })
    let row = ModelRow(id: "chat", name: "Chat", kind: "chat", states: ["bf16": .init(state: .loaded)])
    let stream = await aggregator.stream { await aggregator.snapshot(rows: [row], memory: .init()) }
    let consumer = Task { for await _ in stream {} }
    defer { release.signal(); consumer.cancel() }
    #expect(await Task.detached { activityWaitForSampler(started) }.value)
    let id = aggregator.begin(model: "chat", op: "chat", requester: "Arsenal")
    aggregator.receive(id, .json(["event": "progress", "phase": "decode", "generated_tokens": 2]))
    let snapshot = await aggregator.snapshot(rows: [row], memory: .init())
    #expect(snapshot.models.first?.requests.first?.generatedTokens == 2)
    #expect(snapshot.machine.thermal == nil)
}

@Test func activityExcludesWorkersOutsideTheResidentSupervisor() async throws {
    let worker = WorkerProcess(), supervisor = Supervisor()
    try await worker.start(runtimeFixtureLaunch())
    let pid = await worker.pid
    #expect(await supervisor.workerPIDs().contains(pid))
    await worker.unload()
    #expect(await !supervisor.workerPIDs().contains(pid))
}

private actor ActivityCompletion {
    var finished = false
    func finish() { finished = true }
}
private actor ActivityGenerationGate {
    var entered = false
    private var waiter: CheckedContinuation<Void, Never>?
    func pause() async {
        guard !entered else { return }
        entered = true
        await withCheckedContinuation { waiter = $0 }
    }
    func release() { waiter?.resume(); waiter = nil }
}

@Test func activityRefusedMemoryRequestDoesNotBlockUpdate() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat", "voices"])
    defer { fixture.stop() }
    await fixture.service.shutdown()
    let file = fixture.paths.support.appendingPathComponent("activity-memory.json")
    try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 1290))]).write(to: file)
    let service = try await RealModelsService.open(
        catalogData: fixture.catalogData, signature: fixture.signature, publicKey: fixture.publicKey, source: fixture.source,
        kinds: KindSpec.defaults, paths: fixture.paths, artifacts: StubWorkerArtifact(), client: fixture.client,
        memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: 128_000))
    for id in ["fixture-chat", "fixture-voices"] { try await service.get(id: id, precision: "bf16") }
    try await service.setMemorySettings(.init(whenFull: .queue))
    await service.setLoadPeak("fixture-voices", gb: 1.28)
    let gate = ActivityGenerationGate()
    let active = Task {
        try await service.run(id: "fixture-chat", op: "chat", params: ["prompt": "hello", "stream": true], sink: { _ in await gate.pause() })
    }
    defer { active.cancel() }
    for _ in 0..<250 {
        if await gate.entered { break }
        try await Task.sleep(for: .milliseconds(20))
    }
    #expect(await gate.entered)
    let runFinished = ActivityCompletion(), updateFinished = ActivityCompletion()
    let run = Task {
        let result: Result<JSONObject, any Error>
        do {
            result = .success(
                try await ActivityRequestContext.$requester.withValue("Queued fixture") {
                    try await service.run(id: "fixture-voices", op: "speak", params: ["text": "hello", "stream": false])
                })
        } catch { result = .failure(error) }
        await runFinished.finish(); return result
    }
    defer { run.cancel() }
    let refused = await run.value
    switch refused {
    case .success: Issue.record("Memory shortage was admitted")
    case .failure(let error): #expect((error as? WorkerFailure)?.code == "insufficient_memory")
    }
    #expect(await service.memoryWaiters.isEmpty)
    #expect(await service.activity().models.first { $0.id == "fixture-voices" }?.requests.isEmpty != false)
    let update = Task {
        let result: Result<[String], any Error>
        do { result = .success(try await service.beginUpdate()) } catch { result = .failure(error) }
        await updateFinished.finish(); return result
    }
    defer { update.cancel() }
    for _ in 0..<250 {
        if await service.updating { break }
        try await Task.sleep(for: .milliseconds(20))
    }
    #expect(await service.updating)
    await gate.release()
    for _ in 0..<250 {
        if await runFinished.finished, await updateFinished.finished { break }
        try await Task.sleep(for: .milliseconds(20))
    }
    let runCompleted = await runFinished.finished, updateCompleted = await updateFinished.finished
    let completed = runCompleted && updateCompleted
    // Release/cancel before reporting a failure so a broken wakeup cannot hang the test suite.
    if !completed { run.cancel(); update.cancel(); await service.cancelUpdate() }
    #expect(completed)
    if completed {
        _ = try await active.value
        switch await run.value {
        case .success: Issue.record("Update admitted the refused memory request")
        case .failure(let error): #expect((error as? WorkerFailure)?.code == "insufficient_memory")
        }
        #expect(try await update.value.get() == ["fixture-chat"])
        do {
            _ = try await service.run(id: "fixture-chat", op: "chat", params: ["prompt": "new"])
            Issue.record("Update admitted a new request")
        } catch { #expect((error as? WorkerFailure)?.code == "updating") }
        #expect(await service.activity().models.allSatisfy { $0.requests.isEmpty && $0.queued == 0 })
    }
    await service.cancelUpdate(); await service.shutdown()
}

/// The machine is read every second while subscribed, while running requests' progress keeps flowing
/// four times a second without another machine reading.
@Test(.timeLimit(.minutes(1))) func theMachineIsReadEverySecondWhileRequestsStayLive() async throws {
    let interval = ActivityAggregator().machineInterval
    #expect(interval == .seconds(1))
    let clock = ActivityTestClock()
    let aggregator = ActivityAggregator(sample: { .init(thermal: "nominal") }, machineInterval: interval, sleep: { try await clock.sleep($0) })
    let row = ModelRow(id: "m", name: "M", kind: "chat", states: ["bf16": .init(state: .loaded)])
    let counter = ActivityTestSnapshots()
    let request = aggregator.begin(model: "m", op: "chat", requester: "Fixture")
    defer { aggregator.finish(request) }
    let stream = await aggregator.stream {
        await counter.tick(); return await aggregator.snapshot(rows: [row], memory: .init())
    }
    let consumer = Task { for await _ in stream { await Task.yield() } }
    defer { consumer.cancel() }
    await clock.wait(for: .seconds(1), count: 1)
    #expect(await aggregator.sampleCount == 1)
    for tick in 1...3 {
        await clock.wait(for: .milliseconds(250), count: 1)
        await clock.fire(.milliseconds(250))
        await clock.wait(for: .milliseconds(250), count: 1)
        #expect(await aggregator.requestTickCount == tick)
        #expect(await aggregator.sampleCount == 1)
    }
    for sample in 2...4 {
        await clock.fire(.seconds(1))
        await clock.wait(for: .seconds(1), count: 1)
        #expect(await aggregator.sampleCount == sample)
    }
    #expect(await counter.count >= 6)
    consumer.cancel(); await consumer.value
    await clock.wait(for: .seconds(1), count: 0)
    await clock.wait(for: .milliseconds(250), count: 0)
}

/// A first request a second or more after the counts last changed publishes at once and starts the request clock; a
/// burst behind it waits for the clock's next tick; the last end, a second later, publishes at once and stops the clock.
/// Idle, only the machine timer runs; without a subscriber, nothing.
@Test(.timeLimit(.minutes(1))) func activityEventsPublishAtOnceAndBurstsWaitForTheTick() async throws {
    let clock = ActivityTestClock(), time = ActivityTestTime(100)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(30), sleep: { try await clock.sleep($0) }, now: { time.value })
    let row = ModelRow(id: "m", name: "M", kind: "decisions", states: ["bf16": .init(state: .loaded)])
    let stream = await aggregator.stream { await aggregator.snapshot(rows: [row], memory: .init()) }
    let consumer = Task { for await _ in stream {} }
    defer { consumer.cancel() }
    // The provider returns before the publish lands on the actor: wait for the count itself.
    func waitPublished(_ count: Int) async -> Int {
        for _ in 0..<2000 { if await aggregator.publishCount >= count { break }; await Task.yield() }
        return await aggregator.publishCount
    }
    await clock.wait(for: .seconds(30), count: 1)
    #expect(await waitPublished(1) == 1)
    #expect(await clock.pendingCount == 1) // only the machine timer, while idle
    time.value = 101.5
    let first = aggregator.begin(model: "m", op: "decide", requester: "Fixture")
    #expect(await waitPublished(2) == 2)
    await clock.wait(for: .milliseconds(250), count: 1)
    // A hundred more at once: no publish until the tick.
    let burst = (0..<100).map { _ in aggregator.begin(model: "m", op: "decide", requester: "Fixture") }
    for _ in 0..<200 { await Task.yield() }
    #expect(await aggregator.publishCount == 2)
    time.value = 101.75
    await clock.fire(.milliseconds(250))
    #expect(await waitPublished(3) == 3)
    #expect(await aggregator.snapshot(rows: [row], memory: .init()).models.first?.queued == 101)
    // Ends while others still run wait for the tick too; the last end publishes at once and stops the clock.
    aggregator.finish(first)
    for id in burst.dropLast() { aggregator.finish(id, .cancelled) }
    for _ in 0..<200 { await Task.yield() }
    #expect(await aggregator.publishCount == 3)
    time.value = 102.6
    aggregator.finish(burst[burst.count - 1], .cancelled)
    #expect(await waitPublished(4) == 4)
    await clock.wait(for: .milliseconds(250), count: 0)
    #expect(await clock.pendingCount == 1)
    consumer.cancel(); await consumer.value
    await clock.wait(for: .seconds(30), count: 0)
    #expect(await aggregator.subscriberCount == 0)
    #expect(await clock.pendingCount == 0)
    let samples = await aggregator.sampleCount, publishes = await aggregator.publishCount
    aggregator.changed()
    for _ in 0..<50 { await Task.yield() }
    await clock.fire(.seconds(30)); await clock.fire(.milliseconds(250))
    #expect(await aggregator.sampleCount == samples)
    #expect(await aggregator.publishCount == publishes)
}

/// Counts before values: under a fake clock, the exact running/waiting split for a runner that reports its queue and for
/// one that reports nothing until it starts (decisions), then the minute's exact counters, means and busy seconds.
@Test func activityCountsRunningWaitingAndTheLastMinute() async {
    let time = ActivityTestTime(1000.25)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let rows = ["chat", "decisions", "images"].map { ModelRow(id: $0, name: $0, kind: $0, states: ["bf16": .init(state: .loaded)]) }
    func model(_ id: String) async -> ModelActivity? { await aggregator.snapshot(rows: rows, memory: .init()).models.first { $0.id == id } }
    func progress(_ id: UUID, _ fields: JSONObject) { aggregator.receive(id, .json(["event": "progress"].merging(fields) { a, _ in a })) }

    // A runner that reports its queue: 48 sent, all acknowledged as queued, 8 started.
    let reported = (0..<48).map { _ in aggregator.begin(model: "chat", op: "chat", requester: "Fixture") }
    for id in reported { aggregator.phase(id, "prefill"); progress(id, ["phase": "queued"]) }
    for id in reported.prefix(8) { progress(id, ["phase": "decode"]) }
    var chat = await model("chat")
    #expect(chat?.requests.count == 48 && chat?.running == 8 && chat?.queued == 40)
    #expect(chat?.requests.filter { $0.phase == "decode" }.count == 8)

    // A serial runner (decisions) reports nothing until it starts one: all 48 it was sent wait…
    let serial = (0..<48).map { _ in aggregator.begin(model: "decisions", op: "decide", requester: "Fixture") }
    for id in serial { aggregator.phase(id, "step") }
    var decisions = await model("decisions")
    #expect(decisions?.requests.count == 48 && decisions?.running == 0 && decisions?.queued == 48)
    #expect(decisions?.requests.allSatisfy { $0.phase == "queued" } == true && decisions?.state == "queued")
    // …and once it reports a start, only the started ones run.
    progress(serial[3], ["phase": "prefill"])
    decisions = await model("decisions")
    #expect(decisions?.running == 1 && decisions?.queued == 47 && decisions?.state == "prefill")
    #expect(decisions?.requests.filter { $0.phase == "prefill" }.map(\.id) == [serial[3]])
    #expect(decisions?.recent == nil)

    // Five finish 0.5 s after acceptance, each started 0.1 s after it; two fail; one is cancelled (not counted).
    time.value += 0.1
    for id in serial[4..<9] { progress(id, ["phase": "prefill"]) }
    time.value += 0.4
    for id in serial[4..<9] { aggregator.finish(id, .finished(["answers": [:]])) }
    for id in serial[9..<11] { aggregator.finish(id, .failed) }
    aggregator.finish(serial[11], .cancelled)
    // The second under way is not counted yet.
    #expect(await model("decisions")?.recent == nil)
    time.value = 1002.1
    decisions = await model("decisions")
    let recent = decisions?.recent
    #expect(decisions?.requests.count == 40 && decisions?.running == 1 && decisions?.queued == 39)
    #expect(recent?.finished == 5 && recent?.failed == 2 && recent?.windowS == 60 && recent?.busy.count == 60)
    #expect(abs((recent?.meanRunS ?? 0) - 0.4) < 1e-9 && abs((recent?.meanWaitS ?? 0) - 0.1) < 1e-9)
    // Busy since serial[3] started at 1000.25: three quarters of second 1000, all of 1001, nothing before.
    #expect(abs((recent?.busy[58] ?? 0) - 0.75) < 1e-9 && recent?.busy[59] == 1 && recent?.busy.prefix(58).allSatisfy { $0 == 0 } == true)
    #expect(abs((recent?.busyS ?? 0) - 1.75) < 1e-9)

    // Chat: tokens counted by their increase per request, in the second they were written.
    progress(reported[0], ["generated_tokens": 10]); progress(reported[1], ["generated_tokens": 4])
    time.value = 1003.5
    progress(reported[0], ["generated_tokens": 25]); progress(reported[0], ["generated_tokens": 25])
    time.value = 1004.0
    chat = await model("chat")
    #expect(chat?.recent?.tokens == 29 && chat?.recent?.finished == 0)
    // Eight streams side by side are one busy model, never more than the whole second.
    #expect(abs((chat?.recent?.busyS ?? 0) - 3.75) < 1e-9 && chat?.recent?.busy.max() == 1)

    // Images: steps, and the images a result returned.
    let image = aggregator.begin(model: "images", op: "generate", requester: "Fixture")
    progress(image, ["phase": "step", "step": 1, "total": 4]); progress(image, ["step": 3])
    time.value = 1004.5
    aggregator.finish(image, .finished(["images": ["a.png", "b.png"]]))
    time.value = 1005.0
    let images = await model("images")?.recent
    #expect(images?.steps == 3 && images?.images == 2 && images?.finished == 1 && images?.busy.last == 0.5)

    // A minute later the models still working are busy all of it; once they stop, a minute after that, it is empty.
    time.value = 1066.0
    let still = await model("decisions")?.recent
    #expect(still?.busyS == 60 && still?.finished == 0)
    #expect(await model("images")?.recent == nil)
    for id in reported + serial { aggregator.finish(id, .cancelled) }
    time.value = 1127.0
    #expect(await model("decisions")?.recent == nil)
    #expect(await model("chat")?.recent == nil)
}

/// Speech: audio seconds come from the runner's sample counts once the request's stream announced its rate; without one,
/// from the result's duration. Never double-counted.
@Test func activityCountsAudioSecondsWithoutDoubleCounting() async {
    let time = ActivityTestTime(50)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let row = ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .loaded)])
    func progress(_ id: UUID, _ samples: Int) { aggregator.receive(id, .json(["event": "progress", "phase": "step", "audio_samples": .number(Double(samples))])) }
    let first = aggregator.begin(model: "voice", op: "speak", requester: "Fixture")
    progress(first, 24_000)
    #expect(await aggregator.snapshot(rows: [row], memory: .init()).models.first?.requests.first?.audioS == nil)
    progress(first, 48_000)
    aggregator.finish(first, .finished(["duration_s": 2.0]))
    let second = aggregator.begin(model: "voice", op: "speak", requester: "Fixture")
    aggregator.receive(second, .json(["event": "audio_begin", "stream": 1, "sample_rate": 24_000, "channels": 1]))
    progress(second, 24_000)
    #expect(await aggregator.snapshot(rows: [row], memory: .init()).models.first?.requests.first?.audioS == 1.0)
    progress(second, 36_000)
    aggregator.finish(second, .finished(["duration_s": 1.5]))
    time.value = 51
    let recent = await aggregator.snapshot(rows: [row], memory: .init()).models.first?.recent
    #expect(recent?.finished == 2 && recent?.audioS == 3.5)
}

/// The AppleSMC struct must be the kernel's 80 bytes with its data at offset 48; on a Mac whose sensors Arsenal knows
/// (M5 Max), a reading is a plausible die temperature.
@Test func smcTemperaturesReadThisMacsDies() {
    #expect(SMCTemperatures.parameterLayout.size == 80 && SMCTemperatures.parameterLayout.bytesOffset == 48)
    let reading = SMCTemperatures().read()
    print("SMC CPU \(reading.cpu.map { String(format: "%.1f", $0) } ?? "none") °C, GPU \(reading.gpu.map { String(format: "%.1f", $0) } ?? "none") °C")
    for value in [reading.cpu, reading.gpu].compactMap({ $0 }) { #expect(value > 5 && value < 130) }
}

/// `ARSENAL_MEASURE_SAMPLER=<seconds>`: the real machine sampler's cost. Times single readings, then this process's CPU
/// time over the window with one subscriber (the open panel) and with none (closed). Prints the figures; not run by
/// default.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MEASURE_SAMPLER"] != nil))
func measureTheMachineSampler() async throws {
    let seconds = Double(ProcessInfo.processInfo.environment["ARSENAL_MEASURE_SAMPLER"] ?? "") ?? 60
    func cpuMS() -> Double {
        var usage = rusage(); getrusage(RUSAGE_SELF, &usage)
        return Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000 + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000
    }
    var sampler = MachineSampler()
    var times: [Double] = []
    for i in 0..<11 {
        let start = ContinuousClock.now
        let m = sampler.sample()
        let ms = Double((ContinuousClock.now - start).components.attoseconds) / 1e15 + Double((ContinuousClock.now - start).components.seconds) * 1000
        if i == 0 {
            print(
                String(
                    format: "first reading (SMC key discovery) %.1f ms: cpu %@ gpu %@ cpuT %@ gpuT %@ used %@ MB", ms, "\(m.cpuBusyPct ?? -1)", "\(m.gpuBusyPct ?? -1)",
                    "\(m.cpuTempC ?? -1)", "\(m.gpuTempC ?? -1)", "\(m.memoryUsedMB ?? -1)"))
        } else {
            times.append(ms)
        }
        try await Task.sleep(for: .milliseconds(200))
    }
    times.sort()
    print(String(format: "later readings: median %.1f ms, max %.1f ms (n=%d)", times[times.count / 2], times.last ?? 0, times.count))
    let aggregator = ActivityAggregator()
    try await Task.sleep(for: .seconds(2))
    let closedStart = cpuMS()
    try await Task.sleep(for: .seconds(seconds))
    let closed = cpuMS() - closedStart
    let stream = await aggregator.stream { await aggregator.snapshot(rows: [], memory: .init()) }
    let consumer = Task { for await _ in stream { await Task.yield() } }
    try await Task.sleep(for: .seconds(1))
    let openStart = cpuMS(), samplesStart = await aggregator.sampleCount
    try await Task.sleep(for: .seconds(seconds))
    let open = cpuMS() - openStart, samples = await aggregator.sampleCount - samplesStart
    consumer.cancel(); await consumer.value
    print(String(format: "closed: %.1f ms CPU in %.0f s (test process baseline); open: %.1f ms CPU in %.0f s with %d machine readings", closed, seconds, open, seconds, samples))
}

/// One client, one spelling: MCP slugs and X-Arsenal-Client names read the same everywhere.
@Test func clientNamesAreNormalisedOnce() {
    #expect(ActivityRequestContext.displayName("claude-code") == "Claude Code")
    #expect(ActivityRequestContext.displayName("Claude Code") == "Claude Code")
    #expect(ActivityRequestContext.displayName("codex-mcp-client") == "Codex")
    #expect(ActivityRequestContext.displayName("my-agent") == "My Agent")
    #expect(ActivityRequestContext.displayName("Arsenal") == "Arsenal" && ActivityRequestContext.displayName("API") == "API")
    #expect(ActivityRequestContext.identity(headers: ["x-arsenal-client": "claude-code"]) == "Claude Code")
}

/// A settable clock for the request counters.
private final class ActivityTestTime: @unchecked Sendable {
    private let lock = NSLock()
    private var current: Double
    init(_ value: Double) { current = value }
    var value: Double {
        get { lock.withLock { current } }
        set { lock.withLock { current = newValue } }
    }
}

/// Each timer advances only when fired. Pending/cancelled timer events provide barriers between assertions.
private actor ActivityTestClock {
    private struct Sleeper { var delay: Duration; var continuation: CheckedContinuation<Void, any Error> }
    private struct Observer { var delay: Duration; var count: Int; var continuation: CheckedContinuation<Void, Never> }
    private var sleepers: [UUID: Sleeper] = [:]
    private var observers: [Observer] = []
    var pendingCount: Int { sleepers.count }
    func pending(_ delay: Duration) -> Int { sleepers.values.filter { $0.delay == delay }.count }
    func sleep(_ delay: Duration) async throws {
        let id = UUID()
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try await withCheckedThrowingContinuation { continuation in
                sleepers[id] = Sleeper(delay: delay, continuation: continuation)
                notify()
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
    }
    func wait(for delay: Duration, count: Int) async {
        if pending(delay) == count { return }
        await withCheckedContinuation { observers.append(Observer(delay: delay, count: count, continuation: $0)) }
    }
    func fire(_ delay: Duration) {
        for (id, sleeper) in sleepers where sleeper.delay == delay {
            sleepers[id] = nil; sleeper.continuation.resume()
        }
        notify()
    }
    private func cancel(_ id: UUID) {
        sleepers.removeValue(forKey: id)?.continuation.resume(throwing: CancellationError())
        notify()
    }
    private func notify() {
        let ready = observers.filter { pending($0.delay) == $0.count }
        observers.removeAll { pending($0.delay) == $0.count }
        for observer in ready { observer.continuation.resume() }
    }
}

private actor ActivityTestSnapshots {
    var count = 0
    private var observers: [(Int, CheckedContinuation<Void, Never>)] = []
    func tick() {
        count += 1
        let ready = observers.filter { count > $0.0 }
        observers.removeAll { count > $0.0 }
        for (_, continuation) in ready { continuation.resume() }
    }
    func wait(after value: Int) async {
        if count > value { return }
        await withCheckedContinuation { observers.append((value, $0)) }
    }
}

// MARK: Round 2 (code review r1 of the activity-live lane)

private func progressFrame(_ fields: JSONObject) -> WireFrame { .json(["event": "progress"].merging(fields) { a, _ in a }) }
/// The clock's pending timers reach `count` within two seconds; false (a failed expectation, not a hung test) otherwise.
private func reached(_ clock: ActivityTestClock, _ delay: Duration, _ count: Int) async -> Bool {
    for _ in 0..<400 {
        if await clock.pending(delay) == count { return true }
        try? await Task.sleep(for: .milliseconds(5))
    }
    return false
}
private func countsOf(_ aggregator: ActivityAggregator, _ rows: [ModelRow], _ id: String) async -> [Int]? {
    await aggregator.snapshot(rows: rows, memory: .init()).models.first { $0.id == id }.map { [$0.requests.count, $0.running, $0.queued] }
}

/// Review #1: a request runs only once its runner says it started. The service handing it to a serial runner is not a
/// start, and cancelling the running one does not promote the next before the runner reports it.
@Test func onlyARunnerReportedStartCountsAsRunning() async {
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { 100 })
    let rows = [ModelRow(id: "d", name: "D", kind: "decisions", states: ["bf16": .init(state: .loaded)])]
    let first = aggregator.begin(model: "d", op: "decide", requester: "Fixture"), second = aggregator.begin(model: "d", op: "decide", requester: "Fixture")
    aggregator.phase(first, "step"); aggregator.phase(second, "step")
    #expect(await countsOf(aggregator, rows, "d") == [2, 0, 2])
    #expect(await aggregator.snapshot(rows: rows, memory: .init()).models.first?.requests.allSatisfy { $0.phase == "queued" } == true)
    aggregator.receive(first, progressFrame(["phase": "prefill"]))
    #expect(await countsOf(aggregator, rows, "d") == [2, 1, 1])
    aggregator.finish(first, .cancelled)
    #expect(await countsOf(aggregator, rows, "d") == [1, 0, 1])
    aggregator.receive(second, progressFrame(["phase": "prefill"]))
    #expect(await countsOf(aggregator, rows, "d") == [1, 1, 0])
    aggregator.finish(second)
}

/// Review #2: a request waiting for its model to load waits; the model's own loading state stays separate.
@Test func requestsWaitingForAModelLoadCountAsWaiting() async {
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { 100 })
    let rows = [ModelRow(id: "c", name: "C", kind: "chat", states: ["bf16": .init(state: .loading)])]
    let id = aggregator.begin(model: "c", op: "chat", requester: "Fixture")
    aggregator.phase(id, "loading")
    #expect(await countsOf(aggregator, rows, "c") == [1, 0, 1])
    #expect(await aggregator.snapshot(rows: rows, memory: .init()).models.first?.state == "loading")
    aggregator.finish(id, .cancelled)
}

/// Review #3: a voice design's audio is inside its preview.
@Test func aVoiceDesignsPreviewAudioIsCounted() async {
    let time = ActivityTestTime(20)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let rows = [ModelRow(id: "v", name: "V", kind: "voices", states: ["bf16": .init(state: .loaded)])]
    let id = aggregator.begin(model: "v", op: "design_voice", requester: "Fixture")
    aggregator.receive(id, progressFrame(["phase": "step"]))
    aggregator.finish(id, .finished(["voice_id": "preview-1", "preview": ["duration_s": 2.5, "sample_rate": 24_000, "path": "/tmp/p.wav"], "saved": false]))
    time.value = 21
    let recent = await aggregator.snapshot(rows: rows, memory: .init()).models.first?.recent
    #expect(recent?.finished == 1 && recent?.audioS == 2.5)
}

/// Review #4: audio seconds use each request's own sample rate (announced by its stream or returned with its result),
/// never one learned from an earlier request.
@Test func audioSecondsUseEachRequestsOwnSampleRate() async {
    let time = ActivityTestTime(30)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let rows = [ModelRow(id: "v", name: "V", kind: "voices", states: ["bf16": .init(state: .loaded)])]
    func samples(_ id: UUID, _ n: Int) { aggregator.receive(id, progressFrame(["phase": "step", "audio_samples": .number(Double(n))])) }
    func live() async -> Double? { await aggregator.snapshot(rows: rows, memory: .init()).models.first?.requests.first?.audioS }
    // Streamed at 24 kHz: the stream announces its rate, so audio counts while it plays.
    let first = aggregator.begin(model: "v", op: "speak", requester: "Fixture")
    aggregator.receive(first, .json(["event": "audio_begin", "stream": 1, "sample_rate": 24_000, "channels": 1, "encoding": "pcm_s16le"]))
    samples(first, 24_000)
    #expect(await live() == 1.0)
    samples(first, 48_000)
    aggregator.finish(first, .finished(["duration_s": 2.0, "sample_rate": 24_000]))
    // A 48 kHz file job: no stream, so its rate arrives with its result; 48,000 samples are one second, not two.
    let second = aggregator.begin(model: "v", op: "speak", requester: "Fixture")
    samples(second, 48_000)
    #expect(await live() == nil)
    aggregator.finish(second, .finished(["duration_s": 1.0, "sample_rate": 48_000]))
    // Streamed at 48 kHz.
    let third = aggregator.begin(model: "v", op: "speak", requester: "Fixture")
    aggregator.receive(third, .json(["event": "audio_begin", "stream": 2, "sample_rate": 48_000, "channels": 1, "encoding": "pcm_s16le"]))
    samples(third, 48_000)
    #expect(await live() == 1.0)
    aggregator.finish(third, .finished(["duration_s": 1.0, "sample_rate": 48_000]))
    time.value = 31
    let recent = await aggregator.snapshot(rows: rows, memory: .init()).models.first?.recent
    #expect(recent?.finished == 3 && recent?.audioS == 4.0)
}

/// Review #5: a tool window following its own request (progress only) never starts the machine readings; the Activity
/// panel does, and closing it stops them while the tool window keeps its progress.
@Test(.timeLimit(.minutes(1))) func aProgressOnlySubscriberNeverReadsTheMachine() async throws {
    let clock = ActivityTestClock()
    let aggregator = ActivityAggregator(sample: { .init(thermal: "nominal") }, machineInterval: .seconds(1), sleep: { try await clock.sleep($0) })
    let row = ModelRow(id: "m", name: "M", kind: "chat", states: ["bf16": .init(state: .loaded)])
    let tool = await aggregator.stream(machine: false) { await aggregator.snapshot(rows: [row], memory: .init()) }
    let toolConsumer = Task { for await _ in tool {} }
    defer { toolConsumer.cancel() }
    let request = aggregator.begin(model: "m", op: "chat", requester: "Tool")
    defer { aggregator.finish(request) }
    #expect(await reached(clock, .milliseconds(250), 1))
    #expect(await clock.pending(.seconds(1)) == 0)
    #expect(await aggregator.sampleCount == 0)
    let panel = await aggregator.stream(machine: true) { await aggregator.snapshot(rows: [row], memory: .init()) }
    let panelConsumer = Task { for await _ in panel {} }
    #expect(await reached(clock, .seconds(1), 1))
    #expect(await aggregator.sampleCount == 1)
    panelConsumer.cancel(); await panelConsumer.value
    #expect(await reached(clock, .seconds(1), 0))
    let ticks = await aggregator.requestTickCount
    await clock.fire(.milliseconds(250))
    #expect(await reached(clock, .milliseconds(250), 1))
    #expect(await aggregator.requestTickCount == ticks + 1)
    #expect(await aggregator.sampleCount == 1)
    #expect(await aggregator.subscriberCount == 1)
}

/// Review #6: a model's minute, batch and audio history go when the minute has passed or the model is gone.
@Test func endedMinutesAndRemovedModelsLeaveNoHistory() async {
    let time = ActivityTestTime(10)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let a = ModelRow(id: "a", name: "A", kind: "chat", states: ["bf16": .init(state: .loaded)])
    let b = ModelRow(id: "b", name: "B", kind: "voices", states: ["bf16": .init(state: .loaded)])
    let chat = aggregator.begin(model: "a", op: "chat", requester: "Fixture")
    aggregator.receive(chat, progressFrame(["phase": "decode", "generated_tokens": 5, "batch": 2]))
    aggregator.finish(chat)
    let speech = aggregator.begin(model: "b", op: "speak", requester: "Fixture")
    aggregator.receive(speech, progressFrame(["phase": "step", "audio_samples": 24_000]))
    aggregator.finish(speech, .finished(["duration_s": 1.0, "sample_rate": 24_000]))
    time.value = 11
    #expect(await aggregator.snapshot(rows: [a, b], memory: .init()).models.compactMap(\.recent).count == 2)
    #expect(await aggregator.trackedModels == 2)
    _ = await aggregator.snapshot(rows: [a], memory: .init())
    #expect(await aggregator.trackedModels == 1)
    time.value = 80
    #expect(await aggregator.snapshot(rows: [a], memory: .init()).models.first?.recent == nil)
    #expect(await aggregator.trackedModels == 0)
}

/// Requests begun, started and ended from many threads at once keep exact counts.
@Test func concurrentRequestsKeepExactCounts() async {
    let time = ActivityTestTime(500.25)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let rows = [ModelRow(id: "d", name: "D", kind: "decisions", states: ["bf16": .init(state: .loaded)])]
    // Per request, by i % 5: 0 waits, 1 runs, 2 finishes, 3 fails, 4 is cancelled before it starts.
    DispatchQueue.concurrentPerform(iterations: 400) { i in
        let id = aggregator.begin(model: "d", op: "decide", requester: "Fixture")
        aggregator.phase(id, "step")
        if i % 5 == 4 { aggregator.finish(id, .cancelled); return }
        guard i % 5 != 0 else { return }
        aggregator.receive(id, progressFrame(["phase": "prefill"]))
        if i % 5 == 2 { aggregator.finish(id, .finished(["answers": [:]])) }
        if i % 5 == 3 { aggregator.finish(id, .failed) }
    }
    time.value = 501.5
    let model = await aggregator.snapshot(rows: rows, memory: .init()).models.first
    #expect(model?.requests.count == 160 && model?.running == 80 && model?.queued == 80)
    #expect(model?.recent?.finished == 80 && model?.recent?.failed == 80)
}

/// A request that fails (a crashed or timed-out runner) stops running at once and counts as failed.
@Test func aFailedRequestStopsRunningAndCounts() async {
    let time = ActivityTestTime(40)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let rows = [ModelRow(id: "c", name: "C", kind: "chat", states: ["bf16": .init(state: .loaded)])]
    let id = aggregator.begin(model: "c", op: "chat", requester: "Fixture")
    aggregator.receive(id, progressFrame(["phase": "decode", "generated_tokens": 3]))
    #expect(await countsOf(aggregator, rows, "c") == [1, 1, 0])
    aggregator.finish(id, .failed)
    #expect(await countsOf(aggregator, rows, "c") == [0, 0, 0])
    time.value = 41
    let recent = await aggregator.snapshot(rows: rows, memory: .init()).models.first?.recent
    #expect(recent?.failed == 1 && recent?.finished == 0 && recent?.tokens == 3)
}

private struct CrashingFixtureArtifact: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        var launch = runtimeFixtureLaunch()
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let code = """
            import os,sys
            sys.path.insert(0, \(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers").path)))
            from arsenal_worker import Worker
            w=Worker('chat',runner='native')
            w.op('load')(lambda r: {'memory':{'footprint_mb':256,'mlx_active_mb':0,'mlx_cache_mb':0}})
            @w.op('chat')
            def chat(r):
                w.emit(r,'progress',phase='decode',prompt_tokens=10,cached_tokens=0,generated_tokens=2)
                os._exit(3)
            w.run()
            """
        launch.arguments = ["-I", "-c", code]; return launch
    }
}

/// A runner that crashes mid-request: the request leaves Activity and counts as failed, nothing stays running.
@Test func aCrashedRunnersRequestLeavesActivityAsFailed() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: CrashingFixtureArtifact())
    defer { fixture.stop() }
    try await fixture.service.get(id: "fixture-chat", precision: "bf16")
    try await fixture.service.load(id: "fixture-chat")
    await #expect(throws: (any Error).self) {
        try await fixture.service.run(id: "fixture-chat", op: "chat", params: ["messages": [["role": "user", "content": "Hi"]]])
    }
    let aggregator = await fixture.service.activityAggregator
    #expect(aggregator.requestCount == 0)
    // The model is no longer resident; its minute still holds the failure for when it is shown again.
    let rows = await fixture.service.rows().map { row -> ModelRow in
        var row = row; row.states[row.selectedPrecision] = .init(state: .loaded); return row
    }
    try await Task.sleep(for: .milliseconds(1100))
    let model = await aggregator.snapshot(rows: rows, memory: .init()).models.first { $0.id == "fixture-chat" }
    #expect(model?.running == 0 && model?.queued == 0 && model?.recent?.failed == 1)
    await fixture.service.shutdown()
}

private actor ActivityTestPublished {
    var values: [ActivitySnapshot] = []
    func add(_ value: ActivitySnapshot) { values.append(value) }
}

/// Design review #9: on a fast queue the counts and the minute change once a second, with the machine reading; the
/// request clock in between carries progress (request lines, step bars) with the counts held.
@Test(.timeLimit(.minutes(1))) func countsChangeOnceASecondWhileProgressFlows() async throws {
    let clock = ActivityTestClock(), time = ActivityTestTime(200)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await clock.sleep($0) }, now: { time.value })
    let row = ModelRow(id: "c", name: "C", kind: "chat", states: ["bf16": .init(state: .loaded)])
    let ids = (0..<10).map { _ in aggregator.begin(model: "c", op: "chat", requester: "Fixture") }
    aggregator.receive(ids[0], progressFrame(["phase": "decode", "generated_tokens": 1]))
    let seen = ActivityTestPublished()
    let stream = await aggregator.stream { await aggregator.snapshot(rows: [row], memory: .init()) }
    let consumer = Task { for await value in stream { await seen.add(value) } }
    defer { consumer.cancel() }
    func published(_ count: Int) async -> ModelActivity? {
        for _ in 0..<4000 { if await aggregator.publishCount >= count { break }; await Task.yield() }
        for _ in 0..<4000 { if await seen.values.count >= count { break }; await Task.yield() }
        return await seen.values.last?.models.first
    }
    #expect(await reached(clock, .seconds(1), 1))
    #expect(await reached(clock, .milliseconds(250), 1))
    var model = await published(1)
    #expect(model?.running == 1 && model?.queued == 9)
    // A quarter second later the first is done and the second writes: its line updates, the counts wait.
    time.value = 200.25
    aggregator.finish(ids[0])
    aggregator.receive(ids[1], progressFrame(["phase": "decode", "generated_tokens": 3]))
    await clock.fire(.milliseconds(250))
    model = await published(2)
    #expect(model?.requests.count == 9 && model?.requests.first?.generatedTokens == 3)
    #expect(model?.running == 1 && model?.queued == 9 && model?.recent == nil)
    // The second's machine reading publishes everything as it is.
    time.value = 201.0
    await clock.fire(.seconds(1))
    model = await published(3)
    #expect(model?.running == 1 && model?.queued == 8 && model?.recent?.finished == 1)
    for id in ids { aggregator.finish(id, .cancelled) }
}

/// Two requests overlapping on one model are one busy stretch; a gap between requests is idle; a stretch longer than the
/// window is clamped to it.
@Test func busySecondsFollowTheUnionOfRunningRequests() async {
    let time = ActivityTestTime(300)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let rows = [ModelRow(id: "c", name: "C", kind: "chat", states: ["bf16": .init(state: .loaded)])]
    let a = aggregator.begin(model: "c", op: "chat", requester: "A"), b = aggregator.begin(model: "c", op: "chat", requester: "B")
    aggregator.receive(a, progressFrame(["phase": "decode"]))
    time.value = 300.5
    aggregator.receive(b, progressFrame(["phase": "decode"]))
    time.value = 301.25
    aggregator.finish(a)
    time.value = 301.5
    aggregator.finish(b)
    let c = aggregator.begin(model: "c", op: "chat", requester: "C")
    time.value = 302.5
    aggregator.receive(c, progressFrame(["phase": "decode"]))
    time.value = 302.75
    aggregator.finish(c)
    time.value = 303.0
    let busy = await aggregator.snapshot(rows: rows, memory: .init()).models.first?.recent?.busy
    #expect(busy?.suffix(3) == [1, 0.5, 0.25])
    #expect(busy?.prefix(57).allSatisfy { $0 == 0 } == true)
}

// MARK: Round 3 (code review r2 of the activity-live lane)

/// Code review r2 #1: request boundaries obey the once-a-second counts too. Ten short requests one after another inside a
/// second publish once at the first start and once, coalesced, when the second is up; never once per start and end.
@Test(.timeLimit(.minutes(1))) func requestBoundariesPublishAtMostOnceASecond() async throws {
    let clock = ActivityTestClock(), time = ActivityTestTime(100)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(30), sleep: { try await clock.sleep($0) }, now: { time.value })
    let row = ModelRow(id: "d", name: "D", kind: "decisions", states: ["bf16": .init(state: .loaded)])
    let seen = ActivityTestPublished()
    let stream = await aggregator.stream { await aggregator.snapshot(rows: [row], memory: .init()) }
    let consumer = Task { for await value in stream { await seen.add(value) } }
    defer { consumer.cancel() }
    func settle() async { for _ in 0..<300 { await Task.yield() } }
    func published(_ count: Int) async -> Int {
        for _ in 0..<4000 { if await aggregator.publishCount >= count { break }; await Task.yield() }
        for _ in 0..<4000 { if await seen.values.count >= count { break }; await Task.yield() }
        return await aggregator.publishCount
    }
    #expect(await reached(clock, .seconds(30), 1))
    #expect(await published(1) == 1)
    for i in 0..<10 {
        time.value = 101.5 + Double(i) * 0.04
        let id = aggregator.begin(model: "d", op: "decide", requester: "Fixture")
        await settle()
        aggregator.receive(id, progressFrame(["phase": "decide"]))
        time.value += 0.02
        aggregator.finish(id)
        await settle()
    }
    // The first start published at once; nine more starts and ten ends inside the same second did not.
    #expect(await aggregator.publishCount == 2)
    // The request clock carries one final publish once the second is up, then stops.
    #expect(await reached(clock, .milliseconds(250), 1))
    await clock.fire(.milliseconds(250))
    await settle()
    #expect(await aggregator.publishCount == 2)
    time.value = 102.6
    #expect(await reached(clock, .milliseconds(250), 1))
    await clock.fire(.milliseconds(250))
    #expect(await published(3) == 3)
    #expect(await reached(clock, .milliseconds(250), 0))
    let last = await seen.values.last?.models.first
    #expect(last?.state == "idle" && last?.running == 0 && last?.queued == 0 && last?.recent?.finished == 10)
}

/// Code review r2 #2: a result that corrects a request's sample rate rescales only that request's audio still in the
/// minute. Its expired seconds are not taken out of the ones that remain, and another request's audio is untouched.
@Test func aCorrectedSampleRateRescalesOnlyThatRequestsRetainedAudio() async {
    let time = ActivityTestTime(1000)
    let aggregator = ActivityAggregator(sample: { .init() }, machineInterval: .seconds(1), sleep: { try await Task.sleep(for: $0) }, now: { time.value })
    let rows = [ModelRow(id: "v", name: "V", kind: "voices", states: ["bf16": .init(state: .loaded)])]
    // Announced at 24 kHz, really 48 kHz: 80 s of audio at 48,000 samples a second count double while they stream.
    let long = aggregator.begin(model: "v", op: "speak", requester: "Fixture")
    aggregator.receive(long, .json(["event": "audio_begin", "stream": 1, "sample_rate": 24_000, "channels": 1]))
    for second in 1...80 {
        time.value = 999.5 + Double(second)
        aggregator.receive(long, progressFrame(["phase": "step", "audio_samples": .number(Double(second * 48_000))]))
    }
    // Another request's second of audio, in the last second of the long one.
    let short = aggregator.begin(model: "v", op: "speak", requester: "Fixture")
    aggregator.receive(short, .json(["event": "audio_begin", "stream": 2, "sample_rate": 24_000, "channels": 1]))
    aggregator.receive(short, progressFrame(["phase": "step", "audio_samples": 24_000]))
    aggregator.finish(short, .finished(["duration_s": 1.0, "sample_rate": 24_000]))
    time.value = 1080
    aggregator.finish(long, .finished(["duration_s": 80.0, "sample_rate": 48_000]))
    time.value = 1081
    // The minute is seconds 1021…1080: 59 of the long request's seconds at 1 s each, and the short request's 1 s.
    let audio = await aggregator.snapshot(rows: rows, memory: .init()).models.first?.recent?.audioS ?? 0
    #expect(abs(audio - 60) < 1e-9, "audio \(audio)")
}
