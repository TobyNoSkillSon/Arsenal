// SPDX-License-Identifier: MIT
import Foundation
import Testing
import Darwin
import os
import ArsenalWire
@testable import ArsenalCore

/// A simulated Mac for the Mac-wide memory stop: memory in use is the footprints of the watched processes (stub workers
/// and an outside process) plus `offset`, which starts at minus their footprints at setup, so the limit reads as growth.
final class SimulatedMac: Sendable {
    private let state = OSAllocatedUnfairLock(initialState: (pids: [Int32](), offset: Int64(0)))
    func watch(_ pid: Int32) { state.withLock { $0.pids.append(pid) } }
    /// From now on, memory in use counts only growth beyond the current footprints, plus `extra` (another app).
    func zero(extra: Int64 = 0) {
        let now = Int64(raw())
        state.withLock { $0.offset = extra - now }
    }
    func setExtra(_ extra: Int64, baseline: UInt64) { state.withLock { $0.offset = extra - Int64(baseline) } }
    func raw() -> UInt64 { state.withLock { $0.pids }.reduce(0) { $0 + WorkerMemoryCeiling.footprint(pid: $1) } }
    var inUse: UInt64 { UInt64(max(0, Int64(raw()) + state.withLock { $0.offset })) }
    func guardWithLimit(megabytes: Double) -> MacMemoryGuard { MacMemoryGuard(limitBytes: UInt64(megabytes * 1e6), inUseBytes: { [self] in inUse }) }
}

/// Waits until the process has grown by `megabytes` (a request's growth phase is about half a second).
private func waitForGrowth(_ pid: Int32, from start: UInt64, megabytes: Double) async throws {
    let deadline = ContinuousClock.now + .seconds(20)
    while WorkerMemoryCeiling.footprint(pid: pid) < start + UInt64(megabytes * 1e6) {
        guard ContinuousClock.now < deadline else { throw WorkerFailure(code: "fixture", message: "Stub worker did not grow in time.") }
        try await Task.sleep(for: .milliseconds(20))
    }
}

@Suite struct WorkerMemoryCeilingTests {
    private func requirement(_ gb: Double?, basis: LoadMemoryRequirement.Basis = .forecast) -> LoadMemoryRequirement {
        var value = LoadMemoryRequirement(row: nil, precision: "fp16", forecast: nil)
        value.neededGB = gb; value.basis = gb == nil ? .unknown : basis
        return value
    }

    /// The per-worker bound is only a runaway backstop: from a measured request peak (4×, at least 16 GB above it), or
    /// a Load Anyway resident's expected peak (2×, at least 4 GB above). A Load peak or forecast describes no request,
    /// so it sets no per-worker bound: those models stop only at the Mac-wide limit.
    @Test func runawayBackstopNeedsAMeasuredRequestPeak() {
        let measured = WorkerMemoryCeiling(name: "TeleOCR", requirement: requirement(1.8, basis: .peak), precision: "int4", forced: false)
        #expect(measured.runawayMB == 17_800); #expect(measured.measuredPeak); #expect(!measured.forced)
        let large = WorkerMemoryCeiling(name: "Big", requirement: requirement(30, basis: .peak), precision: "fp16", forced: false)
        #expect(large.runawayMB == 120_000)
        for basis in [LoadMemoryRequirement.Basis.loadPeak, .forecast, .resident] {
            let unmeasured = WorkerMemoryCeiling(name: "SAM 3.1", requirement: requirement(1.8, basis: basis), precision: "fp16", forced: false)
            #expect(unmeasured.runawayMB == nil); #expect(unmeasured.expectedMB == 1800)
        }
        #expect(WorkerMemoryCeiling(name: "New", requirement: requirement(nil), precision: nil, forced: false).runawayLimitBytes == nil)
        let forced = WorkerMemoryCeiling(name: "TeleOCR", requirement: requirement(3.5), precision: "fp16", forced: true)
        #expect(forced.forced); #expect(forced.runawayMB == 7500); #expect(!forced.measuredPeak)
        #expect(WorkerMemoryCeiling(name: "New", requirement: requirement(nil), precision: nil, forced: true).runawayMB == nil)
    }

    @Test func stopMessagesSayWhatIsTrueAndBlameNeitherInputNorOutput() {
        let runaway = WorkerMemoryCeiling(name: "TeleOCR", requirement: requirement(1.8, basis: .peak), precision: "int4", forced: false)
            .runawayFailure(footprintBytes: 18_140_000_000, limitBytes: 17_800_000_000)
        #expect(runaway.code == "memory"); #expect(runaway.details["status"]?.number == 507)
        #expect(runaway.details["reason"]?.string == "runaway")
        #expect(runaway.details["footprint_gb"]?.number == 18.14); #expect(runaway.details["ceiling_gb"]?.number == 17.8)
        #expect(
            runaway.message
                == "Arsenal stopped TeleOCR: it grew to 16.9 GB, past its 16.6 GB runaway limit (its measured peak is 1.68 GB at INT4). The model was unloaded; try again.")
        let mac = WorkerMemoryCeiling(name: "FLUX.2 Klein", expectedMB: 9_900, precision: "fp16")
            .macFailure(footprintBytes: 31_000_000_000, startBytes: 9_000_000_000, inUseBytes: 108_200_000_000, limitBytes: 107_370_000_000)
        #expect(mac.details["reason"]?.string == "mac_memory_limit")
        #expect(mac.details["mac_in_use_gb"]?.number == 108.2); #expect(mac.details["mac_limit_gb"]?.number == 107.37)
        #expect(mac.details["start_gb"]?.number == 9); #expect(mac.details["expected_gb"]?.number == 9.9)
        #expect(
            mac.message
                == "Arsenal stopped FLUX.2 Klein: the Mac had 101 GB of memory in use, past its 100 GB limit (of this Mac's 128 GB), "
                + "and FLUX.2 Klein had grown from 8.38 GB to 28.9 GB during this request (it is expected to need about 9.22 GB at FP16). "
                + "The model was unloaded; try again when more memory is free.")
        for failure in [runaway, mac] { #expect(!failure.message.contains("output")); #expect(!failure.message.contains("input")) }
    }

    /// The choice itself, on recorded footprints: only workers that grew during their requests and were not loaded with
    /// Load Anyway; the one furthest past its expected peak plus 1 GB (its start without a peak); one stop at a time.
    @Test func macWideDecisionPicksTheWorkerFurthestPastItsExpectedPeak() {
        func member(_ name: String, expected: Double?, start: Double, now: Double, forced: Bool = false) -> MacMemoryGuard.Member {
            MacMemoryGuard.Member(
                ceiling: WorkerMemoryCeiling(name: name, expectedMB: expected.map { $0 * 1000 }, forced: forced), startBytes: UInt64(start * 1e9),
                footprintBytes: UInt64(now * 1e9),
                stop: { _ in })
        }
        let gb: UInt64 = 1_000_000_000
        let (a, b, c, d) = (UUID(), UUID(), UUID(), UUID())
        var state = MacMemoryGuard.State()
        state.members = [
            a: member("Within", expected: 30, start: 20, now: 29), // 2 GB short of expected + 1 GB
            b: member("Unknown", expected: nil, start: 4, now: 9), // 4 GB past its start + 1 GB
            c: member("Idle", expected: 1, start: 40, now: 40.3), // big but barely grew: not a cause
            d: member("Forced", expected: 2, start: 2, now: 30, forced: true) // Load Anyway: never chosen
        ]
        let now = Date()
        #expect(MacMemoryGuard.decide(&state, inUseBytes: 100 * gb, limitBytes: 107 * gb, now: now) == .none)
        guard case .stop(let victim, let failure) = MacMemoryGuard.decide(&state, inUseBytes: 110 * gb, limitBytes: 107 * gb, now: now) else {
            Issue.record("No worker stopped past the limit"); return
        }
        #expect(victim == b); #expect(failure.message.hasPrefix("Arsenal stopped Unknown: the Mac had 102 GB"))
        // While that stop completes, nothing else is chosen; once the worker has left, the next tick reads the Mac again.
        #expect(MacMemoryGuard.decide(&state, inUseBytes: 110 * gb, limitBytes: 107 * gb, now: now) == .none)
        state.members[b] = nil
        guard case .stop(let next, _) = MacMemoryGuard.decide(&state, inUseBytes: 110 * gb, limitBytes: 107 * gb, now: now) else {
            Issue.record("The next decision did not run"); return
        }
        #expect(next == a)
        // A stop whose worker never reported its exit stops blocking after its timeout.
        #expect(MacMemoryGuard.decide(&state, inUseBytes: 110 * gb, limitBytes: 107 * gb, now: now.addingTimeInterval(1)) == .none)
        #expect(MacMemoryGuard.decide(&state, inUseBytes: 110 * gb, limitBytes: 107 * gb, now: now.addingTimeInterval(MacMemoryGuard.stopTimeout + 1)) != .none)
        // Nobody grew: other apps filled the Mac, and no worker is blamed for it.
        var quiet = MacMemoryGuard.State()
        quiet.members = [c: member("Idle", expected: 1, start: 40, now: 40.3)]
        #expect(MacMemoryGuard.decide(&quiet, inUseBytes: 120 * gb, limitBytes: 107 * gb, now: now) == .none)
    }

    @Test func runawayWorkerIsStoppedAtItsBackstop() async throws {
        let mac = SimulatedMac()
        let worker = WorkerProcess(memorySampleSeconds: 0.05, macMemory: mac.guardWithLimit(megabytes: 1e9)); try await worker.start(stubLaunch())
        let pid = await worker.pid
        let started = Date()
        do {
            _ = try await worker.request(
                op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 10, "max_mb": 4096], timeout: 30, memoryCeiling: WorkerMemoryCeiling(name: "Stub", runawayMB: 256))
            Issue.record("A worker growing to 4 GB passed a 256 MB runaway limit")
        } catch let failure as WorkerFailure {
            #expect(failure.code == "memory")
            #expect(failure.details["reason"]?.string == "runaway")
            let footprint = failure.details["footprint_gb"]?.number ?? 0, limit = failure.details["ceiling_gb"]?.number ?? 0
            // Stopped within a few sampling periods of crossing: far below the 4 GB it would have reached.
            // The details are rounded to 0.01 GB, and the guard fires on the raw bytes (footprint > limit): rounding can make
            // the two equal, so the rounded check is >=.
            #expect(footprint >= limit); #expect(limit < 0.5); #expect(footprint < 1.0)
            // The request fails only once the worker is gone, so the caller never sees a stopped worker still running.
            #expect(await worker.exited)
        }
        #expect(Date().timeIntervalSince(started) < 10)
        #expect(await worker.exited); #expect(kill(pid, 0) == -1)
        #expect(await worker.memoryCeilingStops == 1)
        // A deliberate stop is not a crash.
        #expect(await !worker.unexpectedExit)
        await worker.unload(grace: 0.1)
    }

    /// Two models growing at once (finding 1's scenario): together they cross the Mac-wide limit, and the one furthest
    /// past its expected peak stops, even though the other is bigger; the other finishes its request.
    @Test(.timeLimit(.minutes(1))) func twoGrowingWorkersStopTheOneFurthestPastItsExpectedPeak() async throws {
        let mac = SimulatedMac(), guardian = mac.guardWithLimit(megabytes: 1500)
        let big = WorkerProcess(memorySampleSeconds: 0.05, macMemory: guardian), small = WorkerProcess(memorySampleSeconds: 0.05, macMemory: guardian)
        try await big.start(stubLaunch()); try await small.start(stubLaunch())
        let (bigPID, smallPID) = (await big.pid, await small.pid)
        mac.watch(bigPID); mac.watch(smallPID); mac.zero()
        let bigStart = WorkerMemoryCeiling.footprint(pid: bigPID)
        // "Large" grows 768 MB within its expected 2 GB peak and holds; "Small" (no known peak) then grows past its start.
        let large = Task {
            try await big.request(
                op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 10, "max_mb": 768, "hold_ms": 4000], timeout: 30,
                memoryCeiling: WorkerMemoryCeiling(name: "Large", expectedMB: 2048))
        }
        try await waitForGrowth(bigPID, from: bigStart, megabytes: 700)
        do {
            _ = try await small.request(
                op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 10, "max_mb": 1024, "hold_ms": 4000], timeout: 30,
                memoryCeiling: WorkerMemoryCeiling(name: "Small"))
            Issue.record("The Mac passed its limit and nothing stopped")
        } catch let failure as WorkerFailure {
            #expect(failure.code == "memory"); #expect(failure.details["reason"]?.string == "mac_memory_limit")
            #expect(failure.message.hasPrefix("Arsenal stopped Small: the Mac had "))
            #expect((failure.details["mac_in_use_gb"]?.number ?? 0) >= 1.5, "past the 1.5 GB limit (two decimals)")
            #expect(await small.exited)
        }
        let reply = try await large.value
        #expect(reply["grown_mb"]?.number == 768)
        #expect(await !big.exited); #expect(await big.memoryCeilingStops == 0); #expect(await small.memoryCeilingStops == 1)
        #expect(await !small.unexpectedExit)
        await big.unload(); await small.unload(grace: 0.1)
    }

    /// Another process fills the Mac while a request runs: the worker that grew is stopped, a worker that did not grow
    /// is not blamed, and the outside process is left alone.
    @Test(.timeLimit(.minutes(1))) func growingOutsideProcessStopsOnlyAWorkerThatGrew() async throws {
        let mac = SimulatedMac(), guardian = mac.guardWithLimit(megabytes: 1000)
        let grower = WorkerProcess(memorySampleSeconds: 0.05, macMemory: guardian), idle = WorkerProcess(memorySampleSeconds: 0.05, macMemory: guardian)
        // Not an Arsenal worker of this guard: another app.
        let outside = WorkerProcess(macMemory: MacMemoryGuard(limitBytes: .max, inUseBytes: { 0 }))
        for worker in [grower, idle, outside] { try await worker.start(stubLaunch()); mac.watch(await worker.pid) }
        mac.zero()
        let growerPID = await grower.pid, outsidePID = await outside.pid
        let growerStart = WorkerMemoryCeiling.footprint(pid: growerPID)
        let idleRequest = Task {
            try await idle.request(
                op: "test_grow", params: ["max_mb": 0, "hold_ms": 4000], timeout: 30, memoryCeiling: WorkerMemoryCeiling(name: "Idle", expectedMB: 64))
        }
        let growing = Task {
            try await grower.request(
                op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 10, "max_mb": 640, "hold_ms": 4000], timeout: 30,
                memoryCeiling: WorkerMemoryCeiling(name: "Grower", expectedMB: 200, precision: "int4"))
        }
        try await waitForGrowth(growerPID, from: growerStart, megabytes: 600)
        let other = Task { try await outside.request(op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 10, "max_mb": 1024, "hold_ms": 1000], timeout: 30) }
        do {
            _ = try await growing.value
            Issue.record("The Mac passed its limit and the worker that grew was not stopped")
        } catch let failure as WorkerFailure {
            #expect(failure.details["reason"]?.string == "mac_memory_limit")
            #expect(failure.message.contains("the Mac had")); #expect(failure.message.contains("Grower had grown from"))
            #expect(failure.message.contains("expected to need about 0.19 GB at INT4"))
            #expect(await grower.exited)
        }
        #expect(try await idleRequest.value["grown_mb"]?.number == 0)
        #expect(try await other.value["grown_mb"]?.number == 1024)
        #expect(await !idle.exited); #expect(await !outside.exited); #expect(kill(outsidePID, 0) == 0)
        for worker in [grower, idle, outside] { await worker.unload(grace: 0.1) }
    }

    /// The limit is re-read on every tick, never frozen at request start: a request that starts while other apps hold the
    /// Mac past its limit, and grows once they free memory, completes.
    @Test(.timeLimit(.minutes(1))) func macWideLimitFollowsTheMacRatherThanTheRequestStart() async throws {
        let mac = SimulatedMac(), guardian = mac.guardWithLimit(megabytes: 2000)
        let worker = WorkerProcess(memorySampleSeconds: 0.05, macMemory: guardian)
        try await worker.start(stubLaunch()); let pid = await worker.pid
        mac.watch(pid)
        let baseline = mac.raw()
        mac.setExtra(4_000_000_000, baseline: baseline) // other apps: 2 GB past the limit
        #expect(guardian.inUseBytes > guardian.limitBytes)
        let request = Task {
            try await worker.request(
                op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 25, "max_mb": 800, "hold_ms": 500], timeout: 30,
                memoryCeiling: WorkerMemoryCeiling(name: "Stub", expectedMB: 100))
        }
        try await Task.sleep(for: .milliseconds(150))
        mac.setExtra(0, baseline: baseline) // they quit
        #expect(try await request.value["grown_mb"]?.number == 800)
        #expect(await worker.memoryCeilingStops == 0); #expect(await !worker.exited)
        await worker.unload()
    }

    @Test func workerUnderItsLimitsCompletesAndStopsSampling() async throws {
        let mac = SimulatedMac()
        let worker = WorkerProcess(memorySampleSeconds: 0.02, macMemory: mac.guardWithLimit(megabytes: 1e9)); try await worker.start(stubLaunch())
        let reply = try await worker.request(
            op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 10, "max_mb": 96], timeout: 30, memoryCeiling: WorkerMemoryCeiling(name: "Stub", runawayMB: 1024))
        #expect(reply["grown_mb"]?.number == 96)
        try await Task.sleep(for: .milliseconds(100))
        #expect(try await worker.request(op: "ping")["ok"] == true)
        #expect(await !worker.exited); #expect(await worker.memoryCeilingStops == 0)
        await worker.unload()
    }
}

/// Get-time int8/int4 conversion runs in its own worker: it takes part in the same Mac-wide stop.
private struct GrowingConversionArtifact: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        let stub = stubLaunch(model.kind)
        let marker = paths.support.appendingPathComponent("fixture-process.pid")
        let command = "echo $$ > '\(marker.path)'; exec '\(stub.executable.path)' --kind \(model.kind) --convert-grow-mb 768"
        return WorkerLaunch(
            executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", command], kind: model.kind, runner: stub.runner, environment: stub.environment, log: stub.log)
    }
}

@Test(.timeLimit(.minutes(1))) func getTimeConversionStopsAtTheMacWideLimit() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let model = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData).models[0].manifest
    let paths = f.paths
    let guardian = MacMemoryGuard(limitBytes: 600_000_000, inUseBytes: { fixturePID(paths).map { WorkerMemoryCeiling.footprint(pid: $0) } ?? 0 })
    let converter = WorkerPrecisionConverter(artifacts: GrowingConversionArtifact(), paths: f.paths, macMemory: guardian)
    do {
        _ = try await converter.convert(
            model: model, modelDirectory: f.paths.support, sourceDirectory: f.paths.support,
            destinationDirectory: f.paths.support.appendingPathComponent("derived"), precision: "int8", recipe: "fixture")
        Issue.record("A conversion past the Mac-wide limit finished")
    } catch let failure as WorkerFailure {
        #expect(failure.code == "memory"); #expect(failure.details["reason"]?.string == "mac_memory_limit")
        #expect(failure.message.hasPrefix("Arsenal stopped the INT8 conversion of \(model.name): the Mac had "))
    }
    if let pid = fixturePID(f.paths) { #expect(kill(pid, 0) == -1) }
    await f.service.shutdown()
}

extension RuntimeTimingTests {
    @Test static func memoryCeilingStopsNeverQuarantineARunner() async throws {
        let supervisor = Supervisor(crashLimit: 2)
        for _ in 0..<3 {
            _ = try await supervisor.worker(id: "grow", launch: stubLaunch())
            do {
                _ = try await supervisor.request(
                    id: "grow", op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 10, "max_mb": 2048], timeout: 30,
                    memoryCeiling: WorkerMemoryCeiling(name: "Stub", runawayMB: 128))
                Issue.record("Runaway passed its ceiling")
            } catch let failure as WorkerFailure { #expect(failure.code == "memory") }
            #expect(await supervisor.hasExited(id: "grow"))
        }
        // Three stops with a crash limit of two: the runner still starts.
        let worker = try await supervisor.worker(id: "grow", launch: stubLaunch())
        #expect(try await worker.request(op: "ping")["ok"] == true)
        await supervisor.shutdown()
    }
}
