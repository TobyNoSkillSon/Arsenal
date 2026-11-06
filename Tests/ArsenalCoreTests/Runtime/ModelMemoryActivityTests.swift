// SPDX-License-Identifier: MIT
import Foundation
import Testing
import os
import ArsenalWire
@testable import ArsenalCore

/// What a model's memory is for (SeedVR2 showed 8.4 GB while each batch took 23–29 GB more). Activity splits a
/// runner's memory with MLX's own counters and shows the most each job took, from the supervisor's samples and the
/// kernel's footprint maximum over the job.
@Suite struct ModelMemoryActivityTests {
    @Test func aRunnerReportingItsWeightsSplitsWeightsWorkingAndCache() throws {
        let m = try #require(
            ModelMemoryActivity(WorkerMemory(footprintMB: 30_000, mlxActiveMB: 28_000, mlxCacheMB: 1_200, mlxWeightsMB: 8_400, mlxPeakMB: 36_000, jobPeakMB: 31_000)))
        #expect(m.source == "mlx"); #expect(m.totalMB == 30_000)
        #expect(m.weightsMB == 8_400); #expect(m.workingMB == 19_600); #expect(m.cacheMB == 1_200); #expect(m.arraysMB == nil)
        // The peak is the process's footprint peak, never MLX's counter (which also counts allocator memory the process
        // does not hold); MLX's figure is still carried apart.
        #expect(m.peakMB == 31_000); #expect(m.mlxPeakMB == 36_000); #expect(m.unsplitMB == nil)
    }

    /// 1.0.2 showed MLX's peak (41,165 MB) for a SeedVR2 clip whose worker's kernel maximum footprint was 35,620 MB.
    @Test func thePeakIsTheFootprintPeakNotMLXsCounter() throws {
        let seedvr = try #require(
            ModelMemoryActivity(WorkerMemory(footprintMB: 8_972, mlxActiveMB: 8_438, mlxCacheMB: 277, mlxWeightsMB: 8_438, mlxPeakMB: 41_165, jobPeakMB: 35_620)))
        #expect(seedvr.peakMB == 35_620); #expect(seedvr.mlxPeakMB == 41_165)
        // MLX's counter alone (no job window yet) sets no peak.
        #expect(ModelMemoryActivity(WorkerMemory(footprintMB: 9_000, mlxActiveMB: 8_400, mlxCacheMB: 0, mlxPeakMB: 20_000))?.peakMB == nil)
        // Still never below what the process holds now.
        var live = WorkerMemory(footprintMB: 9_000, mlxActiveMB: 8_400, mlxCacheMB: 0, mlxPeakMB: 30_000, jobPeakMB: 12_000)
        live.liveFootprintMB = 13_000
        #expect(ModelMemoryActivity(live)?.peakMB == 13_000)
    }

    /// The kernel's footprint interval: a plausible figure for a running process, never below its footprint, never
    /// falling between two readings, and not above its lifetime maximum.
    @Test func theKernelFootprintIntervalIsReadableAndResettable() throws {
        let child = Process(); child.executableURL = URL(fileURLWithPath: "/bin/sleep"); child.arguments = ["60"] // outlives the test under full-suite load; terminated below
        try child.run(); defer { child.terminate(); child.waitUntilExit() }
        let pid = child.processIdentifier
        // Under full-suite load the child can still be mid-exec (81,992 bytes on 6 Nov): wait for a running sleep.
        let deadline = Date().addingTimeInterval(5)
        while WorkerMemoryCeiling.footprint(pid: pid) <= 100_000, Date() < deadline { usleep(20_000) }
        #expect(WorkerMemoryCeiling.resetFootprintInterval(pid: pid), "the supervisor may reset its own child's interval")
        let now = WorkerMemoryCeiling.footprint(pid: pid), first = WorkerMemoryCeiling.intervalMaxFootprint(pid: pid)
        #expect(first > 100_000 && first < 1_000_000_000, "\(first) bytes")
        #expect(first >= now)
        #expect(WorkerMemoryCeiling.intervalMaxFootprint(pid: pid) >= first)
        #expect(!WorkerMemoryCeiling.resetFootprintInterval(pid: 0) && WorkerMemoryCeiling.intervalMaxFootprint(pid: 0) == 0)
    }

    /// Each job's peak covers that job alone: a spike between two samples (none is taken here) still counts, from the
    /// kernel's maximum, and the next job starts a new interval rather than inheriting it.
    @Test func eachJobsPeakIsTheKernelMaximumOverThatJob() async throws {
        let lives = OSAllocatedUnfairLock(initialState: [WorkerMemory]())
        let worker = WorkerProcess(memorySampleSeconds: 60, macMemory: MacMemoryGuard(limitBytes: .max, inUseBytes: { 0 }))
        try await worker.start(stubLaunch(), live: { m in lives.withLock { $0.append(m) } })
        defer { Task { await worker.unload(grace: 0.2) } }
        _ = try await worker.request(op: "load", params: ["weights_dir": "/tmp"], timeout: 10)
        let before = Double(WorkerMemoryCeiling.footprint(pid: await worker.pid)) / 1e6
        _ = try await worker.request(op: "test_spike", params: ["mb": 256, "hold_ms": 50], timeout: 30)
        let spiked = try #require(await worker.memory.jobPeakMB)
        #expect(spiked >= before + 250, "peak \(spiked) MB, before \(before) MB")
        #expect(lives.withLock { $0.last?.jobPeakMB } == spiked, "the job's final peak reaches Activity")
        let after = Double(WorkerMemoryCeiling.footprint(pid: await worker.pid)) / 1e6
        #expect(after < spiked - 200, "the spike was given back: \(after) MB")
        _ = try await worker.request(op: "test_spike", params: ["mb": 0], timeout: 30)
        let next = try #require(await worker.memory.jobPeakMB)
        #expect(next < spiked - 200, "the next job must not inherit the last one's spike: \(next) MB after \(spiked) MB")
        // Bookkeeping after a job neither opens a window nor raises the closed one from the kernel.
        _ = try await worker.request(op: "count_tokens", params: ["text": "a b"], timeout: 10)
        #expect(await worker.memory.jobPeakMB == next)
    }

    /// Review finding: a job that starts while another is still pending must not restart the kernel interval, or the
    /// first job's spike would vanish from the shared window.
    @Test func aJobStartingDuringAnotherKeepsTheFirstJobsSpike() async throws {
        let worker = WorkerProcess(memorySampleSeconds: 60, macMemory: MacMemoryGuard(limitBytes: .max, inUseBytes: { 0 }))
        try await worker.start(stubLaunch(), live: { _ in })
        defer { Task { await worker.unload(grace: 0.2) } }
        _ = try await worker.request(op: "load", params: ["weights_dir": "/tmp"], timeout: 10)
        let before = Double(WorkerMemoryCeiling.footprint(pid: await worker.pid)) / 1e6
        async let first = worker.request(op: "test_spike", params: ["mb": 256, "hold_ms": 300], timeout: 30)
        try await Task.sleep(for: .milliseconds(100))
        async let second = worker.request(op: "test_spike", params: ["mb": 0], timeout: 30)
        _ = try await (first, second)
        let peak = try #require(await worker.memory.jobPeakMB)
        #expect(peak >= before + 250, "the overlapping second job erased the first job's spike: \(peak) MB, before \(before) MB")
    }

    /// Review finding: a sample taken after the runner's reading must not be split with that reading's MLX figures. The
    /// growth since is its own part, and the total is the sample.
    @Test func aNewerSampleAddsOnlyWhatGrewSinceTheRunnersReading() throws {
        var reading = WorkerMemory(footprintMB: 9_000, mlxActiveMB: 8_400, mlxCacheMB: 100, mlxWeightsMB: 8_400, jobPeakMB: 12_500)
        reading.liveFootprintMB = 12_000
        let m = try #require(ModelMemoryActivity(reading))
        #expect(m.totalMB == 12_000); #expect(m.unsplitMB == 3_000); #expect(m.workingMB == 0); #expect(m.cacheMB == 100)
        #expect(m.peakMB == 12_500)
        // Shrunk since the reading: nothing unsplit, and the total is still the newest sample.
        reading.liveFootprintMB = 8_800
        #expect(ModelMemoryActivity(reading)?.unsplitMB == nil); #expect(ModelMemoryActivity(reading)?.totalMB == 8_800)
    }

    @Test func withoutWeightsMLXMemoryIsOnePieceAndAProcessOnlyRunnerIsItsFootprint() throws {
        let arrays = try #require(ModelMemoryActivity(WorkerMemory(footprintMB: 2_000, mlxActiveMB: 1_500, mlxCacheMB: 100)))
        #expect(arrays.source == "mlx"); #expect(arrays.arraysMB == 1_500); #expect(arrays.weightsMB == nil); #expect(arrays.workingMB == nil)
        #expect(arrays.peakMB == nil, "no job yet: no peak")
        let process = try #require(ModelMemoryActivity(WorkerMemory(footprintMB: 3_000, jobPeakMB: 4_200)))
        #expect(process.source == "process"); #expect(process.cacheMB == nil); #expect(process.peakMB == 4_200)
        // A peak is never below what is held now.
        #expect(ModelMemoryActivity(WorkerMemory(footprintMB: 5_000, jobPeakMB: 4_200))?.peakMB == 5_000)
        // Weights above what MLX holds now (the runner freed some) leave no negative working memory.
        #expect(ModelMemoryActivity(WorkerMemory(footprintMB: 900, mlxActiveMB: 300, mlxCacheMB: 0, mlxWeightsMB: 400))?.workingMB == 0)
        #expect(ModelMemoryActivity(WorkerMemory()) == nil, "nothing loaded: no breakdown")
    }

    @Test func olderRunnersDecodeAndTheSnapshotCarriesTheBreakdown() throws {
        let old = try JSONDecoder().decode(WorkerMemory.self, from: Data(#"{"footprint_mb":120.5,"mlx_active_mb":80.0,"mlx_cache_mb":8.0}"#.utf8))
        #expect(old.mlxWeightsMB == nil && old.mlxPeakMB == nil && old.jobPeakMB == nil)
        let new = try JSONDecoder().decode(
            WorkerMemory.self, from: Data(#"{"footprint_mb":9013,"mlx_active_mb":8437,"mlx_cache_mb":252,"mlx_weights_mb":8437,"mlx_peak_mb":31000}"#.utf8))
        #expect(new.mlxWeightsMB == 8437 && new.mlxPeakMB == 31000)
        let activity = ModelActivity(id: "seedvr2-3b", kind: "restoration", memoryMB: 9013, memory: ModelMemoryActivity(new))
        let json = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(activity)) as? [String: Any])
        let memory = try #require(json["memory"] as? [String: Any])
        #expect(memory["source"] as? String == "mlx"); #expect(memory["weights_mb"] as? Double == 8437); #expect(memory["working_mb"] as? Double == 0)
        #expect(memory["cache_mb"] as? Double == 252)
        // MLX's counter alone sets no peak; it is carried apart.
        #expect(memory["peak_mb"] == nil); #expect(memory["mlx_peak_mb"] as? Double == 31000)
    }

    /// The supervisor's samples during a job reach the live sink (Activity), never the status sink (which makes the menu
    /// re-read everything); the job's peak survives the runner's own closing status.
    @Test func jobReadingsAreLiveAndTheJobPeakIsSampled() async throws {
        let statuses = OSAllocatedUnfairLock(initialState: [WorkerMemory]()), lives = OSAllocatedUnfairLock(initialState: [WorkerMemory]())
        let worker = WorkerProcess(memorySampleSeconds: 0.05, macMemory: MacMemoryGuard(limitBytes: .max, inUseBytes: { 0 }))
        try await worker.start(
            stubLaunch(), status: { m in statuses.withLock { $0.append(m) } }, live: { m in lives.withLock { $0.append(m) } })
        defer { Task { await worker.unload(grace: 0.2) } }
        _ = try await worker.request(op: "load", params: ["weights_dir": "/tmp"], timeout: 10)
        let afterLoad = statuses.withLock { $0.count }
        let before = WorkerMemoryCeiling.footprint(pid: await worker.pid)
        _ = try await worker.request(op: "test_grow", params: ["chunk_mb": 32, "interval_ms": 5, "max_mb": 256, "hold_ms": 400], timeout: 30)
        #expect(statuses.withLock { $0.count } == afterLoad, "a job's samples must not notify the menu")
        let sampled = try #require(lives.withLock { $0.last })
        let peak = try #require(sampled.jobPeakMB)
        #expect(peak >= Double(before) / 1e6 + 200, "peak \(peak) MB, before \(Double(before) / 1e6) MB")
        // A runner's live reading arrives on the live sink with its MLX parts.
        _ = try await worker.request(op: "test_live_status", timeout: 10)
        let live = try #require(lives.withLock { $0.last })
        #expect(live.mlxWeightsMB == 400 && live.mlxPeakMB == 820 && live.jobPeakMB ?? 0 >= peak - 50, "a new window starts from what the stub still holds")
        #expect(statuses.withLock { $0.count } == afterLoad)
        // A runner's ordinary status still notifies, and keeps the peak.
        _ = try await worker.request(op: "status", timeout: 10)
        #expect(statuses.withLock { $0.count } == afterLoad + 1)
        // A new job opens a new peak window from what the worker holds now: the last job's MLX peak is dropped, and the
        // samples stay apart from the runner's own reading.
        _ = try await worker.request(op: "test_live_status", timeout: 10)
        let reported = try #require(lives.withLock { $0.last })
        #expect(reported.mlxPeakMB == 820 && reported.footprintMB == 900)
        _ = try await worker.request(op: "test_grow", params: ["chunk_mb": 16, "interval_ms": 5, "max_mb": 64, "hold_ms": 300], timeout: 30)
        let sample = try #require(lives.withLock { $0.last })
        #expect(sample.mlxPeakMB == nil, "a new window must not keep the previous job's MLX peak")
        #expect(sample.footprintMB == 900, "a sample never overwrites the runner's reading")
        #expect((sample.liveFootprintMB ?? 0) >= Double(before) / 1e6 + 200, "the stub keeps what it grew")
        // Control and bookkeeping operations open no window.
        #expect(!WorkerProcess.isJob("count_tokens") && !WorkerProcess.isJob("measurement_logits") && !WorkerProcess.isJob("unload"))
        // The voice list the app sends to Voices is bookkeeping: it must not reset the last speech job's peak.
        #expect(!WorkerProcess.isJob("voices"))
        #expect(WorkerProcess.isJob("convert") && WorkerProcess.isJob("upscale_video") && WorkerProcess.isJob("load"))
    }
}
