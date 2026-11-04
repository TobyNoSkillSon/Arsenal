// SPDX-License-Identifier: MIT
import Foundation
import Testing
import os
import ArsenalWire
@testable import ArsenalCore

/// What a model's memory is for (SeedVR2 showed 8.4 GB while each batch took 23–29 GB more). Activity splits a
/// runner's memory with MLX's own counters and shows the most each job took, from the supervisor's samples.
@Suite struct ModelMemoryActivityTests {
    @Test func aRunnerReportingItsWeightsSplitsWeightsWorkingAndCache() throws {
        let m = try #require(
            ModelMemoryActivity(WorkerMemory(footprintMB: 30_000, mlxActiveMB: 28_000, mlxCacheMB: 1_200, mlxWeightsMB: 8_400, mlxPeakMB: 36_000, jobPeakMB: 31_000)))
        #expect(m.source == "mlx"); #expect(m.totalMB == 30_000)
        #expect(m.weightsMB == 8_400); #expect(m.workingMB == 19_600); #expect(m.cacheMB == 1_200); #expect(m.arraysMB == nil)
        // MLX's exact peak (which the footprint contains) beats the half-second samples; the two are never added.
        #expect(m.peakMB == 36_000); #expect(m.mlxPeakMB == 36_000); #expect(m.unsplitMB == nil)
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
        #expect(memory["cache_mb"] as? Double == 252); #expect(memory["peak_mb"] as? Double == 31000)
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
