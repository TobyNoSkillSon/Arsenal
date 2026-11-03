// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The probe's no-swap figure for a Mac of `totalMB` with `roomMB` free under the Mac-wide line: admission subtracts the
/// RAM above the line, so a test states room and this adds that outside part back.
func macAvailableMB(roomMB: Double, totalMB: Double = 128_000) -> Double {
    roomMB + totalMB - MacMemoryGuard.safeInUseMB(totalMB: totalMB)
}

extension RealModelsService {
    func setLoadPeak(_ id: String, gb: Double?) {
        var row = modelRows[id]!
        var stats = PrecisionStatSet(kind: ModelStatKind(rawValue: row.kind)!, dtype: "bf16", runner: row.runner)
        if let gb { stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: gb)) }
        row.precisionStats["bf16"] = stats; modelRows[id] = row
    }
    func plantReservation(_ id: String, mb: Double, footprint: Double) {
        loadReservations[id] = mb; footprints[id] = WorkerMemory(footprintMB: footprint)
    }
    func markReleasing(_ id: String) { loadReservationReleases.insert(id) }
}

@Suite struct LoadMemoryAdmissionTests {
    @Test func precisionPeakWinsThenForecastThenUnknown() {
        var row = ModelRow(id: "m", name: "M", kind: "voices", precisions: ["bf16", "int4"])
        var stats = PrecisionStatSet(kind: ModelStatKind(rawValue: row.kind)!, dtype: "bf16", runner: row.runner)
        stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: 20.4)); row.precisionStats["bf16"] = stats
        let forecast = PrecisionMemoryRecipe(residentBytes: ["bf16": 10_000_000_000, "int4": 4_000_000_000], basis: "Declared tensors")
        let low = LoadMemoryAssessment(row: row, precision: "bf16", forecast: forecast, freeMB: 11200)
        #expect(low.neededGB == 20.4 && low.freeGB == 11.2 && !low.fits && low.basis == .peak)
        #expect(LoadMemoryAssessment(row: row, precision: "bf16", forecast: forecast, freeMB: 20400).fits)
        let small = LoadMemoryAssessment(row: row, precision: "int4", forecast: forecast, freeMB: 4000)
        #expect(small.neededGB == 4 && small.fits && small.basis == .forecast)
        let unknown = LoadMemoryAssessment(row: row, precision: "int8", forecast: forecast, freeMB: 0)
        #expect(unknown.neededGB == nil && unknown.fits && unknown.basis == .unknown)
        row.formats = [ModelFormat(id: "int4", bitsPerWeight: 4, source: .derived, runners: [row.runner], ramMB: 5000)]
        #expect(LoadMemoryAssessment(row: row, precision: "int4", forecast: forecast, freeMB: 4000).neededGB == 5)
        row.precisionStats["bf16"]?.runner = "another"
        #expect(LoadMemoryAssessment(row: row, precision: "bf16", forecast: forecast, freeMB: 10000).basis == .forecast)
    }

    @Test func reclaimableCountsDoNotTreatAllInactiveMemoryAsFree() throws {
        // Speculative pages appear in both free and file-backed; don't count them twice.
        #expect(MemoryProbe.reclaimableMB(free: 100, speculative: 20, external: 300, purgeable: 40, pageSize: 1e6) == 420)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try WireJSON.encode([
            "free_count": 100, "speculative_count": 20, "external_page_count": 300,
            "purgeable_count": 40, "inactive_count": 900, "page_size": 1000000, "memorystatus_level": 1
        ]).write(to: root)
        let probe = MemoryProbe(environment: ["ARSENAL_TEST_VM_STATS": root.path], totalMB: 1000)
        #expect(probe.noSwapAvailableMB(loadedMB: 200) == 420) // Already excluded by OS counters; no second subtraction.
    }

    @Test(arguments: [MemoryWhenFull.refuse, .queue])
    func everyLoadRefusesBeforeMutationEvenWithLegacySwapPreference(_ policy: MemoryWhenFull) async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        await f.service.shutdown()
        let file = f.paths.support.appendingPathComponent("memory.json")
        try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 11200))]).write(to: file)
        let service = try await RealModelsService.open(
            catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey,
            source: f.source, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client,
            memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: 128000))
        let id = "fixture-voices"
        try await service.get(id: id, precision: "bf16")
        try await service.setMemorySettings(.init(whenFull: policy))
        var residency = await service.residencySettings(); residency.allowSwap = true
        try await service.setResidencySettings(residency)
        await service.setLoadPeak(id, gb: 20.4)
        // Actual service methods, not just a preflight query. Both explicit and implicit loads refuse.
        for path in 0..<3 {
            do {
                if path == 0 {
                    try await service.load(id: id, precision: "bf16", fast: nil)
                } else if path == 1 {
                    _ = try await service.run(id: id, op: "speak", params: ["text": "Hello", "stream": false])
                } else {
                    try await service.applySettings(id: id)
                }
                Issue.record("Load path \(path) admitted")
            } catch let failure as WorkerFailure {
                let refusal = try #require(LoadMemoryRefusal(details: failure.details))
                #expect(failure.code == "insufficient_memory" && refusal.neededGB == 20.4 && refusal.freeGB == 11.2)
            }
        }
        #expect(await service.memoryWaiters.isEmpty)
        #expect(await service.rows().first?.residentDType == nil)
        try await service.load(id: id, precision: nil, fast: nil, force: true)
        #expect(await service.rows().first?.residentDType == "bf16")
        // Already loaded: use is allowed, and idempotent Load needs no new capacity.
        try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 0))]).write(to: file)
        await service.setLoadPeak(id, gb: 20.4)
        try await service.load(id: id, precision: nil, fast: nil)
        _ = try await service.run(id: id, op: "speak", params: ["text": "Hello", "stream": false])
        await #expect(throws: WorkerFailure.self) { try await service.applySettings(id: id) }
        #expect(await service.rows().first?.residentDType == "bf16") // Refused Reload preserves healthy worker.
        try await service.unload(id: id)
        _ = try await service.run(id: id, op: "speak", params: ["text": "Hello", "stream": false], force: true)
        await service.shutdown()
    }

    @Test func unknownLoadsWithAnExplicitDiagnostic() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let id = "fixture-voices"
        try await f.service.get(id: id, precision: "bf16")
        await f.service.setLoadPeak(id, gb: nil)
        try await f.service.load(id: id, precision: "bf16", fast: nil)
        #expect(await f.service.rows().first?.residentDType == "bf16")
        let row = try #require(await f.service.rows().first)
        #expect(row.loadMemoryNote?.contains("unknown") == true)
        #expect(try JSONDecoder().decode(ModelRow.self, from: JSONEncoder().encode(row)).loadMemoryNote == row.loadMemoryNote)
        await f.service.shutdown()
    }
    @Test func pendingReservationsAndAlreadyLoadedFootprintsAreCountedOnce() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        await f.service.shutdown()
        let file = f.paths.support.appendingPathComponent("memory.json")
        try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 11200))]).write(to: file)
        let service = try await RealModelsService.open(
            catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey,
            source: f.source, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client,
            memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: 128000))
        await service.plantReservation("loaded", mb: 0, footprint: 200)
        await service.plantReservation("loading", mb: 1000, footprint: 400)
        let model = try await service.manifest("fixture-voices")
        let check = await service.loadMemoryAssessment(model, precision: "bf16")
        #expect(check.freeGB == 10) // 11.2 - 0.2 - 0.4 - (1 - 0.4); neither omission nor double subtraction.
        // While unload is suspended, VM may already have reclaimed the worker before its last sample is cleared.
        // Keep the entire replacement reserved through that gap, rather than subtracting the stale footprint.
        await service.markReleasing("loading")
        let duringRelease = await service.loadMemoryAssessment(model, precision: "bf16")
        #expect(duringRelease.freeGB == 9.6)
        await service.shutdown()
    }

    /// Admission compares with the Mac-wide line, not raw free RAM: 39.2 GB is free without swap here, but only 11.2 GB
    /// of it lies under the 100 GB line of a 128 GB Mac (the probe's 28 GB above the line is subtracted).
    @Test func admissionRefusesPastTheMacLineAndAdmitsUpToIt() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        await f.service.shutdown()
        let file = f.paths.support.appendingPathComponent("line-memory.json")
        try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 11200))]).write(to: file)
        let service = try await RealModelsService.open(
            catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey,
            source: f.source, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client,
            memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: 128000))
        let id = "fixture-voices"
        try await service.get(id: id, precision: "bf16")
        await service.setLoadPeak(id, gb: 20.4)
        do {
            try await service.load(id: id, precision: "bf16", fast: nil)
            Issue.record("A Load past the Mac-wide line was admitted")
        } catch let failure as WorkerFailure {
            let refusal = try #require(LoadMemoryRefusal(details: failure.details))
            #expect(failure.code == "insufficient_memory" && refusal.neededGB == 20.4 && refusal.freeGB == 11.2)
            #expect(refusal.limitGB == 100 && refusal.totalGB == 128)
            #expect(refusal.limitGB * 1000 == MacMemoryGuard.safeInUseMB(totalMB: 128000))
        }
        #expect(await service.rows().first?.residentDType == nil)
        // A model that fits exactly under the line loads; the guard would not stop it.
        await service.setLoadPeak(id, gb: 11.2)
        try await service.load(id: id, precision: "bf16", fast: nil)
        #expect(await service.rows().first?.residentDType == "bf16")
        await service.shutdown()
    }

    @Test func declaredForecastRefusesAtZeroFreeAndUnknownStillLoads() async throws {
        for known in [false, true] {
            let extra: [String: TOMLValue] =
                known
                ? [
                    "memory": .table([
                        "basis": .string("Declared tensor payload"),
                        "resident_bytes": .table(["bf16": .integer(20_400_000_000)])
                    ])
                ] : [:]
            let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: extra); defer { f.stop() }
            await f.service.shutdown()
            let file = f.paths.support.appendingPathComponent("memory.json")
            try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 0))]).write(to: file)
            let service = try await RealModelsService.open(
                catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey,
                source: f.source, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client,
                memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: 128000))
            try await service.get(id: "fixture-voices", precision: "bf16")
            do {
                try await service.load(id: "fixture-voices", precision: nil, fast: nil)
                #expect(!known)
            } catch let failure as WorkerFailure {
                #expect(known && failure.code == "insufficient_memory")
                #expect(failure.details["needed_gb"] == 20.4 && failure.details["free_gb"] == 0)
            }
            await service.shutdown()
        }
    }

}
