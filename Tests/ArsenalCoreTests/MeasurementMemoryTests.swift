// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct MeasurementMemoryTests {
    func snapshot(bytes: Int64, precision: String = "bf16") -> SnapshotManifest {
        SnapshotManifest(model: "fixture", revision: "pinned", precision: precision, files: [.init(path: "weights", bytes: bytes, sha256: String(repeating: "a", count: 64))])
    }
    @Test func coldQwenEstimateRefusesButExistingPackedSnapshotFits() throws {
        let source = snapshot(bytes: 55_563_006_776)
        let packed = snapshot(bytes: 15_627_095_656, precision: "int4")
        let cold = try MeasurementMemoryEstimate.megabytes(source: source, installed: nil, assets: [])
        let reused = try MeasurementMemoryEstimate.megabytes(source: source, installed: packed, assets: [])
        #expect(abs(cold - (111_126.013552 + 768 + MeasurementMemoryEstimate.payloadMB)) < 0.001)
        #expect(
            ResidencyPolicy.admission(id: "fixture", estimateMB: cold, availableMB: 90_000, residents: [], allowSwap: false)
                == .refuse(
                    needMB: cold + MemoryProbe.headroomMB, freeMB: 90_000))
        #expect(ResidencyPolicy.admission(id: "fixture", estimateMB: reused, availableMB: 90_000, residents: [], allowSwap: false) == .admit(evict: []))
        #expect(MeasurementMemorySkip(needMB: cold + MemoryProbe.headroomMB).localizedDescription == "skipped: not enough free memory (needs ~112.7 GB)")
        let assets = try MeasurementMemoryEstimate.megabytes(source: source, installed: packed, assets: [snapshot(bytes: 2_000_000_000)])
        #expect(assets == reused + 2000)
    }
    @Test func ordinaryAdmissionIsRecheckedBeforeGetAndLoadWithoutSwapOrWaiting() async throws {
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"])
        defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
        try await fixture.service.get(id: "fixture-chat", precision: "bf16")
        await fixture.service.shutdown()
        let memory = fixture.paths.support.appendingPathComponent("memory-fixture.json")
        try WireJSON.encode(["available_mb": 1]).write(to: memory)
        let service = try await RealModelsService.open(
            catalogData: fixture.catalogData, signature: fixture.signature, publicKey: fixture.publicKey, source: fixture.source,
            kinds: KindSpec.defaults, paths: fixture.paths, artifacts: StubWorkerArtifact(), client: fixture.client,
            memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": memory.path], totalMB: 128_000))
        try await service.setMemorySettings(.init(whenFull: .queue))
        let model = try await service.measurementManifest(id: "fixture-chat")
        let adapter = MeasurementAdapter(service: service, model: model, directory: fixture.paths.support.appendingPathComponent("output"), fixture: true)
        await #expect(throws: MeasurementMemorySkip.self) { try await adapter.prepare(dtype: "int4", sourceDtype: "bf16") }
        #expect(await service.supervisor.residentIDs().isEmpty)
        #expect(await service.loadReservations.isEmpty)
        #expect(await service.rows().first?.states["int4"]?.state == .notInstalled)
        // Reuse an existing derived fixture snapshot, then simulate free memory dropping before Load.
        try await service.get(id: "fixture-chat", precision: "int4")
        let plan = try await service.getPlan(id: "fixture-chat", dtype: "int4")
        try WireJSON.encode(["available_mb": 10_000]).write(to: memory)
        try await service.measurementGet(id: "fixture-chat", precision: "int4", approvedPlan: plan)
        try WireJSON.encode(["available_mb": 1]).write(to: memory)
        await #expect(throws: MeasurementMemorySkip.self) { try await service.measurementLoad(id: "fixture-chat", precision: "int4") }
        #expect(await service.supervisor.residentIDs().isEmpty)
        let queue = PendingBenchmarkQueue(support: fixture.paths.support)
        let request = PendingBenchmark(model: model.id, kind: .chat, dtype: "int4", sourceDtype: "bf16", revision: model.weights.revision, runner: "native", mode: "fast")
        try queue.enqueue([request])
        try queue.recordFailure(id: request.id, reason: MeasurementMemorySkip(needMB: 1548).localizedDescription)
        #expect(try queue.read().first { $0.id == request.id }?.state == "pending")
        #expect(try queue.read().first { $0.id == request.id }?.error == "skipped: not enough free memory (needs ~1.5 GB)")
        try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 10_000))]).write(to: memory)
        _ = try await adapter.prepare(dtype: "int4", sourceDtype: "bf16")
        #expect(await service.supervisor.residentIDs() == [model.id])
        await adapter.shutdown()
    }
}
