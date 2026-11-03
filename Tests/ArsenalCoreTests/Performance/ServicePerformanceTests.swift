// SPDX-License-Identifier: MIT
import Darwin
import Foundation
import Testing
@testable import ArsenalCore

private func threadCount() -> Int {
    var threads: thread_act_array_t?
    var count: mach_msg_type_number_t = 0
    guard task_threads(mach_task_self_, &threads, &count) == KERN_SUCCESS, let threads else { return -1 }
    for i in 0..<Int(count) { mach_port_deallocate(mach_task_self_, threads[i]) }
    vm_deallocate(mach_task_self_, vm_address_t(UInt(bitPattern: threads)), vm_size_t(Int(count) * MemoryLayout<thread_act_t>.stride))
    return Int(count)
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_POLISH_PROBE"] == "1"))
struct PerformanceTests {
    @Test func sharedKindsRefresh() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let kinds = repo.appendingPathComponent("Resources/kinds")
        let files = try SafePath.files(in: kinds).map { path in
            CatalogFile(path: path, sha256: try FileDigest.sha256(kinds.appendingPathComponent(path)), bytes: try FileDigest.bytes(kinds.appendingPathComponent(path)))
        }
        let contents = try Dictionary(uniqueKeysWithValues: files.map { ($0.path, try Data(contentsOf: kinds.appendingPathComponent($0.path))) })
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], kindsDirectory: kinds, kindFiles: files, kindContents: contents)
        defer { fixture.stop() }
        let service = fixture.service
        _ = await service.rows(); _ = await service.catalog(); _ = await service.operations(id: "fixture-chat")
        for sample in 1...3 {
            let started = Date()
            for _ in 0..<30 { _ = await service.rows(); _ = await service.catalog(); _ = await service.operations(id: "fixture-chat") }
            print("PERF shared-kinds sample=\(sample) files=\(files.count) refresh-triplet-ms=\(Date().timeIntervalSince(started) * 1000 / 30)")
        }
        await service.shutdown()
    }
    @Test func idleWorkerThreads() async throws {
        let before = threadCount(), started = Date()
        let workers = try await withThrowingTaskGroup(of: WorkerProcess.self) { group in
            for _ in 0..<16 {
                group.addTask {
                    let worker = WorkerProcess(); try await worker.start(stubLaunch()); return worker
                }
            }
            var workers: [WorkerProcess] = []
            for try await worker in group { workers.append(worker) }
            return workers
        }
        let launchMS = Date().timeIntervalSince(started) * 1000
        try await Task.sleep(for: .milliseconds(200))
        print("PERF idle-workers count=\(workers.count) thread-before=\(before) thread-loaded=\(threadCount()) launch-ms=\(launchMS)")
        let ping = Date()
        for worker in workers { _ = try await worker.request(op: "ping") }
        print("PERF sequential-ping-ms=\(Date().timeIntervalSince(ping) * 1000 / Double(workers.count))")
        await withTaskGroup(of: Void.self) { group in for worker in workers { group.addTask { await worker.unload() } } }
    }
}
