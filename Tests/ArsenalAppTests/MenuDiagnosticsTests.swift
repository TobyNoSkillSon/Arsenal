// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Synchronization
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuDiagnosticsTests {
    private func support() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-log-" + UUID().uuidString)
    }
    private func entries(_ url: URL) throws -> [[String: Any]] {
        try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map {
            try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
        }
    }

    @Test func selectedPrivateVoiceNameNeverReachesTheGeometryFile() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let sentinel = "PRIVATE-SAVED-VOICE-" + UUID().uuidString
        let rows = Fixtures.rows(.loaded)
        let app = AppDelegate(service: FixtureService(rows: rows, supportDirectory: support), paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh()
        let voiceRow = try #require(app.table.rows.first { $0.tools.contains("voice") })
        let choice = VoiceChoice(id: "saved-private", name: sentinel, language: "en")
        app.tools.preview(voices: [voiceRow.id: [choice]], current: [voiceRow.id: choice.id])
        app.rebuildMenu()
        let voice = try #require(app.menu.settings?.items.first { $0.title == "Agent Voice: " + sentinel })
        let submenu = try #require(voice.submenu)
        #expect(voice.identifier == MenuDiagnosticName.voice.itemIdentifier)
        #expect(submenu.items.contains { $0.title == sentinel })
        // A menu's own title is untrusted too, and diagnostics must not replace it.
        submenu.title = sentinel
        submenu.delegate?.menuWillOpen?(submenu)
        // Unidentified contributed submenus must never fall back to their visible titles.
        let unknown = NSMenuItem(title: sentinel, action: nil, keyEquivalent: "")
        unknown.submenu = NSMenu(title: sentinel)
        app.menu.addItem(unknown)
        app.menuDiagnostics.observeSubmenus(in: app.menu)
        unknown.submenu?.delegate?.menuWillOpen?(try #require(unknown.submenu))
        for _ in 0..<5 { await Task.yield() }
        await app.menuDiagnostics.flush()
        let text = try String(contentsOf: app.menuDiagnostics.url, encoding: .utf8)
        #expect(!text.contains(sentinel))
        #expect(submenu.title == sentinel)
        #expect(try entries(app.menuDiagnostics.url).compactMap { $0["menu"] as? String } == ["Voice", "Submenu"])
        for title in [MenuContent.keepLoadedTitle, "Memory"] {
            let menu = try #require(app.menu.settings?.items.first { $0.title == title }?.submenu)
            menu.delegate?.menuWillOpen?(menu)
        }
        for _ in 0..<5 { await Task.yield() }
        await app.menuDiagnostics.flush()
        #expect(try entries(app.menuDiagnostics.url).suffix(2).compactMap { $0["menu"] as? String } == ["Keep Hot", "Memory"])
    }

    @Test func largeIncomingRecordsKeepTheNewestWholeRecordsWithinExactly64KiB() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support)
        var retained: [(id: Int, bytes: Int)] = []
        for (id, padding) in [1000, 40000, 40000, 12000, 53000, 65000, 100, 2000, 40000].enumerated() {
            let record: [String: Any] = ["id": id, "padding": String(repeating: "x", count: padding)]
            let bytes = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]).count + 1
            retained.append((id, bytes))
            while retained.reduce(0, { $0 + $1.bytes }) > MenuDiagnostics.maximumBytes { retained.removeFirst() }
            try log.append(record)
            await log.flush()
            let data = try Data(contentsOf: log.url)
            #expect(data.count == retained.reduce(0, { $0 + $1.bytes }))
            #expect(data.count <= MenuDiagnostics.maximumBytes)
            #expect(data.last == 10)
            #expect(try entries(log.url).compactMap { $0["id"] as? Int } == retained.map(\.id))
            #expect(log.lastError == nil)
        }
        let empty: [String: Any] = ["padding": ""]
        let overhead = try JSONSerialization.data(withJSONObject: empty).count + 1
        try log.append(["padding": String(repeating: "x", count: MenuDiagnostics.maximumBytes - overhead)])
        await log.flush()
        let atLimit = try Data(contentsOf: log.url)
        #expect(atLimit.count == MenuDiagnostics.maximumBytes)
        #expect(try entries(log.url).count == 1)
        #expect(throws: CocoaError.self) {
            try log.append(["padding": String(repeating: "x", count: MenuDiagnostics.maximumBytes - overhead + 1)])
        }
        await log.flush()
        #expect(try Data(contentsOf: log.url) == atLimit)
    }

    @Test func legacyOversizedFileAndIncompleteTailDoNotLeavePartialRecords() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support)
        try FileManager.default.createDirectory(at: log.url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var legacy = Data()
        for id in 0..<20 {
            legacy.append(try JSONSerialization.data(withJSONObject: ["id": id, "padding": String(repeating: "x", count: 7000)]))
            legacy.append(10)
        }
        legacy.append(Data("{\"incomplete\":true".utf8))
        try legacy.write(to: log.url)
        try log.append(["id": 20, "padding": String(repeating: "x", count: 40000)])
        await log.flush()
        #expect(try Data(contentsOf: log.url).count <= MenuDiagnostics.maximumBytes)
        #expect(try entries(log.url).compactMap { $0["id"] as? Int } == [17, 18, 19, 20])
    }

    @Test func geometryCallbacksReturnWhileDiskIsBlockedAndWritesStayOrderedOffMain() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let accesses = Mutex<[Bool]>([])
        let gate = DispatchSemaphore(value: 0)
        let log = MenuDiagnostics(
            support: support,
            onDiskAccess: {
                let first = accesses.withLock { calls in
                    calls.append(Thread.isMainThread); return calls.count == 1
                }
                if first { _ = gate.wait(timeout: .now() + 5) }
            })
        defer { gate.signal() }
        let menu = NSMenu()
        // These are the same synchronous capture calls used by open and scaled-layout callbacks.
        for index in 0..<100 { log.record(menu: menu, name: .models, scale: CGFloat(index) / 128) }
        try await waitUntil { accesses.withLock { $0.count == 1 } }
        #expect(!FileManager.default.fileExists(atPath: log.url.path))
        gate.signal()
        await log.flush()
        let records = try entries(log.url)
        #expect(records.count == 100)
        #expect(records.compactMap { $0["scale"] as? Double } == (0..<100).map { Double($0) / 128 })
        #expect(accesses.withLock { $0.count == 100 && $0.allSatisfy { !$0 } })
        #expect(log.lastError == nil)
    }

    @Test func backgroundWriteFailureIsReportedAndTheNextWriteRecovers() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support)
        try FileManager.default.createDirectory(at: log.url, withIntermediateDirectories: true)
        log.record(menu: NSMenu(), name: .main)
        await log.flush()
        #expect(log.lastError != nil)
        try FileManager.default.removeItem(at: log.url)
        log.record(menu: NSMenu(), name: .activity)
        await log.flush()
        #expect(log.lastError == nil)
        #expect(try entries(log.url).compactMap { $0["menu"] as? String } == ["Activity"])
    }

    @Test func terminationDrainsOnlyCapturedWritesAndCancelsDeferredOpenCallbacks() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let writer = TerminationWriterProbe()
        let app = AppDelegate(service: FixtureService(rows: [], supportDirectory: support), paths: ArsenalPaths(support: support), geometryWriter: writer)
        defer { app.table.stop(); app.tools.closeAll() }
        let log = app.menuDiagnostics
        try log.append(["id": 1])
        try log.append(["id": 2])
        log.recordAfterOpen(menu: NSMenu(), name: .models)
        #expect(writer.records.isEmpty, "Records remain pending until the delegate drains them")
        // Exercise the actual synchronous termination callback, with no dependency on utility-queue/disk timing.
        app.applicationWillTerminate(Notification(name: NSApplication.willTerminateNotification))
        #expect(writer.drains == 1)
        let records = try writer.records.map { try #require(JSONSerialization.jsonObject(with: $0) as? [String: Int]) }
        #expect(records == [["id": 1], ["id": 2]])
        for _ in 0..<5 { await Task.yield() }
        await log.flush()
        #expect(writer.records.count == 2, "Deferred geometry must not be captured during or after termination")
        #expect(log.drainForTermination())
        #expect(writer.records.count == 2)
    }

    @Test func writerUsesTheExistingDeadlineRatherThanRestartingTheBudget() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let gate = DispatchSemaphore(value: 0), entered = Mutex(false)
        let writer = MenuGeometryWriter(
            url: support.appendingPathComponent("geometry.jsonl"),
            onDiskAccess: {
                entered.withLock { $0 = true }; _ = gate.wait(timeout: .now() + 5)
            })
        defer { gate.signal() }
        writer.enqueue(Data("{}\n".utf8)) { _ in }
        try await waitUntil { entered.withLock { $0 } }
        let expired = DispatchTime.now()
        let started = ContinuousClock.now
        let drained = writer.drain(until: expired)
        let elapsed = ContinuousClock.now - started
        #expect(!drained && elapsed < .milliseconds(25), "An exhausted budget must not start another 50 ms wait")
        gate.signal(); await writer.flush()
    }

    @Test func aBlockedDiskCannotHoldTerminationAndItsWritesRemainOffMain() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let gate = DispatchSemaphore(value: 0)
        let accesses = Mutex<[Bool]>([])
        let log = MenuDiagnostics(
            support: support,
            onDiskAccess: {
                accesses.withLock { $0.append(Thread.isMainThread) }
                _ = gate.wait(timeout: .now() + 5)
            })
        defer { gate.signal() }
        try log.append(["id": 1])
        try await waitUntil { accesses.withLock { !$0.isEmpty } }
        // Each AppKit probe would cost 5 ms. Capturing these at quit would exceed the whole budget.
        let menus = (0..<32).map { _ in TerminationMenuProbe() }
        for menu in menus { log.recordAfterOpen(menu: menu, name: .models) }
        let started = ContinuousClock.now
        let drained = log.drainForTermination()
        let elapsed = ContinuousClock.now - started
        #expect(!drained)
        #expect(elapsed >= .milliseconds(50) && elapsed < .milliseconds(75), "Whole quit drain: \(elapsed), budget 50 ms plus 25 ms scheduling tolerance")
        #expect(menus.allSatisfy { $0.captures == 0 })
        #expect(!FileManager.default.fileExists(atPath: log.url.path))
        gate.signal()
        await log.flush()
        #expect(try entries(log.url).compactMap { $0["id"] as? Int } == [1])
        #expect(accesses.withLock { $0 == [false] })
        #expect(log.drainForTermination())
    }
}

@MainActor private final class TerminationMenuProbe: NSMenu {
    nonisolated private let count = Mutex(0)
    var captures: Int { count.withLock { $0 } }
    override nonisolated func accessibilityFrame() -> NSRect {
        count.withLock { $0 += 1 }
        Thread.sleep(forTimeInterval: 0.005)
        return .zero
    }
}

private final class TerminationWriterProbe: MenuGeometryWriting {
    private struct State { var pending: [Data] = []; var records: [Data] = []; var drains = 0 }
    private let state = Mutex(State())
    var records: [Data] { state.withLock { $0.records } }
    var drains: Int { state.withLock { $0.drains } }
    func enqueue(_ line: Data, completion: @escaping @MainActor @Sendable (String?) -> Void) {
        state.withLock { $0.pending.append(line) }
    }
    func flush() async {}
    func drain(until deadline: DispatchTime) -> Bool {
        state.withLock { value in
            value.drains += 1; value.records.append(contentsOf: value.pending); value.pending = []
        }
        return true
    }
}
