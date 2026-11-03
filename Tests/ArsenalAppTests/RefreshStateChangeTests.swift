// SPDX-License-Identifier: MIT
import Foundation
import Combine
import ArsenalCore
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct RefreshStateChangeTests {
    enum Change: CaseIterable {
        case residency, loadCompletion, newModel, downloadCompletion
        var before: [ModelRow] {
            let state: ModelState =
                switch self {
                case .residency, .newModel: .loaded
                case .loadCompletion: .loading
                case .downloadCompletion: .downloading
                }
            return [ModelRow(id: "fixture", name: "Fixture", kind: "images", states: ["bf16": .init(state: state)])]
        }
        var after: [ModelRow] {
            var rows = before
            switch self {
            case .residency, .downloadCompletion: rows[0].states["bf16"] = .init(state: .ready)
            case .loadCompletion: rows[0].states["bf16"] = .init(state: .loaded)
            case .newModel: rows.append(ModelRow(id: "added", name: "Added", kind: "images", states: ["bf16": .init(state: .ready)]))
            }
            return rows
        }
    }

    @Test(arguments: Change.allCases)
    func stateChangeDuringAwaitedMenuCallbackGetsExactlyOneTrailingRefresh(change: Change) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-refresh-change-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = RefreshReadService()
        await service.changeRows(change.before)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
        defer { app.table.stop(); app.tools.closeAll() }
        let callbacks = RefreshCallbackSignal()
        let delivered = RefreshCatalogSignal(table: app.table)
        app.table.start {
            callbacks.arrive()
            await app.refreshMenuState()
            if callbacks.count == 1 {
                #expect(app.table.rows == change.before)
                let status = await service.changeRows(change.after)
                await delivered.wait(for: status.source)
                #expect(app.table.rows == change.before, "The published snapshot predates the independent change")
            }
        }
        await callbacks.wait(for: 2)
        await app.table.waitUntilSettled()
        #expect(app.table.rows == change.after)
        #expect(callbacks.count == 2)
        #expect(await service.counts[.rows] == 2)
    }

    @Test(arguments: Change.allCases, [false, true])
    func stateChangeAfterRowsSnapshotSurvivesBothReadKinds(change: Change, rowsOnly: Bool) async throws {
        let service = RefreshReadService()
        await service.changeRows(change.before)
        let table = TableController(service: service)
        defer { table.stop() }
        let callbacks = RefreshCallbackSignal()
        let delivered = RefreshCatalogSignal(table: table)
        if !rowsOnly { await service.holdNextRowsRead() }
        table.start { callbacks.arrive() }
        if rowsOnly {
            await callbacks.wait(for: 1)
            await table.waitUntilSettled()
            await service.holdNextRowsRead()
            await service.progressChanged()
        }
        await service.waitForPausedRows()
        let status = await service.changeRows(change.after)
        await delivered.wait(for: status.source)
        await service.releaseRowsRead()
        await callbacks.wait(for: 2)
        await table.waitUntilSettled()
        #expect(table.rows == change.after)
        let expectedReads = rowsOnly ? 3 : 2
        #expect(callbacks.count == 2)
        #expect(await service.counts[.rows] == expectedReads)
    }

    @Test func errorDuringRowsOnlyReadReachesTheFooterWithoutRetrying() async throws {
        let service = RefreshReadService()
        let table = TableController(service: service)
        defer { table.stop() }
        let callbacks = RefreshCallbackSignal()
        table.start { callbacks.arrive() }
        await callbacks.wait(for: 1)
        await table.waitUntilSettled()
        await service.setRowsError("Rows-only failure.")
        await service.progressChanged()
        await service.waitForRowsRead(2)
        await table.waitUntilSettled()
        #expect(table.lastError == "Fixture: Rows-only failure.")
        #expect(callbacks.count == 1)
        #expect(await service.counts[.rows] == 2)
    }

}

/// The catalog event is last in the fixture's three-event burst. Await its publication while the original
/// refresh is held so actor load cannot turn the intended mid-refresh burst into several later changes.
@MainActor private final class RefreshCatalogSignal {
    private var received: Set<String> = []
    private var waiters: [(String, CheckedContinuation<Void, Never>)] = []
    private var observation: AnyCancellable?
    init(table: TableController) {
        observation = table.$catalog.sink { [weak self] status in
            guard let self, let source = status?.source else { return }
            received.insert(source)
            let ready = waiters.filter { $0.0 == source }
            waiters.removeAll { $0.0 == source }
            for (_, waiter) in ready { waiter.resume() }
        }
    }
    func wait(for source: String) async {
        guard !received.contains(source) else { return }
        await withCheckedContinuation { waiters.append((source, $0)) }
    }
}

@MainActor private final class RefreshCallbackSignal {
    private(set) var count = 0
    private var waiters: [(Int, CheckedContinuation<Void, Never>)] = []
    func arrive() {
        count += 1
        let ready = waiters.filter { $0.0 <= count }
        waiters.removeAll { $0.0 <= count }
        for (_, waiter) in ready { waiter.resume() }
    }
    func wait(for count: Int) async {
        guard self.count < count else { return }
        await withCheckedContinuation { waiters.append((count, $0)) }
    }
}
