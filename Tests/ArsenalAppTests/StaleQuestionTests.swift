// SPDX-License-Identifier: MIT
import Combine
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// VM run 2 Nov: Remove on a Get waiting at a gated repository asked "Delete FLUX 3 Image Klein? Frees up to 0 KB", the
/// question stayed up after the model was removed over MCP, and Quit did nothing while it was up. Headless: the
/// decisions, the watcher and the quit sequence; the real alert closing and quitting is the guest's check.
@MainActor @Suite(.menuState, .serialized) final class StaleQuestionTests {
    private func live(_ rows: [ModelRow]) async -> (TableController, FixtureService) {
        let service = FixtureService(rows: rows, kinds: Fixtures.fallbackKinds)
        let c = TableController(service: service, chip: "M5 Max")
        await c.refresh()
        return (c, service)
    }
    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
    /// One turn of the main queue: blocks queued before it have run.
    private func mainQueueTurn() async { await withCheckedContinuation { done in DispatchQueue.main.async { done.resume() } } }
    private static let gate = GateWait(repo: "black-forest-labs/FLUX.3-klein", page: "https://huggingface.co/black-forest-labs/FLUX.3-klein", hasToken: false, mode: "auto")

    @Test func removingAGetWaitingAtTheGateWithNothingOnDiskAsksNothing() async throws {
        let waiting = ModelRow(
            id: "g", name: "Gated", kind: "voices", precisions: ["bf16"],
            states: ["bf16": .init(state: .failed, error: "Accept the terms.", phase: PrecisionState.gatedPhase, gate: Self.gate)], selectedPrecision: "bf16")
        let (c, service) = await live([waiting])
        var asked = 0
        c.confirmDeleteWeights = { _, _ in
            asked += 1; return true
        }
        #expect(!c.hasWeights(c.rows[0]))
        c.remove(c.rows[0])
        try await waitUntil { c.rows.isEmpty }
        #expect(asked == 0)
        #expect(await service.calls == ["remove g keep"])
        #expect(await service.deletes.isEmpty)
    }

    @Test func aFewBytesNeverReadZeroKB() {
        #expect(formatBytes(0) == "0 KB")
        #expect(formatBytes(250) == "1 KB")
        #expect(formatBytes(1_500) == "2 KB")
        #expect(formatBytes(341_000_000) == "341 MB")
    }

    @Test func aDeleteQuestionStandsOnlyWhileItsRowHasWeights() async throws {
        let have = ModelRow(id: "b", name: "B", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        let (c, service) = await live([have])
        let row = c.rows[0]
        #expect(c.deleteQuestionStands(row))
        let copy = DeleteConfirm.Copy(label: "16-bit", downloadable: true, dtype: "fp16")
        #expect(c.deleteQuestionStands(row, copy: copy))
        #expect(!c.deleteQuestionStands(row, copy: DeleteConfirm.Copy(label: "4-bit", downloadable: true, dtype: "int4")))
        // Removed elsewhere (MCP's arsenal_remove deletes the weights): the subject is gone.
        try await service.remove(id: "b", deleteWeights: true)
        await c.refresh()
        #expect(!c.deleteQuestionStands(row))
        #expect(!c.deleteQuestionStands(row, copy: copy))
    }

    @Test func aQuestionWhoseModelIsRemovedElsewhereClosesItselfOnce() async throws {
        let have = ModelRow(id: "b", name: "B", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        let keep = ModelRow(id: "k", name: "K", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        let (c, service) = await live([have, keep])
        let row = c.rows.first { $0.id == "b" }!
        var closed = 0
        let watch = StaleQuestion(c.objectWillChange, stands: { c.deleteQuestionStands(row) }, close: { closed += 1 })
        defer { watch.end() }
        // Another row changing leaves the question up.
        try await service.remove(id: "k", deleteWeights: true)
        await c.refresh(); await mainQueueTurn()
        #expect(closed == 0)
        try await service.remove(id: "b", deleteWeights: true)
        await c.refresh(); await mainQueueTurn()
        #expect(closed == 1)
        // Later changes do not close it again.
        await c.refresh(); await mainQueueTurn()
        #expect(closed == 1)
    }

    @Test func anEndedWatchNeverCloses() async throws {
        let subject = ObservableObjectPublisher()
        var closed = 0
        let watch = StaleQuestion(subject, stands: { false }, close: { closed += 1 })
        watch.end()
        subject.send(); await mainQueueTurn()
        #expect(closed == 0)
    }

    /// A fake modal stack: `cancel` ends the top session one turn later, as AppKit's loop unwinds after an event.
    private final class Modals {
        var stack: [NSObject]
        var cancels = 0, terminations: [Bool] = []
        /// Stops the top session ignores before it ends.
        var ignored = 0
        var pending: [@MainActor () -> Void] = []
        init(_ count: Int) { stack = (0..<count).map { _ in NSObject() } }
        @MainActor var hooks: AppModal.Hooks {
            AppModal.Hooks(
                modal: { self.stack.last },
                cancel: {
                    self.cancels += 1
                    if self.ignored > 0 { self.ignored -= 1; return }
                    let top = self.stack.last; self.pending.append { self.stack.removeAll { $0 === top } }
                },
                terminate: { self.terminations.append($0) }, next: { work in self.pending.append(work) })
        }
        @MainActor func drain() {
            var turns = 0
            while !pending.isEmpty, turns < 100 {
                let now = pending; pending = []; turns += 1
                for work in now { work() }
            }
        }
    }

    @Test func quitWithNoQuestionUpTerminatesAtOnce() {
        let m = Modals(0)
        AppModal.quit(m.hooks)
        #expect(m.cancels == 0 && m.terminations == [false])
    }

    @Test func quitEndsEachOpenQuestionAsCancelThenTerminates() {
        let m = Modals(2) // a question over another (the menu's alert over an open panel)
        AppModal.quit(m.hooks)
        #expect(m.terminations.isEmpty, "never terminates while a modal session runs")
        m.drain()
        #expect(m.cancels == 2, "each session is ended once, never the one beneath twice")
        #expect(m.stack.isEmpty && m.terminations == [true])
    }

    @Test func aQuestionThatIgnoresTheStopIsEndedAgain() {
        let m = Modals(1)
        m.ignored = 1
        AppModal.quit(m.hooks)
        m.drain()
        #expect(m.cancels == 2 && m.stack.isEmpty && m.terminations == [true])
    }
}
