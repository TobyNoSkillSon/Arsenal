// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The clock of a gated Get's wait: each scheduled retry waits here until the test fires it, or is cancelled. No sleeping.
final class ManualSleeper: @unchecked Sendable {
    private let lock = NSLock()
    private var waiting: [(id: UUID, continuation: CheckedContinuation<Void, Error>)] = []
    private var cancelled = 0
    private var immediate = false
    /// Every retry runs at once (no counter time at all).
    static func immediate() -> ManualSleeper { let s = ManualSleeper(); s.immediate = true; return s }
    var sleep: GateRetry.Sleep {
        { [self] _ in
            if lock.withLock({ immediate }) { await Task.yield(); return }
            let id = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                    let gone = lock.withLock { () -> Bool in
                        if Task.isCancelled { return true }
                        waiting.append((id, c)); return false
                    }
                    if gone { c.resume(throwing: CancellationError()) }
                }
            } onCancel: {
                let c = lock.withLock { () -> CheckedContinuation<Void, Error>? in
                    guard let i = waiting.firstIndex(where: { $0.id == id }) else { return nil }
                    cancelled += 1; return waiting.remove(at: i).continuation
                }
                c?.resume(throwing: CancellationError())
            }
        }
    }
    var pending: Int { lock.withLock { waiting.count } }
    var cancellations: Int { lock.withLock { cancelled } }
    /// Ends the oldest wait: its retry runs.
    func fire() { lock.withLock { waiting.isEmpty ? nil : waiting.removeFirst().continuation }?.resume() }
}

/// Pages the service asked the browser to open.
final class OpenedPages: @unchecked Sendable {
    private let lock = NSLock(); private var pages: [[String]] = []
    func record(_ urls: [URL]) { lock.withLock { pages.append(urls.map(\.absoluteString)) } }
    var all: [[String]] { lock.withLock { pages } }
}

func eventually(_ seconds: Double = 10, _ condition: () async -> Bool) async throws {
    let end = Date().addingTimeInterval(seconds)
    while Date() < end {
        if await condition() { return }
        try await Task.sleep(for: .milliseconds(10))
    }
    Issue.record("condition not met in \(seconds) s")
}

@Suite struct GateRetryTests {
    let gate = HFGateError(repo: "org/model", hasToken: true, mode: "auto")

    @Test func episodeOpensOnceCountsAndGivesUpAtTheLimit() async throws {
        let sleeper = ManualSleeper()
        let now = Date(timeIntervalSince1970: 1_000)
        let retry = GateRetry(interval: .seconds(30), limit: 3, sleep: sleeper.sleep, now: { now })
        let first = retry.refused("m", gate: gate) {}
        #expect(first.began && first.wait.tries == 0 && first.wait.retryAt == now.addingTimeInterval(30))
        #expect(first.wait.page == "https://huggingface.co/org/model" && !first.wait.gaveUp)
        for tries in 1...2 {
            let next = retry.refused("m", gate: gate) {}
            #expect(!next.began && next.wait.tries == tries && !next.wait.gaveUp)
        }
        let last = retry.refused("m", gate: gate) {}
        #expect(!last.began && last.wait.tries == 3 && last.wait.gaveUp && last.wait.retryAt == nil)
        #expect(!retry.isWaiting("m"))
        // Each refusal replaced the previous counter (a counter cancelled before it started never reaches the sleeper);
        // the last one left none.
        try await eventually { sleeper.pending == 0 }
        try await Task.sleep(for: .milliseconds(50))
        #expect(sleeper.pending == 0)
        // A Get after giving up starts a new wait (the pages open again).
        let again = retry.refused("m", gate: gate) {}
        #expect(again.began)
        retry.end("m")
    }

    @Test func counterFiresTheRetryAndAClickCancelsIt() async throws {
        let sleeper = ManualSleeper()
        let retry = GateRetry(interval: .seconds(30), limit: 10, sleep: sleeper.sleep)
        let fired = OpenedPages()
        _ = retry.refused("m", gate: gate) { fired.record([]) }
        try await eventually { sleeper.pending == 1 }
        sleeper.fire()
        try await eventually { fired.all.count == 1 }
        let pending = retry.hasPendingRetry("m"), waiting = retry.isWaiting("m")
        #expect(!pending && waiting)
        // Refused again: a new counter; a click (a Get started now) removes it and its retry never runs.
        _ = retry.refused("m", gate: gate) { fired.record([]) }
        try await eventually { sleeper.pending == 1 }
        retry.retryingNow("m")
        try await eventually { sleeper.cancellations == 1 && sleeper.pending == 0 }
        let stillWaiting = retry.isWaiting("m")
        #expect(fired.all.count == 1 && stillWaiting)
        retry.end("m")
        #expect(!retry.isWaiting("m"))
    }
}

/// The service's side: a Get refused by a gated repository opens the pages once, counts down, retries, and reaches the
/// normal download when access is granted. Fixture catalog and loopback file server; no model runs.
@Suite(.serialized) struct GatedGetTests {
    static let id = "fixture-voices"
    static let secret = "hf_TESTONLYsecretValue0123456789"

    func fixture(refusals: Int, token: String?, sleeper: ManualSleeper, limit: Int = 10) async throws -> (LocalCatalogFixture, OpenedPages) {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"], gatedRequests: refusals, token: token)
        let pages = OpenedPages()
        await f.service.setGateHandling(retry: GateRetry(interval: .seconds(30), limit: limit, sleep: sleeper.sleep), open: { pages.record($0) })
        return (f, pages)
    }
    func state(_ f: LocalCatalogFixture) async -> PrecisionState? { await f.service.rows().first { $0.id == Self.id }?.states["bf16"] }
    func refusals(_ f: LocalCatalogFixture) -> Int {
        let url = f.paths.support.deletingLastPathComponent().appendingPathComponent("hf/refusals")
        return ((try? String(contentsOf: url, encoding: .utf8)) ?? "").split(separator: "\n").count
    }

    @Test func noTokenOpensTokenAndModelPagesAndCountsDown() async throws {
        let sleeper = ManualSleeper()
        let (f, pages) = try await fixture(refusals: -1, token: nil, sleeper: sleeper); defer { f.stop() }
        let error = await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        #expect(error?.hasToken == false)
        let s = try #require(await state(f))
        #expect(s.state == .failed && s.phase == PrecisionState.gatedPhase)
        #expect(s.gate?.hasToken == false && s.gate?.tries == 0 && s.gate?.retryAt != nil)
        #expect(pages.all == [["https://huggingface.co/test/fixture", "https://huggingface.co/settings/tokens"]])
        // The waiting row stays on the table.
        #expect(await f.service.rows().contains { $0.id == Self.id })
        await f.service.cancelGet(id: Self.id)
    }

    @Test func tokenNotAcceptedRetriesAutomaticallyAndDownloadsWhenGranted() async throws {
        let sleeper = ManualSleeper()
        let (f, pages) = try await fixture(refusals: 3, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        #expect(await state(f)?.gate?.hasToken == true)
        // Two automatic retries are refused (still waiting, the counter runs again), the third downloads.
        for tries in 1...2 {
            try await eventually { sleeper.pending == 1 }
            sleeper.fire()
            try await eventually { await state(f)?.gate?.tries == tries }
            #expect(await state(f)?.gate?.retryAt != nil)
        }
        try await eventually { sleeper.pending == 1 }
        sleeper.fire()
        try await eventually { await state(f)?.state == .ready }
        #expect(await state(f)?.gate == nil)
        #expect(refusals(f) == 3)
        #expect(pages.all == [["https://huggingface.co/test/fixture"]], "pages open once per wait, never on a retry")
    }

    @Test func limitReachedStopsWithGetAvailableAgain() async throws {
        let (f, pages) = try await fixture(refusals: -1, token: Self.secret, sleeper: .immediate(), limit: 3); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { await state(f)?.gate?.gaveUp == true }
        let s = try #require(await state(f))
        #expect(s.state == .failed && s.gate?.tries == 3 && s.gate?.retryAt == nil)
        #expect(refusals(f) == 4)
        #expect(pages.all.count == 1)
        // Get again: a new wait, and the page opens again.
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { await state(f)?.gate?.gaveUp == true && refusals(f) == 8 }
        #expect(pages.all.count == 2)
    }

    @Test func clickDuringCountdownRetriesNowAndRemovesTheCounter() async throws {
        let sleeper = ManualSleeper()
        let (f, pages) = try await fixture(refusals: 2, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { sleeper.pending == 1 }
        // A click while counting: the counter is cancelled and the Get runs now (refused again: still waiting, a new counter).
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        #expect(sleeper.cancellations == 1)
        #expect(await state(f)?.gate?.tries == 1)
        try await eventually { sleeper.pending == 1 }
        // Access granted: the next click downloads, and nothing is left counting.
        try await f.service.get(id: Self.id, precision: "bf16")
        #expect(await state(f)?.state == .ready)
        try await eventually { sleeper.pending == 0 }
        #expect(sleeper.cancellations == 2)
        #expect(refusals(f) == 2 && pages.all.count == 1)
    }

    @Test func cancelEndsTheWaitAndTheRowLeavesTheTable() async throws {
        let sleeper = ManualSleeper()
        let (f, _) = try await fixture(refusals: -1, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { sleeper.pending == 1 }
        await f.service.cancelGet(id: Self.id)
        try await eventually { sleeper.pending == 0 }
        #expect(await state(f)?.gate == nil)
        #expect(await !f.service.rows().contains { $0.id == Self.id })
    }

    @Test func aRetryRefusedBeforeItStartsWaitsAgainInsteadOfFreezingTheCounter() async throws {
        let sleeper = ManualSleeper()
        let (f, _) = try await fixture(refusals: -1, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { sleeper.pending == 1 }
        // Arsenal prepares an update: the automatic retry is refused before its Get registers.
        _ = try await f.service.beginUpdate()
        sleeper.fire()
        try await eventually { await state(f)?.gate?.tries == 1 && sleeper.pending == 1 }
        #expect(await state(f)?.gate?.retryAt != nil, "a new counter, never one stuck at 0")
        await f.service.cancelUpdate()
        sleeper.fire()
        try await eventually { await state(f)?.gate?.tries == 2 && sleeper.pending == 1 }
        #expect(refusals(f) == 2)
        await f.service.cancelGet(id: Self.id)
    }

    @Test func aWaitBelongsToTheModelSoAnotherPrecisionsGetClearsTheOldRow() async throws {
        let (f, pages) = try await fixture(refusals: -1, token: Self.secret, sleeper: ManualSleeper()); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "int4") }
        let row = try #require(await f.service.rows().first { $0.id == Self.id })
        #expect(row.states["int4"]?.gate?.tries == 1 && row.states["bf16"]?.phase != PrecisionState.gatedPhase)
        #expect(pages.all.count == 1)
        await f.service.cancelGet(id: Self.id)
    }

    @Test func savingATokenRetriesWaitingAndGivenUpGetsAtOnce() async throws {
        let (f, _) = try await fixture(refusals: 4, token: Self.secret, sleeper: .immediate(), limit: 2); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { await state(f)?.gate?.gaveUp == true }
        #expect(refusals(f) == 3)
        // The next request is refused once more, then served: the token save retries the given-up Get by itself.
        await f.service.retryGatedGets()
        try await eventually { await state(f)?.state == .ready }
    }

    @Test func refreshMidCountdownKeepsTheWaitingRowAndItsRetry() async throws {
        let sleeper = ManualSleeper()
        let (f, _) = try await fixture(refusals: -1, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { sleeper.pending == 1 }
        try await f.service.refresh()
        #expect(await state(f)?.gate?.retryAt != nil, "the row still waits after Refresh")
        #expect(await f.service.rows().contains { $0.id == Self.id })
        sleeper.fire()
        try await eventually { await state(f)?.gate?.tries == 1 && sleeper.pending == 1 }
        #expect(refusals(f) == 2)
        await f.service.cancelGet(id: Self.id)
    }

    @Test func aRefusedClickKeepsTheCounterRunning() async throws {
        let sleeper = ManualSleeper()
        let (f, _) = try await fixture(refusals: -1, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { sleeper.pending == 1 }
        _ = try await f.service.beginUpdate()
        await #expect(throws: WorkerFailure.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        let pending = await f.service.gateRetryForTests().hasPendingRetry(Self.id)
        #expect(pending && sleeper.cancellations == 0, "the ring never stops at 0")
        await f.service.cancelUpdate()
        await f.service.cancelGet(id: Self.id)
    }

    @Test func aRetryFromACancelledWaitStartsNothing() async throws {
        let sleeper = ManualSleeper()
        let (f, _) = try await fixture(refusals: -1, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        await f.service.cancelGet(id: Self.id)
        // An automatic retry that was already inside get() when the wait ended: its episode is no longer current.
        await #expect(throws: CancellationError.self) {
            try await GateRetry.$episode.withValue(UUID()) { try await f.service.get(id: Self.id, precision: "bf16") }
        }
        let gate = await state(f)?.gate
        #expect(refusals(f) == 1 && gate == nil)
    }

    @Test func deleteEndsTheWait() async throws {
        let sleeper = ManualSleeper()
        let (f, _) = try await fixture(refusals: -1, token: Self.secret, sleeper: sleeper); defer { f.stop() }
        await #expect(throws: HFGateError.self) { try await f.service.get(id: Self.id, precision: "bf16") }
        try await eventually { sleeper.pending == 1 }
        try await f.service.delete(id: Self.id)
        try await eventually { sleeper.pending == 0 }
        let waiting = await f.service.gateRetryForTests().isWaiting(Self.id)
        #expect(!waiting)
    }

    @Test func tokenNeverReachesLogsRowsOrMessages() async throws {
        let (f, _) = try await fixture(refusals: -1, token: Self.secret, sleeper: .immediate(), limit: 2); defer { f.stop() }
        let events = await f.service.events()
        let collected = Task {
            var texts: [String] = []
            for await event in events { texts.append(String(describing: event)); if texts.count > 200 { break } }
            return texts
        }
        do { try await f.service.get(id: Self.id, precision: "bf16") } catch { #expect(!String(describing: error).contains(Self.secret)) }
        try await eventually { await state(f)?.gate?.gaveUp == true }
        // The token did go to the (loopback) Hub on every request...
        let tokens = try String(contentsOf: f.paths.support.deletingLastPathComponent().appendingPathComponent("hf/tokens"), encoding: .utf8)
        #expect(tokens.split(separator: "\n").allSatisfy { $0 == "token" })
        // ...and nowhere else: not in any file Arsenal wrote, its rows as the API serves them, or its events.
        let rows = String(decoding: try JSONEncoder().encode(await f.service.rows()), as: UTF8.self)
        #expect(!rows.contains(Self.secret))
        collected.cancel()
        #expect(!(await collected.value).joined().contains(Self.secret))
        let support = f.paths.support
        let files = FileManager.default.enumerator(at: support, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL } ?? []
        #expect(!files.isEmpty)
        for file in files where (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            let data = (try? Data(contentsOf: file)) ?? Data()
            #expect(data.range(of: Data(Self.secret.utf8)) == nil, "token in \(file.lastPathComponent)")
        }
    }
}
