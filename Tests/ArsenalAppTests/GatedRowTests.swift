// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// A row whose Get waits for access to a gated Hugging Face model: Get stays Get (it retries now), a countdown ring sits in
/// the trash glyph's place, the line under the name says what to do in a few words, and nothing beside it moves.
@MainActor @Suite(.menuState, .serialized) final class GatedRowTests {
    static let now = Date(timeIntervalSince1970: 2_000_000)
    static let page = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
    static func wait(hasToken: Bool = true, tries: Int = 0, left: Double? = 27, mode: String? = "auto") -> GateWait {
        GateWait(
            repo: "black-forest-labs/FLUX.2-klein-4B", page: page, hasToken: hasToken, mode: mode, tries: tries, limit: 10,
            retryAt: left.map { now.addingTimeInterval($0) })
    }
    static func row(_ wait: GateWait?) -> ModelRow {
        var row = ModelRow(
            id: "flux", name: "FLUX.2 Klein 4B", kind: "images", params: "4B", licence: "Apache-2.0", precisions: ["bf16"], selectedPrecision: "bf16",
            sizeBytes: 15_980_140_843)
        row.purpose = "Image generation"
        row.states["bf16"] =
            wait.map {
                PrecisionState(state: .failed, error: "gated", phase: PrecisionState.gatedPhase, gate: $0)
            } ?? PrecisionState()
        return row
    }

    @Test func eachWaitingStateSaysWhatToDoAndKeepsGet() {
        let waiting = RowPresentation(Self.row(Self.wait()))
        #expect(waiting.action == .get && !waiting.failed && waiting.actionTitle == "Get \u{00b7} 16.0 GB")
        #expect(waiting.countdown == Self.now.addingTimeInterval(27) && !waiting.locked && !waiting.cancellable)
        #expect(waiting.line.text == "Waiting for access" && waiting.line.tone == .pending)
        #expect(waiting.line.help.contains(Self.page) && waiting.line.help.contains("Get retries now"))

        let still = RowPresentation(Self.row(Self.wait(tries: 3)))
        #expect(still.line.text == "Still waiting for access" && still.line.help.contains("Tried 3 of 10."))
        #expect(still.actionTitle == waiting.actionTitle)

        let noToken = RowPresentation(Self.row(Self.wait(hasToken: false)))
        #expect(noToken.line.text == "Add a token in Settings" && noToken.line.tone == .pending)
        #expect(noToken.line.help.contains("Settings \u{203a} Hugging Face Token\u{2026}") && noToken.countdown != nil)

        let gaveUp = RowPresentation(Self.row(Self.wait(tries: 10, left: nil)))
        #expect(gaveUp.line.text == "No access on Hugging Face" && gaveUp.line.tone == .error)
        #expect(gaveUp.countdown == nil && gaveUp.action == .get && gaveUp.actionTitle == "Get \u{00b7} 16.0 GB" && !gaveUp.failed)

        let manual = RowPresentation(Self.row(Self.wait(mode: "manual")))
        #expect(manual.line.help.contains("can take days"))
        #expect(ModelHelp.action(Self.row(Self.wait()), presentation: waiting) == "Retries now")
        #expect(ModelHelp.action(Self.row(Self.wait(left: nil)), presentation: gaveUp) == "Tries the download again")
    }

    @Test func toolWindowsAndTheMenuSayItInAFewWordsWithTryAgain() throws {
        let center = ToolCenter(client: { nil }, playback: FixturePlayback())
        let waiting = try #require(center.readiness(Self.row(Self.wait()), task: .tool(.image)))
        #expect(waiting.note == "Waiting" && waiting.problem.message == "Waiting for access to FLUX.2 Klein 4B on Hugging Face.")
        #expect(waiting.problem.action == .download && waiting.problem.actionTitle == "Try Again")
        let noToken = try #require(center.readiness(Self.row(Self.wait(hasToken: false)), task: .tool(.image)))
        #expect(noToken.problem.message == "FLUX.2 Klein 4B needs a Hugging Face token: add one in Settings.")
        let gaveUp = try #require(center.readiness(Self.row(Self.wait(left: nil)), task: .tool(.image)))
        #expect(gaveUp.note == "No access" && gaveUp.problem.message == "No access to FLUX.2 Klein 4B on Hugging Face yet.")
        #expect(center.menuRecovery(Self.row(Self.wait()), tool: .image)?.reason == "waiting for access")
    }

    @Test func countdownNeitherMovesTheButtonNorTakesItsClick() throws {
        RowActionView.now = { Self.now }
        defer { RowActionView.now = { Date() } }
        var performed = 0, cancelled = 0
        func view(_ countdown: Date?) -> RowActionView {
            let v = RowActionView(frame: NSRect(x: 0, y: 0, width: RowAction.width, height: RowAction.height))
            v.spec = RowAction(
                title: "Get \u{00b7} 16.0 GB", enabled: true, deletable: true, help: "Retries now", onPerform: { performed += 1 }, onDelete: {},
                onCancel: { cancelled += 1 }, countdown: countdown)
            return v
        }
        let plain = view(nil), counting = view(Self.now.addingTimeInterval(27)), late = view(Self.now.addingTimeInterval(9))
        #expect(counting.buttonRect == plain.buttonRect && late.buttonRect == plain.buttonRect, "no jump of the button")
        #expect(counting.secondsLeft == 27 && late.secondsLeft == 9 && plain.secondsLeft == nil)
        #expect(counting.accessibilityLabel() == "Get \u{00b7} 16.0 GB, retries in 27 seconds")
        #expect(counting.toolTipText(at: NSPoint(x: counting.trashRect.midX, y: counting.trashRect.midY)) == RowAction.stopWaitingHelp)
        #expect(counting.toolTipText(at: NSPoint(x: counting.buttonRect.midX, y: counting.buttonRect.midY)) == "Retries now")
        // Clicks through the view's own mouse-down: the button retries now, the ring stops waiting.
        let window = NSWindow(contentRect: counting.frame, styleMask: .borderless, backing: .buffered, defer: true)
        window.isReleasedWhenClosed = false
        window.contentView?.addSubview(counting)
        func click(_ point: NSPoint, _ n: Int) throws {
            let location = counting.convert(point, to: nil)
            counting.mouseDown(
                with: try #require(
                    NSEvent.mouseEvent(
                        with: .leftMouseDown, location: location, modifierFlags: [], timestamp: TimeInterval(n), windowNumber: window.windowNumber, context: nil,
                        eventNumber: n, clickCount: 1, pressure: 1)))
        }
        try click(NSPoint(x: counting.buttonRect.midX, y: counting.buttonRect.midY), 1)
        #expect(performed == 1 && cancelled == 0)
        try click(NSPoint(x: counting.trashRect.midX, y: counting.trashRect.midY), 2)
        #expect(performed == 1 && cancelled == 1)
        #expect(counting.accessibilityCustomActions()?.map(\.name) == [RowAction.stopWaitingHelp])
        window.close()
    }

    @Test func tokenAlertSavesRemovesAndRefusesSpaces() {
        var saved: [String] = [], removed = 0
        let store = HuggingFaceToken.Store(
            write: {
                saved.append($0); return errSecSuccess
            },
            delete: {
                removed += 1; return errSecSuccess
            }, hasToken: { true })
        #expect(HuggingFaceToken.apply(.alertFirstButtonReturn, text: "  hf_abc123\n", store: store) == .saved)
        #expect(saved == ["hf_abc123"])
        #expect(HuggingFaceToken.apply(.alertFirstButtonReturn, text: "", store: store) == .cancelled)
        if case .failed = HuggingFaceToken.apply(.alertFirstButtonReturn, text: "hf_a b", store: store) {} else { Issue.record("a token with a space was saved") }
        #expect(HuggingFaceToken.apply(.alertSecondButtonReturn, text: "hf_x", store: store) == .cancelled)
        #expect(HuggingFaceToken.apply(.alertThirdButtonReturn, text: "", store: store) == .removed)
        #expect(saved == ["hf_abc123"] && removed == 1)
        let refusing = HuggingFaceToken.Store(write: { _ in errSecAuthFailed }, delete: { errSecSuccess }, hasToken: { false })
        if case .failed(let why) = HuggingFaceToken.apply(.alertFirstButtonReturn, text: "hf_secret", store: refusing) {
            #expect(!why.contains("hf_secret"))
        } else {
            Issue.record("a refused save reported success")
        }
    }

    /// Review renders of the four waiting states in the real Models table (2x, dark): run with ARSENAL_GATED_RENDERS=<dir>.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_GATED_RENDERS"] != nil))
    func renderGatedRows() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_GATED_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-gated-renders-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support); Appearance.shared.set(.dark); NSApp.appearance = nil }
        RowActionView.now = { Self.now }
        defer { RowActionView.now = { Date() } }
        let scenes: [(String, GateWait)] = [
            ("waiting", Self.wait()), ("still-waiting", Self.wait(tries: 2, left: 14)), ("no-token", Self.wait(hasToken: false, left: 30)),
            ("gave-up", Self.wait(tries: 10, left: nil))
        ]
        var tips: [String] = []
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            try await Task.sleep(for: .milliseconds(100))
            for (name, wait) in scenes {
                var rows = Fixtures.rows(.loaded)
                let i = try #require(rows.firstIndex { $0.id == "flux2-klein-4b" })
                rows[i].states["bf16"] = PrecisionState(state: .failed, error: "gated", phase: PrecisionState.gatedPhase, gate: wait)
                let c = TableController(service: FixtureService(rows: rows, supportDirectory: support.appendingPathComponent(name + mode.rawValue)), chip: "M5 Max")
                await c.refresh()
                await QA.renderTable(c, to: directory.appendingPathComponent("gated-\(name)-\(mode.rawValue).png"))
                if mode == .dark {
                    let p = c.presentation(try #require(c.rows.first { $0.id == "flux2-klein-4b" }))
                    tips.append("## \(name)\nline: \(p.line.text)\nline hover: \(p.line.help)\nbutton: \(p.actionTitle)\n")
                }
                c.stop()
            }
        }
        try tips.joined(separator: "\n").write(to: directory.appendingPathComponent("gated-tooltips.txt"), atomically: true, encoding: .utf8)
    }
}
