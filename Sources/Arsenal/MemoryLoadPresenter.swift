// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore

/// The one app policy for service and loopback-client memory refusals. Native presentation is injected in tests.
enum MemoryLoadPresenter {
    @TaskLocal static var ownsRequest = false
    /// Where Cancel in the alert leaves the refusal it declined, so the caller can keep its figures ("Not run: needs 6 GB
    /// of memory; 2.9 GB free."). Cancel still throws CancellationError.
    @TaskLocal static var declines: DeclineRecord?
    final class DeclineRecord: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: LoadMemoryRefusal?
        var refusal: LoadMemoryRefusal? {
            get { lock.withLock { stored } }
            set { lock.withLock { stored = newValue } }
        }
    }
    @MainActor static var dismissMenus: () -> Void = {}
    @MainActor static var activate: () -> Void = { NSApp.activate(ignoringOtherApps: true) }
    @MainActor static var respond: (NSAlert) -> NSApplication.ModalResponse = { $0.runModal() }
    /// Unloads Arsenal's own idle models (the alert's third button); the app sets it to the service's Unload.
    @MainActor static var unloadModels: @MainActor @Sendable ([String]) async -> Void = { _ in }
    /// How long freed memory takes to show in the counters admission reads, after the idle workers exit.
    nonisolated(unsafe) static var settle: Duration = .milliseconds(500)
    static let unloadIdleTitle = "Unload Idle Models and Load"
    /// The alert's answer.
    enum Choice { case cancel, anyway, unloadIdle }

    @MainActor static func alert(_ refusal: LoadMemoryRefusal) -> NSAlert {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = refusal.title
        alert.informativeText = refusal.explanation
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Load Anyway")
        // Offered only when unloading Arsenal's own idle models makes room: then the load needs no swap.
        if refusal.unloadingIdleFits { alert.addButton(withTitle: unloadIdleTitle).keyEquivalent = "" }
        alert.buttons[0].keyEquivalent = "\r"
        alert.buttons[0].keyEquivalentModifierMask = []
        alert.buttons[1].keyEquivalent = ""
        alert.window.defaultButtonCell = alert.buttons[0].cell as? NSButtonCell
        alert.window.contentView?.addSubview(EscapeShortcut(cancel: alert.buttons[0]))
        return alert
    }

    /// A button has one key equivalent: assigning Escape removes its Return-default role. Keep the native default
    /// and route Escape through the same Cancel button. This zero-sized view adds no alert content or layout.
    private final class EscapeShortcut: NSView {
        weak var cancel: NSButton?
        init(cancel: NSButton) { self.cancel = cancel; super.init(frame: .zero) }
        required init?(coder: NSCoder) { nil }
        override func performKeyEquivalent(with event: NSEvent) -> Bool {
            guard event.type == .keyDown, event.keyCode == 53, event.modifierFlags.isDisjoint(with: [.command, .option, .control, .shift]),
                let cancel, cancel.isEnabled
            else { return super.performKeyEquivalent(with: event) }
            cancel.performClick(nil); return true
        }
    }

    /// Retry exactly once, with request-scoped force. Cancelling makes no second request.
    static func perform<T>(force: Bool = false, isolation: isolated (any Actor)? = #isolation, _ action: (Bool) async throws -> T) async throws -> T {
        do { return try await $ownsRequest.withValue(true) { try await action(force) } } catch {
            if force { throw error }
            let details: JSONObject?
            if let failure = error as? WorkerFailure, failure.code == "insufficient_memory" {
                details = failure.details
            } else if let failure = error as? ToolError, failure.code == "insufficient_memory" {
                details = failure.details
            } else {
                details = nil
            }
            guard let details, let refusal = LoadMemoryRefusal(details: details) else { throw error }
            try Task.checkCancellation()
            switch await confirm(refusal) {
            case .cancel: declines?.refusal = refusal; throw CancellationError()
            case .anyway:
                try Task.checkCancellation()
                return try await $ownsRequest.withValue(true) { try await action(true) }
            case .unloadIdle:
                // The idle models go, then the same request runs once more without swap; a second refusal is reported.
                await unloadIdle(refusal.idle.map(\.id))
                try await Task.sleep(for: settle)
                try Task.checkCancellation()
                return try await $ownsRequest.withValue(true) { try await action(false) }
            }
        }
    }

    @MainActor private static func unloadIdle(_ ids: [String]) async { await unloadModels(ids) }

    @MainActor private static func confirm(_ refusal: LoadMemoryRefusal) -> Choice {
        dismissMenus()
        activate()
        switch respond(alert(refusal)) {
        case .alertSecondButtonReturn: return .anyway
        case .alertThirdButtonReturn where refusal.unloadingIdleFits: return .unloadIdle
        default: return .cancel
        }
    }
}
