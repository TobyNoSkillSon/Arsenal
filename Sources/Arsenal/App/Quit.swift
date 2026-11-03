// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Foundation

enum Bounded {
    /// Runs `work`; true when it finished within `limit`. Past the limit the caller moves on (quit must not hang on a
    /// stuck shutdown) and `work` keeps running until the process ends.
    static func wait(_ limit: Duration, _ work: @escaping @Sendable () async -> Void) async -> Bool {
        await withCheckedContinuation { continuation in
            let once = Once(continuation)
            let timer = Task {
                try? await Task.sleep(for: limit); if !Task.isCancelled { once.resume(false) }
            }
            Task {
                await work(); timer.cancel(); once.resume(true)
            }
        }
    }
    /// The first of the two finishers resumes the continuation.
    private final class Once: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Bool, Never>?
        init(_ continuation: CheckedContinuation<Bool, Never>) { self.continuation = continuation }
        func resume(_ value: Bool) {
            let pending = lock.withLock { () -> CheckedContinuation<Bool, Never>? in
                defer { continuation = nil }; return continuation
            }
            pending?.resume(returning: value)
        }
    }
}

/// `kill <pid>` (SIGTERM) quits like the menu's Quit, so a scripted stop also shuts the runners down. Event-driven
/// (a dispatch signal source), no polling.
@MainActor enum TerminationSignal {
    private static var source: DispatchSourceSignal?
    private static var quit: (@MainActor () -> Void)?
    static func install(_ quit: @escaping @MainActor () -> Void) {
        guard source == nil else { return }
        self.quit = quit
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        // Not `quit()` here: this handler runs inside the main queue's drain, and terminate's .terminateLater wait spins
        // a nested run loop whose reply (a main-actor Task) needs that same serial queue, so the app hung forever on
        // SIGTERM. A run-loop block runs outside the drain, so the nested loop can service the main queue. Also in the
        // modal-panel mode: with a question up the default mode never runs, and quit ends the question first.
        source.setEventHandler { RunLoop.main.perform(inModes: [.default, .modalPanel]) { MainActor.assumeIsolated { TerminationSignal.quit?() } } }
        source.resume()
        self.source = source
    }
}

/// Arsenal's questions are app-modal (`runModal`), and AppKit does not quit from inside a modal session: with a stale
/// "Delete …?" alert up, Quit in the menu did nothing (VM run, 2 Nov). Quit first ends each open alert or panel as
/// Cancel, then terminates once no modal session runs.
@MainActor enum AppModal {
    struct Hooks {
        /// The modal window now, nil when no modal session runs.
        var modal: @MainActor () -> AnyObject?
        /// Ends the innermost modal session as Cancel.
        var cancel: @MainActor () -> Void
        /// `fromQueue`: called from a main-queue block, so terminate must run outside the queue's drain.
        var terminate: @MainActor (_ fromQueue: Bool) -> Void
        /// Runs the next step on a later turn of the main queue (it is serviced inside modal sessions too).
        var next: @MainActor (@escaping @MainActor () -> Void) -> Void
    }
    static let app = Hooks(
        modal: { NSApp.modalWindow }, cancel: { cancelCurrent() },
        terminate: { fromQueue in
            // Terminate's .terminateLater wait needs the main queue for its reply: never from inside its drain (TerminationSignal).
            guard fromQueue else { NSApp.terminate(nil); return }
            RunLoop.main.perform(inModes: [.default]) { MainActor.assumeIsolated { NSApp.terminate(nil) } }
        },
        next: { work in DispatchQueue.main.async { MainActor.assumeIsolated(work) } })

    /// Ends the innermost modal session as Cancel (`stopModal`, not `abortModal`, whose exception would cross Swift
    /// frames); the posted event wakes its loop, which checks for the stop only after an event.
    static func cancelCurrent() {
        NSApp.stopModal(withCode: .cancel)
        let wake = NSEvent.otherEvent(
            with: .applicationDefined, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil, subtype: 0, data1: 0, data2: 0)
        if let wake { NSApp.postEvent(wake, atStart: false) }
    }

    /// Quit (the menu, ⌘Q, SIGTERM, the updater): every open question answers Cancel, then the app terminates.
    static func quit(_ hooks: Hooks = app) { step(hooks, ended: nil, waited: 0, fromQueue: false) }
    /// Turns to wait for an ended session to unwind before ending it again (one that ignored the stop).
    static let unwindTurns = 50
    private static func step(_ hooks: Hooks, ended: AnyObject?, waited: Int, fromQueue: Bool) {
        guard let modal = hooks.modal() else { hooks.terminate(fromQueue); return }
        // The session just ended may still be unwinding: wait for it rather than end the one beneath it as well.
        let unwinding = modal === ended && waited < unwindTurns
        if !unwinding { hooks.cancel() }
        hooks.next { step(hooks, ended: modal, waited: unwinding ? waited + 1 : 0, fromQueue: true) }
    }
}

/// Quit's menu target. In an app-modal session AppKit sends a menu action (the item or ⌘Q) only to a target whose
/// `worksWhenModal` is true, so with a "Delete …?" question up the menu's Quit never reached `AppModal.quit` (VM run,
/// 2 Nov). Only Quit works during a question; every other item keeps waiting for the answer.
@MainActor final class QuitTarget: NSObject {
    static let shared = QuitTarget()
    @objc var worksWhenModal: Bool { true }
    @objc func quit(_ sender: Any?) { AppModal.quit() }
}
