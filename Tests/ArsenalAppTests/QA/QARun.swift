// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore

// What the QA modes (renders, real cycles) share: a stamped step log, their failure, the end of a run, waiting on the
// table's rows, rendering the table, and the check that no worker outlived the run.

struct QAFailure: LocalizedError {
    let errorDescription: String?
    init(_ message: String?) { errorDescription = message }
}

/// One line per step, stamped with the seconds since the run started; printed as it happens (a run killed by its alarm
/// keeps every step in its log) and written out at the end.
@MainActor final class QALog {
    private let started = Date()
    private(set) var lines: [String] = []
    func say(_ line: String) {
        let stamped = String(format: "%7.2f s  ", Date().timeIntervalSince(started)) + line
        print(stamped); lines.append(stamped)
    }
    func write(to url: URL) { try? lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8) }
}

@MainActor enum QA {
    /// The end of a QA run: the harness test waiting on it resumes.
    static func stopApp() { QAHarness.finish() }

    /// Re-checks `condition` every 100 ms until it holds; past `timeout` the run fails.
    static func wait(timeout: TimeInterval, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !(await condition()) {
            if Date() > deadline { throw QAFailure("timed out") }
            try await Task.sleep(for: .milliseconds(100))
        }
    }

    /// Re-reads the table until the model's row satisfies `done`, logging each change of what the row shows. `each`
    /// sees every reading (a run renders the states it passes through). An error on the row or in the footer fails.
    static func follow(
        _ table: TableController, model: String, log: QALog, step: String, timeout: TimeInterval,
        until done: (RowPresentation) -> Bool, each: (ModelRow) async throws -> Void = { _ in }
    ) async throws {
        var last = ""
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            await table.refresh()
            guard let r = table.rows.first(where: { $0.id == model }) else { throw QAFailure("row vanished") }
            let p = table.presentation(r)
            let line = "\(step): \(p.busyText ?? p.actionTitle) · \(p.status.text)\(p.pending ? " · red" : "")"
            if line != last { log.say(line); last = line }
            if p.status.tone == .error { throw QAFailure(p.status.help) }
            try await each(r)
            if done(p) { return }
            if let error = table.lastError { throw QAFailure(error) }
            if Date() > deadline { throw QAFailure("\(step) did not finish in \(Int(timeout)) s") }
            try await Task.sleep(for: .milliseconds(100))
        }
    }

    /// The Activity submenu's panel as it opens: its fixed width, the height its content asks for, the menu's corner.
    /// `wallpaper`: render over a blurred busy wallpaper under the translucent material instead of the plain material.
    @discardableResult static func renderActivity(_ model: ActivityModel, to url: URL, wallpaper: Bool = false) async -> NSSize {
        let view = ActivityHostingView(rootView: ActivityPanel(model: model))
        // The host lays its content out at its natural size (as ActivityMenu sets it); a bare frame leaves it unplaced.
        view.setNaturalSize(NSSize(width: ActivityPanel.width, height: ActivityPanel.minHeight))
        view.setNaturalSize(NSSize(width: ActivityPanel.width, height: ActivityMenu.height(view)))
        let container = NSView(frame: view.frame.insetBy(dx: -6, dy: -6))
        container.wantsLayer = true
        let p = Appearance.shared.palette
        if p.translucent {
            container.layer?.backgroundColor = MenuMock.material(dark: true).cgColor
            container.layer?.contents = V21RenderDelegate.blurred(V21RenderDelegate.darkPicture(), size: container.frame.size)
            container.layer?.contentsGravity = .resizeAspectFill
        } else {
            MenuMock.applyMaterial(to: container.layer, dark: p.dark, picture: wallpaper ? MenuMock.wallpaper(size: container.frame.size) : nil)
        }
        container.layer?.cornerRadius = 10; container.layer?.masksToBounds = true
        view.frame.origin = NSPoint(x: 6, y: 6)
        container.addSubview(view)
        container.appearance = Appearance.shared.nsAppearance
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in MenuMock.capture(container, to: url) { done.resume() } }
        return view.frame.size
    }

    static func renderTable(_ table: TableController, to url: URL, wallpaper: Bool = false) async {
        let size = NSSize(width: ModelTable.width(table) + 24, height: ModelTable.height(table) + 20)
        let backdrop = wallpaper ? MenuMock.wallpaper(size: size) : nil
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            RenderDelegate.renderTable(table, to: url, backdrop: backdrop) { done.resume() }
        }
    }

    /// No runner process outlived `step` (Unload, Remove): this app has no child processes left.
    static func noWorkersLeft(after step: String, log: QALog) throws {
        let children = Process(); children.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        children.arguments = ["-P", String(ProcessInfo.processInfo.processIdentifier)]
        let pipe = Pipe(); children.standardOutput = pipe
        try children.run(); children.waitUntilExit()
        let left = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).split(separator: "\n")
        log.say("worker processes left after \(step): \(left.count)")
        if !left.isEmpty { throw QAFailure("a worker process outlived \(step): \(left)") }
    }
}
