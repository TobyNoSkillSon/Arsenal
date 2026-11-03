// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore

/// `Arsenal --render-v2 <dir>`: the v2 surfaces at real size from the real catalog (ARSENAL_CATALOG, ARSENAL_SUPPORT_DIR
/// as the app uses them; a fresh support dir is day one). Table membership and settings go through the service; the
/// loaded/pending states are set on the service's own rows for the picture (no GPU, no worker). Writes v2-*.png and
/// v2-tooltips.txt. No screen interaction: every window is drawn offscreen.
@MainActor final class V2RenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private var log: [String] = []
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.appearance = NSAppearance(named: .darkAqua)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { print("render-v2 FAILED: \(TableController.message(error))") }
            QA.stopApp()
        }
    }

    private func run() async throws {
        let (service, error) = await ServiceFactory.make(paths: ArsenalPaths())
        if let error { throw QAFailure(error) }
        let table = TableController(service: service)
        await table.refresh()
        say("day one: \(table.rows.count) rows; catalog \(table.catalog.map(FooterText.catalog) ?? "?")")
        try await shot(table, "day-one")

        // The browser has its own renders since v2.1 (`--render-v21`); the two suggested picks are added as it would.
        for id in ["kokoro-82m", "qwen3.8-27b"] { try await service.add(id: id) }
        var tips: [String] = []

        // The table with rows: more added through the service (Add's path), all as on a first Get.
        for id in ["flux2-klein-4b", "laya-english", "stable-audio-3-small-sfx", "re-use"] { try await service.add(id: id) }
        await table.refresh()
        say("table: " + table.groups.map { "\($0.kind.title) \($0.rows.count)" }.joined(separator: ", "))
        try await shot(table, "table-rows")
        table.renderHoverPinned = true
        try await shot(table, "table-pinned-hover")
        table.renderHoverPinned = false

        // Settings open in their own window (render-settings); the table shows the rows.
        guard table.rows.contains(where: { $0.id == "qwen3.8-27b" }), let kokoro = table.rows.first(where: { $0.id == "kokoro-82m" }) else {
            throw QAFailure("qwen3.8-27b / kokoro-82m missing from the table")
        }

        // Loaded states for the picture: Kokoro loaded fp16 with int8 clicked (a dtype change), Qwen loaded int4 with
        // its context changed (a load-time setting, needsReload) — both red with Reload.
        var rows = table.rows
        func edit(_ id: String, _ change: (inout ModelRow) -> Void) { if let i = rows.firstIndex(where: { $0.id == id }) { change(&rows[i]) } }
        edit("kokoro-82m") { r in
            r.states["fp16"] = .init(state: .loaded); r.states["int8"] = .init(state: .ready); r.selectedPrecision = "fp16"
        }
        edit("qwen3.8-27b") { r in
            r.states["int4"] = .init(state: .loaded); r.selectedPrecision = "int4"; r.needsReload = true
        }
        table.apply(rows: rows, kinds: table.kinds, catalog: table.catalog)
        if let i = table.settings["qwen3.8-27b"]?.firstIndex(where: { $0.applies == "load" }), let s = table.settings["qwen3.8-27b"]?[i] {
            let other = s.choices.first { $0 != s.value } ?? s.value
            table.settings["qwen3.8-27b"]?[i].value = other; table.settings["qwen3.8-27b"]?[i].pending = true
        }
        if let k = table.rows.first(where: { $0.id == "kokoro-82m" }) { table.select(k, column: "8") }
        try await shot(table, "red-row")

        // Quiet rows with no Fast kernels: Kokoro back at its loaded fp16, Qwen loaded with its loaded context (blue,
        // not red), FLUX not loaded; Qwen and FLUX show the switch pinned down at Exact.
        edit("qwen3.8-27b") { r in r.needsReload = false }
        table.apply(rows: rows, kinds: table.kinds, catalog: table.catalog)
        if let k = table.rows.first(where: { $0.id == "kokoro-82m" }) { table.select(k, column: "16") }
        if let i = table.settings["qwen3.8-27b"]?.firstIndex(where: { $0.applies == "load" }) {
            table.settings["qwen3.8-27b"]?[i].value = table.settings["qwen3.8-27b"]![i].defaultValue; table.settings["qwen3.8-27b"]?[i].pending = false
        }
        say("quiet: " + table.rows.map { "\($0.id) \(table.presentation($0).pending ? "red" : table.presentation($0).hot ? "blue" : "plain")" }.joined(separator: ", "))
        try await shot(table, "fast-pinned-exact")

        tips.append("\n## Table (red-row state)\n")
        tips.append("[Get more models]\n\(PinnedRow.help)\n")
        tips.append("[gear]\n\(GearCell.help)\n")
        for row in table.rows {
            let p = table.presentation(row)
            tips.append(
                "[\(row.name) · Action \(p.actionTitle)\(p.pending ? " · red" : "")]\n\(ModelHelp.action(row, presentation: p, changes: table.pendingChanges(row), loadChoices: table.loadChoices(row)))\n"
            )
            for s in table.settings[row.id] ?? [] { tips.append("[\(row.name) · setting \(s.title) (\(s.applies), \(s.type))]\n\(s.tooltip)\n") }
        }
        tips.append("[trash]\n\(RowAction.deleteHelp)\n")
        let remove = ModelsMenu.removeAlert(kokoro)
        tips.append("[Remove alert]\n\(remove.messageText)\n\(remove.informativeText)\n\(remove.buttons.map(\.title).joined(separator: " · "))\n")

        // The one-time code consent prompt for a local model (the stub's local fixture row: a dropped folder with a
        // native local runner whose contents have not been approved).
        let alert = ModelsMenu.codeAlert(StubModelsService.localFixture)
        tips.append("[Consent alert]\n\(alert.messageText)\n\(alert.informativeText)\n\(alert.buttons.map(\.title).joined(separator: " · "))\n")
        let mock = AlertMock(alert)
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.capture(mock, to: directory.appendingPathComponent("v2-consent-prompt.png")) { done.resume() }
        }
        say("wrote v2-consent-prompt.png")
        try (tips.joined(separator: "\n") + "\n").write(to: directory.appendingPathComponent("v2-tooltips.txt"), atomically: true, encoding: .utf8)
        try log.joined(separator: "\n").write(to: directory.appendingPathComponent("v2-renders.txt"), atomically: true, encoding: .utf8)
    }

    private func say(_ line: String) { print(line); log.append(line) }

    private func shot(_ table: TableController, _ name: String) async throws {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            RenderDelegate.renderTable(table, to: directory.appendingPathComponent("v2-\(name).png")) { done.resume() }
        }
        say("wrote v2-\(name).png")
    }

}

/// An NSAlert drawn the way macOS lays it out (icon, bold message, informative text, buttons right to left), with
/// the alert's own strings and real buttons: a real alert panel cannot be cached offscreen (only its icon draws).
final class AlertMock: NSView {
    init(_ alert: NSAlert) {
        let width: CGFloat = 260, pad: CGFloat = 16
        let icon = NSImageView(image: NSImage(named: NSImage.applicationIconName) ?? NSImage())
        let title = NSTextField(wrappingLabelWithString: alert.messageText)
        title.font = .boldSystemFont(ofSize: 13); title.alignment = .center
        let info = NSTextField(wrappingLabelWithString: alert.informativeText)
        info.font = .systemFont(ofSize: 11); info.alignment = .center; info.textColor = .labelColor
        let buttons = alert.buttons.map { b -> NSButton in
            let button = NSButton(title: b.title, target: nil, action: nil); button.bezelStyle = .push; button.controlSize = .large
            button.keyEquivalent = b === alert.buttons.first ? "\r" : ""; return button
        }
        for v in [title, info] { v.preferredMaxLayoutWidth = width - 2 * pad; v.frame.size = v.fittingSize }
        let iconSize: CGFloat = 64, buttonHeight: CGFloat = 28
        let height = pad + iconSize + 12 + title.frame.height + 8 + info.frame.height + 16 + CGFloat(buttons.count) * (buttonHeight + 8) + pad - 8
        super.init(frame: NSRect(x: 0, y: 0, width: width, height: height))
        wantsLayer = true
        layer?.backgroundColor = NSColor(calibratedRed: 0.17, green: 0.17, blue: 0.18, alpha: 1).cgColor
        layer?.cornerRadius = 14; layer?.borderWidth = 1; layer?.borderColor = NSColor.separatorColor.cgColor
        appearance = NSAppearance(named: .darkAqua)
        var y = height - pad - iconSize
        icon.frame = NSRect(x: (width - iconSize) / 2, y: y, width: iconSize, height: iconSize)
        y -= 12 + title.frame.height; title.frame.origin = NSPoint(x: (width - title.frame.width) / 2, y: y)
        y -= 8 + info.frame.height; info.frame.origin = NSPoint(x: (width - info.frame.width) / 2, y: y)
        y -= 16
        // Stacked full width as macOS lays out a narrow alert, the default (first added) on top.
        for b in buttons {
            y -= buttonHeight; b.frame = NSRect(x: pad, y: y, width: width - 2 * pad, height: buttonHeight); y -= 8
        }
        for v in [icon, title, info] + buttons { addSubview(v) }
    }
    required init?(coder: NSCoder) { nil }
}
