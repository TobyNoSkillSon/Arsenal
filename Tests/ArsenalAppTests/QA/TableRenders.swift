// Render harness adapted from Projects/Vella Sources/Vella/QA/TableRenders.swift @ f4c5c34 (Vella; relicensed MIT by its author, 2 Oct 2026).
// `Arsenal --render-table <dir>` draws the table in each fixture state and the status menu offscreen, at real size and
// 2× pixels, to PNGs; table-tooltips.txt lists every cell's tooltip (a PNG cannot hover). No service work, no files
// outside <dir>, no screen interaction.
import AppKit
@testable import Arsenal
import ArsenalCore
import SwiftUI

@MainActor final class RenderDelegate: NSObject, NSApplicationDelegate {
    let directory: URL
    init(directory: URL) { self.directory = directory }

    struct Shot {
        var name: String
        var rows: [ModelRow]
        var order: [String] = []
        var previews: [(String, String)] = []
        var hover: String?
        var hoverGroup: String?
        var drag: (String, Int)?
        var error: String?
    }

    static func shots() -> [Shot] {
        var shots = Fixtures.Scene.allCases.map { Shot(name: $0.rawValue, rows: Fixtures.rows($0)) }
        if let i = shots.firstIndex(where: { $0.name == Fixtures.Scene.reload.rawValue }) {
            // Kokoro loaded at bf16 with int8 clicked (green Reload); Laya on disk under the pointer (Load + trash).
            shots[i].previews = [("kokoro-82m", "8")]
            shots[i].hover = "laya-english"
        }
        var moved = Shot(name: "moved-group", rows: Fixtures.rows(.loaded), order: ["chat", "voices", "decisions", "images"])
        moved.hoverGroup = "chat"
        shots.append(moved)
        var dragging = Shot(name: "dragging-group", rows: Fixtures.rows(.loaded))
        dragging.drag = ("images", 0)
        shots.append(dragging)
        return shots
    }

    static func controller(_ shot: Shot) -> TableController {
        let c = TableController(service: FixtureService(rows: shot.rows), chip: "M5 Max")
        c.apply(rows: shot.rows, kinds: Fixtures.kinds, catalog: CatalogStatus(source: "local", pinnedCommit: "f600b7d4c1e2", verified: true))
        if !shot.order.isEmpty {
            for (slot, id) in shot.order.enumerated() { c.moveGroup(id, toSlot: slot) }
        }
        for (id, column) in shot.previews { if let r = shot.rows.first(where: { $0.id == id }) { c.select(r, column: column) } }
        c.renderHover = shot.hover; c.renderHoverGroup = shot.hoverGroup
        c.drag = shot.drag.map { (group: $0.0, slot: $0.1) }
        c.lastError = shot.error
        return c
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.appearance = NSAppearance(named: .darkAqua)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        writeTooltips()
        render(Self.shots(), 0)
    }

    private func render(_ shots: [Shot], _ index: Int) {
        guard index < shots.count else { renderMenu { QA.stopApp() }; return }
        let shot = shots[index]
        Self.renderTable(Self.controller(shot), to: directory.appendingPathComponent("stub-table-\(shot.name).png")) { [self] in render(shots, index + 1) }
    }

    /// The table on a stand-in for the menu's native material, as inside the Models… submenu (the table draws no fill).
    /// `backdrop`: what the translucent material shows through (a blurred desktop picture), under the material's colour.
    static func renderTable(_ controller: TableController, to url: URL, backdrop: NSImage? = nil, done: @escaping () -> Void) {
        let table = NSHostingView(rootView: ModelsMenuView(controller: controller))
        table.frame = NSRect(x: 0, y: 0, width: ModelTable.width(controller), height: ModelTable.height(controller))
        let container = NSView(frame: table.frame.insetBy(dx: -12, dy: -10))
        container.wantsLayer = true
        let p = Appearance.shared.palette
        if p.translucent {
            container.layer?.backgroundColor = MenuMock.material(dark: true).cgColor
            if let backdrop { container.layer?.contents = backdrop; container.layer?.contentsGravity = .resizeAspectFill }
        } else {
            MenuMock.applyMaterial(to: container.layer, dark: p.dark, picture: backdrop)
        }
        container.layer?.cornerRadius = 10; container.layer?.masksToBounds = true
        table.frame.origin = NSPoint(x: 12, y: 10)
        container.addSubview(table)
        container.appearance = Appearance.shared.nsAppearance
        MenuMock.capture(container, to: url, prepare: { MenuMock.standInMaterials(in: container, dark: p.dark, picture: backdrop) }, done: done)
    }

    /// The status menu with two loaded models contributing actions, plus its Keep Hot and Memory submenus.
    private func renderMenu(done: @escaping () -> Void) {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-render-\(ProcessInfo.processInfo.processIdentifier)")
        let rows = Fixtures.rows(.loaded)
        let app = AppDelegate(service: FixtureService(rows: rows), paths: ArsenalPaths(support: support))
        app.table.apply(rows: rows, kinds: Fixtures.kinds)
        Task { @MainActor in
            await app.setPreview(
                actions: [
                    LoadedAction(id: "kokoro-82m-speak-clipboard", modelID: "kokoro-82m", title: "Speak Clipboard", op: "speak", input: "clipboard"),
                    LoadedAction(id: "pocket-tts-speak-clipboard", modelID: "pocket-tts", title: "Speak Clipboard", op: "speak", input: "clipboard")
                ],
                memory: MemorySnapshot(
                    totalMB: 128_000, availableMB: 86_900, workersMB: 3_400,
                    models: ["kokoro-82m": WorkerMemory(footprintMB: 410), "pocket-tts": WorkerMemory(footprintMB: 520), "laya-english": WorkerMemory(footprintMB: 1_310)]),
                lastResult: "Spoke 4.2 s of audio")
            MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("stub-menu-loaded-actions.png")) { [self] in
                MenuMock.renderSubmenus(of: app.menu, titles: [("Keep Hot", "keep-hot"), ("Memory", "memory")], into: directory, prefix: "stub-menu-") {
                    try? FileManager.default.removeItem(at: support)
                    done()
                }
            }
        }
    }

    /// Every cell's tooltip in the loaded, first-launch, download and local-copy/retention scenes: "[Row · Column]" then the text as shown.
    private func writeTooltips() {
        let text = [Fixtures.Scene.loaded, .firstLaunch, .downloading, .localCopy, .verifying, .retention].map { scene in
            "## \(scene.rawValue)\n\n" + Self.tooltips(Self.controller(Shot(name: scene.rawValue, rows: Fixtures.rows(scene))))
        }.joined(separator: "\n")
        try? text.write(to: directory.appendingPathComponent("stub-table-tooltips.txt"), atomically: true, encoding: .utf8)
    }

    static func tooltips(_ c: TableController) -> String {
        var blocks: [String] = ["[Refresh]\n\(c.refreshHelp)\n"]
        for group in c.groups {
            blocks.append("[\(group.kind.title) · header]\n\(GroupHeader.titleHelp(group.kind))\n")
            for column in TableStats.columns { blocks.append("[\(group.kind.title) · \(column.title)]\n\(TableStats.help(kind: group.kind, column: column.id))\n") }
            for row in group.rows {
                let p = c.presentation(row)
                blocks.append("[\(row.name) · Model]\n\(ModelHelp.text(row, presentation: p))\n")
                blocks.append("[\(row.name) · Status]\n\(p.status.text)" + (p.status.help.isEmpty ? "" : ": " + p.status.help) + "\n")
                for column in PrecisionColumn.all {
                    let tip = ModelHelp.precisionTooltip(row, column: column, presentation: p)
                    let mark = ModelHelp.warnings(row)[column] != nil && ModelHelp.unavailable(row)[column] == nil ? " ▲" : ""
                    blocks.append("[\(row.name) · Precision \(PrecisionColumn.label(row, column))\(mark)]\n\(tip)\n")
                }
                for column in TableFigures.columns {
                    let f = TableFigures.figure(row, precision: p.selected, column: column, host: c.chip, shown: c.shownKinds[row.id])
                    let shown = ([f.value ?? "▲"] + [f.reference, f.delta?.text].compactMap { $0 }).joined(separator: " / ")
                    blocks.append("[\(row.name) · \(column.title) \(shown)]\n\(f.help)\n")
                }
                blocks.append("[\(row.name) · Action \(p.busyText ?? p.actionTitle)]\n\(ModelHelp.action(row, presentation: p))\n")
            }
        }
        return blocks.joined(separator: "\n")
    }
}

extension AppDelegate {
    /// Render harness: the menu's state without a running service.
    func setPreview(actions: [LoadedAction], memory: MemorySnapshot, lastResult: String?) async {
        previewState(actions: actions, memory: memory)
        runner.previewResult(lastResult)
        rebuildMenu()
    }
}
