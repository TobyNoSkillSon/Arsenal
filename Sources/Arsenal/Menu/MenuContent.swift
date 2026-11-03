// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// The status menu's content as data (unit-tested; drawn by AppDelegate.rebuildMenu and the render harness).

enum MenuContent {
    /// Keep Hot's idle windows (Vella's): models you loaded, and models a request loaded on demand.
    static let keepLoadedTitle = "Keep Loaded"
    static let manualHeader = "Models you load"
    static let manualHelp = "Models you loaded yourself with Load in Models; they load again when Arsenal starts."
    static let onDemandHeader = "Models a request loads"
    static let onDemandHelp = "Models a request needed (an agent through MCP, the API or a menu action), so Arsenal loaded them; they do not load again when Arsenal starts."
    static let alwaysHelp = "Never unloaded for being idle; only Unload, or Memory making room for another model, unloads them."
    static func idleChoices(_ current: Int) -> [(minutes: Int, title: String, checked: Bool)] {
        KeepHot.choices.map { ($0, KeepHot.title($0), $0 == current) }
    }
    /// Memory's two modes (Vella's wording).
    static let fitTitle = "Fit in free memory"
    static let fitHelp =
        "Checks free memory before loading: a model loads only if it fits in memory that is free at that moment. Otherwise the least recently used idle model a request loaded on demand is unloaded first; if nothing more can be unloaded, When a model doesn\u{2019}t fit decides. Best effort: memory use can change after the check."
    static let swapTitle = "Allow swap (slower)"
    static let swapHelp = "Loads even when memory is short; macOS moves data to disk and everything, including other apps, can slow down."

    /// When memory is full (C6): refuse at once (the default, for agents) or wait in a queue (for scripts).
    static let whenFullHeader = "When a model doesn\u{2019}t fit"
    static let whenFull: [(MemoryWhenFull, String, String)] = [
        (
            .refuse, "Refuse right away",
            "When nothing more can be unloaded, the request is answered at once: not enough memory, with what is loaded and what it needs. Models you loaded or keep hot are never unloaded for it."
        ),
        (
            .queue, "Wait in queue",
            "When nothing more can be unloaded, the request waits until a generation ends and frees room, then runs; a model it loaded is unloaded after its Keep Hot time. Models you loaded or keep hot are never unloaded for it."
        )
    ]

    /// The ids of rows with a loaded precision, on the table or not (Unload All stops each of them).
    static func loadedIDs(_ rows: [ModelRow]) -> [String] {
        rows.filter { r in r.precisions.contains { r.states[$0]?.state == .loaded } }.map(\.id)
    }

    /// Memory: free memory now, then each loaded model and what it uses.
    static func memoryLines(_ snapshot: MemorySnapshot, rows: [ModelRow]) -> [String] {
        // The room Load admission measures against (under the Mac-wide line), in Activity Monitor's units.
        let room = MacMemoryGuard.roomUnderLineMB(totalMB: snapshot.totalMB, noSwapAvailableMB: snapshot.availableMB)
        var lines = ["Room for models: \(MemoryDisplay.text(decimalGB: max(0, room) / 1000))"]
        for row in rows where row.precisions.contains(where: { row.states[$0]?.state == .loaded }) {
            let mb = snapshot.models[row.id]?.footprintMB
            // The same figure and format as the activity panel's card: the model's memory now, one decimal in GB.
            lines.append(row.name + (mb.flatMap { FigureText.gigabytes(mb: $0) }.map { " \u{00b7} " + $0 } ?? ""))
        }
        if lines.count == 1 { lines.append("Nothing loaded") }
        return lines
    }

    /// Files for agents and people: models are folders, settings are config (v2.2 items 7, 9).
    static let openFiles = "Show Models Folder"
    static let openFilesHelp = "Shows Arsenal's folder in Finder. Drop a model folder into local-models to add it; settings are in config.json."
    static let refresh = "Refresh Models"

    /// The update item (C8): orange while a newer release is offered; its progress while it downloads and installs.
    struct UpdateLine: Equatable { var title: String; var enabled: Bool; var orange: Bool; var help: String }
    static func update(_ state: AppUpdateState) -> UpdateLine? {
        switch state {
        case .none: return nil
        case .available(let version, _, let size):
            return .init(
                title: "Update to \(version)", enabled: true, orange: true,
                help: "Downloads \(formatBytes(size)), waits until nothing is generating, installs, relaunches and loads what was loaded. The previous version is kept.")
        case .downloading(let progress):
            return .init(
                title: "Downloading update \u{00b7} \(Int((progress * 100).rounded())) %", enabled: false, orange: true, help: "The update installs when nothing is generating.")
        case .ready: return .init(title: "Update waits for generations to finish", enabled: false, orange: true, help: "Arsenal installs it as soon as nothing is generating.")
        case .installing: return .init(title: "Installing update\u{2026}", enabled: false, orange: true, help: "Arsenal relaunches when it is done.")
        case .failed(let reason): return .init(title: "Update failed", enabled: true, orange: false, help: reason + "\nClick to try again.")
        }
    }

    /// The ungrouped form names the model on the action line. The status menu does not use it: every model's tools sit under
    /// the model's name as a header, so a row names only its tool and stays short enough to read in full.
    static func toolTitle(_ title: String, model: ModelRow, grouped: Bool) -> String {
        guard !grouped else { return title }
        let action = title.hasSuffix("\u{2026}") ? String(title.dropLast()) : title
        return action + " — " + model.name + "\u{2026}"
    }

    /// Models with menu tools (C1), including their folder's plain actions (Speak Clipboard).
    struct ToolSection: Equatable { var row: ModelRow; var tools: [ToolKind]; var actions: [LoadedAction] }
    /// `resolve`: the tools a row's section offers (ToolCenter.resolvedTools places the generic tool's models in their
    /// capability windows); the folder's own tools by default.
    static func toolSections(_ rows: [ModelRow], sections: [LoadedMenuSection], resolve: (ModelRow) -> [ToolKind] = ToolCenter.tools) -> [ToolSection] {
        rows.compactMap { row in
            let tools = resolve(row)
            guard !tools.isEmpty else { return nil }
            let plain = sections.first { $0.id == row.id && $0.modelID == row.id }?.items ?? []
            return ToolSection(row: row, tools: tools, actions: plain)
        }
    }

    /// The loaded table models' menu entries by section (v2.2 item 8). A folder's declared section keeps its title as a
    /// header and its items need no subtitle; plain `[[menu]]` actions (the section the service names after the model)
    /// show as before, titled as declared with the model's name as subtitle. Models removed from the table contribute
    /// nothing.
    /// `toolModels`: models whose plain actions are already in their tool section.
    static func sections(
        _ sections: [LoadedMenuSection], rows: [ModelRow], toolModels: Set<String> = []
    ) -> [(header: String?, entries: [(action: LoadedAction, title: String, subtitle: String?)])] {
        sections.compactMap { s in
            guard let row = rows.first(where: { $0.id == s.modelID }), row.tools.isEmpty || row.toolsShown, !s.items.isEmpty else { return nil }
            let plain = s.id == s.modelID
            if plain, toolModels.contains(s.modelID) { return nil }
            return (plain ? nil : s.title, s.items.map { ($0, $0.title, plain ? row.name : nil) })
        }
    }

    /// The status item's accessibility description and tooltip: what is loaded.
    static func summary(_ rows: [ModelRow]) -> String {
        let loaded = rows.filter { r in r.precisions.contains { r.states[$0]?.state == .loaded } }
        switch loaded.count {
        case 0: return "Arsenal: no model loaded"
        case 1: return "Arsenal: \(loaded[0].name) loaded"
        default: return "Arsenal: \(loaded.count) models loaded"
        }
    }
}
