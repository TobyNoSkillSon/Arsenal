// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// What a model's settings card draws, in order, as plain values: the app's own rows (Keep loaded, Show in menu, Runner)
// first, then the model's load settings (`applies = "load"`) in its folder's settings.toml order, with that file's
// labels, units and controls. Request settings never appear: the app or tool making the request sets them. The pane
// draws these items and the card is sized from the same list, so its height can never drift from the drawing.

/// How one setting row is drawn: the folder's choices for it, or none (the generic list).
struct PaneStyle: Equatable {
    var label: String?
    var control: SettingsPanel.Control?
    var labels: [String: String] = [:]
    var icons: [String: String] = [:]
    var unit: String?
    var randomize = false
    static let plain = PaneStyle()
    init(label: String? = nil, control: SettingsPanel.Control? = nil, labels: [String: String] = [:], icons: [String: String] = [:], unit: String? = nil, randomize: Bool = false) {
        self.label = label; self.control = control; self.labels = labels; self.icons = icons; self.unit = unit; self.randomize = randomize
    }
    init(_ row: SettingsPanel.Row) {
        self.init(label: row.label, control: row.control, labels: row.labels, icons: row.icons, unit: row.unit, randomize: row.randomize)
    }
}

extension SettingState {
    /// The row's control: the panel's choice, else the type's default (`control`).
    func control(_ style: PaneStyle) -> Control {
        switch style.control {
        case .switch?: return .toggle
        case .segments?: return .segments
        case .menu?: return .list
        case .slider?: return .slider
        case .field?: return ["number", "slider"].contains(type) ? .number : .text
        // The card has no multi-line editor: a text setting is a one-line field.
        case .editor?: return .text
        case .file?: return .file
        case .voice?: return .voice
        case nil: return control
        }
    }
    func title(_ style: PaneStyle) -> String { style.label ?? title }
    /// A value as this row shows it: the panel's name for it, else the generic display.
    func display(_ value: JSONValue, _ style: PaneStyle) -> String { style.labels[SettingsPanel.key(value)] ?? display(value) }
    func unit(_ style: PaneStyle) -> String? { style.unit ?? unit }
    /// The values a segmented row offers: a choice's choices; a toggle's default first, then the other.
    var segmentValues: [JSONValue] {
        if type == "toggle" { let d = defaultValue.bool ?? false; return [.bool(d), .bool(!d)] }
        return choices
    }
}

struct PaneItem: Equatable, Identifiable {
    enum Kind: Equatable {
        case setting(SettingState, PaneStyle)
        /// A preset row whose values are all load settings; `selected` is the preset whose values all match, nil when
        /// none does.
        case presets(label: String, presets: [SettingsPanel.Preset], selected: Int?, pending: Bool)
        /// Why the rows that lock are locked (a Get or Load is running): one line under the first of them.
        case notice(String)
        /// The model's last failed action from its settings (a refused setting, a failed Reload), at the form's foot.
        case error(String)
        case divider
    }
    var id: String
    var kind: Kind
    var isDivider: Bool { kind == .divider }
    /// The lock notice: a short line under a row, drawn in the row's control column.
    var isNote: Bool { if case .notice = kind { return true }; return false }
}

enum PaneLayout {
    /// The app's rows, in this order, above the model's own (ids the table controller synthesises).
    /// Keep loaded, Show in menu, then the runner picker (it is a load-time choice; the switches are the frequent ones).
    static let appOrder = [TableController.keepKey, "__tools", TableController.runnerKey]
    static func isAppRow(_ s: SettingState) -> Bool { s.id.hasPrefix("__") }
    /// What the card shows: the app's rows and the model's load settings. Request settings are the requesting app's.
    static func hosting(_ s: SettingState) -> Bool { isAppRow(s) || s.applies == "load" }
    /// What a running Get or Load locks: load-time settings (the runner included). The app's own rows stay live.
    static func locks(_ s: SettingState) -> Bool { s.applies == "load" }

    /// `lockReason`: set while a Get or Load runs; a notice says it under the rows it locks. `error`: the model's last
    /// failed action, the form's last line.
    static func items(_ states: [SettingState], panel: SettingsPanel?, lockReason: String? = nil, error: String? = nil) -> [PaneItem] {
        // One runner is not a choice: the row shows only as a picker for two or more.
        let app = states.filter { isAppRow($0) && !($0.id == TableController.runnerKey && $0.choices.count < 2) }
            .sorted { (appOrder.firstIndex(of: $0.id) ?? 99) < (appOrder.firstIndex(of: $1.id) ?? 99) }
        let model = states.filter { !isAppRow($0) && hosting($0) }
        var groups: [[PaneItem]] = [rows(app)]
        groups.append(loadRows(model, panel: panel))
        var items: [PaneItem] = []
        for group in groups where !group.isEmpty {
            if !items.isEmpty { items.append(PaneItem(id: "divider-\(items.count)", kind: .divider)) }
            items += group
        }
        // The reason sits under the first locked row, not at the foot.
        if let lockReason, let at = items.firstIndex(where: locked) {
            items.insert(PaneItem(id: "notice", kind: .notice(lockReason)), at: at + 1)
        }
        if let error { items.append(PaneItem(id: "error", kind: .error(error))) }
        return items
    }
    /// Whether a running Get or Load locks this item.
    static func locked(_ item: PaneItem) -> Bool {
        switch item.kind {
        // A plain value (one choice: nothing to change) is never locked: it neither dims nor takes the lock note.
        case .setting(let s, _): return locks(s) && !SettingsPane.plainValue(s)
        case .presets: return true
        default: return false
        }
    }

    private static func rows(_ states: [SettingState], styles: [String: PaneStyle] = [:]) -> [PaneItem] {
        states.map { s in PaneItem(id: s.id, kind: .setting(s, styles[s.id] ?? .plain)) }
    }

    /// The model's load settings in its panel's order (every section, collapsed or not: the card has no More), with the
    /// panel's styles; a preset row only when every value it sets is a load setting; then any load setting the panel
    /// does not name, in declaration order.
    private static func loadRows(_ states: [SettingState], panel: SettingsPanel?) -> [PaneItem] {
        guard let panel else { return rows(states) }
        let byID = Dictionary(states.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        let values = byID.mapValues(\.value)
        var items: [PaneItem] = [], placed = Set<String>()
        for (index, section) in panel.sections.enumerated() where SettingsPanel.holds(section.showWhen, values: values) {
            for row in section.rows where SettingsPanel.holds(row.showWhen, values: values) {
                if let id = row.setting {
                    guard let s = byID[id], placed.insert(id).inserted else { continue }
                    items += rows([s], styles: [id: PaneStyle(row)])
                } else {
                    let keys = Set(row.presets.flatMap(\.values.keys))
                    guard !keys.isEmpty, keys.allSatisfy({ byID[$0] != nil }) else { continue }
                    let set = states.filter { keys.contains($0.id) }
                    placed.formUnion(keys)
                    items.append(
                        PaneItem(
                            id: "presets-\(index)-\(row.label ?? "")",
                            kind: .presets(label: row.label ?? "", presets: row.presets, selected: match(row.presets, values: values), pending: set.contains(where: \.pending))))
                }
            }
        }
        return items + rows(states.filter { !placed.contains($0.id) })
    }

    /// The preset in `presets` whose values all equal the current ones.
    static func match(_ presets: [SettingsPanel.Preset], values: [String: JSONValue]) -> Int? {
        presets.firstIndex { $0.values.allSatisfy { values[$0.key] == $0.value } }
    }
    /// Each setting's row style in the panel, collapsed sections and hidden conditions included.
    static func styles(_ panel: SettingsPanel?) -> [String: PaneStyle] {
        var out: [String: PaneStyle] = [:]
        for row in panel?.sections.flatMap(\.rows) ?? [] { if let id = row.setting { out[id] = PaneStyle(row) } }
        return out
    }

    /// The form's height for these items: padding, rows with their gaps, dividers with their margins, and the Reload
    /// footer when it shows.
    static func height(_ items: [PaneItem], footer: Bool, width: CGFloat = SettingsPane.M.width) -> CGFloat {
        typealias M = SettingsPane.M
        var h = M.top + M.bottom
        for (i, item) in items.enumerated() {
            if i > 0, !item.isDivider, !items[i - 1].isDivider { h += item.isNote ? M.noteGap : M.rowGap }
            switch item.kind {
            case .divider: h += 2 * M.dividerMargin + 1
            case .notice: h += M.noteHeight
            case .error(let text): h += SettingsPane.errorHeight(text, width: width)
            default: h += M.rowHeight
            }
        }
        if footer { h += M.footerHeight }
        return h
    }
}

extension TableController {
    /// A preset: every value it names at once, through the service's all-or-nothing settings write.
    func setPreset(_ row: ModelRow, _ values: [String: JSONValue]) {
        guard values.keys.allSatisfy({ settingEnabled(row, $0) }), let states = settings[row.id] else { return }
        let changed = values.filter { id, value in states.first { $0.id == id }.map { $0.value != value } ?? false }
        guard !changed.isEmpty else { return }
        // Shown at once; the service's answer (and its pending flags) follows.
        rowErrors[row.id] = nil
        for (id, value) in changed { if let i = settings[row.id]?.firstIndex(where: { $0.id == id }) { settings[row.id]?[i].value = value } }
        Task {
            do { try await service.setSettings(id: row.id, values: changed) } catch { fail(row, Self.message(error)) }
            await refresh()
        }
    }
    /// A path setting's folder glyph: the chosen file's path, or nothing when the panel is cancelled.
    func chooseFile(_ row: ModelRow, _ s: SettingState) {
        guard settingEnabled(row, s.id), let path = pickFile(row, s) else { return }
        set(row, s.id, .string(path))
    }
    /// The dice: a random value inside the setting's bounds, on its step.
    func randomize(_ row: ModelRow, _ s: SettingState) {
        guard let low = s.minValue, let high = s.maxValue, high > low else { return }
        let step = s.step ?? 1
        let n = low + (Double.random(in: 0...((high - low) / step)).rounded(.down)) * step
        set(row, s.id, .number(min(high, n)))
    }
}
