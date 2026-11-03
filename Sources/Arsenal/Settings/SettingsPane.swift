// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// A model's hosting settings, the body of its settings card (SettingsCard.swift), drawn on the menu's own material.
//
// Layout (SettingsLayout.swift): the app's rows first (Keep loaded, Show in menu, Runner), a hairline, then the model's
// load settings with its settings.toml's labels, units and controls. Labels sit in a left column, controls end at the
// right edge as in System Settings; no headings. Choices are segments and numbers the compact field: nothing in the card
// opens a menu. Every label and control carries the folder's help sentence plus when it applies as its tooltip. A changed
// load setting of a loaded model shows its label in amber until Reload, at the form's foot, applies it.

/// A model's settings columns: its width (between Activity's 376 and 480 pt) and its label column, fixed before the
/// card opens.
struct PaneGeometry: Equatable {
    var width: CGFloat, label: CGFloat
    var inner: CGFloat { width - 2 * SettingsPane.M.padding }
    var control: CGFloat { inner - label - SettingsPane.M.gap }
    static let standard = PaneGeometry(width: SettingsPane.M.width, label: SettingsPane.M.label)
}

struct SettingsPane: View {
    enum M {
        /// `width` is the widest a model's settings get (`minWidth`, Activity's width, the narrowest); `label` the label
        /// column of the standard geometry. Each model's own come from `geometry`.
        static let width: CGFloat = 480, minWidth: CGFloat = 376, padding: CGFloat = 20, top: CGFloat = 14, bottom: CGFloat = 14, label: CGFloat = 176
        static let gap: CGFloat = 12
        static let control = width - 2 * padding - label - gap
        static let rowHeight: CGFloat = 28, rowGap: CGFloat = 6, dividerMargin: CGFloat = 11
        /// Reload at the form's foot: a gap, then a row holding the table's own small bordered button at the trailing edge.
        static let footerGap: CGFloat = 6, buttonHeight: CGFloat = rowHeight, footerHeight = footerGap + buttonHeight
        static let labelSize: CGFloat = 13, controlSize: CGFloat = 12
        /// Fields are as tall as the segments they sit beside.
        static let fieldHeight: CGFloat = 24, numberField: CGFloat = 76, seedField: CGFloat = 116, glyph: CGFloat = 20
        static let noteHeight: CGFloat = 15, noteGap: CGFloat = 2
        /// A glyph at the form's trailing edge (dice, folder): as wide as the symbol, so it ends on the edge.
        static let trailingGlyph: CGFloat = 16
        /// The file row keeps this room for its folder glyph.
        static let glyphSlot: CGFloat = 26
    }
    @ObservedObject var controller: TableController
    @Environment(\.palette) private var pal
    let row: ModelRow
    /// This model's columns (`SettingsPane.geometry`).
    var geometry: PaneGeometry = .standard

    /// Defaults: a load setting differs from its default (Keep loaded and Show in menu are placement choices, not model
    /// defaults). Reload: the model is loaded with a load setting pending.
    static func footer(_ states: [SettingState], loaded: Bool) -> (defaults: Bool, reload: Bool) {
        (states.contains(where: TableController.restores), loaded && states.contains { $0.pending })
    }
    /// The footer uses the row’s action title and tooltip.
    struct FooterAction: Equatable { var title: String; var help: String }
    static func footerAction(_ row: ModelRow, presentation p: RowPresentation, changes: [String], loadChoices: [String] = []) -> FooterAction {
        FooterAction(title: p.actionTitle, help: ModelHelp.action(row, presentation: p, changes: changes, loadChoices: loadChoices))
    }
    static let restoreHelp = "Restore defaults", restoredHelp = "Everything is at its default"
    /// An empty text setting's placeholder, so an empty field never reads as a blank bar.
    static let emptyText = "None"
    /// A number setting without a default: nothing is sent, the runner picks.
    static let randomText = "Random", autoText = "Auto"
    static let chooseFileHelp = "Choose a file"
    static func fileName(_ path: String) -> String { path.isEmpty ? "" : (path as NSString).lastPathComponent }
    static let randomHelp = "Pin a random value; clear the field to go back to a new one each time"

    static func items(_ states: [SettingState], panel: SettingsPanel?, lockReason: String? = nil, error: String? = nil) -> [PaneItem] {
        PaneLayout.items(states, panel: panel, lockReason: lockReason, error: error)
    }
    /// The form's height for its content (deterministic, so the card is sized before it is drawn).
    static func height(
        _ states: [SettingState], loaded: Bool, withdrawn: Bool = false, panel: SettingsPanel? = nil, lockReason: String? = nil, error: String? = nil,
        width: CGFloat = M.width
    ) -> CGFloat {
        let items = items(states, panel: panel, lockReason: lockReason, error: error)
        return PaneLayout.height(items, footer: footer(states, loaded: loaded).reload || withdrawn, width: width)
    }
    /// The form's height for this row as the controller holds it now.
    @MainActor static func height(_ c: TableController, row: ModelRow) -> CGFloat {
        height(
            c.settings[row.id] ?? [], loaded: c.presentation(row).hot, withdrawn: !row.withdrawnTiers.isEmpty || revocable(row), panel: c.panels[row.id],
            lockReason: lockReason(row), error: c.rowErrors[row.id],
            width: geometry(c, row: row).width)
    }

    /// This row's columns as the controller holds its settings now.
    @MainActor static func geometry(_ c: TableController, row: ModelRow) -> PaneGeometry {
        geometry(c.settings[row.id] ?? [], panel: c.panels[row.id])
    }
    /// The model's columns, as a native menu sizes to its content: the label column takes its widest label, the control
    /// column its widest control (the lock note counted), so nothing in the card changes its width while it is open.
    /// Between `M.minWidth` and `M.width`.
    static func geometry(_ states: [SettingState], panel: SettingsPanel?) -> PaneGeometry {
        let items = items(states, panel: panel, lockReason: lockReason(longest: true))
        let labelFont = NSFont.systemFont(ofSize: M.labelSize), controlFont = NSFont.systemFont(ofSize: M.controlSize)
        let segmentFont = NSFont.systemFont(ofSize: M.controlSize, weight: .medium), noteFont = NSFont.systemFont(ofSize: 11)
        func w(_ text: String, _ font: NSFont) -> CGFloat { ceil((text as NSString).size(withAttributes: [.font: font]).width) }
        func segments(_ labels: [String], icons: [String?]) -> CGFloat {
            let widest = labels.indices.map { i in icons.indices.contains(i) && icons[i] != nil ? 40 : max(52, w(labels[i], segmentFont) + 18) }.max() ?? 0
            return CGFloat(labels.count) * widest
        }
        let column = unitColumn(states, panel: panel)
        var label: CGFloat = 0, control: CGFloat = 0
        for item in items {
            switch item.kind {
            case .setting(let s, let style):
                label = max(label, w(s.title(style), labelFont))
                let need: CGFloat
                if plainValue(s) {
                    need = w(s.display(s.value, style), labelFont)
                } else {
                    switch s.control(style) {
                    case .toggle: need = SettingControl.Switch.size.width
                    case .segments, .list:
                        need = segments(s.segmentValues.map { s.display($0, style) }, icons: s.segmentValues.map { style.icons[SettingsPanel.key($0)] })
                    case .slider: need = 110 + 6 + M.numberField + (column > 0 ? 6 + column : 0)
                    case .number: need = (style.randomize ? M.seedField : M.numberField) + (column > 0 ? 6 + column : 0)
                    case .text, .editor, .voice: need = 200
                    case .file: need = 200 + M.glyphSlot
                    }
                }
                control = max(control, need)
            case .presets(let title, let presets, _, _):
                label = max(label, w(title, labelFont) + 6 + w(customText, controlFont))
                control = max(control, segments(presets.map(\.title), icons: presets.map(\.icon)))
            case .notice(let text):
                control = max(control, w(text, noteFont) + 8 + 16)
            case .divider, .error:
                break
            }
        }
        let width = min(M.width, max(M.minWidth, (2 * M.padding + label + M.gap + control).rounded(.up)))
        return PaneGeometry(width: width, label: min(label, width - 2 * M.padding - M.gap - min(control, M.control)).rounded(.up))
    }
    /// The error line: the image window's (a warning glyph, red, 12 pt), wrapping to at most `errorLines` lines, its
    /// whole text in the tooltip. Measured, so the card is sized before it is drawn.
    nonisolated static let errorLines = 4, errorGlyph: CGFloat = 14
    nonisolated static var errorFont: NSFont { .systemFont(ofSize: M.controlSize) }
    nonisolated static func errorWidth(_ width: CGFloat = M.width) -> CGFloat { width - 2 * M.padding - errorGlyph - 6 }
    nonisolated static func errorHeight(_ text: String, width: CGFloat = M.width) -> CGFloat {
        let line = ceil(errorFont.ascender - errorFont.descender + errorFont.leading)
        let measured = (text as NSString).boundingRect(
            with: NSSize(width: errorWidth(width), height: .greatestFiniteMagnitude), options: [.usesLineFragmentOrigin, .usesFontLeading], attributes: [.font: errorFont]
        ).height
        return max(M.rowHeight, min(ceil(measured), line * CGFloat(errorLines)) + 6)
    }
    /// Restore defaults: on when something can be restored now. Off, its tooltip says why: everything is at its default,
    /// or what differs is locked (a Get or Load runs, Arsenal quits).
    @MainActor static func restore(_ c: TableController, row: ModelRow) -> (enabled: Bool, help: String) {
        if !c.restorable(row).isEmpty { return (true, restoreHelp) }
        guard (c.settings[row.id] ?? []).contains(where: TableController.restores) else { return (false, restoredHelp) }
        if !c.admitting { return (false, quittingHelp) }
        return (false, lockReason(row) ?? TierControl.inUseHelp)
    }
    static let quittingHelp = "Arsenal is quitting"
    /// A local model whose current contents the owner approved: its card offers Revoke.
    static func revocable(_ row: ModelRow) -> Bool { row.local && row.approval != nil && !row.needsCodeConsent }
    /// The longest lock reason, which the card's width keeps room for.
    static func lockReason(longest: Bool) -> String? { longest ? "Locked while downloading" : nil }
    /// Why load-time rows are locked now, in words for the form; nil when nothing runs.
    static func lockReason(_ row: ModelRow) -> String? {
        let states = Set(row.precisions.compactMap { row.states[$0]?.state })
        if states.contains(.downloading) { return "Locked while downloading" }
        if states.contains(.deriving) { return "Locked while preparing" }
        if states.contains(.loading) { return "Locked while loading" }
        return nil
    }
    /// The column after the number fields that carry a unit or the dice: as wide as the widest of those in this form, so
    /// those fields end together.
    static func unitColumn(_ states: [SettingState], panel: SettingsPanel?) -> CGFloat {
        let styles = PaneLayout.styles(panel)
        let font = NSFont.systemFont(ofSize: M.controlSize)
        return states.filter(PaneLayout.hosting).compactMap { s -> CGFloat? in
            let style = styles[s.id] ?? .plain
            guard [.number, .slider].contains(s.control(style)) else { return nil }
            let dice: CGFloat = style.randomize ? M.trailingGlyph : 0
            let unit = s.unit(style).map { ceil(($0 as NSString).size(withAttributes: [.font: font]).width) } ?? 0
            return dice + unit + (dice > 0 && unit > 0 ? 6 : 0)
        }.max() ?? 0
    }

    private var states: [SettingState] { controller.settings[row.id] ?? [] }
    private var p: RowPresentation { controller.presentation(row) }
    private var column: CGFloat { Self.unitColumn(states, panel: controller.panels[row.id]) }

    var body: some View {
        let reload = Self.footer(states, loaded: p.hot).reload
        let withdrawn = ModelHelp.withdrawn(row) != nil
        let revocable = Self.revocable(row)
        let items = Self.items(states, panel: controller.panels[row.id], lockReason: Self.lockReason(row), error: controller.rowErrors[row.id])
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(items.enumerated()), id: \.element.id) { i, item in
                if i > 0, !item.isDivider, !items[i - 1].isDivider { Color.clear.frame(height: item.isNote ? M.noteGap : M.rowGap) }
                view(item)
            }
            if reload || withdrawn || revocable {
                HStack(spacing: M.gap) {
                    Spacer(minLength: 0)
                    // A local model's code approval: forgotten only on this click; the next Get or Load asks again.
                    if revocable {
                        SettingControl.Action(
                            title: "Revoke", enabled: p.busyText == nil && controller.admitting,
                            tooltip: p.busyText == nil ? ModelHelp.revokeHelp(row) : ModelHelp.lockLine(row)
                        ) {
                            controller.revokeCode(row)
                        }
                        .controlID("revoke-code")
                    }
                    // Old precisions on this Mac that the catalog no longer offers: deleted from disk only on this click
                    // (Delete, not Remove: Remove only ever takes a model off the list).
                    if withdrawn {
                        SettingControl.Action(
                            title: "Delete", enabled: p.busyText == nil && controller.admitting,
                            tooltip: p.busyText == nil ? ModelHelp.withdrawnHelp(row) : ModelHelp.lockLine(row)
                        ) {
                            controller.removeWithdrawnTiers(row)
                        }
                        .controlID("remove-withdrawn")
                    }
                    if reload {
                        let action = Self.footerAction(row, presentation: p, changes: controller.pendingChanges(row), loadChoices: controller.loadChoices(row))
                        // The table row's own button (Reload, or Get when an environment is missing), at the trailing edge.
                        SettingControl.Action(title: p.busyText ?? action.title, enabled: p.busyText == nil && controller.admitting, tooltip: action.help) {
                            controller.perform(row)
                        }
                        .controlID("reload")
                    }
                }
                .frame(width: geometry.inner, height: M.buttonHeight, alignment: .trailing).padding(.top, M.footerGap)
            }
        }
        .padding(.horizontal, M.padding).padding(.top, M.top).padding(.bottom, M.bottom)
        .frame(width: geometry.width, alignment: .topLeading)
    }

    @ViewBuilder private func view(_ item: PaneItem) -> some View {
        switch item.kind {
        case .divider:
            Rectangle().fill(pal.hairline.swiftUI).frame(height: 1).padding(.vertical, M.dividerMargin)
        case .setting(let s, let style):
            HStack(spacing: M.gap) {
                label(s.title(style), pending: s.pending, locked: !enabled(s) && !Self.plainValue(s), tip: tip(s))
                // SwiftUI's own enabled state as well: it re-enables a hosted AppKit control from its environment after
                // `updateNSView`, so a locked setting would otherwise stay clickable.
                control(s, style).disabled(!enabled(s)).frame(width: geometry.control, height: M.rowHeight, alignment: .trailing)
                    // AppKit shows no tooltip on a disabled control: a locked one keeps its hover (what it is, why it
                    // is locked) on a shield over it, which also takes the clicks the control refuses.
                    .overlay { if !enabled(s) { TooltipCell(text: tip(s)) } }
            }
        case .presets(let title, let presets, let selected, let pending):
            let on = lockEnabled
            HStack(spacing: M.gap) {
                // Values that match no preset: the row says so beside its label.
                label(title, pending: pending, locked: !on, tip: presetHelp(title), detail: selected == nil ? Self.customText : nil)
                SettingControl.Segments(
                    label: title, labels: presets.map(\.title), icons: presets.map(\.icon), tips: presets.map { presetTip($0) }, selected: selected, enabled: on,
                    tooltip: presetHelp(title)
                ) { i in controller.setPreset(row, presets[i].values) }
                .disabled(!on).frame(width: geometry.control, height: M.rowHeight)
                .overlay { if !on { TooltipCell(text: presetHelp(title)) } }
            }
        case .notice(let text):
            // Why the row above is locked, under it.
            HStack(spacing: 4) {
                Image(systemName: "lock.fill").font(.system(size: 8.5))
                Text(text).font(.system(size: 11)).lineLimit(1)
            }.foregroundStyle(pal.t3.swiftUI)
                .frame(width: geometry.control, height: M.noteHeight, alignment: .leading).padding(.leading, geometry.label + M.gap + 8)
        case .error(let text):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                // The full message on the glyph: the text itself stays selectable, so nothing lies over it.
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 11)).frame(width: Self.errorGlyph).cellTooltip(text)
                Text(text).font(Font(Self.errorFont)).lineLimit(Self.errorLines).textSelection(.enabled)
                    .frame(width: Self.errorWidth(geometry.width), alignment: .leading)
            }
            .foregroundStyle(pal.red.swiftUI).padding(.top, 3)
            .frame(width: geometry.inner, height: Self.errorHeight(text, width: geometry.width), alignment: .topLeading)
            .controlID("error")
        }
    }

    static let customText = "Custom"
    private func label(_ text: String, pending: Bool, locked: Bool, tip: String, detail: String? = nil) -> some View {
        HStack(spacing: 6) {
            Text(text).font(.system(size: M.labelSize)).lineLimit(1)
                .foregroundStyle(pending ? pal.amber.swiftUI : locked ? pal.t3.swiftUI : pal.t1.swiftUI)
            if let detail { Text(detail).font(.system(size: M.controlSize)).foregroundStyle(pal.t3.swiftUI).lineLimit(1) }
        }
        .frame(width: geometry.label, height: M.rowHeight, alignment: .leading)
        .cellTooltip(tip)
    }

    /// A setting with a single choice (one context length, one runner) shows its value as plain text, whatever control
    /// its row names: there is nothing to choose.
    nonisolated static func plainValue(_ s: SettingState) -> Bool { s.type == "choice" && s.choices.count <= 1 }

    private func tip(_ s: SettingState) -> String {
        let text = s.id == TableController.runnerKey && s.choices.count == 1 ? "Only available runner for this model." : s.tooltip
        return !enabled(s) && controller.admitting ? text + "\n" + TierControl.inUseHelp : text
    }
    private func presetTip(_ preset: SettingsPanel.Preset) -> String {
        let parts = preset.values.keys.sorted { a, b in (states.firstIndex { $0.id == a } ?? 0) < (states.firstIndex { $0.id == b } ?? 0) }.map { id in
            let s = states.first { $0.id == id }
            let value = preset.values[id]!
            return "\(s?.title ?? id) \(s?.display(value) ?? SettingsPanel.key(value))\(s?.unit.map { " " + $0 } ?? "")"
        }
        return preset.title + ": " + parts.joined(separator: ", ")
    }
    private func presetHelp(_ title: String) -> String { !lockEnabled ? title + "\n" + TierControl.inUseHelp : title }
    /// The controller's one lock rule (`TableController.settingEnabled`): load-time rows wait while a Get or Load runs.
    private var lockEnabled: Bool { !p.locked && controller.admitting }
    private func enabled(_ s: SettingState) -> Bool { controller.settingEnabled(row, s.id) }

    @ViewBuilder private func control(_ s: SettingState, _ style: PaneStyle) -> some View {
        let set: (JSONValue) -> Void = { controller.set(row, s.id, $0) }
        if Self.plainValue(s) {
            // Nothing to choose, so nothing to lock: the value never dims.
            Text(s.display(s.value, style)).font(.system(size: M.labelSize)).foregroundStyle(pal.t2.swiftUI).lineLimit(1)
                .cellTooltip(tip(s))
        } else {
            switch s.control(style) {
            case .toggle:
                SettingControl.Switch(label: s.title(style), on: s.value.bool ?? false, enabled: enabled(s), tooltip: tip(s)) { set(.bool($0)) }
                    .frame(width: SettingControl.Switch.size.width, height: SettingControl.Switch.size.height)
            case .segments, .list:
                // Every choice is segments, as the precision control is: nothing in the card opens a menu.
                let values = s.segmentValues
                SettingControl.Segments(
                    label: s.title(style), labels: values.map { s.display($0, style) }, icons: values.map { style.icons[SettingsPanel.key($0)] },
                    tips: values.map { s.display($0, style) }, selected: values.firstIndex(of: s.value), enabled: enabled(s), tooltip: tip(s)
                ) {
                    set(values[$0])
                }.frame(width: geometry.control, height: M.rowHeight)
            case .slider:
                HStack(spacing: 6) {
                    SettingControl.Slider(
                        label: s.title(style), value: s.value.number ?? 0, min: s.minValue ?? 0, max: s.maxValue ?? 1, step: s.step, enabled: enabled(s), tooltip: tip(s)
                    ) {
                        if let n = s.number(from: String($0)) { set(.number(n)) }
                    }.frame(height: M.rowHeight)
                    numberField(s, style, width: M.numberField, set: set)
                    accessory(s, style)
                }
            case .number:
                HStack(spacing: 6) {
                    Spacer(minLength: 0)
                    numberField(s, style, width: style.randomize ? M.seedField : M.numberField, set: set)
                    accessory(s, style)
                }
            case .text, .editor, .voice:
                SettingControl.Field(
                    label: s.title(style), text: s.value.string ?? "", placeholder: (s.defaultValue.string ?? "").isEmpty ? Self.emptyText : s.defaultValue.string ?? "",
                    width: geometry.control, alignment: .left, enabled: enabled(s), tooltip: tip(s)
                ) {
                    set(.string($0))
                }.frame(width: geometry.control, height: M.fieldHeight)
            case .file:
                // A path: typed, or chosen with the folder glyph. The file's name shows; its full path is the tooltip.
                // Typing a path replaces it; leaving the name as it was changes nothing.
                let path = s.value.string ?? ""
                let name = Self.fileName(path)
                HStack(spacing: 6) {
                    SettingControl.Field(
                        label: s.title(style), text: name, placeholder: Self.emptyText, width: geometry.control - M.glyphSlot, alignment: .left,
                        enabled: enabled(s), tooltip: path.isEmpty ? tip(s) : path + "\n" + tip(s)
                    ) { text in
                        if text != name { set(.string(text)) }
                    }.frame(height: M.fieldHeight)
                    SettingControl.Glyph(symbol: "folder", tooltip: Self.chooseFileHelp, label: Self.chooseFileHelp, enabled: enabled(s), trailing: true) {
                        controller.chooseFile(row, s)
                    }.frame(width: M.trailingGlyph, height: M.glyph)
                }
            }
        }
    }
    /// A number as a plain field: arrow keys step it, Enter or leaving commits, an invalid entry reverts. A setting
    /// without a default shows "Random" (or "Auto") when empty, and clearing it returns there.
    private func numberField(_ s: SettingState, _ style: PaneStyle, width: CGFloat, set: @escaping (JSONValue) -> Void) -> some View {
        let unset = s.defaultValue == .null
        return SettingControl.Field(
            label: s.title(style), text: s.value == .null ? "" : s.display(s.value), placeholder: unset ? (style.randomize ? Self.randomText : Self.autoText) : "",
            width: width, alignment: .right, enabled: enabled(s), tooltip: tip(s),
            step: { direction, text in
                let base = s.number(from: text) ?? s.value.number ?? s.minValue ?? 0
                return s.number(from: String(base + Double(direction) * (s.step ?? 1))).map { s.display(.number($0)) }
            },
            commit: { text in
                if text.trimmingCharacters(in: .whitespaces).isEmpty, unset { set(.null) } else if let n = s.number(from: text) { set(.number(n)) }
            }
        ).frame(width: width, height: M.fieldHeight)
    }
    /// What follows a number field: its unit (left in the column, beside the field) or the dice (at the form's edge).
    /// A field with neither ends at the edge itself, as the segments do.
    @ViewBuilder private func accessory(_ s: SettingState, _ style: PaneStyle) -> some View {
        let unit = s.unit(style)
        if unit != nil || style.randomize {
            HStack(spacing: 6) {
                if let unit { unitText(unit) }
                if style.randomize {
                    SettingControl.Glyph(symbol: "dice", tooltip: Self.randomHelp, label: Self.randomHelp, enabled: enabled(s), trailing: true) {
                        controller.randomize(row, s)
                    }
                    .frame(width: M.trailingGlyph, height: M.glyph)
                }
            }
            .frame(width: column, height: M.rowHeight, alignment: style.randomize ? .trailing : .leading)
        }
    }
    private func unitText(_ unit: String) -> some View {
        Text(unit).font(.system(size: M.controlSize)).foregroundStyle(pal.t2.swiftUI).fixedSize()
    }
}
