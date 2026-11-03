// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore
import SwiftUI

// A model's settings as a small card beside its gear, inside the Models table. The card is an overlay in the table's own hosted view:
// its trailing edge just left of the gear, top-aligned with the row, clamped inside the table, on the menu's own material
// with a hairline border and a soft shadow. Nothing about the menu changes for it: no item, window, submenu or pop-up,
// and the table never moves, grows or reflows, so AppKit keeps tracking the Models menu exactly as it does for the row's
// Load button.
//
// The gear's click opens it (never hover); the gear again, Esc, a click anywhere else in the table (which then does its
// own job), another row's gear (that row's card takes its place) or the menu closing closes it.
//
// It holds hosting settings only: Keep loaded, Show in menu, Runner and the model's `applies = "load"` settings
// (PaneLayout). Per-request settings belong to the app or tool making the request and to the API.

/// What the open card keeps for itself: whose it is, its columns, the gear it sits beside (in the table's coordinates,
/// top-left origin) and the tallest it may be.
@MainActor final class SettingsState: ObservableObject {
    @Published var id: String?
    @Published var geometry: PaneGeometry = .standard
    @Published var gear: CGRect = .zero
}

@MainActor final class SettingsCard {
    weak var table: TableController?
    let state = SettingsState()
    /// The row whose card is open, nil when none is.
    private(set) var openID: String?
    /// Every gear in the table by row, for its active look (SwiftUI can briefly hold two views for one row while it
    /// rebuilds one).
    private var gears: [String: NSHashTable<GearView>] = [:]
    /// The click that closed the card from the table's hit test: its own delivery to a view must not act on the card again.
    private var handled: EventKey?
    struct EventKey: Equatable {
        var timestamp: TimeInterval, number: Int, window: Int
        init(_ e: NSEvent) { timestamp = e.timestamp; number = e.eventNumber; window = e.windowNumber }
    }

    /// The first line: the model's name and Restore defaults; as tall as a table row, so the name sits level with the row's.
    static let headerHeight: CGFloat = ModelTable.rowHeight
    /// Between the card's trailing edge and the gear's leading edge.
    static let gap: CGFloat = 4
    /// The card keeps this far inside the table's edges.
    static let inset: CGFloat = 4
    static let corner: CGFloat = 10
    /// Opaque grounds close to the menu's own tone, a step apart from the table so the card reads as raised over it.
    static let groundDark = ArsenalPalette.rgb(0x2A2C31), groundLight = ArsenalPalette.rgb(0xF7F7F8)

    // MARK: Gears

    func register(_ gear: GearView, for id: String) {
        for (other, table) in gears where other != id && table.contains(gear) { table.remove(gear) }
        let table = gears[id] ?? NSHashTable<GearView>.weakObjects()
        table.add(gear)
        gears[id] = table
        gear.active = openID == id
    }
    /// The row's gear in a window, else any.
    func gear(_ id: String) -> GearView? {
        let all = gears[id]?.allObjects ?? []
        return all.first { $0.window != nil } ?? all.first
    }
    private func markActive(_ id: String?, _ on: Bool) {
        guard let id else { return }
        for gear in gears[id]?.allObjects ?? [] where gear.active != on { gear.active = on }
    }

    // MARK: Opening and closing

    /// A gear's press: its row's card opens; pressed again, it closes; another row's takes its place.
    func toggle(_ row: ModelRow, from anchor: NSView? = nil) {
        if openID == row.id { close(); return }
        open(row, from: anchor)
    }

    /// Opens the row's card beside its gear. Its width is fixed for as long as it is open.
    func open(_ row: ModelRow, from anchor: NSView? = nil) {
        guard let table, table.rows.contains(where: { $0.id == row.id }) else { return }
        if openID != nil { close() }
        state.geometry = SettingsPane.geometry(table, row: row)
        state.gear = Self.gearRect(table, row: row, anchor: anchor)
        state.id = row.id
        openID = row.id
        markActive(row.id, true)
        table.settingsOpened(row)
    }

    /// Closes the open card (the gear again, Esc, a click elsewhere in the table, the menu closing, a removed model).
    func close() {
        guard let id = openID else { return }
        // A field being typed in keeps what was typed.
        SettingsDrafts.commitAll()
        openID = nil
        state.id = nil
        markActive(id, false)
        table?.settingsClosed(id)
    }
    func close(_ id: String) { if openID == id { close() } }
    /// Closes the card of a model no longer in the table.
    func prune(keeping ids: Set<String>) { if let id = openID, !ids.contains(id) { close() } }

    /// The Models menu closed: the card goes with it.
    func menuClosed() { handled = nil; close() }

    /// A mouse-down the table's hit test is about to deliver to `target` at `point` (table coordinates): outside the card
    /// and not on a gear (a gear toggles or switches the card itself), it closes the card; the click then goes on to its
    /// own target and does its own job. Returns whether it closed the card.
    @discardableResult func tableClick(at point: CGPoint, target: NSView?, event: NSEvent?) -> Bool {
        guard openID != nil, let table else { return false }
        if let event {
            let key = EventKey(event)
            if handled == key { return false }
            handled = key
        }
        if Self.isGear(target) { return false }
        if let frame = frame(table), frame.contains(point) { return false }
        close()
        return true
    }
    private static func isGear(_ view: NSView?) -> Bool {
        var v = view
        while let current = v { if current is GearView { return true }; v = current.superview }
        return false
    }
    /// Esc: closes the card (only the card; the menu stays).
    func escape() -> Bool {
        guard openID != nil else { return false }
        close()
        return true
    }

    // MARK: Size and place

    /// The card's whole height for this row: the name line, its hairline and the form.
    static func height(_ c: TableController, row: ModelRow) -> CGFloat {
        headerHeight + SettingsPane.height(c, row: row)
    }
    /// The open card's frame in the table's coordinates (top-left origin), nil when none is open.
    func frame(_ c: TableController) -> CGRect? {
        guard let id = openID, let row = c.rows.first(where: { $0.id == id }) else { return nil }
        return Self.frame(gear: state.gear, size: CGSize(width: state.geometry.width, height: Self.height(c, row: row)), table: Self.tableSize(c))
    }
    static func tableSize(_ c: TableController) -> CGSize { CGSize(width: ModelTable.width(c), height: ModelTable.height(c)) }

    /// Beside the gear: the trailing edge `gap` left of it, the top level with the row's top; then moved up and right only
    /// as far as the table's bounds need, and never taller than the table (the form scrolls inside beyond that).
    static func frame(gear: CGRect, size: CGSize, table: CGSize) -> CGRect {
        let height = min(size.height, max(0, table.height - 2 * inset)).rounded(.up)
        var x = gear.minX - gap - size.width
        var y = gear.minY
        x = max(inset, min(x, table.width - inset - size.width))
        y = max(inset, min(y, table.height - inset - height))
        return CGRect(x: x.rounded(), y: y.rounded(), width: size.width, height: height)
    }

    /// The gear's rectangle in the table's coordinates (top-left origin): from the gear's own view inside the Models
    /// menu's host (a scrolled table included), else from the table's metrics, which lay its rows out.
    static func gearRect(_ c: TableController, row: ModelRow, anchor: NSView?) -> CGRect {
        if let anchor, let host = anchor.enclosingTableHost {
            let r = anchor.convert(anchor.bounds, to: host.hosting)
            let s = max(host.contentScale, 0.01)
            return CGRect(x: r.minX / s, y: r.minY / s, width: r.width / s, height: r.height / s)
        }
        return ModelTable.gearRect(c, id: row.id) ?? .zero
    }
}

extension NSView {
    /// The Models menu's hosted table view this view is drawn in, if any.
    var enclosingTableHost: MenuTableHostingView? {
        var v = superview
        while let current = v { if let host = current as? MenuTableHostingView { return host }; v = current.superview }
        return nil
    }
}

/// The card: the model's name, then the form (no Restore glyph, no line under the name); on the menu's own material with a
/// hairline border and a soft shadow, in the menu palette. It sits in an overlay of the table, beside its gear.
struct SettingsCardView: View {
    @ObservedObject var controller: TableController
    @ObservedObject var state: SettingsState
    @ObservedObject var appearance = Appearance.shared

    var body: some View {
        let pal = appearance.menuPalette
        if let id = state.id, let row = controller.rows.first(where: { $0.id == id }) {
            let natural = SettingsCard.height(controller, row: row)
            let geometry = state.geometry
            let frame = SettingsCard.frame(gear: state.gear, size: CGSize(width: geometry.width, height: natural), table: SettingsCard.tableSize(controller))
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: SettingsPane.M.gap) {
                    Text(row.name).font(.system(size: 13, weight: .semibold)).lineLimit(1).truncationMode(.middle)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, SettingsPane.M.padding)
                .frame(width: geometry.width, height: SettingsCard.headerHeight)
                if natural > frame.height {
                    // A table of one or two models is shorter than the card: the form scrolls inside it. No scroller
                    // flashes in as the card opens.
                    ScrollView(.vertical) { SettingsPane(controller: controller, row: row, geometry: geometry) }.scrollIndicators(.never)
                } else {
                    SettingsPane(controller: controller, row: row, geometry: geometry)
                }
            }
            .frame(width: frame.width, height: frame.height, alignment: .top)
            // Clicks inside the card that no control takes stop here, never reaching the table under the card.
            .contentShape(Rectangle()).onTapGesture {}
            .background(SettingsCardMaterial(dark: pal.dark))
            .foregroundStyle(pal.t1.swiftUI)
            .environment(\.colorScheme, appearance.colorScheme)
            .environment(\.palette, pal)
            .offset(x: frame.minX, y: frame.minY)
            .accessibilityElement(children: .contain)
            .accessibilityLabel("\(row.name) settings")
        }
    }
}

/// The card's ground: an opaque surface (the menu's material is see-through, so the table showed through the card on
/// screen), rounded, with a hairline border and a soft shadow so it reads as floating over the table. It takes every click inside the card that no
/// control takes, so nothing under the card is reached through it.
struct SettingsCardMaterial: NSViewRepresentable {
    var dark: Bool
    func makeNSView(context: Context) -> SettingsCardBackground { SettingsCardBackground() }
    func updateNSView(_ v: SettingsCardBackground, context: Context) { v.dark = dark }
}

final class SettingsCardBackground: NSView {
    let effect = NSView()
    var dark = true { didSet { if dark != oldValue { needsLayout = true } } }
    override var isFlipped: Bool { true }
    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.masksToBounds = false
        layer?.shadowColor = NSColor.black.cgColor
        layer?.shadowOpacity = 0.28
        layer?.shadowRadius = 9
        layer?.shadowOffset = CGSize(width: 0, height: -3)
        effect.wantsLayer = true
        effect.layer?.cornerRadius = SettingsCard.corner
        effect.layer?.cornerCurve = .continuous
        effect.layer?.masksToBounds = true
        effect.layer?.borderWidth = 1
        addSubview(effect)
    }
    required init?(coder: NSCoder) { nil }
    override func layout() {
        super.layout()
        effect.frame = bounds
        layer?.shadowPath = CGPath(roundedRect: bounds, cornerWidth: SettingsCard.corner, cornerHeight: SettingsCard.corner, transform: nil)
        effect.layer?.backgroundColor = (dark ? SettingsCard.groundDark : SettingsCard.groundLight).cgColor
        effect.layer?.borderColor = (dark ? NSColor.white.withAlphaComponent(0.16) : NSColor.black.withAlphaComponent(0.14)).cgColor
    }
    /// Never the hit view: SwiftUI placed this ground above the card's own controls in hit-testing order, so it took
    /// their clicks and hovers (HoverReachTests). The card's content shape takes the clicks no control takes instead
    /// (SettingsCardView), so nothing under the card is reached through it.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func mouseDown(with event: NSEvent) {}
    override func mouseUp(with event: NSEvent) {}
    override func rightMouseDown(with event: NSEvent) {}
}
