// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore
import SwiftUI

/// A transparent, vibrant host lets NSMenu supply its own material and shadow (Vella's MenuTableHostingView). It also
/// closes a model's settings card for a click elsewhere in the table (the click then goes on to its own target) and for
/// Esc, without ending the menu.
final class MenuTableHostingView: MenuContentView<ModelsMenuView> {
    override var allowsVibrancy: Bool { true }
    weak var card: SettingsCard?
    /// The event being dispatched (tests supply their own).
    var currentEvent: () -> NSEvent? = { NSApp.currentEvent }

    /// AppKit finds a click's target here before delivering it: a mouse-down outside the open card and off every gear
    /// closes the card first. The target is unchanged, so the click does its own job.
    override func hitTest(_ point: NSPoint) -> NSView? {
        let target = super.hitTest(point)
        if let card, card.openID != nil, let event = currentEvent(), [.leftMouseDown, .rightMouseDown, .otherMouseDown].contains(event.type) {
            let local = hosting.convert(point, from: superview)
            let s = max(contentScale, 0.01)
            card.tableClick(at: CGPoint(x: local.x / s, y: local.y / s), target: target, event: event)
        }
        return target
    }
    /// Esc closes the card, not the menu.
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        if event.type == .keyDown, event.keyCode == 53, card?.escape() == true { return true }
        return super.performKeyEquivalent(with: event)
    }
    override func cancelOperation(_ sender: Any?) {
        if card?.escape() == true { return }
        super.cancelOperation(sender)
    }
}

/// The Models submenu's content: the table alone, on the menu's own material (no fill of its own).
struct ModelsMenuView: View {
    @ObservedObject var controller: TableController
    @MainActor static func height(_ c: TableController) -> CGFloat { ModelTable.height(c) }
    @MainActor static func width(_ c: TableController) -> CGFloat { ModelTable.width(c) }

    var body: some View {
        ModelTable(controller: controller)
            .frame(width: Self.width(controller), height: Self.height(controller), alignment: .topLeading)
    }
}

/// The Models item holds the table alone; its allocation can differ from the table's natural size.
@MainActor final class ModelsMenu: NSObject, NSMenuDelegate {
    let controller: TableController
    private weak var tableMenu: NSMenu?
    private weak var tableView: MenuTableHostingView?
    var diagnostics: MenuDiagnostics?
    /// Window presenters are injectable; tests verify dismissal before presentation without opening UI.
    var presentAlert: (NSAlert) -> NSApplication.ModalResponse = {
        NSApp.activate(ignoringOtherApps: true)
        return $0.runModal()
    }

    /// `presentAlert` for a question about something that can go while it is up: once `stands` fails, the alert closes
    /// as if cancelled (a stale "Delete …?" never waits for a click, and never holds up Quit).
    func ask(_ alert: NSAlert, while stands: @escaping @MainActor () -> Bool) -> NSApplication.ModalResponse {
        let watch = StaleQuestion(
            controller.objectWillChange, stands: stands,
            close: { [weak alert] in
                guard let window = alert?.window, NSApp.modalWindow === window else { return }
                AppModal.cancelCurrent()
            })
        defer { watch.end() }
        return presentAlert(alert)
    }

    /// Shows an open panel outside menu tracking for a path setting (tests answer without a window).
    var presentOpenPanel: (NSOpenPanel) -> String? = {
        NSApp.activate(ignoringOtherApps: true)
        return $0.runModal() == .OK ? $0.url?.path : nil
    }

    private var appearanceObservation: AnyCancellable?
    /// Re-reads model folders, local models and config (set by the app; the table's header Refresh calls it).
    var refreshModels: () -> Void = {}

    init(controller: TableController) {
        self.controller = controller
        super.init()
        appearanceObservation = Appearance.shared.$mode.removeDuplicates().sink { [weak self] _ in
            // @Published emits before mode is assigned: apply the native state after the palette changes.
            DispatchQueue.main.async { self?.refreshAppearance() }
        }

        controller.confirmRemove = { [weak self] row in
            guard let self else { return .cancel }
            self.dismiss(.removeConfirmation)
            switch self.presentAlert(Self.removeAlert(row)) {
            case .alertSecondButtonReturn: return .deleteWeights
            default: return .cancel
            }
        }
        // Delete: the model's weights and its table row, one confirmation with the size (Table/DeleteConfirm.swift).
        controller.confirmDeleteWeights = { [weak self] row, bytes in
            guard let self else { return false }
            self.dismiss(.removeConfirmation)
            let copy = DeleteConfirm.copy, alert = DeleteConfirm.alert(row, bytes: bytes)
            return self.ask(alert) { [weak self] in self?.controller.deleteQuestionStands(row, copy: copy) ?? false } == .alertSecondButtonReturn
        }
        controller.confirmCode = { [weak self] row in
            guard let self else { return false }
            self.dismiss(.codeConfirmation)
            return self.presentAlert(Self.codeAlert(row)) == .alertSecondButtonReturn
        }
        controller.pickFile = { [weak self] _, s in
            guard let self else { return nil }
            self.dismiss(.filePicker)
            let panel = NSOpenPanel()
            panel.canChooseDirectories = false; panel.allowsMultipleSelection = false; panel.message = s.title
            return self.presentOpenPanel(panel)
        }
        controller.refreshModels = { [weak self] in self?.refreshModels() }
        controller.openBrowser = { [weak self] in
            self?.dismiss(.modelBrowser)
            self?.showBrowser()
        }
        controller.confirmGet = { [weak self] row, plan in
            guard let self else { return false }
            self.dismiss(.getConfirmation)
            return self.presentAlert(Self.getAlert(row, plan: plan)) == .alertFirstButtonReturn
        }
        controller.unknownGetSize = { [weak self] row, plan in
            guard let self else { return }
            self.dismiss(.unknownGetSize)
            _ = self.presentAlert(Self.unknownSizeAlert(row, plan: plan))
        }
    }

    /// An alert, an open panel or a window takes the menu's place: the Models menu ends first, recording why (its card
    /// closes with it).
    func dismiss(_ site: MenuCancellationSite) {
        tableMenu?.cancelTrackingFrom(site)
    }

    /// Dispatch the repair named by the tool item, never the table's state-dependent toggle
    /// (a resident row's toggle is Unload even when its folder needs approval or setup).
    func prepareTool(_ row: ModelRow, action: String) {
        // The precision the table shows: a saved 8-bit preference is the 16-bit copy now, never int8.
        let dtype = PrecisionColumn.offered(row.selectedPrecision, row)
        guard controller.admitting, ![.downloading, .deriving, .loading].contains(row.states[dtype]?.state ?? .notInstalled), ["Get", "Load"].contains(action)
        else { return }
        // One Get, Load or delete of a model at a time: a second repair while one runs does nothing.
        guard controller.claimWork(row.id) else { return }
        let service = controller.service
        func approve(_ plan: GetPlan, revised: Bool) -> Bool {
            guard plan.totalDownloadBytes != nil else { controller.unknownGetSize(row, plan); return false }
            return revised || TableController.asks(plan) ? controller.confirmGet(row, plan) : true
        }
        Task {
            defer { controller.releaseWork(row.id) }
            do {
                var plan: GetPlan?
                if action == "Get" {
                    plan = try await service.getPlan(id: row.id, dtype: dtype)
                    guard let plan, approve(plan, revised: false) else { return }
                }
                if row.needsCodeConsent {
                    guard let digest = row.contentDigest else { throw WorkerFailure(code: "code_consent", message: "Its folder could not be read, so nothing was run.") }
                    guard controller.confirmCode(row) else { return }
                    try await service.approveLocalCode(id: row.id, contentDigest: digest, approvedBy: .app)
                }
                if var approved = plan {
                    for attempt in 0..<3 {
                        do {
                            try await service.get(id: row.id, precision: dtype, approvedPlan: approved)
                            break
                        } catch let changed as GetPlanChanged {
                            guard attempt < 2 else { throw WorkerFailure(code: "get_plan_changed", message: "What Get needs kept changing; try again.") }
                            approved = changed.revisedPlan
                            guard approve(approved, revised: true) else { return }
                        }
                    }
                } else {
                    controller.memoryNotice = await service.loadMemoryNotice(id: row.id, precision: dtype).map { "\(row.name): \($0)" }
                    try await MemoryLoadPresenter.perform { force in
                        try await service.load(id: row.id, precision: dtype, fast: row.fast, force: force)
                    }
                }
            } catch is CancellationError {
            } catch is HFGateError {
                // A gated model: its row says what to do and counts down to the retry; the pages are open.
            } catch { controller.lastError = "\(row.name): \(TableController.message(error))" }
            await controller.refresh()
        }
    }

    /// Opens the model browser window (set by the app).
    var showBrowser: () -> Void = {}

    /// Remove with weights on this Mac: keep them (Add brings the row back) or delete them too.
    static func removeAlert(_ row: ModelRow) -> NSAlert {
        let alert = NSAlert(); alert.alertStyle = .warning
        alert.messageText = "Delete \(row.name) from this Mac?"
        let size = row.sizeBytes.map { " (\(formatBytes($0)))" } ?? ""
        alert.informativeText = "Its weights\(size) are deleted and it leaves the table; Add downloads it again."
        alert.addButton(withTitle: "Cancel")
        let delete = alert.addButton(withTitle: "Delete"); delete.hasDestructiveAction = true
        return alert
    }

    /// Once per local folder content, before its unreviewed code first runs (CONTRACTS §15, local folders).
    static func codeAlert(_ row: ModelRow) -> NSAlert {
        let alert = NSAlert(); alert.alertStyle = .warning
        alert.messageText = "Run \(row.name)\u{2019}s own code?"
        // Folder-local executables and kernels run with the user's permissions.
        let code =
            row.runnerLanguage.lowercased() == "swift"
            ? "any Swift executable or Metal kernels supplied by this folder were not reviewed or signed by Arsenal, and run with your permissions"
            : "its \(runnerWord(row.runnerLanguage)) runner was not reviewed or signed by Arsenal, and it runs with your permissions"
        // A changed folder the owner approved before: the new digest beside the old, and the owner's yes is all it takes.
        let contents = row.contentDigest.map { "Contents \($0.prefix(8))" } ?? "Contents unreadable"
        let changed = row.approval.flatMap { $0.digest != row.contentDigest ? " Its code changed since you approved contents \($0.digest.prefix(8))." : nil } ?? ""
        alert.informativeText = "\(row.name) is a local model: \(code).\(changed)\n\n\(contents). Arsenal asks again if the folder changes."
        alert.addButton(withTitle: "Cancel"); alert.addButton(withTitle: "Run")
        return alert
    }
    static func runnerWord(_ language: String) -> String {
        switch language.lowercased() {
        case "swift": return "Swift"
        case "": return "own"
        default: return language.prefix(1).uppercased() + language.dropFirst()
        }
    }

    /// Before a Get: one plain sentence, then what will be downloaded with its size (v2.2 item 6). Get is the default
    /// button, Cancel (Esc) the other. A plan of weights alone (a large one) says where they come from instead
    /// (`weightsProvenance`).
    static func getAlert(_ row: ModelRow, plan: GetPlan) -> NSAlert {
        let alert = NSAlert()
        let lines = TableController.downloads(plan)
        let total = plan.totalDownloadBytes.map(formatBytes) ?? "size unknown"
        alert.messageText = "Get \(row.name) \u{00b7} \(total)?"
        if lines.contains(where: { $0.kind != "weights" }) {
            let list = lines.map { "\u{2022} \(plainName($0.name)) \u{00b7} \($0.downloadBytes.map(formatBytes) ?? "size unknown")" }
            alert.informativeText = (["To run it, Arsenal downloads and sets up:"] + list).joined(separator: "\n")
        } else {
            alert.informativeText = weightsProvenance(row, plan: plan)
        }
        alert.addButton(withTitle: "Get"); alert.addButton(withTitle: "Cancel")
        return alert
    }
    /// A plan with a part of unknown size: Arsenal cannot bound the download, so it does not Get. Cancel is the only
    /// button; the list says which part is unknown.
    static func unknownSizeAlert(_ row: ModelRow, plan: GetPlan) -> NSAlert {
        let alert = NSAlert(); alert.alertStyle = .warning
        alert.messageText = "Arsenal can\u{2019}t tell how much \(row.name) needs to download"
        let list = TableController.downloads(plan).map { "\u{2022} \(plainName($0.name)) \u{00b7} \($0.downloadBytes.map(formatBytes) ?? "size unknown")" }
        alert.informativeText = (list + ["Nothing was downloaded."]).joined(separator: "\n")
        alert.addButton(withTitle: "Cancel")
        return alert
    }
    /// Where a weights-only Get's bytes come from, only as far as the row and plan say: who pinned the revision (the
    /// signed catalog, or a local folder Arsenal did not review), the part a copy already on this Mac covers, and what
    /// the format's source makes of them (a vendor format is downloaded ready-made; a derived one is made at Get; a
    /// runner-native one at Load).
    static func weightsProvenance(_ row: ModelRow, plan: GetPlan) -> String {
        let pin = row.local ? "pinned to the revision its local folder names (Arsenal did not review it)" : "pinned to the revision Arsenal reviewed"
        var text = "The weights come from Hugging Face, \(pin), and are checked by hash."
        let local = plan.items.filter { $0.kind == "weights" }.reduce(Int64(0)) { $0 + $1.localCopyBytes }
        if local > 0 { text += " \(formatBytes(local)) of them is taken from a copy already on this Mac." }
        switch row.formats.first(where: { $0.id == plan.dtype })?.source {
        case .derived?: text += " \(plan.dtype) is then made from them on this Mac."
        case .runnerNative?: text += " The runner turns them into \(plan.dtype) when it loads."
        case .vendor?, nil: break
        }
        return text + " Arsenal keeps them until you delete them."
    }

    /// Long library lists are counted instead of spelling out every name.
    static func plainName(_ name: String) -> String {
        guard let and = name.range(of: " and ") else { return name }
        let parts = name[and.upperBound...].split(separator: ",")
        return parts.count > 3 ? "\(name[..<and.lowerBound]) and \(parts.count) libraries" : name
    }

    /// The Models menu closing takes the settings card along.
    func menuDidClose(_ menu: NSMenu) {
        controller.settingsCard.menuClosed()
        controller.discardPreviews()
    }

    /// The table's height can change while the menu is open (a group appears): keep the item view in step.
    func menuWillOpen(_ menu: NSMenu) {
        resize()
        refreshAppearance()
        tableView?.recordAfterLayout()
    }
    private func refreshAppearance() {
        tableMenu?.appearance = Appearance.shared.nsAppearance
        tableView?.appearance = Appearance.shared.nsAppearance
    }

    /// Refresh models (the table's top-right icon): off while one runs (or Arsenal quits); a refusal shows in the table's footer
    /// line. The most recent problem wins: a new refusal replaces an older action error there, and a later error replaces
    /// the refusal (the footer shows `lastError` first).
    func updateRefresh(enabled: Bool, notice: String?) {
        guard controller.refreshEnabled != enabled || controller.refreshNotice != notice else { return }
        if controller.refreshEnabled != enabled { controller.refreshEnabled = enabled }
        if let notice, notice != controller.refreshNotice { controller.lastError = nil }
        if controller.refreshNotice != notice { controller.refreshNotice = notice }
        resize()
    }
    func resize() {
        guard let view = tableView else { return }
        // The controller already knows the table's outer dimensions. Allocate them before its first frame/hit test,
        // rather than briefly fitting populated content to the empty table's old allocation. Geometry measurements
        // still follow view-local changes, including settings controls, through the existing hook.
        let size = NSSize(width: ModelsMenuView.width(controller), height: ModelsMenuView.height(controller))
        view.setNaturalSize(size)
        tableMenu?.minimumWidth = size.width
        view.needsLayout = true
        view.layoutSubtreeIfNeeded()
        view.recordAfterLayout()
    }

    /// The Models… item, made once: the status menu is rebuilt on every change, and a new hosting view would rebuild
    /// the whole SwiftUI table on the next open. The one view follows the controller instead.
    private var root: NSMenuItem?
    func modelItem() -> NSMenuItem {
        if let root { return root }
        let root = NSMenuItem(title: "Models", action: nil, keyEquivalent: "")
        self.root = root
        root.image = NSImage(systemSymbolName: "square.stack.3d.up", accessibilityDescription: nil)
        root.showMenuImage()
        let menu = NSMenu(); menu.autoenablesItems = false; menu.delegate = self; tableMenu = menu
        let item = NSMenuItem()
        let view = MenuTableHostingView(rootView: ModelsMenuView(controller: controller), tracksContentSize: true)
        view.card = controller.settingsCard
        // Every cause of a content-size change uses the same geometry hook, including view-local settings state.
        view.onNaturalSizeChange = { [weak menu] size in menu?.minimumWidth = size.width }
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.clear.cgColor
        view.layer?.isOpaque = false
        // Request natural size; fit to the native visible allocation once attached.
        view.setNaturalSize(NSSize(width: ModelsMenuView.width(controller), height: ModelsMenuView.height(controller)))
        view.onOpenLayout = { [weak self, weak menu] view in
            guard let menu else { return }
            self?.diagnostics?.record(menu: menu, name: .models, view: view, content: view.hosting, scale: view.contentScale)
        }
        menu.minimumWidth = view.frame.width
        item.view = view; menu.addItem(item); root.submenu = menu
        tableView = view
        refreshAppearance()
        return root
    }
}
