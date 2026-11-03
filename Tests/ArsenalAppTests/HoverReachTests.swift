// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// What the view under a point offers AppKit's tooltip machinery, read from AppKit's and SwiftUI's own records, never
/// from an owner callback with an invented tag:
/// - the view AppKit's hit test reaches (AppKit's tooltip manager works only for the view under the pointer);
/// - its `toolTip`, which AppKit registers for the view's bounds; or
/// - a tooltip region it registered with `addToolTip` (`ToolTipRegions`: real tag, rect and user data) that contains the
///   point, answered by its owner with that tag; or
/// - SwiftUI's `.help`, from the host's own dynamic-tooltip lookup (`dynamicToolTipStringAtPoint:trackingRect:`), the
///   query AppKit makes for SwiftUI hovers.
/// AppKit shows no tooltip for a disabled control (the rule this fix relies on, assumed here, not observed). That the
/// tooltip window then appears during live menu tracking is not observed by these offscreen tests: unverified.
@MainActor enum Hover {
    /// The hover text the hit view offers at a point in the window (window coordinates), or nil.
    static func offered(_ window: NSWindow, at point: NSPoint) -> String? {
        guard let root = window.contentView?.superview, let hit = root.hitTest(root.convert(point, from: nil)) else { return nil }
        let local = hit.convert(point, from: nil)
        if let regions = hit as? ToolTipRegions {
            guard let owner = hit as? NSViewToolTipOwner,
                let (tag, region) = regions.toolTipRegions.filter({ $0.value.rect.contains(local) && !$0.value.rect.isEmpty })
                    .min(by: { $0.value.rect.width * $0.value.rect.height < $1.value.rect.width * $1.value.rect.height })
            else { return nil }
            let text = owner.view(hit, stringForToolTip: tag, point: local, userData: region.userData)
            return text.isEmpty ? nil : text
        }
        if let control = hit as? NSSegmentedControl, let cell = control.cell as? NSSegmentedCell {
            let pitch = control.bounds.width / CGFloat(max(1, control.segmentCount))
            let i = Int(local.x / pitch)
            guard control.isEnabled, (0..<control.segmentCount).contains(i), cell.isEnabled(forSegment: i) else { return nil }
            return control.toolTip(forSegment: i)
        }
        if let control = hit as? NSControl, !control.isEnabled { return nil }
        if let tip = hit.toolTip, !tip.isEmpty { return tip }
        return swiftUIHelp(hit, at: local)
    }

    /// SwiftUI's own hover at a point in a host (its coordinates), through the host's dynamic-tooltip manager.
    static func swiftUIHelp(_ host: NSView, at point: NSPoint) -> String? {
        let getter = NSSelectorFromString("_dynamicToolTipManager")
        guard host.responds(to: getter), let manager = host.perform(getter)?.takeUnretainedValue() else { return nil }
        let selector = NSSelectorFromString("dynamicToolTipStringAtPoint:trackingRect:")
        guard manager.responds(to: selector), let imp = class_getMethodImplementation(object_getClass(manager), selector) else { return nil }
        typealias Lookup = @convention(c) (AnyObject, Selector, NSPoint, UnsafeMutablePointer<NSRect>) -> AnyObject?
        var rect = NSRect.zero
        let text = unsafeBitCast(imp, to: Lookup.self)(manager, selector, point, &rect).map { "\($0)" }
        return text?.isEmpty == false ? text : nil
    }

    static func views(_ root: NSView) -> [NSView] { [root] + root.subviews.flatMap(views) }

    /// The class name of the view AppKit's hit test reaches at a window point.
    /// Whether the view AppKit's hit test reaches at a window point is inside a scroll view.
    static func scrolled(_ window: NSWindow, at point: NSPoint) -> Bool {
        guard let root = window.contentView?.superview, let hit = root.hitTest(root.convert(point, from: nil)) else { return false }
        return hit.enclosingScrollView != nil
    }
    static func hitName(_ window: NSWindow, at point: NSPoint) -> String {
        guard let root = window.contentView?.superview, let hit = root.hitTest(root.convert(point, from: nil)) else { return "nothing" }
        return String(describing: type(of: hit))
    }

    /// Every hover under `root` that is not offered at its own centre: each view's `toolTip` and each segment's tooltip
    /// must be what the hit view offers there (`skip` leaves out views another check owns).
    static func unreachable(_ window: NSWindow, in root: NSView, skip: (NSView) -> Bool = { _ in false }) -> [String] {
        var problems: [String] = []
        for view in views(root) where !view.isHiddenOrHasHiddenAncestor && !skip(view) {
            let visible = view.visibleRect
            guard visible.width > 2, visible.height > 2 else { continue }
            if let control = view as? NSSegmentedControl, !control.isEnabled {
                // Locked: AppKit shows none of its own tooltips, so a hover over it must still say something.
                let centre = control.convert(NSPoint(x: control.bounds.midX, y: control.bounds.midY), to: nil)
                if offered(window, at: centre) == nil { problems.append("locked \(type(of: control)) offers nothing") }
                continue
            }
            if let control = view as? NSSegmentedControl {
                let pitch = control.bounds.width / CGFloat(max(1, control.segmentCount))
                for i in 0..<control.segmentCount {
                    guard let tip = control.toolTip(forSegment: i), !tip.isEmpty else { continue }
                    let rect = NSRect(x: CGFloat(i) * pitch, y: control.bounds.minY, width: pitch, height: control.bounds.height).intersection(visible)
                    guard !rect.isEmpty else { continue }
                    let got = offered(window, at: control.convert(NSPoint(x: rect.midX, y: rect.midY), to: nil))
                    if got != tip { problems.append("segment \(i) “\(tip.prefix(40))”: offers \(got.map { "“\($0.prefix(40))”" } ?? "nothing")") }
                }
                continue
            }
            guard let tip = view.toolTip, !tip.isEmpty else { continue }
            let got = offered(window, at: view.convert(NSPoint(x: view.bounds.midX, y: view.bounds.midY), to: nil))
            if got != tip {
                let p = view.convert(NSPoint(x: view.bounds.midX, y: view.bounds.midY), to: nil)
                let hit = window.contentView?.superview.flatMap { $0.hitTest($0.convert(p, from: nil)) }
                let at = hit.map { $0.convert(p, from: nil) } ?? .zero, own = view.convert(view.bounds, to: hit)
                let hitName = (hit.map { String(describing: type(of: $0)).prefix(50) } ?? "nil") + " at \(at) own \(own)"
                problems.append("\(type(of: view)) “\(tip.prefix(40))”: offers \(got.map { "“\($0.prefix(40))”" } ?? "nothing") [hit \(hitName)]")
            }
        }
        return problems
    }
}

/// "there is no information on hover." Every hover in Arsenal's other surfaces on the real event path: the
/// Activity panel (in the menu), the settings card, Get more models and the tool windows, built at their real size in
/// never-shown windows far off every screen.
@MainActor @Suite(.menuState, .serialized) final class HoverReachTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
    }
    /// Hovers whose display stays unverified (see `expectReachable`), by window, for the report.
    static var unverified: [String: [String]] = [:]
    /// Unverified hovers inside a ScrollView's document, each answered by its host's own lookup at its centre.
    static var scrolled: Set<String> = []
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-hover-\(UUID().uuidString)", isDirectory: true)
    deinit { try? FileManager.default.removeItem(at: scratch) }

    private func window(for view: NSView) -> NSWindow {
        let w = NSWindow(contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: view.frame.size), styleMask: .borderless, backing: .buffered, defer: false)
        w.isReleasedWhenClosed = false; w.contentView = view; w.appearance = Appearance.shared.nsAppearance
        view.layoutSubtreeIfNeeded()
        return w
    }
    private func open<V: View>(_ content: V, size: NSSize) async throws -> (ToolWindowController, NSWindow) {
        let wc = ToolWindowController(title: "test", autosave: "", size: size, minSize: size, content: content)
        let w = try #require(wc.window)
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
        w.orderFrontRegardless()
        w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        w.contentView?.layoutSubtreeIfNeeded(); w.displayIfNeeded()
        try await Task.sleep(for: .milliseconds(150))
        return (wc, w)
    }
    private func expectReachable(_ w: NSWindow, _ name: String) throws {
        var problems = Hover.unreachable(w, in: try #require(w.contentView))
        // SwiftUI's own hovers (`windowHelp`): at each one's centre AppKit's hit test must reach its SwiftUI host and the
        // host's own dynamic-tooltip lookup must answer that text (a removed `.help`, or a view laid over it, fails).
        if let wc = w.windowController as? ToolWindowController, let height = w.contentView?.bounds.height {
            let helps = ControlFrames.helps[wc.scope] ?? [:]
            var unverified: [String] = []
            for (text, frame) in helps where frame.width > 2 && frame.height > 2 {
                let point = NSPoint(x: frame.midX, y: height - frame.midY)
                let got = Hover.offered(w, at: point)
                // Known gap, reported rather than hidden: SwiftUI's own container is the hit view over a drop well and over
                // everything in a ScrollView's document (an options column), and whether AppKit then shows the host's
                // SwiftUI hover is not observable here. The host's own lookup must still answer this text at this point.
                if got == nil, Hover.hitName(w, at: point) == "PlatformGroupContainer" {
                    let host = try #require(w.contentView)
                    let answered = Hover.swiftUIHelp(host, at: host.convert(point, from: nil))
                    if answered != text {
                        problems.append("help “\(text.prefix(40))”: its host offers \(answered.map { "“\($0.prefix(40))”" } ?? "nothing") under SwiftUI’s container")
                        continue
                    }
                    if Hover.scrolled(w, at: point) { Self.scrolled.insert(text) }
                    unverified.append(text); continue
                }
                if got != text { problems.append("help “\(text.prefix(40))”: offers \(got.map { "“\($0.prefix(40))”" } ?? "nothing")") }
            }
            if helps.isEmpty { problems.append("no window hovers recorded") }
            if !unverified.isEmpty { Self.unverified[name, default: []].append(contentsOf: unverified) }
        }
        #expect(problems.isEmpty, Comment(rawValue: "\(name): \(problems.count) hovers not offered\n" + problems.joined(separator: "\n")))
    }

    /// The Activity panel in its submenu, with a busy feed (models, requests, the machine strip).
    @Test func activityPanelHovers() async throws {
        let model = ActivityModel()
        model.preview(V3RenderDelegate.busyFeed())
        let view = ActivityHostingView(rootView: ActivityPanel(model: model))
        view.setNaturalSize(NSSize(width: ActivityPanel.width, height: ActivityPanel.minHeight))
        view.setNaturalSize(NSSize(width: ActivityPanel.width, height: ActivityMenu.height(view)))
        let w = window(for: view); defer { w.close() }
        try await Task.sleep(for: .milliseconds(100))
        view.layoutSubtreeIfNeeded()
        #expect(Hover.views(view).contains { $0.toolTip != nil }, "the panel has hovers")
        try expectReachable(w, "Activity")
    }

    /// Each model's settings card in a table of six, idle and while a Get runs (load-time rows locked, saying why).
    @Test func settingsCardHovers() async throws {
        for state in [ModelState.ready, .downloading] {
            let ids = SettingsCardTests.catalogTables.first ?? []
            let open = try await SettingsCardTests.open(ids, support: scratch.appendingPathComponent("\(state)"), state: state)
            defer { open.close() }
            let window = try #require(open.host.window)
            for id in ids {
                try open.host.press(try #require(open.host.gear(id)))
                await open.host.settle()
                // The card's own views only: the platform views SwiftUI stacks above the card's ground (the table under
                // the card, which it covers, has TableHoverTests).
                let hosting = open.host.view.hosting
                let ground = try #require(hosting.subviews.firstIndex { Hover.views($0).contains { $0 is SettingsCardBackground } })
                let problems = Hover.unreachable(window, in: open.host.view) { view in
                    var top: NSView = view
                    while let parent = top.superview, parent !== hosting { top = parent }
                    return top.superview !== hosting || (hosting.subviews.firstIndex(of: top) ?? 0) <= ground
                }
                #expect(problems.isEmpty, Comment(rawValue: "\(id) (\(state)): \(problems.count) hovers not offered\n" + problems.joined(separator: "\n")))
                open.app.table.settingsCard.close()
                await open.host.settle()
            }
        }
    }

    /// Get more models with two expanded rows (runner cards).
    @Test func getMoreModelsHovers() async throws {
        let rows = Fixtures.figureRows()
        let entries = rows.map { CatalogEntry(row: $0, metadata: ["languages": .array([.string("en")])], added: !$0.states.isEmpty) }
        let browser = BrowserController(service: FixtureService(rows: rows, supportDirectory: scratch.appendingPathComponent("browser")), chip: "M5 Max")
        browser.apply(entries: entries, kinds: Fixtures.figureKinds)
        browser.expanded = ["laya-multilingual", "kokoro-82m"]
        let wc = BrowserWindowController(controller: browser)
        let w = try #require(wc.window)
        defer { w.orderOut(nil); w.close() }
        w.setFrame(NSRect(x: -9000, y: -9000, width: 1300, height: 760), display: false)
        wc.list.expansionChanged()
        w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await Task.sleep(for: .milliseconds(400))
        try expectReachable(w, "Get more models")
    }

    /// The tool windows: Chat, Make an Image, Make a Sound, the generic operation tool and Remove Background.
    @Test func toolWindowHovers() async throws {
        let chat = try await open(ChatView(session: ChatSession(model: V3RenderDelegate.qwen, client: { nil })), size: ChatView.size)
        try expectReachable(chat.1, "Chat"); chat.1.close()
        let image = try await open(ImageView(session: ImageSession(model: V3RenderDelegate.flux, client: { nil })), size: ImageView.size)
        try expectReachable(image.1, "Image"); image.1.close()
        let sound = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        sound.prepare()
        let soundWindow = try await open(SoundView(session: sound), size: SoundView.size)
        try expectReachable(soundWindow.1, "Sound"); soundWindow.1.close()
        let operation = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: scratch)
        operation.preview(operations: [LayersFixture.descriptor])
        let op = try await open(OperationView(session: operation, symbol: LayersFixture.kind.icon!), size: OperationView.size)
        try expectReachable(op.1, "Operation"); op.1.close()
        // Remove Background, empty and with a cut-out shown (its Copy, Save and Show in Finder).
        let background = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row))], selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch,
            scratch: scratch)
        background.preview(operations: [CutoutFixture.descriptor])
        let empty = try await open(BackgroundView(session: background), size: BackgroundView.size)
        try expectReachable(empty.1, "Remove Background"); empty.1.close()
        let photo = try BackgroundToolTests.image(scratch.appendingPathComponent("hover/photo.jpg"))
        let reply = try CutoutFixture.run(params: ["image": ["path": .string(photo.path)]], directory: scratch.appendingPathComponent("hover/out"))
        let png = URL(fileURLWithPath: try #require(reply["files"]?.array?.first?["path"]?.string))
        let mask = URL(fileURLWithPath: try #require(reply["files"]?.array?.last?["path"]?.string))
        // With its mask: the Cut-out | Mask pills' hovers too.
        let cutout = Cutout(original: photo, url: png, mask: mask, model: CutoutFixture.id, modelName: "BiRefNet", width: 300, height: 200, seconds: 1.2)
        var earlier = Cutout(original: photo, url: png, mask: nil, model: CutoutFixture.id, modelName: "BiRefNet", width: 300, height: 200, seconds: 1.1)
        earlier.original = scratch.appendingPathComponent("hover/earlier.jpg")
        background.preview(operations: [CutoutFixture.descriptor], input: photo, cutouts: [earlier, cutout], shown: cutout.id)
        let shown = try await open(BackgroundView(session: background), size: BackgroundView.size)
        try expectReachable(shown.1, "Remove Background result"); shown.1.close()
        let wells: Set<String> = ["Drop a file here, or click to choose", BackgroundView.emptyHelp, BackgroundView.chosenHelp]
        #expect(
            Self.unverified.values.flatMap { $0 }.allSatisfy { wells.contains($0) || Self.scrolled.contains($0) },
            "only the drop wells’ and scrolled options’ hovers are unverified: \(Self.unverified)")
    }

    /// No cell in Arsenal carries its hover behind its content any more (TooltipCell's `appKitTooltip`): on this macOS
    /// AppKit's hit test never reaches it there.
    @Test func noHoverSitsBehindItsCell() throws {
        let sources = SettingsRenderDelegate.repo.appendingPathComponent("Sources/Arsenal")
        let files = (FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
        #expect(files.count > 50)
        for file in files where file.lastPathComponent != "TooltipCell.swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            #expect(!text.contains(".appKitTooltip("), "\(file.lastPathComponent) still puts a hover behind its cell")
        }
    }
}
